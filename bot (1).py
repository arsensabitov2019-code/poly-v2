from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime, timezone

import anthropic
import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

TELEGRAM_TOKEN = os.environ["TELEGRAM_TOKEN"]
ANTHROPIC_API_KEY = os.environ["ANTHROPIC_API_KEY"]

anthropic_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

GAMMA_API = "https://gamma-api.polymarket.com"
DATA_API = "https://data-api.polymarket.com"

# Порог "кита" в USD — сделки от этой суммы считаем китовыми
WHALE_THRESHOLD_USD = float(os.environ.get("WHALE_THRESHOLD_USD", "10000"))
# Минимальный размер позиции для топ-холдеров (в USD)
WHALE_POSITION_MIN_USD = float(os.environ.get("WHALE_POSITION_MIN_USD", "5000"))

# ── Polymarket: Gamma (markets) ───────────────────────────────────────────────


async def get_trending_markets(limit: int = 10) -> list:
    url = f"{GAMMA_API}/markets"
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []


async def get_market_by_id(market_id: str) -> dict:
    url = f"{GAMMA_API}/markets/{market_id}"
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url)
        if resp.status_code == 404:
            return {}
        resp.raise_for_status()
        return resp.json()


async def search_markets(query: str, limit: int = 8) -> list:
    url = f"{GAMMA_API}/markets"
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
        "search": query,
    }
    async with httpx.AsyncClient(timeout=20) as client:
        resp = await client.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []


# ── Polymarket: Data API (whales, positions, trades) ──────────────────────────


async def get_market_trades(market_condition_id: str, limit: int = 100) -> list:
    """Получить недавние трейды для рынка по conditionId."""
    url = f"{DATA_API}/trades"
    params = {
        "market": market_condition_id,
        "limit": limit,
        "takerOnly": "true",
    }
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("trades fetch failed: %s", e)
            return []


async def get_market_holders(token_id: str, limit: int = 20) -> list:
    """Получить топ-холдеров для конкретного outcome-токена."""
    url = f"{DATA_API}/holders"
    params = {"market": token_id, "limit": limit}
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                # Иногда обёрнуто в {token, holders:[...]}
                if "holders" in data[0]:
                    return data[0].get("holders") or []
                return data
            return []
        except Exception as e:
            logger.warning("holders fetch failed: %s", e)
            return []


async def get_user_positions(wallet: str, limit: int = 50) -> list:
    """Получить активные позиции конкретного кошелька (кита)."""
    url = f"{DATA_API}/positions"
    params = {
        "user": wallet,
        "limit": limit,
        "sortBy": "CURRENT",
        "sortDirection": "DESC",
    }
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            return data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("positions fetch failed: %s", e)
            return []


# ── Helpers ───────────────────────────────────────────────────────────────────


def _yes_price(market: dict) -> float:
    try:
        prices = market.get("outcomePrices")
        if isinstance(prices, str):
            prices = json.loads(prices)
        if prices:
            return float(prices[0])
    except Exception:
        pass
    return 0.5


def _outcomes(market: dict) -> list:
    try:
        outs = market.get("outcomes")
        if isinstance(outs, str):
            outs = json.loads(outs)
        if outs:
            return outs
    except Exception:
        pass
    return ["Yes", "No"]


def _token_ids(market: dict) -> list:
    """Извлечь clob token IDs из рынка."""
    try:
        toks = market.get("clobTokenIds")
        if isinstance(toks, str):
            toks = json.loads(toks)
        if toks:
            return toks
    except Exception:
        pass
    return []


def _split_message(text: str, limit: int = 4000) -> list:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


def _short_wallet(addr: str) -> str:
    if not addr or len(addr) < 10:
        return addr or "?"
    return f"{addr[:6]}...{addr[-4:]}"


def format_market_list(markets: list) -> tuple:
    if not markets:
        return "Рынки не найдены.", InlineKeyboardMarkup([])

    lines = ["📋 Trending markets on Polymarket:\n"]
    buttons = []

    for i, m in enumerate(markets[:10], 1):
        question = m.get("question", "?")
        yes_p = _yes_price(m)
        vol = float(m.get("volume24hr") or m.get("volume") or 0)
        short_q = question[:55] + ("..." if len(question) > 55 else "")
        lines.append(f"{i}. {short_q}\n   YES: {yes_p:.0%} | Vol: ${vol:,.0f}")
        market_id = str(m.get("id") or m.get("conditionId") or "")
        if market_id:
            buttons.append(
                [
                    InlineKeyboardButton(f"📊 Analyze #{i}", callback_data=f"analyze:{market_id}"),
                    InlineKeyboardButton(f"🐋 Whales #{i}", callback_data=f"whales:{market_id}"),
                ]
            )

    return "\n".join(lines), InlineKeyboardMarkup(buttons)


# ── Whale analytics ───────────────────────────────────────────────────────────


def analyze_whale_trades(trades: list, outcomes: list) -> dict:
    """
    Анализирует список трейдов, выделяет китовые сделки (>= WHALE_THRESHOLD_USD).
    Возвращает агрегированную статистику и список китовых сделок.
    """
    whale_trades = []
    yes_buy_usd = 0.0
    no_buy_usd = 0.0
    total_volume = 0.0

    for t in trades:
        usd = float(t.get("usdcSize") or 0)
        total_volume += usd
        if usd < WHALE_THRESHOLD_USD:
            continue

        side = (t.get("side") or "").upper()
        outcome_idx = int(t.get("outcomeIndex") or 0)
        outcome_name = outcomes[outcome_idx] if outcome_idx < len(outcomes) else f"#{outcome_idx}"
        price = float(t.get("price") or 0)

        # Считаем эффективное направление: BUY YES либо BUY NO
        # SELL YES ~= BUY NO с точки зрения сентимента
        effective_yes = (side == "BUY" and outcome_idx == 0) or (
            side == "SELL" and outcome_idx == 1
        )
        if effective_yes:
            yes_buy_usd += usd
        else:
            no_buy_usd += usd

        whale_trades.append(
            {
                "ts": int(t.get("timestamp") or 0),
                "wallet": t.get("proxyWallet") or "",
                "name": t.get("name") or t.get("pseudonym") or _short_wallet(t.get("proxyWallet") or ""),
                "side": side,
                "outcome": outcome_name,
                "outcome_idx": outcome_idx,
                "price": price,
                "usd": usd,
                "size": float(t.get("size") or 0),
                "effective_yes": effective_yes,
            }
        )

    # Сортируем по размеру убывания
    whale_trades.sort(key=lambda x: x["usd"], reverse=True)

    # Уникальные киты
    unique_whales = len({t["wallet"] for t in whale_trades})

    # Сентимент китов: процент денег в YES
    whale_total = yes_buy_usd + no_buy_usd
    yes_share = (yes_buy_usd / whale_total) if whale_total > 0 else 0.5

    return {
        "whale_trades": whale_trades,
        "whale_count": len(whale_trades),
        "unique_whales": unique_whales,
        "whale_volume_usd": whale_total,
        "total_volume_usd": total_volume,
        "yes_buy_usd": yes_buy_usd,
        "no_buy_usd": no_buy_usd,
        "yes_share": yes_share,
        "whale_share_of_volume": (whale_total / total_volume) if total_volume > 0 else 0,
    }


def analyze_holders(holders_yes: list, holders_no: list, yes_price: float) -> dict:
    """Анализирует крупных холдеров обеих сторон."""
    def _stats(holders, side_price):
        total = 0.0
        big = []
        for h in holders:
            amount = float(h.get("amount") or 0)
            usd_value = amount * side_price
            total += usd_value
            if usd_value >= WHALE_POSITION_MIN_USD:
                big.append(
                    {
                        "wallet": h.get("proxyWallet") or "",
                        "name": h.get("name") or h.get("pseudonym") or _short_wallet(h.get("proxyWallet") or ""),
                        "amount": amount,
                        "usd": usd_value,
                    }
                )
        big.sort(key=lambda x: x["usd"], reverse=True)
        return total, big

    no_price = max(0.001, 1.0 - yes_price)
    yes_total_usd, yes_big = _stats(holders_yes, yes_price)
    no_total_usd, no_big = _stats(holders_no, no_price)

    total = yes_total_usd + no_total_usd
    return {
        "yes_holders_usd": yes_total_usd,
        "no_holders_usd": no_total_usd,
        "yes_share_holders": (yes_total_usd / total) if total > 0 else 0.5,
        "yes_top_whales": yes_big[:5],
        "no_top_whales": no_big[:5],
        "total_top_whales_usd": total,
    }


def calculate_expected_profit(
    bet_usd: float,
    market_price: float,
    true_prob: float,
) -> dict:
    """
    Расчёт ожидаемой прибыли по Kelly и EV.
    market_price — текущая цена YES (0..1).
    true_prob — наша оценка истинной вероятности YES (0..1).
    Возвращает рекомендации для BUY YES и BUY NO.
    """
    def _ev(side_price: float, side_prob: float) -> dict:
        # При покупке за price: выигрыш на $1 ставки = (1/price - 1) при выигрыше, -1 при проигрыше
        if side_price <= 0 or side_price >= 1:
            return {"ev_pct": 0, "kelly": 0, "win_payout": 0, "expected_profit_usd": 0}
        b = (1.0 / side_price) - 1.0  # net odds
        p = side_prob
        q = 1.0 - p
        ev_per_dollar = p * b - q  # ожидаемая прибыль на $1
        # Kelly fraction = (bp - q) / b
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0.0, min(0.25, kelly))  # cap на 25% — quarter Kelly safety
        win_shares = bet_usd / side_price
        return {
            "ev_pct": ev_per_dollar * 100,
            "kelly_pct": kelly * 100,
            "win_payout_usd": win_shares,  # сколько получишь, если выиграл (включая стейк)
            "win_profit_usd": win_shares - bet_usd,
            "expected_profit_usd": ev_per_dollar * bet_usd,
            "win_prob": p,
        }

    yes_calc = _ev(market_price, true_prob)
    no_calc = _ev(1.0 - market_price, 1.0 - true_prob)

    edge_yes = true_prob - market_price
    if edge_yes > 0.03:
        recommendation = "BUY YES"
        primary = yes_calc
    elif edge_yes < -0.03:
        recommendation = "BUY NO"
        primary = no_calc
    else:
        recommendation = "SKIP"
        primary = yes_calc if abs(edge_yes) < 0.001 else (yes_calc if edge_yes > 0 else no_calc)

    return {
        "recommendation": recommendation,
        "edge_pct": edge_yes * 100,
        "yes": yes_calc,
        "no": no_calc,
        "primary": primary,
    }


# ── AI analysis ───────────────────────────────────────────────────────────────


def analyze_market_with_ai(market: dict, whale_data: dict | None = None, holders_data: dict | None = None) -> dict:
    """
    Возвращает dict с полями:
      true_prob (0..1), market_price, edge_pct, recommendation,
      confidence, reasoning_text, full_text
    """
    question = market.get("question", "Unknown market")
    yes_price = _yes_price(market)
    volume = float(market.get("volume") or 0)
    end_date = market.get("endDate", "unknown")

    # Контекст по китам для AI
    whale_context = ""
    if whale_data and whale_data.get("whale_count", 0) > 0:
        wd = whale_data
        whale_context = (
            f"\nWHALE ACTIVITY (last trades, >= ${WHALE_THRESHOLD_USD:,.0f}):\n"
            f"- {wd['whale_count']} whale trades from {wd['unique_whales']} unique wallets\n"
            f"- Total whale volume: ${wd['whale_volume_usd']:,.0f}\n"
            f"- YES side bought: ${wd['yes_buy_usd']:,.0f} ({wd['yes_share']:.0%})\n"
            f"- NO side bought: ${wd['no_buy_usd']:,.0f} ({1-wd['yes_share']:.0%})\n"
            f"- Whales = {wd['whale_share_of_volume']:.0%} of total volume\n"
        )
        # Топ-3 китовых сделки
        for i, t in enumerate(wd["whale_trades"][:3], 1):
            direction = "YES" if t["effective_yes"] else "NO"
            whale_context += (
                f"  Top whale #{i}: ${t['usd']:,.0f} on {direction} "
                f"@ {t['price']:.0%} by {t['name']}\n"
            )

    holders_context = ""
    if holders_data:
        hd = holders_data
        holders_context = (
            f"\nTOP HOLDERS:\n"
            f"- YES side: ${hd['yes_holders_usd']:,.0f} held by top whales\n"
            f"- NO side: ${hd['no_holders_usd']:,.0f} held by top whales\n"
            f"- Big money YES bias: {hd['yes_share_holders']:.0%}\n"
        )

    prompt = (
        "You are a Polymarket prediction-market analyst. Analyze this market using whale activity AND web research.\n\n"
        f"MARKET:\n"
        f"- Question: {question}\n"
        f"- Current YES price: {yes_price:.1%}\n"
        f"- 24h volume: ${volume:,.0f}\n"
        f"- Closes: {end_date}\n"
        f"{whale_context}"
        f"{holders_context}\n"
        "STEPS:\n"
        "1. Use web_search to find recent relevant news/data.\n"
        "2. Weight whale flow heavily — if whales overwhelmingly favor one side AND it aligns with news, signal is strong.\n"
        "3. Estimate the TRUE probability of YES (0-100%).\n"
        "4. Compare with current market price; calculate edge.\n\n"
        "RESPOND in this EXACT format (machine-parseable, no markdown, no extra text):\n"
        "TRUE_PROB: <number 0-100>\n"
        "CONFIDENCE: <Low|Medium|High>\n"
        "WHALE_SIGNAL: <Strong YES|Weak YES|Neutral|Weak NO|Strong NO>\n"
        "RESEARCH: <2-3 sentences on key findings from web>\n"
        "REASONING: <2-3 sentences justifying the probability>\n"
        "RISKS: <1-2 key risks>\n"
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1200,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [
            block.text for block in response.content if hasattr(block, "text") and block.text
        ]
        result_text = "\n".join(text_parts).strip()

        # Парсим
        parsed = {
            "true_prob": yes_price,
            "confidence": "Low",
            "whale_signal": "Neutral",
            "research": "",
            "reasoning": "",
            "risks": "",
            "raw": result_text,
        }
        for line in result_text.splitlines():
            line = line.strip()
            if line.startswith("TRUE_PROB:"):
                try:
                    val = line.split(":", 1)[1].strip().rstrip("%").strip()
                    parsed["true_prob"] = max(0.01, min(0.99, float(val) / 100))
                except Exception:
                    pass
            elif line.startswith("CONFIDENCE:"):
                parsed["confidence"] = line.split(":", 1)[1].strip()
            elif line.startswith("WHALE_SIGNAL:"):
                parsed["whale_signal"] = line.split(":", 1)[1].strip()
            elif line.startswith("RESEARCH:"):
                parsed["research"] = line.split(":", 1)[1].strip()
            elif line.startswith("REASONING:"):
                parsed["reasoning"] = line.split(":", 1)[1].strip()
            elif line.startswith("RISKS:"):
                parsed["risks"] = line.split(":", 1)[1].strip()
        return parsed
    except Exception as e:
        logger.error("AI analysis error: %s", e)
        return {
            "true_prob": yes_price,
            "confidence": "Low",
            "whale_signal": "Neutral",
            "research": f"AI error: {e}",
            "reasoning": "",
            "risks": "",
            "raw": "",
        }


# ── Full pipeline: market + whales + AI + EV ─────────────────────────────────


async def full_market_analysis(market: dict, bet_size_usd: float = 100.0) -> str:
    """Полный анализ: рынок → киты → холдеры → AI → ожидаемая прибыль."""
    question = market.get("question", "?")
    yes_price = _yes_price(market)
    outcomes = _outcomes(market)
    condition_id = market.get("conditionId") or ""
    token_ids = _token_ids(market)

    # Параллельно тянем китов и холдеров
    tasks = []
    if condition_id:
        tasks.append(get_market_trades(condition_id, limit=200))
    else:
        tasks.append(asyncio.sleep(0, result=[]))

    if len(token_ids) >= 2:
        tasks.append(get_market_holders(str(token_ids[0]), limit=20))
        tasks.append(get_market_holders(str(token_ids[1]), limit=20))
    else:
        tasks.append(asyncio.sleep(0, result=[]))
        tasks.append(asyncio.sleep(0, result=[]))

    trades, holders_yes, holders_no = await asyncio.gather(*tasks, return_exceptions=False)

    whale_data = analyze_whale_trades(trades or [], outcomes)
    holders_data = analyze_holders(holders_yes or [], holders_no or [], yes_price)

    # AI с контекстом китов
    ai = analyze_market_with_ai(market, whale_data, holders_data)
    true_prob = ai["true_prob"]

    # EV расчёт
    ev = calculate_expected_profit(bet_size_usd, yes_price, true_prob)

    # Форматирование ответа
    lines = []
    lines.append(f"📊 АНАЛИЗ РЫНКА")
    lines.append(f"━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"❓ {question[:120]}")
    lines.append("")
    lines.append(f"💰 Цена YES: {yes_price:.1%}")
    lines.append(f"🎯 Истинная вер. (AI): {true_prob:.1%}")
    lines.append(f"📈 Edge: {ev['edge_pct']:+.1f}%")
    lines.append(f"🎓 Уверенность: {ai['confidence']}")
    lines.append("")

    # Киты
    if whale_data["whale_count"] > 0:
        wd = whale_data
        lines.append(f"🐋 КИТЫ (сделки ≥ ${WHALE_THRESHOLD_USD:,.0f})")
        lines.append(f"  Сделок: {wd['whale_count']} от {wd['unique_whales']} кошельков")
        lines.append(f"  Объём: ${wd['whale_volume_usd']:,.0f} ({wd['whale_share_of_volume']:.0%} рынка)")
        lines.append(f"  YES: ${wd['yes_buy_usd']:,.0f} ({wd['yes_share']:.0%})")
        lines.append(f"  NO:  ${wd['no_buy_usd']:,.0f} ({1-wd['yes_share']:.0%})")
        lines.append(f"  📡 Сигнал китов: {ai['whale_signal']}")
        lines.append("")
        lines.append("  Топ-3 китовых сделки:")
        for i, t in enumerate(wd["whale_trades"][:3], 1):
            direction = "YES" if t["effective_yes"] else "NO"
            lines.append(
                f"  {i}. ${t['usd']:,.0f} → {direction} @ {t['price']:.0%} "
                f"({t['name']})"
            )
        lines.append("")
    else:
        lines.append(f"🐋 Крупных сделок (≥ ${WHALE_THRESHOLD_USD:,.0f}) не найдено")
        lines.append("")

    # Холдеры
    hd = holders_data
    if hd["total_top_whales_usd"] > 0:
        lines.append(f"💼 КРУПНЫЕ ПОЗИЦИИ")
        lines.append(f"  YES holders: ${hd['yes_holders_usd']:,.0f}")
        lines.append(f"  NO holders:  ${hd['no_holders_usd']:,.0f}")
        lines.append(f"  Smart money YES bias: {hd['yes_share_holders']:.0%}")
        lines.append("")

    # Рекомендация и прибыль
    lines.append(f"━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🎯 РЕКОМЕНДАЦИЯ: {ev['recommendation']}")
    lines.append("")
    if ev["recommendation"] != "SKIP":
        p = ev["primary"]
        side = "YES" if "YES" in ev["recommendation"] else "NO"
        side_price = yes_price if side == "YES" else 1 - yes_price
        lines.append(f"💵 Если поставить ${bet_size_usd:.0f} на {side} @ {side_price:.0%}:")
        lines.append(f"  • Вер. выигрыша: {p['win_prob']:.0%}")
        lines.append(f"  • Если выиграл: +${p['win_profit_usd']:.2f} (всего ${p['win_payout_usd']:.2f})")
        lines.append(f"  • Если проиграл: -${bet_size_usd:.2f}")
        lines.append(f"  • Ожидаемая прибыль (EV): ${p['expected_profit_usd']:+.2f} ({p['ev_pct']:+.1f}%)")
        lines.append(f"  • Kelly (рекоменд. % банка): {p['kelly_pct']:.1f}%")
    lines.append("")
    if ai["research"]:
        lines.append(f"🔍 Research: {ai['research']}")
    if ai["reasoning"]:
        lines.append(f"💭 Логика: {ai['reasoning']}")
    if ai["risks"]:
        lines.append(f"⚠️ Риски: {ai['risks']}")

    slug = market.get("slug", "")
    if slug:
        lines.append("")
        lines.append(f"🔗 https://polymarket.com/event/{slug}")

    return "\n".join(lines)


# ── Telegram handlers ─────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🤖 Polymarket AI Whale-Tracker\n\n"
        "Анализирую рынки + слежу за китами ($10K+) + даю рекомендации с расчётом прибыли.\n\n"
        "Команды:\n"
        "/markets — топ рынки (с кнопками анализа)\n"
        "/search <запрос> — поиск рынков\n"
        "/analyze <id> — глубокий анализ + киты + EV\n"
        "/whales <id> — только китовая активность по рынку\n"
        "/top — лучшие сделки сейчас (top-5 рынков с EV)\n"
        "/wallet <0x...> — посмотреть позиции конкретного кита\n"
        "/setbet <сумма> — задать размер ставки для расчётов (по умолч. $100)\n\n"
        "💡 Совет: /markets → нажми «🐋 Whales #N» → увидишь, куда несут деньги большие игроки"
    )
    await update.message.reply_text(text)


async def cmd_markets(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Загружаю рынки...")
    try:
        markets = await get_trending_markets(10)
        text, keyboard = format_market_list(markets)
        await msg.edit_text(text, reply_markup=keyboard)
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_search(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(ctx.args or []).strip()
    if not query:
        await update.message.reply_text("Использование: /search bitcoin")
        return
    msg = await update.message.reply_text(f"Ищу: {query}...")
    try:
        markets = await search_markets(query)
        text, keyboard = format_market_list(markets)
        await msg.edit_text(text, reply_markup=keyboard)
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    market_id = " ".join(ctx.args or []).strip()
    if not market_id:
        await update.message.reply_text("Использование: /analyze <market_id>")
        return
    await _run_analysis(update, ctx, market_id)


async def cmd_whales(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    market_id = " ".join(ctx.args or []).strip()
    if not market_id:
        await update.message.reply_text("Использование: /whales <market_id>")
        return
    await _run_whales(update, ctx, market_id)


async def cmd_setbet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    if not ctx.args:
        cur = ctx.user_data.get("bet_size", 100.0)
        await update.message.reply_text(f"Текущий размер ставки: ${cur:.0f}\nИзменить: /setbet 250")
        return
    try:
        amount = float(ctx.args[0])
        if amount <= 0 or amount > 100000:
            raise ValueError("range")
        ctx.user_data["bet_size"] = amount
        await update.message.reply_text(f"✅ Размер ставки установлен: ${amount:.0f}")
    except Exception:
        await update.message.reply_text("Неверная сумма. Пример: /setbet 250")


async def cmd_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    wallet = " ".join(ctx.args or []).strip()
    if not wallet or not wallet.startswith("0x"):
        await update.message.reply_text(
            "Использование: /wallet 0x6af75d4e4aaf700450efbac3708cce1665810ff1"
        )
        return
    msg = await update.message.reply_text("Тяну позиции кита...")
    try:
        positions = await get_user_positions(wallet, limit=20)
        if not positions:
            await msg.edit_text("Нет активных позиций или адрес недоступен.")
            return
        lines = [f"🐋 ПОЗИЦИИ КОШЕЛЬКА {_short_wallet(wallet)}\n"]
        total_value = 0.0
        total_pnl = 0.0
        for i, p in enumerate(positions[:15], 1):
            cur_val = float(p.get("currentValue") or 0)
            pnl = float(p.get("cashPnl") or 0)
            pct = float(p.get("percentPnl") or 0)
            outcome = p.get("outcome") or "?"
            title = (p.get("title") or "?")[:60]
            total_value += cur_val
            total_pnl += pnl
            lines.append(
                f"{i}. {title}\n"
                f"   {outcome} | ${cur_val:,.0f} | PnL: ${pnl:+,.0f} ({pct:+.0f}%)"
            )
        lines.append("")
        lines.append(f"💰 Total value: ${total_value:,.0f}")
        lines.append(f"📊 Total PnL: ${total_pnl:+,.0f}")
        for chunk in _split_message("\n".join(lines)):
            await update.message.reply_text(chunk)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Сканирую топ-рынки + анализирую китов... (~60-90 сек)")
    try:
        markets = await get_trending_markets(8)
        if not markets:
            await msg.edit_text("Рынки не найдены.")
            return
        bet_size = ctx.user_data.get("bet_size", 100.0)

        await msg.edit_text("AI + whale-анализ топ-8 рынков, отбираю лучшие сделки...")

        results = []
        for m in markets:
            try:
                analysis = await full_market_analysis(m, bet_size)
                # Извлекаем edge для сортировки
                edge = 0.0
                ev_value = 0.0
                for line in analysis.splitlines():
                    if "Edge:" in line:
                        try:
                            edge = float(line.split("Edge:")[1].strip().rstrip("%").replace("+", ""))
                        except Exception:
                            pass
                    if "Ожидаемая прибыль (EV):" in line:
                        try:
                            part = line.split("EV):")[1].strip()
                            ev_value = float(part.split(" ")[0].replace("$", "").replace("+", ""))
                        except Exception:
                            pass
                results.append({"text": analysis, "edge": abs(edge), "ev": ev_value})
            except Exception as e:
                logger.error("market analysis error: %s", e)

        # Сортируем по абс. edge (лучшие возможности)
        results.sort(key=lambda x: x["edge"], reverse=True)
        top_picks = results[:5]

        header = (
            f"🎯 ТОП-5 ЛУЧШИХ ВОЗМОЖНОСТЕЙ\n"
            f"(размер ставки: ${bet_size:.0f}, отсортировано по edge)\n\n"
        )
        await update.message.reply_text(header)

        for i, r in enumerate(top_picks, 1):
            text = f"#{i} ━━━━━━━━━━━━━━━━━━━━\n{r['text']}"
            for chunk in _split_message(text):
                await update.message.reply_text(chunk)

        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def callback_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    market_id = query.data.split(":", 1)[1]
    await _run_analysis(update, ctx, market_id, via_callback=True)


async def callback_whales(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    market_id = query.data.split(":", 1)[1]
    await _run_whales(update, ctx, market_id, via_callback=True)


async def _run_analysis(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    market_id: str,
    via_callback: bool = False,
) -> None:
    send = update.callback_query.message if via_callback else update.message
    msg = await send.reply_text("🔬 Полный анализ: рынок → киты → AI → EV... (~30 сек)")
    try:
        market = await get_market_by_id(market_id)
        if not market:
            await msg.edit_text("Рынок не найден.")
            return
        bet_size = ctx.user_data.get("bet_size", 100.0)
        full_text = await full_market_analysis(market, bet_size)
        for chunk in _split_message(full_text):
            await send.reply_text(chunk, disable_web_page_preview=True)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка анализа: {e}")


async def _run_whales(
    update: Update,
    ctx: ContextTypes.DEFAULT_TYPE,
    market_id: str,
    via_callback: bool = False,
) -> None:
    send = update.callback_query.message if via_callback else update.message
    msg = await send.reply_text("🐋 Анализирую активность китов...")
    try:
        market = await get_market_by_id(market_id)
        if not market:
            await msg.edit_text("Рынок не найден.")
            return

        condition_id = market.get("conditionId") or ""
        outcomes = _outcomes(market)
        token_ids = _token_ids(market)
        yes_price = _yes_price(market)

        # Параллельно
        tasks = [
            get_market_trades(condition_id, limit=300) if condition_id else asyncio.sleep(0, result=[]),
        ]
        if len(token_ids) >= 2:
            tasks.append(get_market_holders(str(token_ids[0]), limit=20))
            tasks.append(get_market_holders(str(token_ids[1]), limit=20))
        else:
            tasks.append(asyncio.sleep(0, result=[]))
            tasks.append(asyncio.sleep(0, result=[]))

        trades, hy, hn = await asyncio.gather(*tasks)
        whale_data = analyze_whale_trades(trades or [], outcomes)
        holders_data = analyze_holders(hy or [], hn or [], yes_price)

        lines = [f"🐋 АКТИВНОСТЬ КИТОВ", "━━━━━━━━━━━━━━━━━━━━"]
        lines.append(f"❓ {market.get('question','?')[:120]}")
        lines.append(f"💰 YES: {yes_price:.1%}")
        lines.append("")

        if whale_data["whale_count"] == 0:
            lines.append(f"Нет сделок ≥ ${WHALE_THRESHOLD_USD:,.0f}.")
        else:
            wd = whale_data
            lines.append(f"📊 Сделки ≥ ${WHALE_THRESHOLD_USD:,.0f}:")
            lines.append(f"  Всего: {wd['whale_count']} от {wd['unique_whales']} кошельков")
            lines.append(f"  Объём китов: ${wd['whale_volume_usd']:,.0f} ({wd['whale_share_of_volume']:.0%} рынка)")
            lines.append("")
            lines.append(f"  💚 YES: ${wd['yes_buy_usd']:,.0f} ({wd['yes_share']:.0%})")
            lines.append(f"  ❤️  NO:  ${wd['no_buy_usd']:,.0f} ({1-wd['yes_share']:.0%})")
            lines.append("")
            lines.append("📋 Последние крупные сделки:")
            for i, t in enumerate(wd["whale_trades"][:10], 1):
                direction = "YES" if t["effective_yes"] else "NO"
                ts = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
                lines.append(
                    f"  {i}. ${t['usd']:>7,.0f} → {direction} @ {t['price']:.0%} "
                    f"| {t['name']} | {ts} UTC"
                )

        lines.append("")
        if holders_data["total_top_whales_usd"] > 0:
            hd = holders_data
            lines.append("💼 ТОП-ХОЛДЕРЫ (>$5K позиции):")
            lines.append(f"  YES side: ${hd['yes_holders_usd']:,.0f}")
            for w in hd["yes_top_whales"][:3]:
                lines.append(f"    • {w['name']} — ${w['usd']:,.0f}")
            lines.append(f"  NO side: ${hd['no_holders_usd']:,.0f}")
            for w in hd["no_top_whales"][:3]:
                lines.append(f"    • {w['name']} — ${w['usd']:,.0f}")

        slug = market.get("slug", "")
        if slug:
            lines.append("")
            lines.append(f"🔗 https://polymarket.com/event/{slug}")

        for chunk in _split_message("\n".join(lines)):
            await send.reply_text(chunk, disable_web_page_preview=True)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text.strip()
    msg = await update.message.reply_text("Думаю...")
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=800,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a Polymarket trading assistant. "
                        "Answer concisely in the same language as the user. "
                        "Use web_search for current info.\n\n"
                        f"User: {user_text}"
                    ),
                }
            ],
        )
        text_parts = [
            b.text for b in response.content if hasattr(b, "text") and b.text
        ]
        answer = "\n".join(text_parts).strip() or "Не удалось получить ответ."
        for chunk in _split_message(answer):
            await update.message.reply_text(chunk)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("markets", cmd_markets))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("whales", cmd_whales))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("wallet", cmd_wallet))
    app.add_handler(CommandHandler("setbet", cmd_setbet))
    app.add_handler(CallbackQueryHandler(callback_analyze, pattern=r"^analyze:"))
    app.add_handler(CallbackQueryHandler(callback_whales, pattern=r"^whales:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
