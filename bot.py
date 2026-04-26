from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
from datetime import datetime, timezone
from typing import Optional

import anthropic
import httpx
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
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

WHALE_THRESHOLD_USD = float(os.environ.get("WHALE_THRESHOLD_USD", "10000"))
WHALE_POSITION_MIN_USD = float(os.environ.get("WHALE_POSITION_MIN_USD", "5000"))

AI_CACHE_TTL_MIN = int(os.environ.get("AI_CACHE_TTL_MIN", "30"))
MIN_VOLUME_FOR_AI = float(os.environ.get("MIN_VOLUME_FOR_AI", "5000"))
WEB_SEARCH_VOL_THRESHOLD = float(os.environ.get("WEB_SEARCH_VOL_THRESHOLD", "50000"))
TOP_AI_LIMIT = int(os.environ.get("TOP_AI_LIMIT", "3"))

DB_PATH = os.environ.get("DB_PATH", "/tmp/bot.db")

_db_dir = os.path.dirname(DB_PATH)
if _db_dir:
    os.makedirs(_db_dir, exist_ok=True)

# ── Markdown очистка для Telegram ────────────────────────────────────────────


def clean_markdown(text: str) -> str:
    """Telegram default-режим markdown не рендерит — убираем мусор."""
    if not text:
        return text
    # **bold** → bold
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    # *italic* → italic (только парные одиночные)
    text = re.sub(r"(?<!\*)\*([^\*\n]+?)\*(?!\*)", r"\1", text)
    # `code` → code
    text = re.sub(r"`([^`\n]+?)`", r"\1", text)
    # ### headers → жирная строка просто без решёток
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    return text


# ── Database ──────────────────────────────────────────────────────────────────


def db_init() -> None:
    db_dir = os.path.dirname(DB_PATH)
    if db_dir:
        os.makedirs(db_dir, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            market_id TEXT NOT NULL,
            condition_id TEXT,
            slug TEXT,
            question TEXT,
            market_price REAL,
            true_prob REAL,
            edge_pct REAL,
            recommendation TEXT,
            confidence TEXT,
            whale_signal TEXT,
            bet_size REAL,
            ev_pct REAL,
            created_at INTEGER,
            end_date TEXT,
            resolved INTEGER DEFAULT 0,
            won INTEGER,
            actual_outcome TEXT,
            resolved_at INTEGER
        )
        """
    )
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_market ON predictions(market_id, created_at)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_pred_resolved ON predictions(resolved, end_date)")
    cur.execute(
        """
        CREATE TABLE IF NOT EXISTS ai_cache (
            market_id TEXT PRIMARY KEY,
            true_prob REAL,
            confidence TEXT,
            whale_signal TEXT,
            research TEXT,
            reasoning TEXT,
            risks TEXT,
            cached_at INTEGER
        )
        """
    )
    conn.commit()
    conn.close()


def db_save_prediction(rec: dict) -> Optional[int]:
    if rec.get("recommendation") == "SKIP":
        return None
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO predictions
            (market_id, condition_id, slug, question, market_price, true_prob,
             edge_pct, recommendation, confidence, whale_signal, bet_size, ev_pct,
             created_at, end_date)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                rec.get("market_id"),
                rec.get("condition_id"),
                rec.get("slug"),
                rec.get("question"),
                rec.get("market_price"),
                rec.get("true_prob"),
                rec.get("edge_pct"),
                rec.get("recommendation"),
                rec.get("confidence"),
                rec.get("whale_signal"),
                rec.get("bet_size"),
                rec.get("ev_pct"),
                int(time.time()),
                rec.get("end_date"),
            ),
        )
        pred_id = cur.lastrowid
        conn.commit()
        conn.close()
        return pred_id
    except Exception as e:
        logger.error("db save error: %s", e)
        return None


def db_get_cached_ai(market_id: str) -> Optional[dict]:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cutoff = int(time.time()) - AI_CACHE_TTL_MIN * 60
        cur.execute(
            "SELECT true_prob, confidence, whale_signal, research, reasoning, risks "
            "FROM ai_cache WHERE market_id=? AND cached_at>=?",
            (market_id, cutoff),
        )
        row = cur.fetchone()
        conn.close()
        if not row:
            return None
        return {
            "true_prob": row[0],
            "confidence": row[1],
            "whale_signal": row[2],
            "research": row[3] or "",
            "reasoning": row[4] or "",
            "risks": row[5] or "",
            "from_cache": True,
        }
    except Exception:
        return None


def db_save_ai_cache(market_id: str, ai: dict) -> None:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO ai_cache
            (market_id, true_prob, confidence, whale_signal, research, reasoning, risks, cached_at)
            VALUES (?,?,?,?,?,?,?,?)
            """,
            (
                market_id,
                ai.get("true_prob"),
                ai.get("confidence"),
                ai.get("whale_signal"),
                ai.get("research"),
                ai.get("reasoning"),
                ai.get("risks"),
                int(time.time()),
            ),
        )
        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning("ai cache save error: %s", e)


def db_get_winrate_stats() -> dict:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT COUNT(*), SUM(won), SUM(CASE WHEN won=1 THEN bet_size*((1.0/CASE "
            "WHEN recommendation='BUY YES' THEN market_price "
            "ELSE 1.0-market_price END)-1.0) ELSE -bet_size END) "
            "FROM predictions WHERE resolved=1"
        )
        row = cur.fetchone()
        total, wins, total_pnl = row[0] or 0, row[1] or 0, row[2] or 0.0

        cur.execute("SELECT COUNT(*) FROM predictions WHERE resolved=0")
        pending = cur.fetchone()[0] or 0

        cur.execute(
            "SELECT confidence, COUNT(*), SUM(won) FROM predictions "
            "WHERE resolved=1 GROUP BY confidence"
        )
        by_conf = {r[0]: (r[1] or 0, r[2] or 0) for r in cur.fetchall()}

        cur.execute(
            "SELECT whale_signal, COUNT(*), SUM(won) FROM predictions "
            "WHERE resolved=1 GROUP BY whale_signal"
        )
        by_whale = {r[0]: (r[1] or 0, r[2] or 0) for r in cur.fetchall()}

        cur.execute("SELECT AVG(edge_pct) FROM predictions WHERE resolved=1 AND won=1")
        avg_edge_won = cur.fetchone()[0] or 0
        cur.execute("SELECT AVG(edge_pct) FROM predictions WHERE resolved=1 AND won=0")
        avg_edge_lost = cur.fetchone()[0] or 0

        cur.execute(
            "SELECT question, recommendation, won, bet_size, market_price, edge_pct "
            "FROM predictions WHERE resolved=1 ORDER BY resolved_at DESC LIMIT 10"
        )
        recent = cur.fetchall()
        conn.close()

        winrate = (wins / total * 100) if total > 0 else 0
        roi = (total_pnl / (total * 100) * 100) if total > 0 else 0

        return {
            "total": total,
            "wins": wins,
            "losses": total - wins,
            "winrate": winrate,
            "total_pnl": total_pnl,
            "roi_pct": roi,
            "pending": pending,
            "by_confidence": by_conf,
            "by_whale_signal": by_whale,
            "avg_edge_won": avg_edge_won,
            "avg_edge_lost": avg_edge_lost,
            "recent": recent,
        }
    except Exception as e:
        logger.error("winrate stats error: %s", e)
        return {"total": 0, "winrate": 0}


async def db_resolve_pending() -> int:
    try:
        conn = sqlite3.connect(DB_PATH)
        cur = conn.cursor()
        cur.execute(
            "SELECT id, market_id, condition_id, recommendation FROM predictions "
            "WHERE resolved=0 LIMIT 50"
        )
        rows = cur.fetchall()
        conn.close()
        if not rows:
            return 0

        resolved_count = 0
        async with httpx.AsyncClient(timeout=15) as client:
            for pred_id, market_id, _cond_id, recommendation in rows:
                try:
                    resp = await client.get(f"{GAMMA_API}/markets/{market_id}")
                    if resp.status_code != 200:
                        continue
                    m = resp.json()
                    is_closed = m.get("closed") or m.get("archived")
                    prices = m.get("outcomePrices")
                    if isinstance(prices, str):
                        try:
                            prices = json.loads(prices)
                        except Exception:
                            prices = None
                    if not is_closed or not prices:
                        continue
                    try:
                        yes_final = float(prices[0])
                    except Exception:
                        continue
                    if yes_final not in (0.0, 1.0):
                        continue

                    yes_won = yes_final == 1.0
                    if recommendation == "BUY YES":
                        won = 1 if yes_won else 0
                    elif recommendation == "BUY NO":
                        won = 1 if not yes_won else 0
                    else:
                        won = None
                    if won is None:
                        continue

                    actual = "YES" if yes_won else "NO"
                    conn = sqlite3.connect(DB_PATH)
                    cur = conn.cursor()
                    cur.execute(
                        "UPDATE predictions SET resolved=1, won=?, actual_outcome=?, resolved_at=? WHERE id=?",
                        (won, actual, int(time.time()), pred_id),
                    )
                    conn.commit()
                    conn.close()
                    resolved_count += 1
                except Exception as e:
                    logger.warning("resolve %s failed: %s", market_id, e)
        return resolved_count
    except Exception as e:
        logger.error("resolve pending error: %s", e)
        return 0


# ── Polymarket: Gamma ─────────────────────────────────────────────────────────


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


# ── Polymarket: Data API ──────────────────────────────────────────────────────


async def get_market_trades(market_condition_id: str, limit: int = 100) -> list:
    url = f"{DATA_API}/trades"
    params = {"market": market_condition_id, "limit": limit, "takerOnly": "true"}
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
    url = f"{DATA_API}/holders"
    params = {"market": token_id, "limit": limit}
    async with httpx.AsyncClient(timeout=25) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            data = resp.json()
            if isinstance(data, list) and data and isinstance(data[0], dict):
                if "holders" in data[0]:
                    return data[0].get("holders") or []
                return data
            return []
        except Exception as e:
            logger.warning("holders fetch failed: %s", e)
            return []


async def get_user_positions(wallet: str, limit: int = 50) -> list:
    url = f"{DATA_API}/positions"
    params = {"user": wallet, "limit": limit, "sortBy": "CURRENT", "sortDirection": "DESC"}
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


def format_market_list(markets: list, header: str = "📋 Trending markets:") -> tuple:
    if not markets:
        return "Рынки не найдены.", InlineKeyboardMarkup([])
    lines = [f"{header}\n"]
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


# ── Whale analytics ──────────────────────────────────────────────────────────


def analyze_whale_trades(trades: list, outcomes: list) -> dict:
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
        effective_yes = (side == "BUY" and outcome_idx == 0) or (side == "SELL" and outcome_idx == 1)
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

    whale_trades.sort(key=lambda x: x["usd"], reverse=True)
    unique_whales = len({t["wallet"] for t in whale_trades})
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


def whale_signal_strength(whale_data: dict, holders_data: dict) -> tuple:
    yes_share_w = whale_data.get("yes_share", 0.5)
    yes_share_h = holders_data.get("yes_share_holders", 0.5)
    whale_vol = whale_data.get("whale_volume_usd", 0)
    holder_vol = holders_data.get("total_top_whales_usd", 0)
    total_vol = whale_vol + holder_vol
    if total_vol == 0:
        return ("Neutral", 0.0, False)
    weighted = (yes_share_w * whale_vol + yes_share_h * holder_vol) / total_vol
    diff = weighted - 0.5
    if diff > 0.20:
        return ("Strong YES", diff, True)
    if diff > 0.08:
        return ("Weak YES", diff, False)
    if diff < -0.20:
        return ("Strong NO", diff, True)
    if diff < -0.08:
        return ("Weak NO", diff, False)
    return ("Neutral", diff, False)


def calculate_expected_profit(bet_usd: float, market_price: float, true_prob: float) -> dict:
    def _ev(side_price: float, side_prob: float) -> dict:
        if side_price <= 0 or side_price >= 1:
            return {"ev_pct": 0, "kelly_pct": 0, "win_payout_usd": 0,
                    "win_profit_usd": 0, "expected_profit_usd": 0, "win_prob": side_prob}
        b = (1.0 / side_price) - 1.0
        p = side_prob
        q = 1.0 - p
        ev_per_dollar = p * b - q
        kelly = (b * p - q) / b if b > 0 else 0
        kelly = max(0.0, min(0.25, kelly))
        win_shares = bet_usd / side_price
        return {
            "ev_pct": ev_per_dollar * 100,
            "kelly_pct": kelly * 100,
            "win_payout_usd": win_shares,
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
        primary = yes_calc if edge_yes >= 0 else no_calc

    return {
        "recommendation": recommendation,
        "edge_pct": edge_yes * 100,
        "yes": yes_calc,
        "no": no_calc,
        "primary": primary,
    }


# ── AI analysis ──────────────────────────────────────────────────────────────


def analyze_market_with_ai(
    market: dict,
    whale_data: dict | None = None,
    holders_data: dict | None = None,
    use_web_search: bool = True,
) -> dict:
    market_id = str(market.get("id") or market.get("conditionId") or "")
    cached = db_get_cached_ai(market_id) if market_id else None
    if cached:
        logger.info("AI cache hit for %s", market_id)
        return cached

    question = market.get("question", "Unknown")
    yes_price = _yes_price(market)
    volume = float(market.get("volume") or 0)
    end_date = market.get("endDate", "?")

    whale_block = ""
    if whale_data and whale_data.get("whale_count", 0) > 0:
        wd = whale_data
        whale_block = (
            f"WHALES: {wd['whale_count']} trades, ${wd['whale_volume_usd']:,.0f} vol, "
            f"YES {wd['yes_share']:.0%} / NO {1-wd['yes_share']:.0%}"
        )
    holders_block = ""
    if holders_data and holders_data.get("total_top_whales_usd", 0) > 0:
        hd = holders_data
        holders_block = f"HOLDERS YES bias: {hd['yes_share_holders']:.0%}"

    prompt = (
        f"Analyze Polymarket prediction market.\n"
        f"Q: {question}\nYES: {yes_price:.0%} | Vol: ${volume:,.0f} | Ends: {end_date}\n"
        f"{whale_block}\n{holders_block}\n\n"
        f"{'Use web_search briefly if needed. ' if use_web_search else 'Use only the data above; do NOT search the web. '}"
        f"Output in EXACTLY this format (no markdown, no asterisks):\n"
        f"TRUE_PROB: <0-100>\n"
        f"CONFIDENCE: <Low|Medium|High>\n"
        f"WHALE_SIGNAL: <Strong YES|Weak YES|Neutral|Weak NO|Strong NO>\n"
        f"REASONING: <1-2 sentences>\n"
        f"RISKS: <1 sentence>"
    )

    try:
        kwargs = {
            "model": "claude-sonnet-4-5",
            "max_tokens": 500,
            "messages": [{"role": "user", "content": prompt}],
        }
        if use_web_search:
            kwargs["tools"] = [{"type": "web_search_20250305", "name": "web_search"}]

        response = anthropic_client.messages.create(**kwargs)
        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        result_text = "\n".join(text_parts).strip()

        parsed = {
            "true_prob": yes_price,
            "confidence": "Low",
            "whale_signal": "Neutral",
            "research": "",
            "reasoning": "",
            "risks": "",
            "from_cache": False,
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
            elif line.startswith("REASONING:"):
                parsed["reasoning"] = clean_markdown(line.split(":", 1)[1].strip())
            elif line.startswith("RISKS:"):
                parsed["risks"] = clean_markdown(line.split(":", 1)[1].strip())

        if market_id:
            db_save_ai_cache(market_id, parsed)
        return parsed
    except Exception as e:
        logger.error("AI error: %s", e)
        return {
            "true_prob": yes_price,
            "confidence": "Low",
            "whale_signal": "Neutral",
            "research": "",
            "reasoning": f"AI error: {e}",
            "risks": "",
            "from_cache": False,
        }


# ── Pipeline ──────────────────────────────────────────────────────────────────


async def collect_whale_context(market: dict) -> tuple:
    condition_id = market.get("conditionId") or ""
    token_ids = _token_ids(market)
    outcomes = _outcomes(market)

    tasks = [
        get_market_trades(condition_id, limit=200) if condition_id else asyncio.sleep(0, result=[])
    ]
    if len(token_ids) >= 2:
        tasks.append(get_market_holders(str(token_ids[0]), limit=20))
        tasks.append(get_market_holders(str(token_ids[1]), limit=20))
    else:
        tasks.append(asyncio.sleep(0, result=[]))
        tasks.append(asyncio.sleep(0, result=[]))

    trades, holders_yes, holders_no = await asyncio.gather(*tasks)
    whale_data = analyze_whale_trades(trades or [], outcomes)
    holders_data = analyze_holders(holders_yes or [], holders_no or [], _yes_price(market))
    return whale_data, holders_data


async def full_market_analysis(market: dict, bet_size_usd: float = 100.0,
                                save_to_db: bool = True) -> str:
    question = market.get("question", "?")
    yes_price = _yes_price(market)
    volume = float(market.get("volume") or 0)
    market_id = str(market.get("id") or market.get("conditionId") or "")

    whale_data, holders_data = await collect_whale_context(market)

    if volume < MIN_VOLUME_FOR_AI:
        ai = {
            "true_prob": yes_price,
            "confidence": "Low",
            "whale_signal": whale_signal_strength(whale_data, holders_data)[0],
            "research": "",
            "reasoning": "Объём слишком мал — AI пропущен для экономии.",
            "risks": "",
            "from_cache": False,
        }
    else:
        signal_str, _, has_strong = whale_signal_strength(whale_data, holders_data)
        use_web = volume >= WEB_SEARCH_VOL_THRESHOLD or not has_strong
        ai = analyze_market_with_ai(market, whale_data, holders_data, use_web_search=use_web)

    true_prob = ai["true_prob"]
    ev = calculate_expected_profit(bet_size_usd, yes_price, true_prob)

    if save_to_db and ev["recommendation"] != "SKIP" and market_id:
        db_save_prediction(
            {
                "market_id": market_id,
                "condition_id": market.get("conditionId"),
                "slug": market.get("slug"),
                "question": question[:200],
                "market_price": yes_price,
                "true_prob": true_prob,
                "edge_pct": ev["edge_pct"],
                "recommendation": ev["recommendation"],
                "confidence": ai["confidence"],
                "whale_signal": ai["whale_signal"],
                "bet_size": bet_size_usd,
                "ev_pct": ev["primary"]["ev_pct"],
                "end_date": market.get("endDate"),
            }
        )

    cache_tag = " (📁 cached)" if ai.get("from_cache") else ""
    lines = [
        f"📊 АНАЛИЗ РЫНКА{cache_tag}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"❓ {question[:120]}",
        "",
        f"💰 Цена YES: {yes_price:.1%}",
        f"🎯 Истинная вер.: {true_prob:.1%}",
        f"📈 Edge: {ev['edge_pct']:+.1f}%",
        f"🎓 Уверенность: {ai['confidence']}",
        "",
    ]

    if whale_data["whale_count"] > 0:
        wd = whale_data
        lines.append(f"🐋 КИТЫ (≥ ${WHALE_THRESHOLD_USD:,.0f})")
        lines.append(f"  {wd['whale_count']} сделок | {wd['unique_whales']} кошельков | ${wd['whale_volume_usd']:,.0f}")
        lines.append(f"  💚 YES: {wd['yes_share']:.0%}  ❤️ NO: {1-wd['yes_share']:.0%}")
        lines.append(f"  📡 Сигнал: {ai['whale_signal']}")
        lines.append("")
        lines.append("  Топ-3:")
        for i, t in enumerate(wd["whale_trades"][:3], 1):
            direction = "YES" if t["effective_yes"] else "NO"
            lines.append(f"  {i}. ${t['usd']:,.0f} → {direction} @ {t['price']:.0%} ({t['name']})")
        lines.append("")

    lines.append("━━━━━━━━━━━━━━━━━━━━")
    lines.append(f"🎯 РЕКОМЕНДАЦИЯ: {ev['recommendation']}")
    if ev["recommendation"] != "SKIP":
        p = ev["primary"]
        side = "YES" if "YES" in ev["recommendation"] else "NO"
        side_price = yes_price if side == "YES" else 1 - yes_price
        lines.append("")
        lines.append(f"💵 Ставка ${bet_size_usd:.0f} на {side} @ {side_price:.0%}:")
        lines.append(f"  • Вер. выигрыша: {p['win_prob']:.0%}")
        lines.append(f"  • Если выиграл: +${p['win_profit_usd']:.2f}")
        lines.append(f"  • Если проиграл: -${bet_size_usd:.2f}")
        lines.append(f"  • EV: ${p['expected_profit_usd']:+.2f} ({p['ev_pct']:+.1f}%)")
        lines.append(f"  • Kelly: {p['kelly_pct']:.1f}% банка")
    lines.append("")
    if ai["reasoning"]:
        lines.append(f"💭 {ai['reasoning']}")
    if ai["risks"]:
        lines.append(f"⚠️ {ai['risks']}")

    slug = market.get("slug", "")
    if slug:
        lines.append("")
        lines.append(f"🔗 polymarket.com/event/{slug}")

    return "\n".join(lines)


# ── Intent classifier (для свободного текста) ────────────────────────────────


# Простые правила без AI — экономим токены
TOP_KEYWORDS = [
    "топ", "top", "лучшие ставки", "лучшие сделки", "рекомендованные ставк",
    "что ставить", "куда ставить", "лучшие возможност", "best bets", "best trades",
    "что закидыват", "куда закидыват",
]
HELP_KEYWORDS = ["help", "помощь", "что умееш", "что ты умеешь", "команды", "/help"]
STATS_KEYWORDS = ["винрейт", "win rate", "winrate", "статистика", "stats", "мой прогресс"]


def classify_intent(text: str) -> str:
    """
    Возвращает intent: 'top' | 'stats' | 'help' | 'search' | 'general'.
    Использует простые правила, чтобы не тратить токены на классификацию.
    """
    t = text.lower().strip()
    if any(k in t for k in HELP_KEYWORDS):
        return "help"
    if any(k in t for k in STATS_KEYWORDS):
        return "stats"
    if any(k in t for k in TOP_KEYWORDS):
        return "top"
    # Если есть значимые слова (тикер, имя, событие) — пробуем поиск
    # Берём слова длиной 3+, исключаем стоп-слова
    stop = {
        "что", "как", "где", "когда", "это", "будет", "есть", "ли", "или",
        "the", "what", "how", "where", "when", "is", "are", "and",
        "анализ", "проанализируй", "пожалуйста", "сейчас", "давай",
        "вверх", "вниз", "up", "down", "5", "10", "15", "минут", "минутном",
        "тайм", "фрейм", "таймфрейм", "timeframe", "мин",
    }
    words = re.findall(r"[a-zA-Zа-яА-Я0-9]{3,}", t)
    meaningful = [w for w in words if w.lower() not in stop]
    if meaningful:
        return "search"
    return "general"


def extract_search_query(text: str) -> dict:
    """
    Разбирает запрос пользователя.
    Возвращает dict с полями:
      - query: строка для поиска
      - is_updown: True если запрос про up/down рынок
      - ticker: тикер монеты (BTC, ETH...) или None
      - coin_name: название для поиска (bitcoin, ethereum...)
      - timeframe: '5m', '15m', '1h' или None
    """
    t = text.lower()

    # Монеты: русские и английские → (search_name, ticker)
    crypto_map = [
        (["доги коин", "догикоин", "доги", "doge"], "dogecoin", "DOGE"),
        (["биткоин", "биткойн", "битка", "битк", "бтс", "btc"], "bitcoin", "BTC"),
        (["эфириум", "эфир", "eth", "ethereum"], "ethereum", "ETH"),
        (["солана", "соль", "sol", "solana"], "solana", "SOL"),
        (["xrp", "рипл", "ripple"], "xrp", "XRP"),
        (["bnb"], "bnb", "BNB"),
        (["матик", "matic", "polygon"], "polygon", "MATIC"),
        (["ада", "ada", "cardano"], "cardano", "ADA"),
        (["авакс", "avax", "avalanche"], "avalanche", "AVAX"),
        (["линк", "link", "chainlink"], "chainlink", "LINK"),
        (["трон", "trx", "tron"], "tron", "TRX"),
        (["трамп", "trump"], "trump", None),
        (["байден", "biden"], "biden", None),
        (["путин", "putin"], "putin", None),
        (["выборы", "election"], "election", None),
    ]

    # Таймфреймы
    timeframe_map = [
        (["1 мин", "1мин", "1 минут", "одну минут", "одна минут"], "1m"),
        (["5 мин", "5мин", "5 минут", "пять минут"], "5m"),
        (["15 мин", "15мин", "15 минут", "пятнадцать минут"], "15m"),
        (["30 мин", "30мин", "30 минут", "тридцать минут"], "30m"),
        (["1 час", "1час", "час", "hourly", "one hour"], "1h"),
    ]

    # Слова-признаки up/down запроса
    updown_words = {
        "вверх", "вниз", "вырастет", "упадёт", "упадет", "памп", "дамп",
        "up", "down", "pump", "dump", "выше", "ниже", "рост", "падение",
        "минут", "мин", "минуты", "часов", "час", "таймфрейм", "тайм",
        "фрейм", "timeframe", "5m", "15m", "1h",
    }

    found_name = None
    found_ticker = None
    for aliases, name, ticker in crypto_map:
        for alias in aliases:
            if alias in t:
                found_name = name
                found_ticker = ticker
                break
        if found_name:
            break

    found_tf = None
    for aliases, tf in timeframe_map:
        for alias in aliases:
            if alias in t:
                found_tf = tf
                break
        if found_tf:
            break

    words_in_text = set(re.findall(r"[a-zA-Zа-яА-Я]+", t))
    has_updown = bool(words_in_text & updown_words) or found_tf is not None

    if found_name and found_ticker and has_updown:
        return {
            "query": f"{found_ticker} up or down",
            "is_updown": True,
            "ticker": found_ticker,
            "coin_name": found_name,
            "timeframe": found_tf,
        }

    if found_name:
        return {
            "query": found_name,
            "is_updown": False,
            "ticker": None,
            "coin_name": found_name,
            "timeframe": None,
        }

    stop = {
        "что", "как", "где", "когда", "это", "будет", "есть", "ли", "или",
        "анализ", "проанализируй", "сейчас", "давай", "пожалуйста",
        "the", "what", "how", "или", "на",
    }
    words = re.findall(r"[a-zA-Zа-яА-Я0-9]{3,}", t)
    meaningful = [w for w in words if w.lower() not in stop]
    return {
        "query": " ".join(meaningful[:3]),
        "is_updown": False,
        "ticker": None,
        "coin_name": None,
        "timeframe": None,
    }


async def search_updown_markets(ticker: str, timeframe: str | None = None, limit: int = 8) -> list:
    """Ищет up/down рынки для конкретной монеты."""
    ticker_lower = ticker.lower()

    # Пробуем через /events по slug
    url = f"{GAMMA_API}/events"
    params = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": "startDate",
        "ascending": "false",
        "slug": f"{ticker_lower}-updown",
    }
    markets = []
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(url, params=params)
            resp.raise_for_status()
            events = resp.json()
            if isinstance(events, list):
                for event in events:
                    for m in (event.get("markets") or []):
                        if m:
                            markets.append(m)
        except Exception as e:
            logger.warning("events updown fetch failed: %s", e)

    # Фильтруем по таймфрейму
    if timeframe and markets:
        filtered = [m for m in markets if timeframe in (m.get("slug") or "").lower()]
        if filtered:
            return filtered

    if markets:
        return markets

    # Fallback: обычный поиск
    fallback = f"{GAMMA_API}/markets"
    params2 = {
        "limit": limit,
        "active": "true",
        "closed": "false",
        "order": "volume24hr",
        "ascending": "false",
        "search": f"{ticker} up or down",
    }
    async with httpx.AsyncClient(timeout=20) as client:
        try:
            resp = await client.get(fallback, params=params2)
            resp.raise_for_status()
            data = resp.json()
            markets = data if isinstance(data, list) else []
        except Exception as e:
            logger.warning("updown fallback failed: %s", e)

    if timeframe and markets:
        filtered = [m for m in markets if timeframe in (m.get("slug") or "").lower()]
        if filtered:
            return filtered

    return markets


def format_quick_answer(market: dict, ai: dict, yes_price: float) -> str:
    """Короткий ответ: рекомендация, вероятность, риск — без китов и EV."""
    question = market.get("question", "?")
    true_prob = ai["true_prob"]
    ev = calculate_expected_profit(100.0, yes_price, true_prob)
    rec = ev["recommendation"]

    if rec == "BUY YES":
        verdict = "✅ ДА (BUY YES)"
        prob_label = f"Вероятность YES: {true_prob:.0%}"
    elif rec == "BUY NO":
        verdict = "❌ НЕТ (BUY NO)"
        prob_label = f"Вероятность NO: {1-true_prob:.0%}"
    else:
        verdict = "⏸ ПРОПУСТИТЬ"
        prob_label = f"Вероятность YES: {true_prob:.0%}"

    lines = [
        f"📊 {question[:100]}",
        "━━━━━━━━━━━━━━━━━━━━",
        f"💰 Текущая цена YES: {yes_price:.0%}",
        f"🎯 {prob_label}",
        f"📈 Edge: {ev['edge_pct']:+.1f}%",
        "",
        f"🎯 Рекомендация: {verdict}",
        f"🎓 Уверенность: {ai['confidence']}",
        "",
    ]
    if ai.get("reasoning"):
        lines.append(f"💭 {ai['reasoning']}")
    if ai.get("risks"):
        lines.append(f"⚠️ Риск: {ai['risks']}")

    slug = market.get("slug", "")
    if slug:
        lines.append("")
        lines.append(f"🔗 polymarket.com/event/{slug}")

    return "\n".join(lines)




# ── Telegram handlers ─────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "🤖 Polymarket AI Whale-Tracker\n\n"
        "Анализирую рынки Polymarket с учётом активности китов и считаю ожидаемую прибыль.\n\n"
        "Команды:\n"
        "/markets — топ рынки с кнопками\n"
        "/search <query> — поиск рынков\n"
        "/analyze <id> — глубокий анализ рынка\n"
        "/whales <id> — только активность китов\n"
        "/top — лучшие сделки сейчас (с pre-фильтром)\n"
        "/wallet 0x... — позиции конкретного кита\n"
        "/setbet <сумма> — размер ставки (по умолч. $100)\n"
        "/stats — мой win-rate и PnL\n"
        "/sync — обновить статус закрытых рынков\n\n"
        "💡 Можно писать обычным текстом: «трамп», «биткоин», «лучшие ставки» — "
        "сам найду рынок и проанализирую."
    )
    await update.message.reply_text(text)


async def cmd_markets(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Загружаю...")
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
        text, keyboard = format_market_list(markets, header=f"🔍 Найдено по «{query}»:")
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
        await update.message.reply_text(f"Текущий размер: ${cur:.0f}\nИзменить: /setbet 250")
        return
    try:
        amount = float(ctx.args[0])
        if amount <= 0 or amount > 100000:
            raise ValueError("range")
        ctx.user_data["bet_size"] = amount
        await update.message.reply_text(f"✅ Размер ставки: ${amount:.0f}")
    except Exception:
        await update.message.reply_text("Пример: /setbet 250")


async def cmd_wallet(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    wallet = " ".join(ctx.args or []).strip()
    if not wallet or not wallet.startswith("0x"):
        await update.message.reply_text("Использование: /wallet 0x6af75d...")
        return
    msg = await update.message.reply_text("Тяну позиции...")
    try:
        positions = await get_user_positions(wallet, limit=20)
        if not positions:
            await msg.edit_text("Нет активных позиций.")
            return
        lines = [f"🐋 ПОЗИЦИИ {_short_wallet(wallet)}\n"]
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
            lines.append(f"{i}. {title}\n   {outcome} | ${cur_val:,.0f} | PnL: ${pnl:+,.0f} ({pct:+.0f}%)")
        lines.append("")
        lines.append(f"💰 Total: ${total_value:,.0f} | PnL: ${total_pnl:+,.0f}")
        for chunk in _split_message("\n".join(lines)):
            await update.message.reply_text(chunk)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text(
        f"⚡ Сканирую рынки + pre-filter по китам... (AI запустим только для топ-{TOP_AI_LIMIT})"
    )
    try:
        markets = await get_trending_markets(15)
        if not markets:
            await msg.edit_text("Рынки не найдены.")
            return
        bet_size = ctx.user_data.get("bet_size", 100.0)

        await msg.edit_text(f"📡 Фаза 1: Whale-скрининг {len(markets)} рынков...")

        async def screen(m):
            try:
                wd, hd = await collect_whale_context(m)
                signal_str, strength, has_strong = whale_signal_strength(wd, hd)
                yes_price = _yes_price(m)
                rough_true = 0.5 + strength
                rough_edge = abs(rough_true - yes_price)
                return {
                    "market": m,
                    "whale": wd,
                    "holders": hd,
                    "signal": signal_str,
                    "strength": strength,
                    "rough_edge": rough_edge,
                    "has_strong": has_strong,
                    "score": rough_edge * (1.5 if has_strong else 1.0) * (
                        1.0 + min(wd["whale_volume_usd"] / 100000, 1.0)
                    ),
                }
            except Exception as e:
                logger.warning("screen err: %s", e)
                return None

        screened = await asyncio.gather(*(screen(m) for m in markets))
        screened = [s for s in screened if s]
        screened.sort(key=lambda s: s["score"], reverse=True)
        top_candidates = screened[:TOP_AI_LIMIT]

        if not top_candidates:
            await msg.edit_text("Не нашёл интересных сигналов.")
            return

        await msg.edit_text(f"🤖 Фаза 2: AI-анализ топ-{len(top_candidates)} кандидатов...")

        results = []
        for s in top_candidates:
            text = await full_market_analysis(s["market"], bet_size, save_to_db=True)
            results.append(text)

        header = (
            f"🎯 ТОП-{len(results)} ВОЗМОЖНОСТЕЙ\n"
            f"(размер: ${bet_size:.0f}, отобрано из {len(screened)} по китам)\n"
        )
        await update.message.reply_text(header)
        for i, text in enumerate(results, 1):
            full = f"#{i} ━━━━━━━━━━━━━━━━━━━━\n{text}"
            for chunk in _split_message(full):
                await update.message.reply_text(chunk, disable_web_page_preview=True)

        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_stats(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Считаю статистику...")
    try:
        resolved = await db_resolve_pending()
        s = db_get_winrate_stats()

        if s["total"] == 0:
            text = (
                "📊 СТАТИСТИКА БОТА\n━━━━━━━━━━━━━━━━━━━━\n"
                f"Pending (ждут резолва): {s.get('pending', 0)}\n"
                f"Resolved: 0\n\n"
                "Пока нет закрытых рекомендаций для подсчёта win-rate.\n"
                "Бот сохраняет каждую BUY-рекомендацию автоматически."
            )
            if resolved > 0:
                text += f"\n\n✨ Только что разрешено: {resolved}"
            await msg.edit_text(text)
            return

        emoji = "🔥" if s["winrate"] >= 60 else ("✅" if s["winrate"] >= 50 else "⚠️")
        lines = [
            "📊 WIN-RATE БОТА",
            "━━━━━━━━━━━━━━━━━━━━",
            f"{emoji} Win-rate: {s['winrate']:.1f}% ({s['wins']}/{s['total']})",
            f"💰 Total PnL: ${s['total_pnl']:+,.2f}",
            f"📈 ROI: {s['roi_pct']:+.1f}%",
            f"⏳ Pending: {s['pending']}",
            "",
            "📊 По уверенности:",
        ]
        for conf in ("High", "Medium", "Low"):
            if conf in s["by_confidence"]:
                cnt, w = s["by_confidence"][conf]
                wr = (w / cnt * 100) if cnt > 0 else 0
                lines.append(f"  {conf}: {wr:.0f}% ({w}/{cnt})")

        if s["by_whale_signal"]:
            lines.append("")
            lines.append("🐋 По сигналу китов:")
            for sig in ("Strong YES", "Strong NO", "Weak YES", "Weak NO", "Neutral"):
                if sig in s["by_whale_signal"]:
                    cnt, w = s["by_whale_signal"][sig]
                    wr = (w / cnt * 100) if cnt > 0 else 0
                    lines.append(f"  {sig}: {wr:.0f}% ({w}/{cnt})")

        lines.append("")
        lines.append(f"📐 Avg edge: выиграл +{s['avg_edge_won']:.1f}% / проиграл +{s['avg_edge_lost']:.1f}%")

        if s.get("recent"):
            lines.append("")
            lines.append("Последние 5:")
            for q, rec, won, bs, mp, ed in s["recent"][:5]:
                tick = "✅" if won else "❌"
                qt = (q or "?")[:45]
                lines.append(f"  {tick} {rec} (edge {ed:+.0f}%) — {qt}")

        if resolved > 0:
            lines.append("")
            lines.append(f"✨ Только что разрешено: {resolved}")

        await msg.edit_text("\n".join(lines))
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def cmd_sync(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Проверяю закрытые рынки...")
    try:
        n = await db_resolve_pending()
        await msg.edit_text(f"✅ Разрешено: {n}")
    except Exception as e:
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


async def _run_analysis(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                         market_id: str, via_callback: bool = False) -> None:
    send = update.callback_query.message if via_callback else update.message
    msg = await send.reply_text("🔬 Анализ... (~15-30 сек)")
    try:
        market = await get_market_by_id(market_id)
        if not market:
            await msg.edit_text("Рынок не найден.")
            return
        bet_size = ctx.user_data.get("bet_size", 100.0)
        full_text = await full_market_analysis(market, bet_size, save_to_db=True)
        for chunk in _split_message(full_text):
            await send.reply_text(chunk, disable_web_page_preview=True)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def _run_whales(update: Update, ctx: ContextTypes.DEFAULT_TYPE,
                       market_id: str, via_callback: bool = False) -> None:
    send = update.callback_query.message if via_callback else update.message
    msg = await send.reply_text("🐋 Анализирую китов...")
    try:
        market = await get_market_by_id(market_id)
        if not market:
            await msg.edit_text("Рынок не найден.")
            return
        whale_data, holders_data = await collect_whale_context(market)
        yes_price = _yes_price(market)
        signal_str, _, _ = whale_signal_strength(whale_data, holders_data)

        lines = ["🐋 АКТИВНОСТЬ КИТОВ", "━━━━━━━━━━━━━━━━━━━━"]
        lines.append(f"❓ {market.get('question','?')[:120]}")
        lines.append(f"💰 YES: {yes_price:.1%}")
        lines.append(f"📡 Сигнал: {signal_str}")
        lines.append("")

        if whale_data["whale_count"] == 0:
            lines.append(f"Нет сделок ≥ ${WHALE_THRESHOLD_USD:,.0f}.")
        else:
            wd = whale_data
            lines.append(f"📊 Сделки ≥ ${WHALE_THRESHOLD_USD:,.0f}:")
            lines.append(f"  Всего: {wd['whale_count']} | {wd['unique_whales']} кошельков")
            lines.append(f"  Объём: ${wd['whale_volume_usd']:,.0f} ({wd['whale_share_of_volume']:.0%} рынка)")
            lines.append(f"  💚 YES: ${wd['yes_buy_usd']:,.0f} ({wd['yes_share']:.0%})")
            lines.append(f"  ❤️  NO:  ${wd['no_buy_usd']:,.0f} ({1-wd['yes_share']:.0%})")
            lines.append("")
            lines.append("Последние сделки:")
            for i, t in enumerate(wd["whale_trades"][:10], 1):
                direction = "YES" if t["effective_yes"] else "NO"
                ts = datetime.fromtimestamp(t["ts"], tz=timezone.utc).strftime("%m-%d %H:%M")
                lines.append(f"  {i}. ${t['usd']:>7,.0f} → {direction} @ {t['price']:.0%} | {t['name']} | {ts}")

        if holders_data["total_top_whales_usd"] > 0:
            hd = holders_data
            lines.append("")
            lines.append("💼 ТОП-ХОЛДЕРЫ:")
            lines.append(f"  YES: ${hd['yes_holders_usd']:,.0f}")
            for w in hd["yes_top_whales"][:3]:
                lines.append(f"    • {w['name']} — ${w['usd']:,.0f}")
            lines.append(f"  NO: ${hd['no_holders_usd']:,.0f}")
            for w in hd["no_top_whales"][:3]:
                lines.append(f"    • {w['name']} — ${w['usd']:,.0f}")

        slug = market.get("slug", "")
        if slug:
            lines.append("")
            lines.append(f"🔗 polymarket.com/event/{slug}")

        for chunk in _split_message("\n".join(lines)):
            await send.reply_text(chunk, disable_web_page_preview=True)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


# ── ГЛАВНОЕ ИЗМЕНЕНИЕ: умный handle_text ─────────────────────────────────────


async def handle_text(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    """
    Свободный текст:
      - "лучшие ставки", "топ" → запускает /top
      - "винрейт", "статистика" → запускает /stats
      - "помощь" → показывает /start
      - тикер/тема (биткоин, доги, трамп...) → ищет рынки и предлагает анализ
      - всё остальное → короткий ответ AI без web_search
    """
    user_text = update.message.text.strip()
    intent = classify_intent(user_text)
    logger.info("text='%s' → intent=%s", user_text[:50], intent)

    if intent == "help":
        await cmd_start(update, ctx)
        return

    if intent == "stats":
        await cmd_stats(update, ctx)
        return

    if intent == "top":
        await cmd_top(update, ctx)
        return

    if intent == "search":
        parsed = extract_search_query(user_text)
        query = parsed["query"]
        is_updown = parsed["is_updown"]
        ticker = parsed["ticker"]
        timeframe = parsed["timeframe"]

        if not query:
            await update.message.reply_text(
                "Не понял запрос. Попробуй: «биткоин», «трамп», «выборы» или /markets"
            )
            return

        if is_updown and ticker:
            # ── UP/DOWN: находим рынок и сразу даём чёткий ответ ──
            tf_label = f" {timeframe}" if timeframe else ""
            msg = await update.message.reply_text(f"⚡ Ищу {ticker}{tf_label} Up/Down рынок...")
            try:
                markets = await search_updown_markets(ticker, timeframe, limit=8)
                if not markets:
                    await msg.edit_text(
                        f"Не нашёл активный {ticker} Up/Down рынок на Polymarket.\n"
                        f"Попробуй /search {ticker.lower()} up or down"
                    )
                    return
                market = markets[0]
                yes_price = _yes_price(market)
                await msg.edit_text(f"🤖 Анализирую {ticker}{tf_label}...")
                ai = analyze_market_with_ai(market, use_web_search=False)
                answer = format_quick_answer(market, ai, yes_price)
                for chunk in _split_message(answer):
                    await update.message.reply_text(chunk, disable_web_page_preview=True)
                await msg.delete()
            except Exception as e:
                logger.error(e)
                await msg.edit_text(f"Ошибка: {e}")
            return

        # ── Обычный поиск: список рынков с кнопками ──
        msg = await update.message.reply_text(f"🔍 Ищу рынки по «{query}»...")
        try:
            markets = await search_markets(query, limit=10)
            if not markets:
                await msg.edit_text(
                    f"По запросу «{query}» рынков на Polymarket не найдено.\n"
                    f"Попробуй /markets — топ всех активных рынков."
                )
                return
            bet_size = ctx.user_data.get("bet_size", 100.0)
            if len(markets) == 1:
                await msg.edit_text(f"Нашёл 1 рынок по «{query}», анализирую...")
                full_text = await full_market_analysis(markets[0], bet_size, save_to_db=True)
                for chunk in _split_message(full_text):
                    await update.message.reply_text(chunk, disable_web_page_preview=True)
                await msg.delete()
                return
            text, keyboard = format_market_list(
                markets, header=f"🔍 Найдено по «{query}» ({len(markets)} рынков):"
            )
            text += "\n\n💡 Нажми «📊 Analyze» для полного анализа с китами и EV"
            await msg.edit_text(text, reply_markup=keyboard)
        except Exception as e:
            logger.error(e)
            await msg.edit_text(f"Ошибка поиска: {e}")
        return

    # General — короткий ответ без web_search
    msg = await update.message.reply_text("Думаю...")
    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=300,
            messages=[
                {
                    "role": "user",
                    "content": (
                        "You are a Telegram bot for Polymarket prediction markets. "
                        "Answer briefly in user's language WITHOUT markdown (no **, no *, no #). "
                        "Suggest commands: /markets /top /search <query> /stats. "
                        "Don't make up data — if user asks about specific market, tell them to use /search.\n\n"
                        f"User: {user_text}"
                    ),
                }
            ],
        )
        text_parts = [b.text for b in response.content if hasattr(b, "text") and b.text]
        answer = clean_markdown("\n".join(text_parts).strip()) or "Не удалось получить ответ."
        for chunk in _split_message(answer):
            await update.message.reply_text(chunk)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


# ── Background job ────────────────────────────────────────────────────────────


async def auto_resolve_job(ctx: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        n = await db_resolve_pending()
        if n > 0:
            logger.info("Auto-resolved %d predictions", n)
    except Exception as e:
        logger.warning("auto resolve error: %s", e)


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    db_init()
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_start))
    app.add_handler(CommandHandler("markets", cmd_markets))
    app.add_handler(CommandHandler("search", cmd_search))
    app.add_handler(CommandHandler("analyze", cmd_analyze))
    app.add_handler(CommandHandler("whales", cmd_whales))
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CommandHandler("wallet", cmd_wallet))
    app.add_handler(CommandHandler("setbet", cmd_setbet))
    app.add_handler(CommandHandler("stats", cmd_stats))
    app.add_handler(CommandHandler("sync", cmd_sync))
    app.add_handler(CallbackQueryHandler(callback_analyze, pattern=r"^analyze:"))
    app.add_handler(CallbackQueryHandler(callback_whales, pattern=r"^whales:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))

    if app.job_queue:
        app.job_queue.run_repeating(auto_resolve_job, interval=3600, first=600)

    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
