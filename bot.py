from __future__ import annotations

import json
import logging
import os

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

# ── Polymarket helpers ────────────────────────────────────────────────────────


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


def _split_message(text: str, limit: int = 4000) -> list:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        chunks.append(text[:limit])
        text = text[limit:]
    return chunks


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
                [InlineKeyboardButton(f"Analyze #{i}", callback_data=f"analyze:{market_id}")]
            )

    return "\n".join(lines), InlineKeyboardMarkup(buttons)


# ── AI analysis ───────────────────────────────────────────────────────────────


def analyze_market_with_ai(market: dict) -> str:
    question = market.get("question", "Unknown market")
    yes_price = _yes_price(market)
    volume = float(market.get("volume") or 0)
    end_date = market.get("endDate", "unknown")

    prompt = (
        "You are a prediction market analyst. Analyze this Polymarket market.\n\n"
        f"MARKET:\n"
        f"- Question: {question}\n"
        f"- Current YES price: {yes_price:.1%}\n"
        f"- 24h volume: ${volume:,.0f}\n"
        f"- Closes: {end_date}\n\n"
        "STEPS:\n"
        "1. Use web_search to find recent relevant news/data.\n"
        "2. Estimate the TRUE probability of YES.\n"
        "3. Compare with current market price.\n"
        "4. Give recommendation: BUY YES / BUY NO / SKIP.\n\n"
        "RESPOND in this exact plain text format (no markdown):\n"
        "MARKET: <short title>\n"
        "RESEARCH: <2-3 sentences on findings>\n"
        "TRUE PROB: <X%>\n"
        "MARKET PRICE: <X%>\n"
        "EDGE: <+X% or -X%>\n"
        "RECOMMENDATION: <BUY YES / BUY NO / SKIP>\n"
        "CONFIDENCE: <Low / Medium / High>\n"
        "REASONING: <2-3 sentences>\n"
        "RISKS: <1-2 key risks>\n"
    )

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=1000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}],
        )
        text_parts = [
            block.text for block in response.content if hasattr(block, "text") and block.text
        ]
        result = "\n".join(text_parts).strip()
        return result or "AI не смог сгенерировать анализ."
    except Exception as e:
        logger.error("AI analysis error: %s", e)
        return f"Ошибка анализа: {e}"


# ── Telegram handlers ─────────────────────────────────────────────────────────


async def cmd_start(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    text = (
        "Polymarket AI Assistant\n\n"
        "Анализирую рынки Polymarket с помощью ИИ и веб-поиска.\n\n"
        "Команды:\n"
        "/markets - топ рынки\n"
        "/search <запрос> - поиск рынков\n"
        "/analyze <id> - глубокий анализ\n"
        "/top - лучшие возможности сейчас\n\n"
        "Совет: используй /markets и нажми Analyze"
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
    await _run_analysis(update, market_id)


async def cmd_top(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    msg = await update.message.reply_text("Ищу лучшие возможности... (~30-60 сек)")
    try:
        markets = await get_trending_markets(5)
        if not markets:
            await msg.edit_text("Рынки не найдены.")
            return
        await msg.edit_text("Запускаю AI анализ топ-5 рынков...")
        results = []
        for m in markets:
            analysis = analyze_market_with_ai(m)
            results.append(analysis)

        separator = "\n\n" + "-" * 30 + "\n\n"
        full = separator.join(results)
        for chunk in _split_message(full):
            await update.message.reply_text(chunk)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка: {e}")


async def callback_analyze(update: Update, ctx: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()
    market_id = query.data.split(":", 1)[1]
    await _run_analysis(update, market_id, via_callback=True)


async def _run_analysis(
    update: Update, market_id: str, via_callback: bool = False
) -> None:
    send = update.callback_query.message if via_callback else update.message
    msg = await send.reply_text("Анализирую рынок... (~20 сек)")
    try:
        market = await get_market_by_id(market_id)
        if not market:
            await msg.edit_text("Рынок не найден. Проверьте ID.")
            return
        analysis = analyze_market_with_ai(market)
        slug = market.get("slug", market_id)
        url = f"https://polymarket.com/event/{slug}"
        full = f"{analysis}\n\nПосмотреть: {url}"
        for chunk in _split_message(full):
            await send.reply_text(chunk, disable_web_page_preview=True)
        await msg.delete()
    except Exception as e:
        logger.error(e)
        await msg.edit_text(f"Ошибка анализа: {e}")


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
    app.add_handler(CommandHandler("top", cmd_top))
    app.add_handler(CallbackQueryHandler(callback_analyze, pattern=r"^analyze:"))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    logger.info("Bot started.")
    app.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
