"""
Создатель — движок этапа ⓪ «Оффер».

Вход: идея фаундера (2-5 предложений).
Выход: 3 варианта оффера с разными акцентами, каждый — полный набор
слотов для smoke-лендинга + запросы Яндекс Директа + пороги вердикта.

Здесь тот самый момент из видения: «уже на этапе идеи ИИ меняет акценты» —
модель не пересказывает идею, а предлагает три РАЗНЫХ угла атаки на боль.

Детерминированных решений здесь нет — это честный LLM-вызов; зато выход
строго типизирован и валидируется, а вердикты дальше по этапам выносит
детерминированный слой (см. VISION: LLM формулирует, правила решают).
"""

from __future__ import annotations

import json
import logging
import os

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
MODEL = os.environ.get("SOZDATEL_MODEL", "claude-sonnet-4-6")
MAX_IDEA_CHARS = 2000

SYSTEM = """Ты — продуктовый стратег Создателя, сервиса проверки стартап-идей.
Фаундер приносит сырую идею. Твоя работа — заострить её в 3 РАЗНЫХ оффера
для smoke-теста через Яндекс Директ. Разных — значит с разными акцентами:
другая грань боли, другая аудитория или другой момент использования.
Не три пересказа одной фразы.

Жёсткие правила:
1. Оффер обязан отвечать на боль, которую УЖЕ ИЩУТ в поиске. Если идею
   не ищут — честно скажи это в поле warning и предложи ближайшую ищимую
   формулировку.
2. Простой русский. Заголовок — про результат человека, не про технологию.
3. Никаких обещаний заработка, лечения, гарантий. Ничего про детей,
   медицину, чужие персональные данные — если идея туда ведёт, warning.
4. direct_queries — реальные фразы, как их набирают в Яндексе (8-10 штук,
   без операторов).
5. demo_review/demo_reply или аналог — конкретика с деталями, не «Отличный
   сервис!». Это пример работы будущего продукта в мире пользователя.

Ответь ТОЛЬКО валидным JSON без markdown-обёртки:
{
 "sharpened_note": "1-2 предложения: что ты изменил в акцентах и почему",
 "warning": "" | "текст предупреждения, если идея слабо ищется или рискованна",
 "offers": [
   {
     "angle": "короткое имя угла атаки",
     "idea_id": "латиницей_v1",
     "product_name": "рабочее имя продукта, 1-2 слова",
     "eyebrow": "для кого, 3-6 слов",
     "h1": "заголовок до 8 слов, можно с <em>акцентом</em>",
     "sub": "подзаголовок 1-2 предложения: что получает человек",
     "pains": [ {"h2": "...", "p": "..."}, {"h2": "...", "p": "..."},
                {"h2": "как это будет работать", "p": "..."} ],
     "demo_left_label": "подпись левой части демо (например: отзыв № 4 812)",
     "demo_left_text": "входной пример из мира пользователя",
     "demo_right_text": "что выдаёт продукт (2-4 предложения, конкретно)",
     "direct_queries": ["...", "..."],
     "lead_rate_signal": 0.08, "lead_rate_dead": 0.04, "click_target": 40
   }, ... ровно 3 оффера ...
 ]
}"""


class OfferEngineError(Exception):
    pass


def _validate(data: dict) -> dict:
    if not isinstance(data, dict) or "offers" not in data:
        raise OfferEngineError("нет поля offers")
    offers = data["offers"]
    if not isinstance(offers, list) or len(offers) != 3:
        raise OfferEngineError("нужно ровно 3 оффера")
    required = ["angle", "idea_id", "product_name", "eyebrow", "h1", "sub",
                "pains", "demo_left_label", "demo_left_text",
                "demo_right_text", "direct_queries"]
    for o in offers:
        for key in required:
            if not o.get(key):
                raise OfferEngineError(f"в оффере нет поля {key}")
        if len(o["pains"]) != 3:
            raise OfferEngineError("pains должен содержать 3 блока")
        if not (5 <= len(o["direct_queries"]) <= 12):
            raise OfferEngineError("direct_queries: 5-12 фраз")
        o.setdefault("lead_rate_signal", 0.08)
        o.setdefault("lead_rate_dead", 0.04)
        o.setdefault("click_target", 40)
    data.setdefault("sharpened_note", "")
    data.setdefault("warning", "")
    return data


async def sharpen_idea(idea: str, api_key: str | None = None, *, _post=None, _attempt: int = 1) -> dict:
    """
    Идея → {"sharpened_note", "warning", "offers":[3]}. Бросает
    OfferEngineError с человеческим текстом при любой проблеме.
    _post — инъекция для тестов.
    """
    idea = (idea or "").strip()[:MAX_IDEA_CHARS]
    if len(idea) < 20:
        raise OfferEngineError("Опишите идею хотя бы парой предложений: кому и чем она помогает.")

    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key and _post is None:
        raise OfferEngineError("Сервер не настроен: не задан ANTHROPIC_API_KEY.")

    payload = {
        "model": MODEL,
        # 3 полных оффера на русском -- это 4-6k токенов (кириллица дорогая),
        # 3000 обрезало ответ посреди JSON (баг первого прод-запуска 2026-07-09).
        "max_tokens": 8000,
        "system": SYSTEM,
        "messages": [{"role": "user", "content": f"Идея фаундера:\n{idea}"}],
    }
    try:
        if _post is not None:
            data = await _post(payload)
        else:
            async with httpx.AsyncClient(timeout=60.0) as client:
                resp = await client.post(
                    ANTHROPIC_URL, json=payload,
                    headers={"x-api-key": api_key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                )
                if resp.status_code != 200:
                    logger.warning("offer engine HTTP %s: %s", resp.status_code, resp.text[:200])
                    raise OfferEngineError("Не получилось обратиться к ИИ. Попробуйте ещё раз через минуту.")
                data = resp.json()
        text = "\n".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _validate(json.loads(text))
    except OfferEngineError:
        raise
    except json.JSONDecodeError:
        logger.exception("offer engine: bad JSON (attempt %s)", _attempt)
        if _attempt == 1:
            return await sharpen_idea(idea, api_key, _post=_post, _attempt=2)
        raise OfferEngineError("ИИ ответил в неожиданном формате. Попробуйте ещё раз.")
    except Exception:
        logger.exception("offer engine failed")
        raise OfferEngineError("Не получилось заострить идею. Попробуйте ещё раз через минуту.")
