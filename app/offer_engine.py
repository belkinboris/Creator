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

import httpx

from app import llm_adapter

logger = logging.getLogger(__name__)

MAX_IDEA_CHARS = 2000
MAX_OFFER_TOKENS = 8000

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
5. Демо-карточка лендинга показывает работу продукта: СЛЕВА вход из мира
   пользователя, СПРАВА — что выдаёт продукт. Выбери подачу под идею:
   «диалог» (входящее сообщение/отзыв/вопрос → ответ продукта),
   «было→стало» (сырьё/бриф → готовый результат),
   «вход→разбор» (документ/запись → структурированный итог).
   Заполни ВСЕ подписи карточки под мир идеи, без наследия «отзывов»:
   - demo_left_label: заголовок слева («бриф игрока № 214», «встреча 47 мин»)
   - demo_left_badge: короткий бейдж/статус слева или "" («★☆☆☆☆», «входящий бриф»)
   - demo_left_text: сам вход, с деталями
   - demo_left_meta: подпись автора/времени («Марина, вчера в 23:47», «заказчик, сегодня»)
   - demo_right_tag: плашка справа («концепт готов · 3 варианта», «протокол готов»)
   - demo_right_text: выход продукта, 2-4 предложения, конкретно
   - demo_head_right: правый верхний штамп («готово за 3 сек»)
   Конкретика с деталями, не «Отличный сервис!».

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
     "demo_left_label": "...", "demo_left_badge": "...", 
     "demo_left_text": "...", "demo_left_meta": "...",
     "demo_right_tag": "...", "demo_right_text": "...",
     "demo_head_right": "...",
     "direct_queries": ["...", "..."],
     "lead_rate_signal": 0.04, "lead_rate_dead": 0.02, "click_target": 40
   }, ... ровно 3 оффера ...
 ]
}"""


class OfferEngineError(Exception):
    """Текст исключения уходит ПРЯМО на экран человека: `/api/sharpen`
    отдаёт `str(e)`, а `result.html` показывает его как есть. Значит он
    обязан быть человеческим. Техническая причина живёт отдельно, в `.tech`,
    и уходит в лог — чинить по ней нам, а не посетителю (A19).
    """

    def __init__(self, message: str, *, tech: str = ""):
        super().__init__(message)
        self.tech = tech


# Одна фраза на все структурные осечки модели: человеку незачем знать, какое
# поле недосчитала проверка, ему нужно понять, что делать дальше.
_MALFORMED = "ИИ вернул неполный ответ. Попробуйте ещё раз — обычно со второго раза получается."


def _validate(data: dict) -> dict:
    if not isinstance(data, dict) or "offers" not in data:
        raise OfferEngineError(_MALFORMED, tech="нет поля offers")
    offers = data["offers"]
    if not isinstance(offers, list) or len(offers) != 3:
        raise OfferEngineError(_MALFORMED, tech=f"ожидали 3 варианта, пришло {len(offers) if isinstance(offers, list) else type(offers).__name__}")
    required = ["angle", "idea_id", "product_name", "eyebrow", "h1", "sub",
                "pains", "demo_left_label", "demo_left_text",
                "demo_right_text", "direct_queries"]
    for o in offers:
        for key in required:
            if not o.get(key):
                raise OfferEngineError(_MALFORMED, tech=f"в варианте нет поля {key}")
        if len(o["pains"]) != 3:
            raise OfferEngineError(_MALFORMED, tech="pains: нужно 3 блока")
        if not (5 <= len(o["direct_queries"]) <= 12):
            raise OfferEngineError(_MALFORMED, tech="direct_queries: нужно 5-12 фраз")
        o.setdefault("demo_left_badge", "")
        o.setdefault("demo_left_meta", "")
        o.setdefault("demo_right_tag", "результат · черновик готов")
        o.setdefault("demo_head_right", "готово за секунды")
        o.setdefault("lead_rate_signal", 0.04)
        o.setdefault("lead_rate_dead", 0.02)
        o.setdefault("click_target", 40)
    data.setdefault("sharpened_note", "")
    data.setdefault("warning", "")
    return data


async def sharpen_idea(idea: str, *, _post=None, _attempt: int = 1) -> dict:
    """
    Идея → {"sharpened_note", "warning", "offers":[3]}. Бросает
    OfferEngineError с человеческим текстом при любой проблеме.
    _post(provider, payload) -> raw_response_dict — инъекция для тестов,
    прокидывается в llm_adapter.call без изменений.
    """
    idea = (idea or "").strip()[:MAX_IDEA_CHARS]
    if len(idea) < 20:
        raise OfferEngineError("Опишите идею хотя бы парой предложений: кому и чем она помогает.")

    try:
        # 3 полных оффера на русском -- это 4-6k токенов (кириллица дорогая),
        # 3000 обрезало ответ посреди JSON (баг первого прод-запуска 2026-07-09).
        # 8000 токенов ответа могут генерироваться дольше минуты -- таймаут
        # транспорта внутри llm_adapter учитывает это (180s).
        text = await llm_adapter.call(
            SYSTEM, f"Идея фаундера:\n{idea}", MAX_OFFER_TOKENS, _post=_post,
        )
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _validate(json.loads(text))
    except OfferEngineError as exc:
        if exc.tech:
            logger.warning("offer engine: %s (attempt %s)", exc.tech, _attempt)
            if _attempt == 1:
                return await sharpen_idea(idea, _post=_post, _attempt=2)
        raise
    except llm_adapter.LLMAdapterError as exc:
        raise OfferEngineError(str(exc))
    except json.JSONDecodeError:
        logger.exception("offer engine: bad JSON (attempt %s)", _attempt)
        if _attempt == 1:
            return await sharpen_idea(idea, _post=_post, _attempt=2)
        raise OfferEngineError("ИИ ответил в неожиданном формате. Попробуйте ещё раз.")
    except httpx.TimeoutException:
        logger.warning("offer engine: timeout (attempt %s)", _attempt)
        if _attempt == 1:
            return await sharpen_idea(idea, _post=_post, _attempt=2)
        raise OfferEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
    except Exception:
        logger.exception("offer engine failed")
        raise OfferEngineError("Не получилось заострить идею. Попробуйте ещё раз через минуту.")
