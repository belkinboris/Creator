"""
Создатель — движок платного отчёта/бизнес-плана.

Отличие от дженерик-ИИ-генераторов бизнес-планов (их уже полно бесплатных
в сети — ai.tochka.com и т.п.): наш отчёт строится НЕ с нуля фантазией
модели, а на данных, которые проект уже честно собрал на бесплатном этапе
проверки спроса — реальная частотность Вордстата, реальные конкуренты из
выдачи, оценка по 4 шкалам. Это единственное отличие, которое стоит денег;
без него отчёт — то же самое, что бесплатные генераторы уже делают.

Два тарифа:
  quick — 4 секции, быстрый разбор
  full  — все 8 секций, подробно

Как offer_engine.py: честный LLM-вызов, строго типизированный и
провалидированный выход, детерминированной логики здесь нет.
"""

from __future__ import annotations

import json
import logging

import httpx

from app import llm_adapter

logger = logging.getLogger(__name__)

MAX_IDEA_CHARS = 2000
MAX_TOKENS_QUICK = 3000
MAX_TOKENS_FULL = 7000

# (ключ, заголовок) — порядок фиксирован, quick берёт первые 4.
ALL_SECTIONS = [
    ("summary", "Резюме проекта"),
    ("market", "Спрос и рынок"),
    ("competitors", "Конкуренты и позиционирование"),
    ("verdict", "Вердикт"),
    ("audience", "Целевая аудитория"),
    ("finance", "Финансовая модель"),
    ("risks", "Риски и как их снижать"),
    ("launch", "План запуска — по этапам"),
]
QUICK_KEYS = ["summary", "market", "competitors", "verdict"]

STAGE_NAMES = ["Идея", "Спрос", "Проверочная страница", "Реклама",
               "Заявки", "Первые продажи", "Повторяемость", "Удержание"]


def _system_prompt(tier: str) -> str:
    keys = QUICK_KEYS if tier == "quick" else [k for k, _ in ALL_SECTIONS]
    sections_spec = "\n".join(f'  "{k}": "текст секции «{title}»"' for k, title in ALL_SECTIONS if k in keys)
    return f"""Ты — аналитик Создателя, сервиса проверки бизнес-идей. Пишешь платный
отчёт по идее пользователя. Тебе переданы РЕАЛЬНЫЕ данные проверки спроса —
частотность Вордстата, реальные конкуренты из выдачи Яндекса, оценка идеи.
Используй эти цифры и названия конкурентов буквально, не выдумывай другие.

Жёсткие правила:
1. Никакого маркетингового жаргона, никаких общих фраз ("отличная идея!",
   "рынок огромен"). Конкретика, цифры, названия — то, за что стоит платить.
2. Секция "finance" — явно помечай как оценку/прикидку, не как гарантию;
   если данных для расчёта мало, честно пиши "оценка приблизительная,
   уточните на реальных продажах".
3. Секция "launch" — план из шагов на пути 0→7: {", ".join(STAGE_NAMES)}.
   Идея и Спрос уже пройдены пользователем бесплатно — план начинай с
   ближайшего следующего шага (обычно "Проверочная страница").
4. Секция "verdict" — честный вывод: запускать / доработать / не запускать
   в текущем виде. Не бойся написать "не запускать", если данные это
   показывают — вердикт без права на нет — не вердикт.
5. Каждая секция — 2-5 абзацев обычным текстом, абзацы разделяй \\n\\n.
   Без markdown-символов (**, ##, списков через *) внутри текста секций.

Ответь ТОЛЬКО валидным JSON без markdown-обёртки:
{{
 "sections": {{
{sections_spec}
 }}
}}"""


class ReportEngineError(Exception):
    """Человекочитаемая ошибка — показывается пользователю как есть."""


def _validate(data: dict, tier: str) -> dict:
    if not isinstance(data, dict) or "sections" not in data:
        raise ReportEngineError("нет поля sections")
    sections = data["sections"]
    if not isinstance(sections, dict):
        raise ReportEngineError("sections должен быть объектом")
    keys = QUICK_KEYS if tier == "quick" else [k for k, _ in ALL_SECTIONS]
    out = []
    for key, title in ALL_SECTIONS:
        if key not in keys:
            continue
        body = sections.get(key)
        if not body or not str(body).strip():
            raise ReportEngineError(f"в отчёте нет секции {key}")
        out.append({"key": key, "title": title, "body": str(body).strip()})
    return {"sections": out}


async def generate_report(idea: str, demand_data: dict, tier: str = "quick",
                          chosen_offer: dict | None = None, *, _post=None, _attempt: int = 1) -> dict:
    """
    Идея + данные бесплатной проверки спроса -> отчёт из секций (dict с
    "sections": [{"key","title","body"}, ...]). Бросает ReportEngineError
    с человеческим текстом при любой проблеме.
    """
    idea = (idea or "").strip()[:MAX_IDEA_CHARS]
    if len(idea) < 15:
        raise ReportEngineError("Идея слишком короткая для отчёта.")
    if tier not in ("quick", "full"):
        tier = "quick"

    context = {
        "идея": idea,
        "частотности": demand_data.get("formulations", []),
        "вердикт_спроса": demand_data.get("verdict", {}),
        "конкуренты_в_выдаче": (demand_data.get("competitors") or {}).get("top", []),
        "страниц_в_выдаче": (demand_data.get("competitors") or {}).get("found"),
        "оценка_по_шкалам": demand_data.get("scores", []),
        "общий_балл": demand_data.get("overall"),
    }
    if chosen_offer:
        context["выбранное_позиционирование"] = chosen_offer
    user_msg = f"Идея и данные проверки спроса:\n{json.dumps(context, ensure_ascii=False)}"
    max_tokens = MAX_TOKENS_QUICK if tier == "quick" else MAX_TOKENS_FULL

    try:
        text = await llm_adapter.call(_system_prompt(tier), user_msg, max_tokens, _post=_post)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _validate(json.loads(text), tier)
    except ReportEngineError:
        raise
    except llm_adapter.LLMAdapterError as exc:
        raise ReportEngineError(str(exc))
    except json.JSONDecodeError:
        logger.exception("report engine: bad JSON (attempt %s)", _attempt)
        if _attempt == 1:
            return await generate_report(idea, demand_data, tier, chosen_offer, _post=_post, _attempt=2)
        raise ReportEngineError("ИИ ответил в неожиданном формате. Попробуйте ещё раз.")
    except httpx.TimeoutException:
        logger.warning("report engine: timeout (attempt %s)", _attempt)
        if _attempt == 1:
            return await generate_report(idea, demand_data, tier, chosen_offer, _post=_post, _attempt=2)
        raise ReportEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
    except Exception:
        logger.exception("report engine failed")
        raise ReportEngineError("Не получилось собрать отчёт. Попробуйте ещё раз через минуту.")
