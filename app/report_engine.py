"""
Создатель — движок платного отчёта/бизнес-плана.

Отличие от дженерик-ИИ-генераторов бизнес-планов (их уже полно бесплатных
в сети — ai.tochka.com и т.п.): наш отчёт строится НЕ с нуля фантазией
модели, а на данных, которые проект уже честно собрал на бесплатном этапе
проверки спроса — реальная частотность Вордстата, реальные конкуренты из
выдачи, оценка по 4 шкалам. Это единственное отличие, которое стоит денег;
без него отчёт — то же самое, что бесплатные генераторы уже делают.

Два тарифа:
  quick — 4 секции + 2 риска, быстрый разбор
  full  — все 8 секций + 3 риска, подробно

Модель: НЕ Anthropic/Claude -- проект российский, обработка персональных
данных не может уезжать за границу (152-ФЗ о трансграничной передаче), и
Claude API в любом случае заблокирован Роскомнадзором для конечных
пользователей в РФ. Вместо смены провайдера форсируем более сильную модель
внутри уже используемого Yandex AI Studio (см. _call_llm) -- YandexGPT 5.1
Pro вместо дефолтного дешёвого DeepSeek Flash остального проекта: отчёт
платный, здесь важно качество аналитики, а не скорость/цена бесплатных
этапов. Вся обработка остаётся на инфраструктуре Yandex Cloud.

Как offer_engine.py: честный LLM-вызов, строго типизированный и
провалидированный выход, детерминированной логики здесь нет.
"""

from __future__ import annotations

import json
import logging
import os

import httpx

from app import llm_adapter

logger = logging.getLogger(__name__)

MAX_IDEA_CHARS = 2000
MAX_TOKENS_QUICK = 4000
MAX_TOKENS_FULL = 12000

# YandexGPT 5.1 Pro (версия "rc" в каталоге Yandex AI Studio) -- заметно
# сильнее дефолтного deepseek-v4-flash остального проекта, остаётся на
# инфраструктуре Yandex Cloud. Один env var -- можно поменять без деплоя,
# если Yandex переведёт "rc" на другую модель.
SOZDATEL_REPORT_MODEL = os.environ.get("SOZDATEL_REPORT_MODEL", "yandexgpt/rc")

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

# Та же 7-шаговая шкала, что в app/main.py (STAGE_NAMES) -- "Проверочная
# страница" и "Реклама" объединены в "Тест на реальных людях", см. комментарий
# там же. Копия, а не импорт: report_engine не тянет зависимость на main.
STAGE_NAMES = ["Идея", "Спрос", "Тест на реальных людях",
               "Заявки", "Первые продажи", "Повторяемость", "Удержание"]


def _risk_count(tier: str) -> int:
    return 2 if tier == "quick" else 3


# Одна и та же идея приходит к нам с двух очень разных сторон, и мерить их
# одной линейкой -- прямой путь к ощущению «отчёт не стоил денег»:
#   business        -- фаундер проверяет идею перед вложением сил и денег;
#   social_contract -- человек (часто самозанятый/безработный) готовит
#                      обоснование для комиссии соцзащиты под выплату ~350к.
# Для второго венчурная логика («не масштабируется», «нет кратного роста»)
# не просто бесполезна -- она даёт низкий балл и вердикт «не запускать»
# там, где устойчивый доход одной семьи и есть цель. См. /social-contract.
PURPOSE_BUSINESS = "business"
PURPOSE_SOCIAL_CONTRACT = "social_contract"
PURPOSES = (PURPOSE_BUSINESS, PURPOSE_SOCIAL_CONTRACT)

_PERSONA = {
    PURPOSE_BUSINESS: """Ты — жёсткий аналитик венчурного фонда. Зарабатываешь тем, что
находишь структурные проблемы в идеях ДО того, как в них вложат время и
деньги — а не тем, что хвалишь идеи.""",
    PURPOSE_SOCIAL_CONTRACT: """Ты — экономист, который готовит обоснование бизнес-плана для комиссии
по социальному контракту. Заявитель — обычный человек, часто самозанятый
или безработный, который просит у государства единовременную выплату
(до ~350 000 ₽) на своё дело. Комиссия отказывает не из-за «недостаточной
амбициозности», а из-за необоснованных сумм в смете, нереалистичной выручки
и непонимания, на что конкретно уйдут деньги.
Венчурные критерии здесь НЕ применяй: масштабируемость, кратный рост, выход
на другие рынки — неуместны и вредны. Устойчивый доход одного человека или
семьи — это нормальная и достаточная цель. Но честность сохраняй: если дело
в текущем виде не прокормит заявителя или смета не сойдётся — скажи прямо,
лучше узнать это до комиссии, чем получить отказ.""",
}

_VIABILITY_SPEC = {
    PURPOSE_BUSINESS: """честная оценка жизнеспособности идеи В
   ТЕКУЩЕМ виде, не оптимистичная скидка. 85+ — редкость, такой балл отдельно
   обоснуй в viability_summary.""",
    PURPOSE_SOCIAL_CONTRACT: """насколько дело реально прокормит
   заявителя и насколько смета выдержит вопросы комиссии. Оценивай не
   амбициозность, а реалистичность: сходится ли экономика, хватит ли выплаты
   на старт, есть ли платящие клиенты рядом. Крепкое локальное дело без роста
   — это высокий балл, а не низкий.""",
}

_FINANCE_SPEC = {
    PURPOSE_BUSINESS: """   - "finance" — ОБЯЗАТЕЛЬНО с конкретными числами в рублях: стартовые
     затраты по статьям, постоянные и переменные расходы в месяц, средний чек,
     сколько клиентов в месяц нужно для выхода в ноль, срок окупаемости.
     Отказ считать («недостаточно данных для расчёта») недопустим — расчёт
     это и есть то, за что заплатили. Мало вводных — прими разумные для этой
     ниши в России допущения и назови их прямо: «считаю при среднем чеке N ₽,
     конверсии M%, аренде K ₽/мес». Отдельной фразой пометь, что это оценка
     по допущениям, а не гарантия.""",
    PURPOSE_SOCIAL_CONTRACT: """   - "finance" — главная секция, комиссия читает её внимательнее всего.
     Обязательно и с конкретными суммами в рублях:
     (а) смета расходов — на что пойдут деньги выплаты, построчно, с суммами
         и итогом; итог не должен превышать ~350 000 ₽, а если делу столько не
         нужно — честно назови меньшую сумму, это только усиливает заявку;
     (б) план доходов на первый год — средний чек, число клиентов в месяц,
         как выручка растёт от месяца к месяцу;
     (в) регулярные расходы в месяц и точка безубыточности;
     (г) срок окупаемости и сколько останется заявителю на руки.
     Отказ считать («недостаточно данных») недопустим: прими разумные для
     России и этой ниши цифры и назови допущения явно. Отдельной фразой
     пометь, что это расчёт-обоснование по допущениям, а не гарантия дохода.""",
}

_LAUNCH_SPEC = {
    PURPOSE_BUSINESS: f"""   - "launch" — план из 7 этапов пути (нумеруй с 1 по 7): {", ".join(STAGE_NAMES)}.
     Идея и Спрос уже пройдены пользователем бесплатно — план начинай с
     ближайшего следующего шага, то есть с «{STAGE_NAMES[2]}».""",
    PURPOSE_SOCIAL_CONTRACT: """   - "launch" — практический план запуска дела, без венчурных этапов и
     проверки гипотез: что подготовить до подачи заявления, что сделать сразу
     после одобрения (оформить статус самозанятого или ИП, закупить по смете),
     где брать первых клиентов, и как отчитаться перед соцзащитой за
     потраченные деньги — за отчётность спрашивают, о ней часто забывают.""",
}


def _system_prompt(tier: str, purpose: str = PURPOSE_BUSINESS) -> str:
    if purpose not in PURPOSES:
        purpose = PURPOSE_BUSINESS
    keys = QUICK_KEYS if tier == "quick" else [k for k, _ in ALL_SECTIONS]
    sections_spec = "\n".join(f'    "{k}": "текст секции «{title}»"' for k, title in ALL_SECTIONS if k in keys)
    risk_count = _risk_count(tier)
    return f"""{_PERSONA[purpose]} Тебе дали идею и РЕАЛЬНЫЕ данные
бесплатной проверки спроса: частотность Вордстата, реальные конкуренты из
выдачи Яндекса, оценка идеи. Используй эти цифры и названия конкурентов
буквально — не выдумывай другие.

Как писать:
- Ищи структурные причины, почему модель может не сработать: похожие паттерны
  в этой нише, как реально зарабатывают деньги похожие бизнесы, что обычно
  убивает такие проекты. Не список нейтральных наблюдений, а причинно-
  следственные цепочки: «если X не решить — то Y», «это работает только там,
  где Z, иначе — дорогая декорация без экономики».
- Если идея сильная — говори прямо и объясняй, почему именно ЭТА идея
  убедительна, а не любая похожая на неё формулировка общих слов.
- Никакого маркетингового жаргона и общих фраз («отличная идея!», «рынок
  огромен», «нужно протестировать гипотезу»). Конкретика, цифры, названия.
- Плоский текст без markdown (**, ##, списки через *) внутри значений JSON.

Обязательные элементы ответа:
1. viability_score — целое 1-100: {_VIABILITY_SPEC[purpose]}
2. viability_summary — 1-2 предложения: главная причина именно такого балла.
3. top_risks — ровно {risk_count} структурных риска, каждый:
   {{"title": "короткое ёмкое название риска, 3-6 слов",
     "body": "конкретное объяснение, почему это реально может провалить
     идею, 2-3 предложения"}}.
   Риски — о структуре бизнеса (повторяемость покупок, стоимость привлечения
   клиента, легко ли скопировать конкурентам, что убивает такие ниши обычно),
   а не общие фразы вида «может не быть спроса».
4. sections — секции ниже, 2-5 абзацев каждая (\\n\\n между абзацами):
{sections_spec}
{_FINANCE_SPEC[purpose]}
{_LAUNCH_SPEC[purpose]}
   - "verdict" — прямой вывод: запускать / дорабатывать / не запускать в
     текущем виде. Вердикт без права сказать «нет» — не вердикт.

Ответь ТОЛЬКО валидным JSON без markdown-обёртки:
{{
 "viability_score": 0,
 "viability_summary": "...",
 "top_risks": [{{"title": "...", "body": "..."}}, ...],
 "sections": {{
{sections_spec}
 }}
}}"""


class ReportEngineError(Exception):
    """Человекочитаемая ошибка — показывается пользователю как есть."""


def _validate(data: dict, tier: str) -> dict:
    if not isinstance(data, dict):
        raise ReportEngineError("ответ не JSON-объект")

    score = data.get("viability_score")
    if not isinstance(score, (int, float)) or not (1 <= score <= 100):
        raise ReportEngineError("нет корректного viability_score")

    summary = str(data.get("viability_summary") or "").strip()
    if not summary:
        raise ReportEngineError("нет viability_summary")

    expected_risks = _risk_count(tier)
    risks_raw = data.get("top_risks")
    if not isinstance(risks_raw, list) or len(risks_raw) < expected_risks:
        raise ReportEngineError("недостаточно top_risks")
    risks = []
    for r in risks_raw[:expected_risks]:
        title = str((r or {}).get("title") or "").strip()
        body = str((r or {}).get("body") or "").strip()
        if not title or not body:
            raise ReportEngineError("риск без title/body")
        risks.append({"title": title, "body": body})

    sections = data.get("sections")
    if not isinstance(sections, dict):
        raise ReportEngineError("sections должен быть объектом")
    keys = QUICK_KEYS if tier == "quick" else [k for k, _ in ALL_SECTIONS]
    out_sections = []
    for key, title in ALL_SECTIONS:
        if key not in keys:
            continue
        body = sections.get(key)
        if not body or not str(body).strip():
            raise ReportEngineError(f"в отчёте нет секции {key}")
        out_sections.append({"key": key, "title": title, "body": str(body).strip()})

    return {
        "viability_score": int(score),
        "viability_summary": summary,
        "top_risks": risks,
        "sections": out_sections,
    }


async def _call_llm(system: str, user: str, max_tokens: int, *, _post=None) -> str:
    """Платный отчёт использует более сильную модель Yandex AI Studio, чем
    дефолтный дешёвый DeepSeek Flash остального проекта -- качество текста
    определяет, оправдана ли цена. Провайдер не переопределяем (остаётся
    "yandex" / LLM_PROVIDER) -- Anthropic/Claude для этого проекта не
    используется в принципе (см. докстринг модуля)."""
    return await llm_adapter.call(system, user, max_tokens, model=SOZDATEL_REPORT_MODEL, _post=_post)


async def generate_report(idea: str, demand_data: dict, tier: str = "quick",
                          chosen_offer: dict | None = None, purpose: str = PURPOSE_BUSINESS,
                          *, _post=None, _attempt: int = 1) -> dict:
    """
    Идея + данные бесплатной проверки спроса -> отчёт: viability_score,
    viability_summary, top_risks, sections. Бросает ReportEngineError с
    человеческим текстом при любой проблеме.

    purpose меняет и оптику, и требования к секциям (см. PURPOSES): для
    соцконтракта это обоснование сметы для комиссии, а не венчурный разбор.
    """
    idea = (idea or "").strip()[:MAX_IDEA_CHARS]
    if len(idea) < 15:
        raise ReportEngineError("Идея слишком короткая для отчёта.")
    if tier not in ("quick", "full"):
        tier = "quick"
    if purpose not in PURPOSES:
        purpose = PURPOSE_BUSINESS

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
        text = await _call_llm(_system_prompt(tier, purpose), user_msg, max_tokens, _post=_post)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        return _validate(json.loads(text), tier)
    except ReportEngineError:
        raise
    except llm_adapter.LLMAdapterError as exc:
        raise ReportEngineError(str(exc))
    except json.JSONDecodeError:
        logger.exception("report engine: bad JSON (attempt %s)", _attempt)
        if _attempt == 1:
            return await generate_report(idea, demand_data, tier, chosen_offer, purpose, _post=_post, _attempt=2)
        raise ReportEngineError("ИИ ответил в неожиданном формате. Попробуйте ещё раз.")
    except httpx.TimeoutException:
        logger.warning("report engine: timeout (attempt %s)", _attempt)
        if _attempt == 1:
            return await generate_report(idea, demand_data, tier, chosen_offer, purpose, _post=_post, _attempt=2)
        raise ReportEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
    except Exception:
        logger.exception("report engine failed")
        raise ReportEngineError("Не получилось собрать отчёт. Попробуйте ещё раз через минуту.")
