"""
Ступень «Спрос» -- бесплатная проверка идеи до лендинга и рекламы.

Три источника, каждый деградирует независимо (пользователь всегда получает
максимум из доступного, а не ошибку 500):

1. Формулировки -- LLM (llm_adapter) переводит описание идеи в 3 коротких
   поисковых запроса, как их набрал бы клиент в Яндексе.
2. Частотность -- ДВА независимых пути, пробуются по очереди (2026-07):
   а) официальный Wordstat API (api.wordstat.yandex.net, Bearer OAuth-токен
      из приложения на oauth.yandex.ru с доступом «Вордстат» + одобрение
      Яндекса) -- отдельный продукт, включается через YANDEX_WORDSTAT_OAUTH_TOKEN;
   б) прокси внутри Yandex Cloud Search API v2
      (searchapi.api.cloud.yandex.net/v2/wordstat/topRequests), авторизация
      YANDEX_API_KEY + YANDEX_FOLDER_ID -- прежний путь, оставлен как есть.
   Раньше здесь было написано, что путь (а) упразднён и слит в (б) -- это
   не подтвердилось на практике (см. /api/diag/yandex): похоже, это два
   разных продукта Яндекса, и уверенности в эквивалентности нет. Поэтому
   оба пути живут параллельно, а не взаимоисключают друг друга.
3. Конкуренты -- Yandex Search API v2, /v2/web/search; сервисному аккаунту
   нужна роль search-api.webSearch.user -- тот же паттерн, что в
   yandex_search.py АвтоПоста.
"""

from __future__ import annotations

import base64
import json
import logging
import os
import random
import re
import xml.etree.ElementTree as ET

import httpx

from app import llm_adapter

logger = logging.getLogger(__name__)

WORDSTAT_URL = os.environ.get(
    "YANDEX_WORDSTAT_URL", "https://searchapi.api.cloud.yandex.net/v2/wordstat/topRequests"
)
SEARCH_URL = os.environ.get(
    "YANDEX_SEARCH_URL", "https://searchapi.api.cloud.yandex.net/v2/web/search"
)
# Официальный Wordstat API (не Cloud Search API) -- отдельная авторизация,
# см. docstring модуля. Путь эндпоинта не подтверждён официальной докой
# (недоступна для чтения на момент написания) -- если Яндекс вернёт 404,
# поправить YANDEX_WORDSTAT_OAUTH_PATH без переката кода.
WORDSTAT_OAUTH_URL = os.environ.get(
    "YANDEX_WORDSTAT_OAUTH_URL", "https://api.wordstat.yandex.net"
)
WORDSTAT_OAUTH_PATH = os.environ.get("YANDEX_WORDSTAT_OAUTH_PATH", "/v1/topRequests")
RUSSIA_REGION = "225"  # geo-код России (строкой -- см. примеры в доке Wordstat API)

MAX_IDEA_CHARS = 300
FORMULATIONS_COUNT = 3

# Пороги вердикта по суммарной месячной частотности лучшей формулировки.
# Калибровка: <300/мес -- спроса в поиске почти нет; 300..3000 -- нишевый
# спрос (проверять стоит); >3000 -- спрос явно есть.
THRESHOLD_NICHE = 300
THRESHOLD_STRONG = 3000

_FORMULATIONS_SYSTEM = (
    "Ты помогаешь проверить спрос на бизнес-идею через статистику поиска Яндекса. "
    "По описанию идеи составь ровно 3 коротких поисковых запроса (2-4 слова каждый), "
    "которыми потенциальный КЛИЕНТ искал бы такую услугу или товар. "
    "Это должны быть массовые, ходовые формулировки -- как реально пишут в Яндексе, "
    "а не точный пересказ идеи: убирай уточнения вроде точного возраста, района, "
    "состава услуги -- если с ними никто не ищет, пользы в них нет. Если идея узкая, "
    "хотя бы одну из трёх формулировок сделай более широкой (родовая категория товара "
    "или услуги), чтобы не мерить спрос на нулевой хвост запросов. Запросы должны быть "
    "разными по формулировке, без названий брендов, без кавычек, строчными буквами. "
    "Ответь ТОЛЬКО JSON-массивом из 3 строк, без пояснений."
)

_IDEA_SYSTEM = (
    "Придумай одну конкретную бизнес-идею для России: понятная услуга или продукт "
    "для ясной аудитории. Одно-два предложения, обычными словами, без названий брендов, "
    "без слов «стартап», «платформа», «экосистема». "
    "Ответь ТОЛЬКО текстом идеи, без кавычек и пояснений."
)

# Модель со своим промптом склонна сходиться к одной и той же «самой вероятной»
# идее при повторных вызовах, даже с ненулевой температурой (замечено на
# практике -- детские мастер-классы повторялись слишком часто). Рандомизируем
# нишу в самом промпте, а не полагаемся только на temperature.
_IDEA_NICHES = [
    "быт и уборка", "малый бизнес и услуги для бизнеса", "здоровье и медицина",
    "ремонт и стройка", "еда и напитки", "обучение и репетиторство",
    "питомцы и животные", "авто и мото", "красота и уход за собой",
    "спорт и фитнес", "путешествия и досуг", "работа с документами и бюрократией",
    "садоводство и дача", "техника и электроника", "одежда и вещи",
]


async def generate_idea(*, _post=None) -> str:
    """Одна идея для тех, кто пришёл без своей. Ниша выбирается случайно и
    передаётся прямо в промпт -- иначе модель при одинаковом запросе слишком
    часто возвращает один и тот же «типичный» ответ."""
    niche = random.choice(_IDEA_NICHES)
    try:
        text = await llm_adapter.call(_IDEA_SYSTEM, f"Придумай идею в сфере: {niche}.", 300, _post=_post)
        idea = text.strip().strip('"«»').strip()
        if len(idea) < 15:
            raise ValueError("too short")
        return idea[:MAX_IDEA_CHARS]
    except llm_adapter.LLMAdapterError as exc:
        raise DemandError(str(exc))
    except Exception:
        logger.warning("generate_idea failed", exc_info=True)
        raise DemandError("Не получилось придумать идею. Попробуйте ещё раз.")


class DemandError(Exception):
    """Человекочитаемая ошибка -- показывается пользователю как есть."""


async def generate_formulations(idea: str, *, _post=None) -> list[str]:
    """Идея -> 3 поисковых формулировки. Единственный обязательный шаг:
    без формулировок проверять нечего, поэтому ошибки здесь не глотаем."""
    idea = (idea or "").strip()[:MAX_IDEA_CHARS]
    if len(idea) < 15:
        raise DemandError("Опишите идею хотя бы одним предложением: кому и чем она помогает.")
    try:
        text = await llm_adapter.call(_FORMULATIONS_SYSTEM, f"Идея:\n{idea}", 500, _post=_post)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        phrases = [str(p).strip().lower() for p in data if str(p).strip()]
        if not phrases:
            raise ValueError("empty")
        return phrases[:FORMULATIONS_COUNT]
    except DemandError:
        raise
    except llm_adapter.LLMAdapterError as exc:
        raise DemandError(str(exc))
    except Exception:
        logger.warning("generate_formulations: не удалось разобрать ответ LLM", exc_info=True)
        raise DemandError("Не получилось разобрать идею. Попробуйте переформулировать и повторить.")


async def _wordstat_oauth_raw(phrase: str, *, _post=None) -> dict:
    """Сырой вызов официального Wordstat API (Bearer OAuth) -- отдельный
    продукт от Cloud Search API, см. docstring модуля. Включается только
    если задан YANDEX_WORDSTAT_OAUTH_TOKEN -- иначе штатно пропускается,
    вообще не трогая сеть (и не трогая инъекцию _post в тестах)."""
    token = os.environ.get("YANDEX_WORDSTAT_OAUTH_TOKEN")
    if not token:
        return {"ok": False, "skipped": "YANDEX_WORDSTAT_OAUTH_TOKEN не задан"}
    payload = {"phrase": phrase, "regions": [RUSSIA_REGION]}
    try:
        if _post is not None:
            data = await _post("wordstat_oauth", payload)
            return {"ok": True, "data": data}
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            resp = await client.post(
                f"{WORDSTAT_OAUTH_URL}{WORDSTAT_OAUTH_PATH}", json=payload,
                headers={"Authorization": f"Bearer {token}",
                         "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return {"ok": False, "status": resp.status_code, "body": resp.text[:500]}
            return {"ok": True, "data": resp.json()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


WORDSTAT_RELATED_COUNT = 20  # сколько похожих формулировок просить у Вордстата вместе с totalCount


async def _wordstat_cloud_raw(phrase: str, *, _post=None) -> dict:
    """Сырой вызов прежнего пути -- Wordstat-прокси внутри Yandex Cloud
    Search API v2, авторизация Api-Key сервисного аккаунта."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if (not api_key or not folder_id) and _post is None:
        return {"ok": False, "skipped": "YANDEX_API_KEY/YANDEX_FOLDER_ID не заданы"}
    # num_phrases обязателен (1..2000) -- без него API отвечал 400 "Value must
    # be in the range of 1 to 2000" на КАЖДЫЙ запрос. Раньше здесь стоял
    # минимум (1) в убеждении, что нам нужен только totalCount самой фразы --
    # это оказалось недальновидно (см. кастдев: LLM угадала «создание
    # рекламного видео» 157/мес, а реальный ходовой запрос для той же идеи --
    # «нейросеть для рекламы», 902/мес). Публичные примеры использования этого
    # эндпоинта используют camelCase (numPhrases) -- шлём оба варианта имени
    # поля, чтобы не зависеть от неподтверждённой офдокой схемы (см. докстринг
    # модуля), и просим больше похожих формулировок, чтобы реально их видеть.
    payload = {"phrase": phrase, "regions": [RUSSIA_REGION], "folderId": folder_id,
               "num_phrases": WORDSTAT_RELATED_COUNT, "numPhrases": WORDSTAT_RELATED_COUNT}
    try:
        if _post is not None:
            data = await _post("wordstat", payload)
            return {"ok": True, "data": data}
        async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
            resp = await client.post(
                WORDSTAT_URL, json=payload,
                headers={"Authorization": f"Api-Key {api_key}",
                         "Content-Type": "application/json"},
            )
            if resp.status_code != 200:
                return {"ok": False, "status": resp.status_code, "body": resp.text[:200]}
            return {"ok": True, "data": resp.json()}
    except Exception as exc:
        return {"ok": False, "error": repr(exc)}


def _best_related(data: dict, fallback_phrase: str, fallback_count: int | None) -> dict:
    """Cloud Search API отдаёт вместе с totalCount список похожих формулировок
    (topRequests: [{"phrase", "count"}, ...]) -- то, что и так предлагает сам
    Вордстат рядом с точной фразой. Если среди них частотность выше, чем у
    дословно запрошенной фразы -- значит наша (LLM-угаданная) формулировка
    промахнулась мимо реального ходового запроса, а Вордстат его тут же
    показывает. Берём максимум, а не то, что дословно спросили -- и ЗАПОМИНАЕМ
    формулировку-победителя (не только число): показывать чужой счёт под
    исходной фразой было бы нечестно -- человек вручную проверит именно её
    в Вордстате, увидит другое число и решит, что сервис врёт или сломан."""
    best_phrase, best_count = fallback_phrase, fallback_count
    for item in data.get("topRequests") or []:
        if not isinstance(item, dict):
            continue
        phrase, count = item.get("phrase"), item.get("count")
        if (isinstance(phrase, str) and phrase.strip()
                and isinstance(count, (int, float)) and (best_count is None or count > best_count)):
            best_phrase, best_count = phrase.strip().lower(), int(count)
    return {"phrase": best_phrase, "count": best_count}


async def wordstat_best(phrase: str, *, _post=None) -> dict:
    """Частотность формулировки + при необходимости более популярная похожая
    формулировка от самого Вордстата. Возвращает {"phrase": str, "count":
    int|None} -- phrase в ответе может отличаться от входной, см. _best_related.
    Пробует официальный Wordstat API первым (если сконфигурирован), при
    неуспехе -- прежний Cloud Search API путь. count=None = оба пути
    недоступны/без данных -- штатная деградация, не ошибка."""
    oauth = await _wordstat_oauth_raw(phrase, _post=_post)
    if oauth.get("ok"):
        count = (oauth.get("data") or {}).get("totalCount")
        if count is not None:
            return {"phrase": phrase, "count": int(count)}
    elif "status" in oauth or "error" in oauth:
        logger.warning("wordstat oauth path failed for %r: %s", phrase, oauth)

    cloud = await _wordstat_cloud_raw(phrase, _post=_post)
    if cloud.get("ok"):
        data = cloud.get("data") or {}
        count = data.get("totalCount")
        count = int(count) if count is not None else None
        return _best_related(data, phrase, count)
    if "status" in cloud or "error" in cloud:
        logger.warning("wordstat cloud path failed for %r: %s", phrase, cloud)
    return {"phrase": phrase, "count": None}


async def wordstat_count(phrase: str, *, _post=None) -> int | None:
    """Только частотность, без данных о формулировке-победителе -- обратная
    совместимость для вызовов, которым не нужна замена фразы."""
    return (await wordstat_best(phrase, _post=_post))["count"]


async def diagnose(phrase: str = "купить слона", *, _post=None) -> dict:
    """Отладка интеграции с Яндексом для владельца (owner-only ручка
    /api/diag/yandex): сырые ответы ОБОИХ путей Вордстата, без глотания
    ошибок -- чтобы увидеть точную причину «нет данных» вместо гадания."""
    oauth = await _wordstat_oauth_raw(phrase, _post=_post)
    cloud = await _wordstat_cloud_raw(phrase, _post=_post)
    return {
        "env": {
            "yandex_api_key_set": bool(os.environ.get("YANDEX_API_KEY")),
            "yandex_folder_id_set": bool(os.environ.get("YANDEX_FOLDER_ID")),
            "wordstat_oauth_token_set": bool(os.environ.get("YANDEX_WORDSTAT_OAUTH_TOKEN")),
        },
        "wordstat_oauth_api": oauth,
        "wordstat_cloud_api": cloud,
    }


def _parse_search_xml(xml_text: str) -> dict:
    """Из XML выдачи достаём число найденных документов и топ-3 (title, domain)."""
    root = ET.fromstring(xml_text)
    found = None
    for f in root.iter("found"):
        if f.get("priority") == "all" and f.text and f.text.isdigit():
            found = int(f.text)
            break
        if found is None and f.text and f.text.isdigit():
            found = int(f.text)
    top = []
    for doc in root.iter("doc"):
        url = doc.findtext("url") or ""
        title_el = doc.find("title")
        title = "".join(title_el.itertext()).strip() if title_el is not None else ""
        domain = re.sub(r"^https?://(www\.)?", "", url).split("/")[0]
        if domain:
            top.append({"title": title[:120], "domain": domain})
        if len(top) >= 3:
            break
    return {"found": found, "top": top}


async def competitors(phrase: str, *, _post=None) -> dict:
    """Кто уже в выдаче по фразе. Всё fail-soft: {'found': None, 'top': []}
    при любой проблеме -- сервис продолжает работать без блока конкурентов."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    empty = {"found": None, "top": []}
    if (_post is None) and (not api_key or not folder_id):
        return empty
    payload = {
        "query": {"searchType": "SEARCH_TYPE_RU", "queryText": phrase, "folderId": folder_id},
    }
    try:
        if _post is not None:
            data = await _post("search", payload)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
                resp = await client.post(
                    SEARCH_URL, json=payload,
                    headers={"Authorization": f"Api-Key {api_key}"},
                )
                if resp.status_code != 200:
                    logger.warning("search HTTP %s: %s", resp.status_code, resp.text[:200])
                    return empty
                data = resp.json()
        raw = data.get("rawData")
        if not raw:
            return empty
        return _parse_search_xml(base64.b64decode(raw).decode("utf-8", errors="replace"))
    except Exception:
        logger.warning("competitors failed for %r", phrase, exc_info=True)
        return empty


def _verdict(best_count: int | None) -> dict:
    """Что означают цифры спроса -- словами, которыми человек думает сам.

    Вердикт СООБЩАЕТ находку и не предписывает следующий шаг: шаг у двух
    аудиторий разный (фаундеру -- проверка на людях, получателю соцконтракта
    -- бизнес-план для комиссии), и его уже подбирает CTA страницы по
    DemandCheck.purpose. Раньше здесь стояло «Стоит проверить на живом
    трафике» -- и жаргон, и совет мимо половины аудитории.
    """
    if best_count is None:
        return {"level": "unknown",
                "text": "Данные Яндекса о числе запросов сейчас недоступны — цифр по этой идее "
                        "у нас пока нет. Попробуйте позже или проверьте фразы сами "
                        "на wordstat.yandex.ru."}
    if best_count >= THRESHOLD_STRONG:
        return {"level": "strong",
                "text": f"Спрос есть: самую популярную из этих фраз ищут {best_count:,} раз в месяц.".replace(",", " ")}
    if best_count >= THRESHOLD_NICHE:
        return {"level": "niche",
                "text": "Спрос небольшой, но он есть: людей мало, и каждый клиент будет на счету."}
    return {"level": "weak",
            "text": "В поиске эту идею почти не ищут. Возможно, люди называют её другими "
                    "словами — попробуйте переформулировать. Если формулировка верная, "
                    "клиентов придётся находить самому: из поиска они не придут."}


async def check_demand(idea: str, *, _post=None) -> dict:
    """Полная бесплатная проверка: формулировки -> частотности -> конкуренты
    по лучшей формулировке -> вердикт."""
    phrases = await generate_formulations(idea, _post=_post)
    results = [await wordstat_best(p, _post=_post) for p in phrases]
    rows = []
    for p, r in zip(phrases, results):
        row = {"phrase": p, "count": r["count"]}
        if r["count"] is not None and r["phrase"] != p:
            # Вордстат сам предложил формулировку с большей частотностью --
            # показываем её отдельно, а не молча приписываем чужой счёт
            # исходной фразе (см. wordstat_best/_best_related).
            row["matched_phrase"] = r["phrase"]
        rows.append(row)
    counts = [r["count"] for r in rows]
    known = [c for c in counts if c is not None]
    best_idx = counts.index(max(known)) if known else 0
    # Конкурентов и "лучшую формулировку" ищем по реально ходовой фразе, если
    # Вордстат её подсказал -- иначе искали бы конкурентов не по тому запросу,
    # который на самом деле приносит трафик.
    search_phrase = rows[best_idx].get("matched_phrase") or phrases[best_idx]
    comp = await competitors(search_phrase, _post=_post)
    best = max(known) if known else None
    llm_scores = await score_idea(idea, rows, comp, _post=_post)
    scores = [{"key": "demand", "label": "Спрос", "value": _demand_score(best), "note": ""}]
    scores += llm_scores or []
    # Один общий балл читается за секунду; 4 шкалы -- расшифровка под ним.
    rated = [s for s in scores if s["value"] is not None]
    demand_value = scores[0]["value"] if scores and scores[0]["key"] == "demand" else None
    overall = None
    # Спрос -- ворота, а не рядовая 1/4 средней (см. ниже), но ворота не
    # открыть, если неизвестно, есть ли за ними хоть что-то: demand_value is
    # None значит Вордстат не смог ответить (нет ключа, сеть, квота) -- это
    # НЕ то же самое, что "спрос есть и он маленький" (тогда demand_value=1,
    # а не None). Раньше при недоступном Вордстате код молча усреднял
    # оставшиеся три LLM-шкалы и показывал уверенное число с подписью
    # «среднее по ЧЕТЫРЁМ шкалам» -- притом что реально участвовали три, а
    # четвёртая (спрос) не имела значения вообще. Рядом вердикт честно писал
    # «данных нет», а число рядом внушало обратное -- то же самое искажение,
    # ради которого вообще придумана эта система ворот, просто с другой
    # причиной пустого числителя.
    if rated and demand_value is not None:
        weakest = min(rated, key=lambda s: s["value"])
        avg = round(sum(s["value"] for s in rated) / len(rated))
        # Спрос -- это ворота, а не рядовая 1/4 средней: если в поиске идею
        # почти не ищут, общий балл не может быть выше балла спроса, каким бы
        # хорошим ни казался разбор конкуренции/своевременности/реализуемости --
        # без спроса они не имеют значения.
        value = min(avg, demand_value)
        # Откуда взялось число -- словами. Без этого правило «спрос это ворота»
        # выглядит как ошибка счёта: владелец сам спросил на живом прогоне,
        # как при почти нулевом спросе идея получила 6/10 (B3 в PRODUCT_ROADMAP).
        if demand_value < avg:
            basis = (f"Среднее по четырём шкалам — {avg} из 10, но итог опущен до балла "
                     "спроса: выше того, насколько идею ищут, она подняться не может.")
        else:
            basis = "Среднее по четырём шкалам: спрос, конкуренция, своевременность, реализуемость."
        overall = {"value": value, "weakest": weakest["label"], "basis": basis}
    return {
        "formulations": rows,
        "best_phrase": search_phrase,
        "verdict": _verdict(best),
        "competitors": comp,
        "scores": scores,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# Оценка идеи по 4 шкалам (стиль DimeADozen, адаптированный под РФ):
# «Спрос» -- детерминированно из цифр Вордстата (данные, не мнение);
# остальные три -- LLM с контекстом реальных конкурентов из выдачи.
# ---------------------------------------------------------------------------

_SCORE_SYSTEM = (
    "Оцени бизнес-идею для российского рынка по трём шкалам от 1 до 10:\n"
    "competition -- насколько легко выделиться (10 = ниша свободна, 1 = рынок забит сильными игроками);\n"
    "timing -- своевременность (10 = рынок готов именно сейчас, 1 = слишком рано или поздно);\n"
    "execution -- реализуемость силами одного человека или маленькой команды (10 = можно запустить за недели).\n"
    "Учитывай переданные данные о конкурентах в выдаче. К каждой шкале -- одно короткое пояснение "
    "обычными словами (до 12 слов). Ответь ТОЛЬКО JSON вида "
    '{"competition": n, "timing": n, "execution": n, '
    '"notes": {"competition": "...", "timing": "...", "execution": "..."}} без пояснений вокруг.'
)

_SCORE_LABELS = (("competition", "Конкуренция"), ("timing", "Своевременность"), ("execution", "Реализуемость"))


def _demand_score(best_count: int | None) -> int | None:
    """Шкала спроса из частотности -- по данным, без участия модели."""
    if best_count is None:
        return None
    for threshold, score in ((50_000, 10), (20_000, 9), (THRESHOLD_STRONG, 8),
                             (1_000, 6), (THRESHOLD_NICHE, 4), (50, 2)):
        if best_count >= threshold:
            return score
    return 1


async def score_idea(idea: str, rows: list, comp: dict, *, _post=None) -> list | None:
    """Три LLM-шкалы с контекстом реальной выдачи. None при любой проблеме --
    блок оценки просто не показывается, проверка спроса работает без него."""
    context = json.dumps({
        "идея": idea[:MAX_IDEA_CHARS],
        "частотности": rows,
        "конкуренты_в_выдаче": comp.get("top", []),
        "страниц_в_выдаче": comp.get("found"),
    }, ensure_ascii=False)
    try:
        text = await llm_adapter.call(_SCORE_SYSTEM, context, 800, _post=_post)
        text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(text)
        notes = data.get("notes") or {}
        out = []
        for key, label in _SCORE_LABELS:
            value = int(data[key])
            if not 1 <= value <= 10:
                raise ValueError(f"{key} out of range")
            out.append({"key": key, "label": label, "value": value,
                        "note": str(notes.get(key, ""))[:140]})
        return out
    except Exception:
        logger.warning("score_idea failed", exc_info=True)
        return None
