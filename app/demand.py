"""
Ступень «Спрос» -- бесплатная проверка идеи до лендинга и рекламы.

Три источника, каждый деградирует независимо (пользователь всегда получает
максимум из доступного, а не ошибку 500):

1. Формулировки -- LLM (llm_adapter) переводит описание идеи в 3 коротких
   поисковых запроса, как их набрал бы клиент в Яндексе.
2. Частотность -- Wordstat API внутри Yandex Cloud Search API v2
   (searchapi.api.cloud.yandex.net/v2/wordstat/topRequests). ВАЖНО: старый
   отдельный OAuth-доступ к Вордстату (oauth.yandex.ru, ClientID/secret,
   заявка в поддержку) Яндекс упразднил -- функциональность перенесена
   в Search API и авторизуется тем же YANDEX_API_KEY + YANDEX_FOLDER_ID,
   что и остальные Yandex-вызовы. Отдельный токен заводить не нужно.
3. Конкуренты -- тот же Yandex Search API v2, только /v2/web/search;
   сервисному аккаунту нужна роль search-api.webSearch.user -- тот же
   паттерн, что в yandex_search.py АвтоПоста.
"""

from __future__ import annotations

import base64
import json
import logging
import os
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
    "которыми потенциальный КЛИЕНТ искал бы такую услугу или товар. Запросы должны быть "
    "разными по формулировке, без названий брендов, без кавычек, строчными буквами. "
    "Ответь ТОЛЬКО JSON-массивом из 3 строк, без пояснений."
)

_IDEA_SYSTEM = (
    "Придумай одну конкретную бизнес-идею для России: понятная услуга или продукт "
    "для ясной аудитории. Одно-два предложения, обычными словами, без названий брендов, "
    "без слов «стартап», «платформа», «экосистема». Каждый раз — другая ниша: "
    "быт, малый бизнес, здоровье, дети, ремонт, еда, обучение, питомцы, авто и т.д. "
    "Ответь ТОЛЬКО текстом идеи, без кавычек и пояснений."
)


async def generate_idea(*, _post=None) -> str:
    """Одна идея для тех, кто пришёл без своей. Немного повышенная температура
    задана на уровне LLM-конфига; разнообразие ниш просим в промпте."""
    try:
        text = await llm_adapter.call(_IDEA_SYSTEM, "Придумай идею.", 300, _post=_post)
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


async def wordstat_count(phrase: str, *, _post=None) -> int | None:
    """Месячная частотность фразы по России. None = сервис недоступен
    (нет ключа/папки, нет квоты, сеть) -- это штатная деградация, не ошибка."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if (not api_key or not folder_id) and _post is None:
        return None
    payload = {"phrase": phrase, "regions": [RUSSIA_REGION], "folderId": folder_id}
    try:
        if _post is not None:
            data = await _post("wordstat", payload)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
                resp = await client.post(
                    WORDSTAT_URL, json=payload,
                    headers={"Authorization": f"Api-Key {api_key}",
                             "Content-Type": "application/json"},
                )
                if resp.status_code != 200:
                    logger.warning("wordstat HTTP %s: %s", resp.status_code, resp.text[:200])
                    return None
                data = resp.json()
        count = data.get("totalCount")
        return int(count) if count is not None else None
    except Exception:
        logger.warning("wordstat_count failed for %r", phrase, exc_info=True)
        return None


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
    if best_count is None:
        return {"level": "unknown",
                "text": "Частотность сейчас недоступна — проверьте формулировки в Вордстате вручную."}
    if best_count >= THRESHOLD_STRONG:
        return {"level": "strong",
                "text": f"Спрос есть: лучшую формулировку ищут {best_count:,} раз в месяц.".replace(",", " ")}
    if best_count >= THRESHOLD_NICHE:
        return {"level": "niche",
                "text": "Нишевый спрос: людей немного, но они есть. Стоит проверить на живом трафике."}
    return {"level": "weak",
            "text": "В поиске идею почти не ищут. Это не приговор — но продавать придётся «в холодную». "
                    "Попробуйте переформулировать идею или сузить аудиторию."}


async def check_demand(idea: str, *, _post=None) -> dict:
    """Полная бесплатная проверка: формулировки -> частотности -> конкуренты
    по лучшей формулировке -> вердикт."""
    phrases = await generate_formulations(idea, _post=_post)
    counts = [await wordstat_count(p, _post=_post) for p in phrases]
    rows = [{"phrase": p, "count": c} for p, c in zip(phrases, counts)]
    known = [c for c in counts if c is not None]
    best_idx = counts.index(max(known)) if known else 0
    comp = await competitors(phrases[best_idx], _post=_post)
    return {
        "formulations": rows,
        "best_phrase": phrases[best_idx],
        "verdict": _verdict(max(known) if known else None),
        "competitors": comp,
    }
