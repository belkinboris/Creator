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


async def _wordstat_cloud_raw(phrase: str, *, _post=None) -> dict:
    """Сырой вызов прежнего пути -- Wordstat-прокси внутри Yandex Cloud
    Search API v2, авторизация Api-Key сервисного аккаунта."""
    api_key = os.environ.get("YANDEX_API_KEY")
    folder_id = os.environ.get("YANDEX_FOLDER_ID")
    if (not api_key or not folder_id) and _post is None:
        return {"ok": False, "skipped": "YANDEX_API_KEY/YANDEX_FOLDER_ID не заданы"}
    payload = {"phrase": phrase, "regions": [RUSSIA_REGION], "folderId": folder_id}
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


async def wordstat_count(phrase: str, *, _post=None) -> int | None:
    """Месячная частотность фразы по России. Пробует официальный Wordstat
    API первым (если сконфигурирован), при неуспехе -- прежний Cloud Search
    API путь. None = оба пути недоступны/без данных -- штатная деградация,
    не ошибка."""
    oauth = await _wordstat_oauth_raw(phrase, _post=_post)
    if oauth.get("ok"):
        count = (oauth.get("data") or {}).get("totalCount")
        if count is not None:
            return int(count)
    elif "status" in oauth or "error" in oauth:
        logger.warning("wordstat oauth path failed for %r: %s", phrase, oauth)

    cloud = await _wordstat_cloud_raw(phrase, _post=_post)
    if cloud.get("ok"):
        count = (cloud.get("data") or {}).get("totalCount")
        return int(count) if count is not None else None
    if "status" in cloud or "error" in cloud:
        logger.warning("wordstat cloud path failed for %r: %s", phrase, cloud)
    return None


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
    best = max(known) if known else None
    llm_scores = await score_idea(idea, rows, comp, _post=_post)
    scores = [{"key": "demand", "label": "Спрос", "value": _demand_score(best), "note": ""}]
    scores += llm_scores or []
    # Один общий балл читается за секунду; 4 шкалы -- расшифровка под ним.
    rated = [s for s in scores if s["value"] is not None]
    overall = None
    if rated:
        weakest = min(rated, key=lambda s: s["value"])
        overall = {"value": round(sum(s["value"] for s in rated) / len(rated)),
                   "weakest": weakest["label"]}
    return {
        "formulations": rows,
        "best_phrase": phrases[best_idx],
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
