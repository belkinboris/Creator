"""
Роутер LLM-провайдера -- тот же паттерн, что уже работает в TruePost
(generator.py) и Компасе: одна точка входа, переключение провайдера одной
переменной окружения без деплоя кода.

LLM_PROVIDER=yandex (по умолчанию после переезда на RF-инфраструктуру) --
DeepSeek v4 Flash через Yandex AI Studio Responses API.
LLM_PROVIDER=anthropic -- путь отката на Claude, оставлен рабочим на случай
проблем с DeepSeek/Yandex; переключается без правки кода.

Важные грабли DeepSeek через Yandex (см. rf-migration.md):
- Это Responses API (/v1/responses), а НЕ /v1/chat/completions.
- thinking-режим у DeepSeek всегда включён, Yandex не даёт его отключить --
  поэтому просим max_tokens + THINKING_BUDGET токенов, иначе ответ
  обрезается посреди скрытого reasoning, не успев дойти до текста.
- В ответе блоки type=="reasoning" -- это скрытые мысли модели, не текст
  ответа; их нужно явно отфильтровывать, а не просто брать output[0].
"""

from __future__ import annotations

import logging
import os

import httpx

logger = logging.getLogger(__name__)

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "yandex")

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
ANTHROPIC_MODEL = os.environ.get("SOZDATEL_MODEL", "claude-sonnet-4-6")

YANDEX_URL = "https://ai.api.cloud.yandex.net/v1/responses"
YANDEX_FOLDER_ID = os.environ.get("YANDEX_FOLDER_ID", "")
YANDEX_MODEL = os.environ.get("YANDEX_MODEL", "deepseek-v4-flash/latest")
YANDEX_THINKING_BUDGET = 8000  # см. YANDEX lessons в rf-migration.md
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.7"))

# DeepSeek/не-Claude модели без явного запрета норовят добавить markdown
# внутрь значений JSON-полей или шаблонные вступления -- жёстко просим этого
# не делать, отдельно от основного SYSTEM-промпта движка.
NON_CLAUDE_HARDENING = (
    "\n\nВАЖНО про формат ответа: никаких markdown-символов (**, ##, ```) "
    "внутри значений JSON-полей; никаких шаблонных вступлений; каждое поле "
    "заполняй конкретикой, а не общими фразами."
)


class LLMAdapterError(Exception):
    """Человекочитаемая ошибка обращения к LLM -- показывается пользователю."""


def _anthropic_payload(system: str, user: str, max_tokens: int) -> dict:
    return {
        "model": ANTHROPIC_MODEL,
        "max_tokens": max_tokens,
        "system": system,
        "messages": [{"role": "user", "content": user}],
    }


def _yandex_payload(system: str, user: str, max_tokens: int, *, model: str | None = None) -> dict:
    if not YANDEX_FOLDER_ID:
        raise LLMAdapterError("Сервер не настроен: не задан YANDEX_FOLDER_ID.")
    return {
        "model": f"gpt://{YANDEX_FOLDER_ID}/{model or YANDEX_MODEL}",
        "instructions": system + NON_CLAUDE_HARDENING,
        "input": user,
        "max_output_tokens": max_tokens + YANDEX_THINKING_BUDGET,
        "temperature": LLM_TEMPERATURE,
    }


def _extract_anthropic_text(data: dict) -> str:
    return "\n".join(
        b.get("text", "") for b in data.get("content", []) if b.get("type") == "text"
    )


def _extract_yandex_text(data: dict) -> str:
    parts: list[str] = []
    for block in data.get("output", []):
        if block.get("type") == "reasoning":
            continue  # скрытые thinking-токены -- не текст ответа
        for item in block.get("content", []) or []:
            if item.get("type") in ("output_text", "text"):
                parts.append(item.get("text", ""))
    return "\n".join(parts)


async def call(system: str, user: str, max_tokens: int, *, provider: str | None = None,
                model: str | None = None, _post=None) -> str:
    """
    Единая точка вызова LLM для всего приложения.

    Возвращает сырой текст ответа -- разбор markdown-обёртки и json.loads
    остаются на вызывающей стороне (это бизнес-логика конкретного движка,
    не транспорт).

    provider -- необязательный оверрайд глобального LLM_PROVIDER для одного
    конкретного вызова. Anthropic/Claude оставлен как путь отката для сценариев
    без ограничений на трансграничную передачу данных (см. комментарий вверху
    файла) -- для Создателя не используется: РФ-проект, вся обработка должна
    оставаться на инфраструктуре Yandex Cloud (152-ФЗ, Claude заблокирован
    Роскомнадзором).

    model -- оверрайд YANDEX_MODEL для одного конкретного вызова (например,
    report_engine.py платного отчёта использует более сильную модель Yandex
    AI Studio, чем дефолтный дешёвый DeepSeek Flash всего остального проекта).

    _post(provider, payload) -> raw_response_dict -- инъекция для тестов:
    подставляет то, что вернул бы соответствующий API, без сети.
    """
    provider = provider or LLM_PROVIDER

    if provider == "anthropic":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key and _post is None:
            raise LLMAdapterError("Сервер не настроен: не задан ANTHROPIC_API_KEY.")
        payload = _anthropic_payload(system, user, max_tokens)
        if _post is not None:
            data = await _post(provider, payload)
        else:
            timeout = httpx.Timeout(180.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    ANTHROPIC_URL, json=payload,
                    headers={"x-api-key": api_key,
                             "anthropic-version": "2023-06-01",
                             "content-type": "application/json"},
                )
                if resp.status_code != 200:
                    logger.warning("llm_adapter anthropic HTTP %s: %s", resp.status_code, resp.text[:200])
                    raise LLMAdapterError("Не получилось обратиться к ИИ. Попробуйте ещё раз через минуту.")
                data = resp.json()
        return _extract_anthropic_text(data)

    elif provider == "yandex":
        api_key = os.environ.get("YANDEX_API_KEY")
        if not api_key and _post is None:
            raise LLMAdapterError("Сервер не настроен: не задан YANDEX_API_KEY.")
        payload = _yandex_payload(system, user, max_tokens, model=model)
        if _post is not None:
            data = await _post(provider, payload)
        else:
            timeout = httpx.Timeout(180.0, connect=10.0)
            async with httpx.AsyncClient(timeout=timeout) as client:
                resp = await client.post(
                    YANDEX_URL, json=payload,
                    headers={"Authorization": f"Api-Key {api_key}",
                             "content-type": "application/json"},
                )
                if resp.status_code != 200:
                    logger.warning("llm_adapter yandex HTTP %s: %s", resp.status_code, resp.text[:200])
                    raise LLMAdapterError("Не получилось обратиться к ИИ. Попробуйте ещё раз через минуту.")
                data = resp.json()
        return _extract_yandex_text(data)

    else:
        raise LLMAdapterError(f"Неизвестный LLM_PROVIDER: {provider!r}")
