"""
ЮКасса для Создателя -- тот же провайдер, что уже работает в TruePost.

Деградация без ключей: если YOOKASSA_SHOP_ID / YOOKASSA_SECRET_KEY не заданы,
configured() == False и заказ живого теста сохраняется как заявка без оплаты --
сайт работает до настройки кассы, ничего не падает.

Безопасность вебхука: тело вебхука НЕ считается доверенным. По payment_id из
события мы сами запрашиваем платёж у ЮКассы (GET /v3/payments/{id}) и верим
только этому ответу -- подделать его без секретного ключа нельзя.
"""

from __future__ import annotations

import logging
import os
import re
import uuid

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://api.yookassa.ru/v3/payments"

# Чек 54-ФЗ: без объекта receipt ЮКасса отклоняет платёж целиком (HTTP 400
# "Receipt is missing or illegal") -- это не опция, а обязательное поле для
# счетов с включённой фискализацией. vat_code зависит от налогового режима
# продавца -- 1 = "без НДС" (УСН "доходы", самозанятый, патент и т.п.).
YOOKASSA_VAT_CODE = int(os.environ.get("YOOKASSA_VAT_CODE", "1"))

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_PHONE_RE = re.compile(r"^\+?\d[\d\s\-()]{6,}\d$")


class PaymentsError(Exception):
    """Человекочитаемая ошибка оплаты."""


def configured() -> bool:
    return bool(os.environ.get("YOOKASSA_SHOP_ID") and os.environ.get("YOOKASSA_SECRET_KEY"))


def _auth() -> tuple[str, str]:
    return (os.environ.get("YOOKASSA_SHOP_ID", ""), os.environ.get("YOOKASSA_SECRET_KEY", ""))


def valid_receipt_contact(contact: str) -> bool:
    """ЮКасса на этом магазине отклоняет платёж целиком без customer.email
    или customer.phone в чеке (см. _receipt) -- значит "телеграм или почта"
    для платных заказов больше не любой текст ≥5 символов, а именно
    email/телефон. Вызывающая сторона проверяет это ДО create_payment,
    чтобы вернуть понятную ошибку, а не 502 после неудачного похода в API."""
    contact = (contact or "").strip()
    return bool(_EMAIL_RE.match(contact) or _PHONE_RE.match(contact))


def _receipt(amount_rub: int, description: str, contact: str) -> dict:
    """Чек для одной услуги. contact у нас — «телеграм или почта», без
    жёсткого формата: если похоже на email/телефон, кладём в customer (тогда
    ЮКасса ещё и отправит чек покупателю), иначе -- чек всё равно валиден,
    просто без адресата доставки."""
    item = {
        "description": description[:128],
        "quantity": "1.00",
        "amount": {"value": f"{amount_rub}.00", "currency": "RUB"},
        "vat_code": YOOKASSA_VAT_CODE,
        "payment_subject": "service",
        "payment_mode": "full_payment",
    }
    receipt: dict = {"items": [item]}
    contact = (contact or "").strip()
    if _EMAIL_RE.match(contact):
        receipt["customer"] = {"email": contact}
    elif _PHONE_RE.match(contact):
        receipt["customer"] = {"phone": re.sub(r"[^\d+]", "", contact)}
    return receipt


async def create_payment(order_id: int, amount_rub: int, description: str,
                         return_url: str, *, kind: str = "livetest", contact: str = "", _post=None) -> tuple[str, str]:
    """Создаёт платёж -> (payment_id, confirmation_url). Idempotence-Key
    привязан к заказу: повторный клик не создаст второй платёж.

    kind различает таблицы заказов (livetest / report) в вебхуке -- id из
    LiveTestOrder и ReportPurchase не глобально уникальны между собой."""
    payload = {
        "amount": {"value": f"{amount_rub}.00", "currency": "RUB"},
        "capture": True,
        "confirmation": {"type": "redirect", "return_url": return_url},
        "description": description[:128],
        "metadata": {"order_id": str(order_id), "kind": kind},
        "receipt": _receipt(amount_rub, description, contact),
    }
    try:
        if _post is not None:
            data = await _post("create", payload)
        else:
            async with httpx.AsyncClient(timeout=httpx.Timeout(20.0, connect=5.0)) as client:
                resp = await client.post(
                    API_URL, json=payload, auth=_auth(),
                    headers={"Idempotence-Key": f"sozdatel-{kind}-{order_id}",
                             "Content-Type": "application/json"},
                )
                if resp.status_code not in (200, 201):
                    logger.warning("yookassa create HTTP %s: %s", resp.status_code, resp.text[:300])
                    raise PaymentsError("Не получилось создать оплату. Попробуйте ещё раз через минуту.")
                data = resp.json()
        url = (data.get("confirmation") or {}).get("confirmation_url")
        pid = data.get("id")
        if not url or not pid:
            raise ValueError("no confirmation url")
        return pid, url
    except PaymentsError:
        raise
    except Exception:
        logger.warning("yookassa create failed", exc_info=True)
        raise PaymentsError("Не получилось создать оплату. Попробуйте ещё раз через минуту.")


async def fetch_payment(payment_id: str, *, _post=None) -> dict:
    """Статус платежа напрямую у ЮКассы -- единственный доверенный источник."""
    # payment_id идёт в путь URL -- пропускаем только безопасный формат
    if not payment_id or not all(c.isalnum() or c in "-_" for c in payment_id):
        return {}
    try:
        if _post is not None:
            return await _post("fetch", {"payment_id": payment_id})
        async with httpx.AsyncClient(timeout=httpx.Timeout(15.0, connect=5.0)) as client:
            resp = await client.get(f"{API_URL}/{payment_id}", auth=_auth())
            if resp.status_code != 200:
                logger.warning("yookassa fetch HTTP %s", resp.status_code)
                return {}
            return resp.json()
    except Exception:
        logger.warning("yookassa fetch failed", exc_info=True)
        return {}
