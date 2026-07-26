"""
Почта для Создателя. Два сценария:

1. Письмо со ссылкой входа в личный кабинет покупателя (без пароля).
   contact уже обязателен для чека оплаты (см. payments.py) -- почта у нас
   уже есть на каждый платный заказ, magic-link даёт способ вернуться к
   своим проектам/отчётам без учётной записи с паролем.
2. Уведомление ВЛАДЕЛЬЦУ о том, что требует его вмешательства (см.
   notify_owner): например, оплата прошла, а отчёт не собрался. Без этого
   единственный, кто узнаёт о сбое доставки платной услуги -- сам
   покупатель, а он про это не сообщит.

Обычный SMTP-ящик (reg.ru Mail-1), не транзакционный сервис -- объём
писем маленький: одно письмо на попытку входа, рассылок нет.

Деградация без настроек: если SOZDATEL_SMTP_* не заданы, configured() ==
False -- вызывающая сторона решает, что делать (см. main.py).
"""

from __future__ import annotations

import logging
import os
import smtplib
from email.message import EmailMessage

logger = logging.getLogger(__name__)


class MailerError(Exception):
    """Человекочитаемая ошибка отправки письма."""


def configured() -> bool:
    return bool(
        os.environ.get("SOZDATEL_SMTP_HOST")
        and os.environ.get("SOZDATEL_SMTP_USER")
        and os.environ.get("SOZDATEL_SMTP_PASSWORD")
    )


def owner_email() -> str:
    """Куда писать владельцу. Пусто -- уведомления просто не уходят."""
    return (os.environ.get("SOZDATEL_OWNER_EMAIL") or "").strip()


def notify_owner(subject: str, body: str, *, _send=None) -> bool:
    """Уведомление владельцу о событии, требующем вмешательства.

    НИКОГДА не бросает исключение: это побочный канал, и сбой уведомления не
    имеет права ломать путь пользователя (принцип «деградация вместо ошибки»).
    Возвращает True, если письмо ушло -- вызывающая сторона по этому флагу
    решает, помечать ли событие как «владелец уже знает».
    """
    to = owner_email()
    if not to:
        logger.info("notify_owner: SOZDATEL_OWNER_EMAIL не задан, пропускаем")
        return False
    if not configured() and _send is None:
        logger.info("notify_owner: SMTP не настроен, пропускаем")
        return False
    try:
        send(to, subject, body, _send=_send)
        return True
    except Exception:
        logger.warning("notify_owner failed", exc_info=True)
        return False


def send(to: str, subject: str, body: str, *, _send=None) -> None:
    """Отправляет одно текстовое письмо.

    _send(msg: EmailMessage) -- инъекция для тестов: подставляет то, что
    сделал бы реальный SMTP, без сети. Без неё и без настроек -- MailerError,
    а не молчаливая деградация: письмо со ссылкой входа не опция, а весь смысл
    вызова этой функции.
    """
    host = os.environ.get("SOZDATEL_SMTP_HOST", "")
    port = int(os.environ.get("SOZDATEL_SMTP_PORT", "465"))
    user = os.environ.get("SOZDATEL_SMTP_USER", "")
    password = os.environ.get("SOZDATEL_SMTP_PASSWORD", "")
    if not (host and user and password) and _send is None:
        raise MailerError("Почта не настроена на сервере.")

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)

    if _send is not None:
        _send(msg)
        return
    try:
        with smtplib.SMTP_SSL(host, port, timeout=15) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
    except Exception:
        logger.warning("mailer send failed", exc_info=True)
        raise MailerError("Не получилось отправить письмо. Попробуйте ещё раз через минуту.")
