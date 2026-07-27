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
import socket
import ssl
from email.message import EmailMessage

logger = logging.getLogger(__name__)


SSL_PORT = 465        # шифрование с первого байта; всё остальное -- STARTTLS
SMTP_TIMEOUT = 15


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


def looks_like_email(contact: str) -> bool:
    """Письмо можно отправить только на почту. Контакт для чека 54-ФЗ
    разрешает и телефон (см. payments.valid_receipt_contact), поэтому
    отправитель обязан спросить, а не пытаться и падать."""
    c = (contact or "").strip()
    if "@" not in c or c.startswith("@") or " " in c:
        return False
    local, _, domain = c.partition("@")
    return bool(local) and "." in domain and not domain.startswith(".") \
        and not domain.endswith(".")


def notify_buyer(to: str, subject: str, body: str, *, _send=None) -> bool:
    """Письмо ПОКУПАТЕЛЮ о его собственном заказе.

    Отдельно от notify_owner: адрес приходит параметром, а не из окружения,
    и контакт может оказаться телефоном -- тогда письма просто нет.

    Как и владельческое, НИКОГДА не бросает: человек уже заплатил, и сбой
    SMTP не имеет права превратиться в ошибку на его экране (принцип
    «деградация вместо ошибки»). Возвращает True, если письмо ушло --
    вызывающая сторона по этому флагу решает, помечать ли заказ.
    """
    if not looks_like_email(to):
        logger.info("notify_buyer: контакт не похож на почту, пропускаем")
        return False
    if not configured() and _send is None:
        logger.info("notify_buyer: SMTP не настроен, пропускаем")
        return False
    try:
        send(to, subject, body, _send=_send)
        return True
    except Exception:
        logger.warning("notify_buyer failed", exc_info=True)
        return False


def _explain(exc: Exception, port: int) -> str:
    """Ошибку SMTP — словами, которыми человек сможет что-то сделать.

    Владелец настраивает почту в чужой панели, где «разбегаются глаза»:
    получить в ответ `SMTPAuthenticationError (535, b'5.7.8 ...')` значит
    остаться ровно там же, где был.
    """
    if isinstance(exc, smtplib.SMTPAuthenticationError):
        return ("Сервер не принял логин или пароль. Логин — это адрес ящика целиком "
                "(вида info@вашдомен.ru), пароль — от самого ящика, не от панели reg.ru.")
    if isinstance(exc, smtplib.SMTPRecipientsRefused):
        return "Сервер отказался принимать письмо для этого адреса — проверьте адрес получателя."
    if isinstance(exc, (socket.timeout, TimeoutError)):
        return (f"Сервер не ответил за {SMTP_TIMEOUT} секунд. Обычно это неверный адрес "
                "сервера или закрытый порт.")
    if isinstance(exc, socket.gaierror):
        return "Такого адреса сервера не существует — проверьте SOZDATEL_SMTP_HOST."
    if isinstance(exc, ConnectionRefusedError):
        return f"Сервер отказал в соединении на порт {port} — проверьте номер порта."
    if isinstance(exc, ssl.SSLError):
        other = 587 if port == SSL_PORT else SSL_PORT
        return (f"Шифрование не сошлось на порту {port}. Обычно это значит, что порт "
                f"не тот: попробуйте {other}.")
    return "Неожиданная ошибка отправки — текст ниже, в поле technical."


def diagnose(to: str = "", *, _send=None) -> dict:
    """Owner-only проверка почты: что настроено и уходит ли письмо на самом деле.

    Без неё владелец настраивает четыре переменные вслепую и узнаёт результат
    только по тому, пожаловался ли покупатель. Тот же приём, что уже выручил
    с Вордстатом (см. demand.diagnose): показать сырую причину вместо гадания.

    Пароль НЕ возвращаем ни при каких условиях — только факт, что он задан.
    """
    host = os.environ.get("SOZDATEL_SMTP_HOST", "")
    port_raw = os.environ.get("SOZDATEL_SMTP_PORT", "465")
    user = os.environ.get("SOZDATEL_SMTP_USER", "")
    password = os.environ.get("SOZDATEL_SMTP_PASSWORD", "")
    try:
        port = int(port_raw)
    except ValueError:
        port = -1

    out = {
        "settings": {
            "host": host or "(не задан)",
            "port": port_raw if port > 0 else f"(не число: {port_raw!r})",
            "user": user or "(не задан)",
            "password_set": bool(password),
            "owner_email": owner_email() or "(не задан)",
            "mode": "SSL" if port == SSL_PORT else "STARTTLS",
        },
        "configured": configured(),
        "problems": [],
        "test_send": None,
    }
    for name, value in (("SOZDATEL_SMTP_HOST", host), ("SOZDATEL_SMTP_USER", user),
                        ("SOZDATEL_SMTP_PASSWORD", password)):
        if not value:
            out["problems"].append(f"{name} не задана — без неё письма не уходят.")
    if port <= 0:
        out["problems"].append(f"SOZDATEL_SMTP_PORT должна быть числом, сейчас {port_raw!r}.")
    if user and not looks_like_email(user):
        out["problems"].append(
            "SOZDATEL_SMTP_USER должен быть адресом ящика целиком, а не именем пользователя.")
    if not owner_email():
        out["problems"].append(
            "SOZDATEL_OWNER_EMAIL не задана — письма об оплатах и сбоях никуда не уйдут.")

    if not to:
        return out
    if not looks_like_email(to):
        out["test_send"] = {"to": to, "ok": False,
                            "error": "Это не похоже на почтовый адрес."}
        return out
    if not out["configured"] and _send is None:
        out["test_send"] = {"to": to, "ok": False,
                            "error": "Почта не настроена — отправлять нечем."}
        return out
    try:
        send(to, "Создатель: проверка почты",
             "Это тестовое письмо. Если вы его видите — отправка с сервера работает.\n"
             "Проверьте заодно, не попало ли оно в спам.\n", _send=_send)
        out["test_send"] = {"to": to, "ok": True,
                            "error": None,
                            "next": "Письмо ушло. Проверьте ящик и папку «Спам»."}
    except Exception as e:                       # noqa: BLE001 -- диагностика ловит всё
        cause = e.__cause__ or e.__context__ or e
        out["test_send"] = {"to": to, "ok": False,
                            "error": _explain(cause, port),
                            "technical": f"{type(cause).__name__}: {cause}"[:500]}
    return out


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
        _transmit(host, port, user, password, msg)
    except Exception:
        logger.warning("mailer send failed", exc_info=True)
        raise MailerError("Не получилось отправить письмо. Попробуйте ещё раз через минуту.")


def _transmit(host: str, port: int, user: str, password: str, msg: EmailMessage) -> None:
    """Отдаёт письмо серверу тем способом, которого требует порт.

    Провайдеры дают два адреса на выбор: 465 говорит по SSL с первого байта,
    587 начинает открыто и поднимает шифрование командой STARTTLS. Код умел
    только первый, и владелец, вписавший в настройки 587 (reg.ru показывает
    оба), получал бы невнятную ошибку SSL вместо работающей почты -- при
    полностью верных логине и пароле. Порт здесь и решает, а не догадка.
    """
    if port == SSL_PORT:
        with smtplib.SMTP_SSL(host, port, timeout=SMTP_TIMEOUT) as smtp:
            smtp.login(user, password)
            smtp.send_message(msg)
        return
    with smtplib.SMTP(host, port, timeout=SMTP_TIMEOUT) as smtp:
        smtp.ehlo()
        smtp.starttls()
        smtp.ehlo()
        smtp.login(user, password)
        smtp.send_message(msg)
