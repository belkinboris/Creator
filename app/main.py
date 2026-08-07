"""
Создатель v0.1 — веб-приложение нулевой стадии.

Что уже работает (этапы ⓪→①):
  1. Фаундер вводит идею → /api/offers → 3 заострённых оффера (LLM).
  2. Выбор оффера → /api/launch → генерируется smoke-лендинг из шаблона,
     сохраняется в БД и СРАЗУ хостится по адресу /l/{idea_id}.
  3. Лендинг шлёт события page_view / lead_submitted в /api/smoke-event —
     Создатель сам их собирает (никакого стороннего трекинга).
  4. /api/verdict/{idea_id} — детерминированный вердикт по порогам
     (сигнал есть / спроса нет / другой оффер / рано судить).

Отдельный репозиторий и деплой (Railway), с Аналитиком Воронки не
смешивается — интеграция позже через его connector (см. SPEC_SMOKE_MODE).

env: LLM_PROVIDER=yandex|anthropic (по умолчанию yandex; в режиме yandex
обязательны YANDEX_API_KEY и YANDEX_FOLDER_ID, в anthropic --
ANTHROPIC_API_KEY), DATABASE_URL (по умолчанию sqlite).
"""

from __future__ import annotations

import hashlib
import html
import json
import logging
import os
import re
import secrets
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from pydantic import BaseModel
from sqlmodel import Field, Session, SQLModel, create_engine, select

BASE_DIR = Path(__file__).resolve().parent
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///./sozdatel.db")
_engine_kwargs = {}
if DATABASE_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}
    if DATABASE_URL in ("sqlite://", "sqlite:///:memory:"):
        from sqlalchemy.pool import StaticPool
        _engine_kwargs["poolclass"] = StaticPool  # одна БД на все соединения (тесты)
else:
    # Postgres: держим тёплый пул. Без pre_ping первый запрос после простоя
    # ждёт таймаута мёртвого соединения -- отсюда были 10-секундные страницы.
    _engine_kwargs.update(
        pool_pre_ping=True,      # проверять живость соединения перед выдачей
        pool_recycle=280,        # пересоздавать раз в ~5 мин (Railway рвёт idle)
        pool_size=5, max_overflow=5,
        connect_args={"connect_timeout": 5},
    )
engine = create_engine(DATABASE_URL, **_engine_kwargs)

from app.offer_engine import OfferEngineError, sharpen_idea  # noqa: E402
from app.demand import DemandError, check_demand, generate_idea, diagnose  # noqa: E402
from app.report_engine import (  # noqa: E402
    ReportEngineError, generate_core, generate_section, generate_report,
    ALL_SECTIONS, QUICK_KEYS, SECTION_GROUPS, section_keys, section_title,
    PURPOSES as report_purposes,
)
from app import audiences  # noqa: E402
from app import payments  # noqa: E402
from app import mailer  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sozdatel")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


# Оплата не пришла за это время -- считаем попытку брошенной. Вычисляется на
# лету при чтении (created_at + порог < сейчас), не мутирует БД: ни воркеров,
# ни крона нет, статус просто перестаёт звать на оплату уже неживую ссылку.
PENDING_PAYMENT_TIMEOUT_MINUTES = 20
PURPOSE_DEFAULT = "business"

# Пороги вердикта теста на реальных людях -- ЕДИНСТВЕННЫЙ источник правды.
# Раньше числа были зашиты в модель, а витрины называли совсем другие: главная
# обещала «выше 2,5% — идея живая», плейбук — «дождитесь ~100 визитов», тогда
# как движок считал по 8% и 40 визитам. Человек с 3% читал на главной «идея
# живая», а в кабинете видел «СПРОСА НЕТ» -- прямое нарушение принципа 3.
# Рекламный бюджет в цену живого теста НЕ входит (так и записано в оферте) --
# человек платит его Яндексу напрямую. До оплаты об этом не говорилось нигде,
# хотя это удваивает-утраивает реальную стоимость шага (A7 в PRODUCT_ROADMAP).
AD_BUDGET_HINT = "3–5 тысяч ₽"

CLICK_TARGET = 40       # раньше этого числа визитов цифры -- шум, не результат
# A8: 8%/4% были ориентиром на тёплый трафик, для холодного Директа завышены --
# владелец решил снизить перед началом рекламы (2026-07-28). Серая зона
# сохранена той же относительной ширины (была 4-8%, стала 2-4%).
SIGNAL_RATE = 0.04      # заявок/визитов, с которых интерес считается настоящим
DEAD_RATE = 0.02        # и ниже -- интереса нет


def _plural(n: int, one: str, few: str, many: str) -> str:
    """«1 заявка», «3 заявки», «5 заявок». Без этого вердикт писал «1 заявок»
    -- мелочь, по которой сразу видно, что текст собран машиной."""
    n = abs(n)
    if n % 100 in range(11, 15):
        return many
    last = n % 10
    return one if last == 1 else few if last in (2, 3, 4) else many


def _pct(rate: float) -> str:
    """8% / 2,5% -- дробные без хвоста .0 и с запятой, как принято в русском."""
    v = round(rate * 100, 1)
    return (f"{v:.1f}".rstrip("0").rstrip(".").replace(".", ",")) + "%"


def _effective_status(status: str, created_at: datetime) -> str:
    if status == "pending_payment" and utcnow() - created_at > timedelta(minutes=PENDING_PAYMENT_TIMEOUT_MINUTES):
        return "expired"
    return status


# ---------------------------------------------------------------------------
# Модели
# ---------------------------------------------------------------------------

class SmokeProject(SQLModel, table=True):
    """Одна идея на этапе ①. Хранит выбранный оффер и сгенерированный лендинг."""
    id: Optional[int] = Field(default=None, primary_key=True)
    idea_id: str = Field(index=True, unique=True)
    product_name: str
    idea_text: str
    offer_json: str          # выбранный оффер целиком (для повторных генераций)
    landing_html: str        # захощенный лендинг
    # Отпечаток шаблона, из которого страница собрана. Правка шаблона живые
    # страницы НЕ трогает (и не должна: менять страницу под работающей
    # рекламой значит менять то, что измеряешь). Но владелец обязан видеть,
    # что страница отстала, и уметь пересобрать её сам -- A20.
    template_hash: str = ""
    click_target: int = CLICK_TARGET
    lead_rate_signal: float = SIGNAL_RATE
    lead_rate_dead: float = DEAD_RATE
    status: str = "running"  # running | signal | dead | gray
    created_at: datetime = Field(default_factory=utcnow)
    contact: str = ""        # почта покупателя -- показывается в его личном кабинете (/account)


# Шкала пути покупателя Создателя 0..6 -- те же названия на главной, в
# кабинете покупателя (/account) и на /r//p/. "Проверочная страница" и
# "Реклама" объединены в один шаг "Тест на реальных людях" -- по кастдев-
# фидбеку это ОДИН платный этап с точки зрения покупателя (мы собираем
# страницу, он по нашей инструкции запускает рекламу), раздельная нумерация
# только запутывала. НЕ путать с TRACKED_STAGE_NAMES ниже -- это разные
# сущности с разной длиной шкалы.
STAGE_NAMES = ["Идея", "Спрос", "Тест на реальных людях",
               "Заявки", "Первые продажи", "Повторяемость", "Удержание"]

# Шкала для TrackedProject (внешние проекты владельца, не Создателя, см.
# докстринг класса) -- сознательно НЕ объединена с STAGE_NAMES выше и
# заморожена в исходном виде из 8 названий. TrackedProject.stage -- сырое
# целое число, уже хранящееся в БД для существующих внешних проектов
# (например АвтоПост); если бы эта шкала менялась вместе с STAGE_NAMES,
# старые записи стали бы указывать не на те этапы (а stage=7 вообще упал бы
# по IndexError). Общий язык с покупательской шкалой этим двум сущностям не
# нужен -- у внешнего проекта нет привязки к тому, как именно устроена
# воронка Создателя.
TRACKED_STAGE_NAMES = ["Идея", "Спрос", "Проверочная страница", "Реклама",
                       "Заявки", "Первые продажи", "Повторяемость", "Удержание"]


class TrackedProject(SQLModel, table=True):
    """Внешний проект в кабинете: живёт не в Создателе (например, АвтоПост
    ведёт Аналитик в Telegram), но виден на общей карте портфеля со своим
    этапом. Мост, а не переезд: ссылка ведёт в родной интерфейс проекта."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    stage: int = 0                 # 0..7, индекс в TRACKED_STAGE_NAMES
    status_note: str = ""          # одна строка: что происходит сейчас
    external_link: str = ""        # куда идти за деталями (бот, кабинет)
    created_at: datetime = Field(default_factory=utcnow)


class DemandCheck(SQLModel, table=True):
    """Каждая бесплатная проверка спроса: счётчик + страница результата /r/<...>."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
    # Адрес страницы результата. Раньше в нём стоял порядковый номер, и
    # чужую идею можно было прочитать, набрав соседний: 42 -> 41. Ссылкой на
    # результат люди делятся намеренно, поэтому вход не требуем -- просто
    # делаем адрес неугадываемым (E6, тот же приём, что у ReportPurchase).
    public_id: str = Field(default_factory=lambda: secrets.token_urlsafe(9), index=True)
    idea: str = ""
    best_count: Optional[int] = None
    result_json: str = ""
    # Пусто, пока человек не привязал бесплатную проверку к кабинету -- см.
    # POST /api/demand/{id}/save и автопривязку на /r/ для уже вошедших.
    contact: str = ""
    # С какой стороны человек пришёл: "business" (главная, фаундер) или
    # "social_contract" (лендинг /social-contract, выплата от государства).
    # Определяет оптику платного отчёта -- см. PURPOSES в report_engine.
    purpose: str = "business"
    # Бесплатный образец платного разбора: балл, объяснение, названные риски
    # и один настоящий раздел. Генерируется лениво при первом открытии
    # /report/{id} без оплаты и кэшируется навсегда -- см. _ensure_sample.
    sample_json: str = ""
    # JSON выбранного на /r/ заострённого позиционирования. Живёт здесь, а не
    # на заказе, потому что отчёт заказывают уже со страницы /report/{check_id},
    # которая знает только check_id: иначе выбор человека до отчёта не доезжает.
    chosen_offer: str = ""


class LiveTestOrder(SQLModel, table=True):
    """Заказ этапа 2 «живой тест». Статусы: new (заявка без оплаты),
    pending_payment, paid."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
    check_id: Optional[int] = None
    idea: str = ""
    contact: str = ""
    status: str = "new"
    payment_id: str = ""
    amount: int = 0
    chosen_offer: str = ""    # JSON: полный оффер, выбранный на /r/{id} -- см. LAUNCH_REQUIRED_FIELDS
    paid_notified: bool = False   # владельцу сообщили об оплате/заявке
    buyer_notified: bool = False    # покупателю сообщили о его заказе -- см. ReportPurchase
    idea_id: Optional[str] = None   # проставляется автозапуском/владельцем -- ссылка на запущенный SmokeProject


class ReportPurchase(SQLModel, table=True):
    """Заказ отчёта/бизнес-плана: quick (990₽) или full (2990₽). Отчёт
    генерируется лениво при первом открытии /report/{id} после оплаты --
    без фоновых воркеров, тот же принцип, что и весь проект."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
    check_id: Optional[int] = None
    idea: str = ""
    tier: str = "quick"       # quick | full
    contact: str = ""
    status: str = "new"       # new | pending_payment | paid
    payment_id: str = ""
    amount: int = 0
    report_json: str = ""     # заполняется лениво после оплаты
    # Оплата прошла, а генерация упала -- худший сценарий платного продукта.
    # Причина видна владельцу в /desk. Два РАЗНЫХ флага уведомлений: об
    # оплате и о сбое доставки. Один на двоих означал бы, что письмо об
    # оплате гасит более важное письмо о том, что услуга не оказана.
    gen_error: str = ""
    fail_notified: bool = False   # владельцу сообщили о сорванной доставке
    paid_notified: bool = False   # владельцу сообщили о самой оплате/заявке
    # Письмо ПОКУПАТЕЛЮ -- отдельный флаг, не общий с владельческим: письмо
    # владельцу и письмо покупателю решают разные задачи, и успех одного не
    # имеет права погасить второе (тот же урок, что fail_notified в A2).
    buyer_notified: bool = False
    # Публичный пример на /example. Настоящий сгенерированный отчёт, а не
    # написанный руками: показывать более гладкий текст, чем отдаёт движок,
    # значит продавать не то, что отдаём (принцип 3). Помечает владелец.
    is_example: bool = False
    # Ключ от собственного отчёта. Страница /report/{check_id} адресуется
    # порядковым номером проверки, то есть чужой оплаченный бизнес-план
    # открывался перебором: 42 -> 41. Токен уходит в return_url оплаты и в
    # ссылку из кабинета; кто вошёл в кабинет своей почтой, проходит и без
    # него (см. _report_access_ok).
    access_token: str = Field(default_factory=lambda: secrets.token_urlsafe(16))


class MagicLinkToken(SQLModel, table=True):
    """Одноразовая ссылка входа в /account -- письмом на contact, без пароля.
    Короткий срок жизни (см. MAGIC_LINK_TTL_MINUTES): это только подтверждение
    почты, долгую сессию после перехода по ссылке несёт AccountSession."""
    id: Optional[int] = Field(default=None, primary_key=True)
    token: str = Field(index=True, unique=True)
    contact: str
    created_at: datetime = Field(default_factory=utcnow)
    used: bool = False


class AccountSession(SQLModel, table=True):
    """Долгая сессия после перехода по magic-link -- токен лежит в cookie
    браузера, contact ищется по нему при каждом заходе на /account."""
    id: Optional[int] = Field(default=None, primary_key=True)
    token: str = Field(index=True, unique=True)
    contact: str
    created_at: datetime = Field(default_factory=utcnow)


class SmokeEvent(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    idea: str = Field(index=True)
    event: str               # page_view | lead_submitted
    source: str = ""
    campaign: str = ""
    content: str = ""
    term: str = ""
    contact: str = ""        # только у lead_submitted; добровольный контакт
    created_at: datetime = Field(default_factory=utcnow)


SQLModel.metadata.create_all(engine)
# create_all не добавляет колонки в существующие таблицы -- добиваем вручную.
# `ADD COLUMN IF NOT EXISTS` -- диалект Postgres; SQLite (dev) на него падает
# синтаксической ошибкой (проверено: sqlite3 3.45 не понимает эту форму даже
# в новой версии -- это не вопрос возраста движка). На свежей sqlite-БД это
# было незаметно: create_all создаёт таблицу сразу со всеми колонками, ALTER
# просто не нужен. Но на БД, пережившей хотя бы одно добавление колонки в
# прошлой сессии (ровно то, для чего dev вообще держит файл `sozdatel.db`
# между запусками, а не только `sqlite://` в памяти, как в тестах) -- первый
# же ALTER падал с "syntax error", единственный `except: pass` глотал ВСЮ
# оставшуюся часть списка разом, и следующее же обращение к новой колонке
# роняло страницу 500-й (принцип 7 требует ровно обратного). Проверяем
# наличие колонки через инспектор БД (работает одинаково для sqlite и
# Postgres) и добавляем только то, чего действительно не хватает -- по
# одной колонке за раз, чтобы сбой одной не глушил остальные.
_MIGRATIONS = [
    ("demandcheck", "result_json", "VARCHAR DEFAULT ''"),
    ("livetestorder", "chosen_offer", "VARCHAR DEFAULT ''"),
    ("smokeproject", "contact", "VARCHAR DEFAULT ''"),
    ("livetestorder", "idea_id", "VARCHAR"),
    ("demandcheck", "contact", "VARCHAR DEFAULT ''"),
    ("demandcheck", "purpose", "VARCHAR DEFAULT 'business'"),
    ("reportpurchase", "gen_error", "VARCHAR DEFAULT ''"),
    ("reportpurchase", "fail_notified", "BOOLEAN DEFAULT FALSE"),
    ("reportpurchase", "paid_notified", "BOOLEAN DEFAULT FALSE"),
    ("livetestorder", "paid_notified", "BOOLEAN DEFAULT FALSE"),
    ("demandcheck", "chosen_offer", "VARCHAR DEFAULT ''"),
    ("reportpurchase", "is_example", "BOOLEAN DEFAULT FALSE"),
    ("demandcheck", "sample_json", "VARCHAR DEFAULT ''"),
    ("reportpurchase", "access_token", "VARCHAR DEFAULT ''"),
    ("reportpurchase", "buyer_notified", "BOOLEAN DEFAULT FALSE"),
    ("demandcheck", "public_id", "VARCHAR DEFAULT ''"),
    ("livetestorder", "buyer_notified", "BOOLEAN DEFAULT FALSE"),
    ("smokeproject", "template_hash", "VARCHAR DEFAULT ''"),
]
try:
    from sqlalchemy import inspect as _sa_inspect, text as _sqltext
    _inspector = _sa_inspect(engine)
    _existing_cols = {
        _table: {c["name"] for c in _inspector.get_columns(_table)}
        for _table in {m[0] for m in _MIGRATIONS}
    }
    with engine.connect() as _c:
        for _table, _col, _ddl in _MIGRATIONS:
            if _col in _existing_cols.get(_table, set()):
                continue
            try:
                _c.execute(_sqltext(f"ALTER TABLE {_table} ADD COLUMN {_col} {_ddl}"))
                _c.commit()
            except Exception:
                logging.getLogger(__name__).warning(
                    "миграция не прошла: %s.%s", _table, _col, exc_info=True)
except Exception:
    logging.getLogger(__name__).warning("проверка колонок при старте не удалась", exc_info=True)

try:  # Проверки, сделанные до появления неугадываемого адреса, иначе
    # остались бы доступны по порядковому номеру (E6). Досыпаем по строке.
    with Session(engine) as _s:
        _old = _s.exec(select(DemandCheck).where(
            (DemandCheck.public_id == "") | (DemandCheck.public_id.is_(None)))).all()
        for _row in _old:
            _row.public_id = secrets.token_urlsafe(9)
            _s.add(_row)
        if _old:
            _s.commit()
except Exception:
    logging.getLogger(__name__).warning("backfill public_id failed", exc_info=True)

try:  # Покупки, оформленные до появления токена, остались бы без ключа от
    # собственного отчёта: в кабинет по своей почте владелец покупки войдёт,
    # а прямая ссылка перестала бы работать. Досыпаем по строке, не пачкой --
    # токен на то и токен, что у каждой покупки свой.
    with Session(engine) as _s:
        _old = _s.exec(select(ReportPurchase).where(
            (ReportPurchase.access_token == "") | (ReportPurchase.access_token.is_(None)))).all()
        for _row in _old:
            _row.access_token = secrets.token_urlsafe(16)
            _s.add(_row)
        if _old:
            _s.commit()
except Exception:
    logging.getLogger(__name__).warning("backfill access_token failed", exc_info=True)

# Прогреваемые на старте страницы -- список вручную, а не сканирование
# static/: часть файлов там не HTML-шаблоны (fonts/, README). Отдельная
# константа, а не тело _lifespan, чтобы тест мог проверить, что каждое имя
# всё ещё существует на диске -- переименование/удаление файла (как
# social-contract.html -> audience-landing.html в F2) иначе тихо переставало
# прогреваться: исключение в _lifespan поймано и лишь залогировано (принцип
# 7), так что сам сервис не падает, просто первый визит снова платит за всё.
PREWARM_STATIC_PAGES = ("index.html", "project.html", "guide-direct.html", "result.html",
                        "report.html", "audience-landing.html", "account.html", "verify.html")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Прогрев при старте: соединение с БД и статика читаются ДО первого
    запроса пользователя. Без этого первый визит платил за всё сразу."""
    try:
        with Session(engine) as s:
            s.exec(select(SmokeProject.id).limit(1)).first()
    except Exception:
        logger.exception("warm-up db failed (non-fatal)")
    for name in PREWARM_STATIC_PAGES:
        try:
            _static(name)
        except Exception:
            logger.exception("warm-up static %s failed", name)
    yield


app = FastAPI(title="Создатель", version="1.0.2", lifespan=_lifespan)

# Ключ владельца: закрывает генерацию офферов, запуск и удаление лендингов.
# Публичными остаются только /l/{id}, /api/smoke-event, /health -- им и
# положено быть открытыми (их дергают браузеры посетителей лендингов).
# Пока Создателем пользуется один владелец, этого достаточно; полноценные
# аккаунты -- этап внешних пользователей (P2 в VISION).
OWNER_KEY = os.environ.get("SOZDATEL_OWNER_KEY", "")

# Счётчик Яндекс.Метрики -- не секрет (виден в исходнике любой страницы), но
# задаётся через env, как и остальная конфигурация проекта: нет ID -- нет
# вставки кода, дев/тесты работают без счётчика.
YANDEX_METRIKA_ID = os.environ.get("YANDEX_METRIKA_ID", "")


def _check_owner(request: Request) -> None:
    if not OWNER_KEY:
        raise HTTPException(503, "Сервер не настроен: задайте SOZDATEL_OWNER_KEY в переменных окружения.")
    provided = request.headers.get("X-Owner-Key") or request.query_params.get("key") or ""
    if provided != OWNER_KEY:
        raise HTTPException(401, "Нужен ключ владельца (X-Owner-Key).")


def _is_owner(request: Request) -> bool:
    """Тот же ключ, что и в _check_owner, но без исключения -- нужен там, где
    владельцу показывается что-то сверх обычной страницы, а посторонний
    должен просто увидеть обычную."""
    provided = request.headers.get("X-Owner-Key") or request.query_params.get("key") or ""
    return bool(OWNER_KEY and provided == OWNER_KEY)


def _project_access_ok(request: Request, proj: "SmokeProject") -> bool:
    """Доступ к цифрам проекта на /p/{id}: владелец по ключу (как везде на
    /desk), либо покупатель по своей сессии кабинета -- /p/ теперь открыт
    и из /account, а не только из /desk, значит нельзя требовать секретный
    ключ владельца у обычного покупателя."""
    provided = request.headers.get("X-Owner-Key") or request.query_params.get("key") or ""
    if OWNER_KEY and provided == OWNER_KEY:
        return True
    contact = _current_contact(request)
    return bool(contact and proj.contact and contact == proj.contact)


def _find_check(s: Session, key: str, request: Request):
    """Проверка спроса по адресу страницы. Возвращает (запись, надо_ли_редирект).

    Адрес — неугадываемый `public_id`. Порядковый номер тоже принимаем, но
    ТОЛЬКО у того, кто и так имеет право видеть эту проверку: владельца по
    ключу или хозяина проверки по сессии кабинета. Им же отдаём редирект на
    канонический адрес, чтобы старая закладка сама починилась.

    Постороннему с порядковым номером — ничего: именно так чужая идея и
    читалась перебором 42 -> 41 (E6).
    """
    rec = s.exec(select(DemandCheck).where(DemandCheck.public_id == key)).first()
    if rec:
        return rec, False
    if not key.isdigit():
        return None, False
    rec = s.get(DemandCheck, int(key))
    if not rec:
        return None, False
    contact = _current_contact(request)
    mine = bool(contact and rec.contact and contact == rec.contact)
    if _is_owner(request) or mine:
        return rec, True
    return None, False


def _report_access_ok(request: Request, purchase: "ReportPurchase") -> bool:
    """Кому открывать ОПЛАЧЕННЫЙ отчёт на /report/{check_id}.

    Страница адресуется порядковым номером проверки спроса, и до этого
    оплаченный бизнес-план отдавался любому, кто наберёт номер: 42 -> 41.
    Утекал не только текст за 2990 ₽, но и чужие деньги -- посторонний мог
    через /api/report/{id}/section гонять генерацию по чужой покупке.

    Три двери, все три -- у настоящего покупателя:
      · владелец по ключу (как везде);
      · ссылка с токеном -- уходит в return_url оплаты и в кабинет;
      · сессия кабинета с той же почтой, на которую оформлен заказ, --
        человек, потерявший ссылку, входит по magic-link и открывает отчёт
        без всякого токена.
    """
    if _is_owner(request):
        return True
    token = (request.query_params.get("t") or "").strip()
    if token and purchase.access_token and secrets.compare_digest(token, purchase.access_token):
        return True
    contact = _current_contact(request)
    return bool(contact and purchase.contact and contact == purchase.contact)


def _report_link(purchase: "ReportPurchase") -> str:
    """Ссылка на отчёт для его покупателя -- по неугадываемому адресу
    проверки и сразу с токеном, чтобы скопированная из кабинета ссылка
    открывалась и в другом браузере."""
    if not purchase.check_id:
        return ""
    with Session(engine) as s:
        rec = s.get(DemandCheck, purchase.check_id)
    if not rec:
        return ""
    tok = purchase.access_token or ""
    return f"/report/{rec.public_id}" + (f"?t={tok}" if tok else "")


# ---------------------------------------------------------------------------
# Этап ⓪: идея → офферы
# ---------------------------------------------------------------------------

class IdeaIn(BaseModel):
    idea: str
    # Откуда пришёл человек -- /social-contract шлёт "social_contract",
    # обычная главная ничего не шлёт и получает дефолт (см. DemandCheck.purpose).
    purpose: str = "business"


@app.post("/api/offers")
async def offers(data: IdeaIn, request: Request):
    _check_owner(request)
    try:
        result = await sharpen_idea(data.idea, purpose=data.purpose)
        return {"ok": True, **_polish_offers(result)}
    except OfferEngineError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Этап ①: бесплатная проверка спроса (публичная — это вход воронки)
# ---------------------------------------------------------------------------

@app.post("/api/idea")
async def idea_suggest(request: Request):
    """«Придумать за меня» — для тех, кто пришёл без идеи (вход воронки)."""
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    if _llm_fuse_blown():
        raise HTTPException(429, _FUSE_MSG)
    try:
        return {"ok": True, "idea": await generate_idea()}
    except DemandError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/demand")
async def demand_check(data: IdeaIn, request: Request):
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    if _llm_fuse_blown():
        raise HTTPException(429, _FUSE_MSG)
    try:
        result = await check_demand(data.idea)
    except DemandError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    check_id = public_id = None
    try:  # сохранение не должно уметь ломать ответ пользователю
        known = [f["count"] for f in result["formulations"] if f["count"] is not None]
        # Уже вошедший в кабинет человек получает автопривязку без лишних
        # действий -- проверка сразу видна в /account, без отдельного "Сохранить".
        contact = _current_contact(request) or ""
        purpose = data.purpose if data.purpose in report_purposes else "business"
        rec = DemandCheck(idea=data.idea[:300], best_count=max(known) if known else None,
                          result_json=json.dumps(result, ensure_ascii=False), contact=contact,
                          purpose=purpose)
        with Session(engine) as s:
            s.add(rec); s.commit(); s.refresh(rec)
            check_id, public_id = rec.id, rec.public_id
    except Exception:
        logging.getLogger(__name__).warning("demand check not persisted", exc_info=True)
    # public_id -- то, из чего витрина собирает адрес /r/. Номер записи
    # остаётся для API (заказы, сохранение), но в адрес больше не попадает.
    return {"ok": True, "id": check_id, "public_id": public_id, **result}


def _polish_offers(result: dict) -> dict:
    """Приводит объяснение боли к виду предложения перед выдачей карточек.

    Модель пишет `p` как придётся -- «Обещают три недели, шьют полтора месяца»
    без точки рядом с вариантом, где точка есть. В карточке эти строки стоят
    друг под другом, и разнобой читается как небрежность (B9). Название боли
    (`h2`) не трогаем: это заголовок, точка ему не нужна.
    """
    offers = result.get("offers")
    if not isinstance(offers, list):
        return result
    fixed = []
    for o in offers:
        pains = o.get("pains") if isinstance(o, dict) else None
        if not isinstance(pains, list):
            fixed.append(o)
            continue
        fixed.append({**o, "pains": [
            {**pn, "p": _as_sentence(pn.get("p", ""))} if isinstance(pn, dict) else pn
            for pn in pains]})
    return {**result, "offers": fixed}


@app.post("/api/sharpen")
async def sharpen(data: IdeaIn, request: Request):
    """Бесплатное заострение идеи в 3 варианта позиционирования — по кнопке
    на странице результата, не на каждый визит (LLM-вызов тяжёлый и долгий)."""
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    if _llm_fuse_blown():
        raise HTTPException(429, _FUSE_MSG)
    try:
        result = await sharpen_idea(data.idea, purpose=data.purpose)
        return {"ok": True, **_polish_offers(result)}
    except OfferEngineError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


LIVE_TEST_PRICE = int(os.environ.get("SOZDATEL_LIVE_TEST_PRICE", "1490"))


@app.get("/r/{rid}", response_class=HTMLResponse)
def result_page(rid: str, request: Request):
    """Страница результата проверки: инструмент, а не витрина. Узкая полоска
    преемственности вместо всего пути 0->7; отсюда же -- заказ живого теста."""
    with Session(engine) as s:
        rec, redirect = _find_check(s, rid, request)
    if not rec or not rec.result_json:
        return _lost_page()
    if redirect:
        return RedirectResponse(f"/r/{rec.public_id}", status_code=307)
    tpl = _static("result.html")
    # Человек уже проходил эту проверку -- лента не должна начинаться заново.
    # Кастдев 2026-08-02: «нажимаю на идею «открыть», и мне будто заново всё
    # до заострения проходить надо». Прогресс не в localStorage, а в самих
    # данных: если позиционирование выбрано или по проверке уже что-то
    # заказано, шаги до финального человек прошёл наверняка -- на любом его
    # устройстве и в любом браузере.
    with Session(engine) as s:
        resume = bool(rec.chosen_offer) or bool(
            s.exec(select(ReportPurchase.id).where(ReportPurchase.check_id == rec.id)).first()
            or s.exec(select(LiveTestOrder.id).where(LiveTestOrder.check_id == rec.id)).first())
    # Что человек выбрал в прошлый раз -- чтобы свёрнутый шаг заострения при
    # возврате не был безымянной галочкой. Битый JSON тут не повод ронять
    # страницу: без подписи, но с лентой (принцип 7).
    chosen_h1 = ""
    if rec.chosen_offer:
        try:
            chosen_h1 = re.sub(r"<[^>]+>", "", str((json.loads(rec.chosen_offer) or {}).get("h1", "")))[:120]
        except (ValueError, TypeError, AttributeError):
            chosen_h1 = ""
    # Подписи шкал приводим к одному виду на выдаче: так чинятся и проверки,
    # сохранённые до этой правки (B9).
    polished = json.dumps(_polish_scores(json.loads(rec.result_json)), ensure_ascii=False)
    safe_json = polished.replace("</", "<\\/")
    idea_json = json.dumps(rec.idea, ensure_ascii=False).replace("</", "<\\/")
    html_out = (tpl
        .replace("__CHECK_ID__", str(rec.id))
        .replace("__PUBLIC_ID__", rec.public_id)
        .replace("__PRICE__", str(LIVE_TEST_PRICE))
        .replace("__AD_BUDGET__", AD_BUDGET_HINT)
        .replace("__PAY_ENABLED__", "true" if payments.configured() else "false")
        .replace("__IDEA__", html.escape(rec.idea))
        .replace("__IDEA_JSON__", idea_json)
        .replace("__RESULT_JSON__", safe_json)
        .replace("__SAVED__", "true" if rec.contact else "false")
        # Человек с /social-contract пришёл за бизнес-планом для комиссии, а
        # не за рекламным тестом -- страница результата разворачивает финальный
        # шаг под него, см. PURPOSE в result.html.
        .replace("__PURPOSE_JSON__", json.dumps(rec.purpose, ensure_ascii=False))
        .replace("__RESUME__", "true" if resume else "false")
        .replace("__CHOSEN_H1_JSON__", json.dumps(chosen_h1, ensure_ascii=False).replace("</", "<\\/"))
        .replace("__AUDIENCE_JSON__",
                 json.dumps(audiences.for_page(rec.purpose), ensure_ascii=False)))
    return HTMLResponse(_fill_server_values(html_out, rec.purpose))


class LiveTestIn(SQLModel):
    check_id: Optional[int] = None
    contact: str = ""
    chosen_offer: Optional[dict] = None


@app.post("/api/live-test")
async def live_test_order(data: LiveTestIn, request: Request):
    """Заказ этапа 2. С настроенной кассой -> ссылка на оплату;
    без ключей ЮКассы -> заявка (свяжемся вручную)."""
    contact = (data.contact or "").strip()
    if len(contact) < 5:
        return JSONResponse({"ok": False, "error": "Оставьте почту или телефон — чтобы нам было куда вернуться с результатом."}, status_code=400)
    # Оплата требует чек (54-ФЗ) с email/телефоном покупателя -- телеграм-хэндл
    # тут не годится, ЮКасса отклонит платёж на API-стороне. Проверяем формат
    # ДО похода в ЮКассу, а не после 400 из чужого API.
    if payments.configured() and not payments.valid_receipt_contact(contact):
        return JSONResponse({"ok": False, "error": "Для оплаты нужна почта или телефон — на них пришлём чек. Телеграм для этого не подходит."}, status_code=400)
    idea = ""
    # Полный оффер (не только angle/h1/sub) -- нужен целиком для автозапуска
    # проекта сразу при оплате, см. LAUNCH_REQUIRED_FIELDS и yookassa_webhook.
    chosen_offer_json = json.dumps(data.chosen_offer, ensure_ascii=False)[:6000] if data.chosen_offer else ""
    with Session(engine) as s:
        if data.check_id:
            rec = s.get(DemandCheck, data.check_id)
            idea = rec.idea if rec else ""
        order = LiveTestOrder(check_id=data.check_id, idea=idea, contact=contact[:200],
                              amount=LIVE_TEST_PRICE, chosen_offer=chosen_offer_json,
                              status="pending_payment" if payments.configured() else "new")
        s.add(order); s.commit(); s.refresh(order)
        order_id = order.id
    if not payments.configured():
        # Заявку без оплаты доводит владелец руками -- значит он должен о ней
        # узнать, а не обнаружить, открыв /desk через несколько дней.
        if _notify_owner_order(request, what="живой тест", order_id=order_id, idea=idea,
                               contact=contact, amount=LIVE_TEST_PRICE, paid=False):
            _mark_notified(LiveTestOrder, order_id)
        if _notify_buyer_order(request, kind="livetest", order_id=order_id, idea=idea,
                               contact=contact, amount=LIVE_TEST_PRICE, paid=False):
            _mark_notified(LiveTestOrder, order_id, field="buyer_notified")
        return {"ok": True, "paid": False,
                "message": "Заявка принята. Свяжемся в течение дня и соберём проверочную "
                           "страницу под вашу идею — рекламу вы запустите сами по нашей инструкции."}
    try:
        base = str(request.base_url).rstrip("/")
        # Без check_id ссылка /r/ ведёт в никуда (404) -- возвращаем на главную,
        # а не на битую страницу результата.
        return_url = f"{base}/r/{data.check_id}?paid=1" if data.check_id else f"{base}/?paid=1"
        pid, url = await payments.create_payment(
            order_id, LIVE_TEST_PRICE, f"Создатель · живой тест идеи (заказ {order_id})",
            return_url, kind="livetest", contact=contact)
        with Session(engine) as s:
            order = s.get(LiveTestOrder, order_id)
            order.payment_id = pid; s.add(order); s.commit()
        return {"ok": True, "paid": True, "confirmation_url": url}
    except payments.PaymentsError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.post("/api/yookassa/notify")
@app.post("/api/yookassa/webhook")
async def yookassa_webhook(request: Request):
    """Телу вебхука не верим: перепроверяем платёж напрямую у ЮКассы."""
    try:
        body = await request.json()
    except Exception:
        return {"ok": True}
    pid = ((body.get("object") or {}).get("id")) or ""
    if not pid:
        return {"ok": True}
    payment = await payments.fetch_payment(pid)
    if payment.get("status") != "succeeded":
        return {"ok": True}
    meta = payment.get("metadata") or {}
    order_id = meta.get("order_id")
    kind = meta.get("kind", "livetest")   # старые платежи до kind -- считаем livetest
    model = {"livetest": LiveTestOrder, "report": ReportPurchase}.get(kind, LiveTestOrder)
    notify = None
    buyer = None
    try:
        with Session(engine) as s:
            order = s.get(model, int(order_id)) if order_id else None
            if order and order.status != "paid":
                order.status = "paid"; s.add(order); s.commit()
            if order is not None and not order.buyer_notified:
                # Письмо покупателю собирается отдельно от владельческого и по
                # своему флагу: у них разные адресаты и разные задачи.
                buyer = {"kind": kind, "order_id": order.id, "idea": order.idea,
                         "contact": order.contact, "amount": order.amount,
                         "tier": getattr(order, "tier", ""),
                         "link": _report_link(order) if kind == "report" else ""}
            if order is not None and not order.paid_notified:
                # Собираем данные письма ВНУТРИ сессии, а шлём после неё:
                # SMTP может отвечать секундами, держать на нём транзакцию
                # и ответ вебхуку ЮКассы незачем.
                if kind == "report":
                    label = REPORT_PRICES.get(order.tier, {}).get("label", order.tier)
                    notify = {"what": f"отчёт «{label}»", "order_id": order.id,
                              "idea": order.idea, "contact": order.contact,
                              "amount": order.amount,
                              "link": _report_link(order)}
                else:
                    notify = {"what": "живой тест", "order_id": order.id,
                              "idea": order.idea, "contact": order.contact,
                              "amount": order.amount, "link": ""}
            # Автозапуск: если на /r/ выбрали заострённый вариант (полный
            # оффер, не только angle/h1/sub -- см. pickOffer в result.html),
            # запускаем проект сразу при оплате, без ручного вмешательства
            # владельца. contact уже есть в заказе -- проект сразу виден
            # покупателю в /account. Идеи без выбранного варианта (пропустили
            # заострение) по-прежнему запускает владелец вручную.
            if kind == "livetest" and isinstance(order, LiveTestOrder) and not order.idea_id and order.chosen_offer:
                try:
                    offer = json.loads(order.chosen_offer)
                except Exception:
                    offer = None
                if isinstance(offer, dict) and all(offer.get(k) for k in LAUNCH_REQUIRED_FIELDS):
                    proj = _launch_offer(s, offer, order.idea, contact=order.contact)
                    order.idea_id = proj.idea_id
                    s.add(order); s.commit()
    except Exception:
        logging.getLogger(__name__).warning("webhook order update failed", exc_info=True)
    if notify and _notify_owner_order(request, paid=True, **notify):
        _mark_notified(model, notify["order_id"])
    if buyer and _notify_buyer_order(request, paid=True, **buyer):
        _mark_notified(model, buyer["order_id"], field="buyer_notified")
    return {"ok": True}


# ---------------------------------------------------------------------------
# Отчёт/бизнес-план: платный разбор идеи поверх уже посчитанных данных спроса
# ---------------------------------------------------------------------------

REPORT_PRICES = {
    "quick": {"price": 990, "was": 1490, "label": "Быстрый разбор"},
    # "Полный отчёт" переименован в "Бизнес-план" -- этот тариф добавляет
    # именно то, что бизнес-планом и называют (финансы, план запуска), а не
    # ещё немного текста к тому же отчёту. Точнее описывает содержимое.
    "full": {"price": 2990, "was": 3990, "label": "Бизнес-план"},
}


_SENTENCE_END = ".!?…:"


def _as_caption(text: str) -> str:
    """Подпись под числом, а не предложение: без точки в конце.

    Заметки к шкалам пишет модель, и пишет как придётся -- «Рынок растёт.» с
    точкой рядом с «Начать можно одной» без. В сетке из четырёх ячеек это
    читается как небрежность (B9). Восклицательный и вопросительный знаки не
    трогаем: если они там есть, это часть смысла, а не пунктуационная случайность.
    """
    return str(text or "").strip().rstrip(".").strip()


def _as_sentence(text: str) -> str:
    """Тот же фрагмент, но отдельным абзацем -- там точка обязательна, иначе
    абзац выглядит оборванным (тизер отчёта)."""
    t = str(text or "").strip()
    return t if not t or t[-1] in _SENTENCE_END else t + "."


def _target_top_count(demand_data: dict) -> int | None:
    """Максимум частотности по ЦЕЛЕВЫМ фразам -- та же логика, что в
    check_demand: родовая категория (`kind == "category"`) это спрос на
    рынок целиком, не на идею, и в цифру спроса не входит. Строка без
    `kind` (старые записи в БД) читается как целевая. Если целевых с числом
    не осталось -- берём что есть: широкая цифра честнее, чем «данных нет»
    при живых числах на экране."""
    rows = demand_data.get("formulations") or []
    target = [f["count"] for f in rows
              if f.get("count") is not None and f.get("kind") != "category"]
    if target:
        return max(target)
    known = [f["count"] for f in rows if f.get("count") is not None]
    return max(known) if known else None


def _demand_caption(demand_data: dict) -> str:
    """Подпись к шкале спроса -- из её же частотности, без участия модели.

    `check_demand` кладёт сюда пустую строку: три остальные шкалы объясняет
    модель, а спрос считается по данным, и объяснять его было некому. В
    результате из четырёх ячеек одна стояла голой -- причём самая важная:
    спрос единственный посчитан по реальным цифрам Яндекса и он же потолок
    общего балла. Числа берём готовые, ничего не выдумываем (принцип 1).

    Максимум -- по целевым фразам (_target_top_count): раньше подпись брала
    max по ВСЕМ строкам и рядом с баллом, посчитанным по целевым 23 123,
    писала «400 978 запросов в месяц» из родовой категории.
    """
    top = _target_top_count(demand_data)
    if top is None:
        return "Частотность недоступна"
    return f"{top:,}".replace(",", "\u00a0") + " запросов в месяц"


def _polish_scores(demand_data: dict) -> dict:
    """Приводит подписи шкал к одному виду ПЕРЕД выдачей страницы.

    Именно на выдаче, а не при записи: так чинятся и уже сохранённые проверки,
    и правило живёт в одном месте на одном языке, а не копией в шаблоне.
    """
    scores = demand_data.get("scores")
    if not isinstance(scores, list):
        return demand_data
    out = dict(demand_data)
    fixed = []
    for sc in scores:
        if not isinstance(sc, dict):
            fixed.append(sc)
            continue
        row = dict(sc)
        note = _as_caption(row.get("note", ""))
        if not note and row.get("key") == "demand":
            note = _demand_caption(demand_data)
        row["note"] = note
        fixed.append(row)
    out["scores"] = fixed
    return out


def _report_preview(demand_data: dict) -> dict:
    """Бесплатный тизер отчёта — из уже посчитанных данных проверки спроса,
    без новых вызовов LLM (заметки по шкалам уже сгенерированы бесплатным
    шагом check_demand). Показывается всегда, вне зависимости от оплаты.
    Текстовый разбор, а не витрина из голых цифр -- см. кастдев-фидбек
    (dimeadozen как ориентир на содержательный анализ, не «воду»)."""
    v = demand_data.get("verdict") or {}
    overall = demand_data.get("overall") or {}
    # По целевым фразам, не по всем: max по всем строкам брал родовую
    # категорию, и тизер называл «спросом» цифру рынка целиком.
    top = _target_top_count(demand_data)
    comp = demand_data.get("competitors") or {}
    top_names = [c.get("domain") or c.get("title") or "" for c in (comp.get("top") or [])[:3]]
    # В тизере заметки идут отдельными абзацами -- там нужна точка, в отличие
    # от подписи под числом на /r/ (см. _as_caption/_as_sentence).
    notes = {s["key"]: _as_sentence(s.get("note", ""))
             for s in (demand_data.get("scores") or [])}
    return {
        "best_count": top,
        "verdict_text": v.get("text", ""),
        "verdict_level": v.get("level", "unknown"),
        "overall_value": overall.get("value"),
        "weakest": overall.get("weakest", ""),
        "competitors_count": len(comp.get("top") or []),
        "top_competitor_names": [n for n in top_names if n],
        "competition_note": notes.get("competition", ""),
        "timing_note": notes.get("timing", ""),
        "execution_note": notes.get("execution", ""),
    }


# Раздел, который отдаётся бесплатно целиком. Резюме — самый ценный раздел, и
# именно поэтому он в образце: человек не купит разбор, о качестве которого
# не может судить. Раньше бесплатная часть была пересказом цифр со страницы
# спроса — ни строчки анализа, оценить нечего, платить не за что.
SAMPLE_SECTION = "summary"


async def _ensure_sample(check_id: int) -> dict | None:
    """Бесплатный образец платного разбора: балл, объяснение, названные риски
    и один настоящий раздел целиком.

    Считается ОДИН раз НА АУДИТОРИЮ и кэшируется навсегда: это два вызова
    модели, и платит за них владелец. Генерируем лениво — только когда
    человек реально открыл страницу отчёта, то есть думает о покупке.
    Сбой не имеет права уронить страницу: без образца она просто остаётся
    такой, какой была (принцип 7 — деградация вместо ошибки).

    Кэш — словарь `{purpose: sample}`, а не один сэмпл на проверку: человек
    может переключить оптику на `/r/` (POST /api/demand/{id}/purpose) уже
    ПОСЛЕ того, как образец сгенерировался под старую аудиторию, и увидеть
    венчурный образец на витрине соцконтракта — тот самый образец, который
    существует ровно для того, чтобы убедить купить (принцип 3, найдено
    кастдев-проходом по платному пути 2026-07-29). Ключей ровно столько,
    сколько аудиторий (сейчас три) — переключение туда-обратно не может
    стоить владельцу больше одной генерации на каждую.
    """
    with Session(engine) as s:
        rec = s.get(DemandCheck, check_id)
        if not rec or not rec.result_json:
            return None
        purpose = rec.purpose or "business"
        cached = {}
        if rec.sample_json:
            try:
                parsed = json.loads(rec.sample_json)
                if isinstance(parsed, dict):
                    cached = parsed
            except ValueError:
                pass
        if purpose in cached:
            return cached[purpose]
        idea = rec.idea
        chosen = _chosen_offer(rec)
        demand_data = json.loads(rec.result_json)

    # Образец — бесплатная генерация за счёт владельца: при сгоревшем дневном
    # предохранителе страница просто остаётся без образца (принцип 7).
    if _llm_fuse_blown():
        return None

    try:
        core = await generate_core(idea, demand_data, "quick",
                                   chosen_offer=chosen, purpose=purpose)
        section = await generate_section(SAMPLE_SECTION, idea, demand_data, "quick",
                                         chosen_offer=chosen, purpose=purpose)
    except ReportEngineError:
        logging.getLogger(__name__).info("sample not built for check %s", check_id,
                                         exc_info=True)
        return None

    sample = {**core, "section": section}
    with Session(engine) as s:
        fresh = s.get(DemandCheck, check_id)
        if fresh:
            store = {}
            if fresh.sample_json:
                try:
                    parsed = json.loads(fresh.sample_json)
                    if isinstance(parsed, dict):
                        store = parsed
                except ValueError:
                    pass
            if purpose not in store:
                store[purpose] = sample
                fresh.sample_json = json.dumps(store, ensure_ascii=False)
                s.add(fresh); s.commit()
    return sample


PREVIEW_STATUS = "preview"   # владельческий прогон без оплаты, см. _owner_preview


def _best_report_purchase(s: Session, check_id: int, *, include_preview: bool = False):
    """Самая полная ОПЛАЧЕННАЯ покупка отчёта для этой проверки спроса --
    full перекрывает quick, если куплены оба.

    include_preview включается ТОЛЬКО для владельца: превью открывает платный
    отчёт бесплатно, и если пустить его сюда для всех, человек, чья проверка
    попала во владельческий прогон, получит бизнес-план за 2990 ₽ даром.
    """
    allowed = ["paid", PREVIEW_STATUS] if include_preview else ["paid"]
    rows = s.exec(select(ReportPurchase).where(
        ReportPurchase.check_id == check_id, ReportPurchase.status.in_(allowed)
    ).order_by(ReportPurchase.created_at.desc())).all()
    if not rows:
        return None
    full = [r for r in rows if r.tier == "full"]
    return full[0] if full else rows[0]


class ReportIn(SQLModel):
    check_id: Optional[int] = None
    tier: str = "quick"
    contact: str = ""


@app.post("/api/report")
async def report_order(data: ReportIn, request: Request):
    """Заказ отчёта/бизнес-плана. Нужны данные бесплатной проверки спроса --
    без них отчёту не на чем строиться, в отличие от живого теста."""
    contact = (data.contact or "").strip()
    if len(contact) < 5:
        return JSONResponse({"ok": False, "error": "Оставьте почту или телефон — чтобы вернуться к отчёту."}, status_code=400)
    if payments.configured() and not payments.valid_receipt_contact(contact):
        return JSONResponse({"ok": False, "error": "Для оплаты нужна почта или телефон — на них пришлём чек. Телеграм для этого не подходит."}, status_code=400)
    tier = data.tier if data.tier in REPORT_PRICES else "quick"
    if not data.check_id:
        return JSONResponse({"ok": False, "error": "Сначала пройдите бесплатную проверку спроса."}, status_code=400)
    with Session(engine) as s:
        rec = s.get(DemandCheck, data.check_id)
        if not rec or not rec.result_json:
            return JSONResponse({"ok": False, "error": "Проверка спроса не найдена."}, status_code=404)
        idea = rec.idea
        price = REPORT_PRICES[tier]["price"]
        order = ReportPurchase(check_id=data.check_id, idea=idea, tier=tier, contact=contact[:200],
                               amount=price, status="pending_payment" if payments.configured() else "new")
        s.add(order); s.commit(); s.refresh(order)
        order_id, order_link = order.id, _report_link(order)
    if not payments.configured():
        if _notify_owner_order(request, what=f"отчёт «{REPORT_PRICES[tier]['label']}»",
                               order_id=order_id, idea=idea, contact=contact,
                               amount=price, paid=False, link=order_link):
            _mark_notified(ReportPurchase, order_id)
        if _notify_buyer_order(request, kind="report", order_id=order_id, idea=idea,
                               contact=contact, amount=price, paid=False, tier=tier,
                               link=order_link):
            _mark_notified(ReportPurchase, order_id, field="buyer_notified")
        return {"ok": True, "paid": False,
                "message": "Заявка принята. Мы соберём отчёт вручную и пришлём в течение дня."}
    try:
        base = str(request.base_url).rstrip("/")
        # Токен в адресе возврата -- это и есть ключ покупателя от своего
        # отчёта: вернувшись с оплаты, он открывает страницу сразу, ещё не
        # заходя в кабинет.
        sep = "&" if "?" in order_link else "?"
        pid, url = await payments.create_payment(
            order_id, REPORT_PRICES[tier]["price"], f"Создатель · отчёт по идее (заказ {order_id})",
            f"{base}{order_link}{sep}paid=1", kind="report", contact=contact)
        with Session(engine) as s:
            order = s.get(ReportPurchase, order_id)
            order.payment_id = pid; s.add(order); s.commit()
        return {"ok": True, "paid": True, "confirmation_url": url}
    except payments.PaymentsError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)


@app.get("/api/report/{rid}/status")
def report_status(rid: int):
    """Лёгкий поллинг после редиректа с оплаты -- вебхук может прийти
    на пару секунд позже, чем пользователь вернётся на страницу."""
    with Session(engine) as s:
        purchase = _best_report_purchase(s, rid)
    return {"paid": bool(purchase), "tier": purchase.tier if purchase else None}


def _notify_owner_order(request: Request, *, what: str, order_id: int, idea: str,
                        contact: str, amount: int, paid: bool, link: str = "") -> bool:
    """Письмо владельцу о новом заказе -- оплаченном или заявке без оплаты.

    До этого владелец узнавал о деньгах и заявках, только открыв /desk
    глазами: продукт с платным рекламным трафиком так работать не может
    (A2 в PRODUCT_ROADMAP). Никогда не бросает -- см. mailer.notify_owner,
    сбой уведомления не имеет права сломать оплату или заявку.
    """
    base = str(request.base_url).rstrip("/")
    head = "оплачено" if paid else "заявка без оплаты"
    body = (f"{what}\n"
            f"Заказ №{order_id}\n"
            f"Идея: {(idea or '—')[:200]}\n"
            f"Контакт: {contact or '—'}\n")
    if paid:
        body += f"Сумма: {amount} ₽\n"
    else:
        body += "Оплаты не было: связаться и довести вручную.\n"
    if link:
        body += f"\n{base}{link}\n"
    body += f"\nВсе заказы: {base}/desk\n"
    return mailer.notify_owner(f"Создатель: {head} — {what}", body)


def _notify_buyer_order(request: Request, *, kind: str, order_id: int, idea: str,
                        contact: str, amount: int, paid: bool, tier: str = "",
                        link: str = "") -> bool:
    """Письмо ПОКУПАТЕЛЮ о его собственном заказе.

    До этого при оплате письмо уходило владельцу, а человеку, отдавшему
    990-2990 ₽, -- ничего: только фискальный чек от ЮКассы, то есть чек, а не
    ссылка на продукт (A10 в PRODUCT_ROADMAP). Между тем отчёт собирается по
    разделам минутами, и единственным следом покупки была вкладка в браузере.

    Письмо отвечает ровно на три вопроса, которые возникают сразу после
    оплаты: что именно куплено, где это лежит и что делать, если ссылка
    потерялась. Никогда не бросает -- см. mailer.notify_buyer.
    """
    base = str(request.base_url).rstrip("/")
    idea_line = f"Идея: {(idea or '—')[:200]}\n"
    if kind == "report":
        label = REPORT_PRICES.get(tier, {}).get("label", "отчёт")
        subject = (f"Создатель: {label.lower()} по вашей идее — "
                   + ("оплата принята" if paid else "заявка принята"))
        if paid:
            body = (f"Оплата принята, спасибо.\n\n"
                    f"Что оплачено: {label}, {amount} ₽ (заказ №{order_id})\n"
                    f"{idea_line}\n"
                    f"Ваш разбор здесь:\n{base}{link}\n\n"
                    "Он собирается по разделам и занимает несколько минут. "
                    "Страницу можно закрыть: собранное сохраняется, при следующем "
                    "открытии сборка продолжится сама.\n\n")
        else:
            body = (f"Заявка принята.\n\n"
                    f"Что заказано: {label} (заказ №{order_id})\n"
                    f"{idea_line}\n"
                    "Мы соберём разбор и пришлём его в течение дня.\n\n")
    else:
        subject = ("Создатель: тест на реальных людях — "
                   + ("оплата принята" if paid else "заявка принята"))
        head = (f"Оплата принята, спасибо.\n\nЧто оплачено: тест на реальных людях, "
                f"{amount} ₽ (заказ №{order_id})\n" if paid
                else f"Заявка принята.\n\nЧто заказано: тест на реальных людях "
                     f"(заказ №{order_id})\n")
        body = (head + f"{idea_line}\n"
                "Мы собираем проверочную страницу под вашу идею. Готовая страница и "
                "пошаговая инструкция по запуску рекламы появятся в личном кабинете.\n\n"
                # Про отдельный бюджет человек узнавал уже после оплаты -- ровно то,
                # что чинила A7. Письмо -- последнее место, где об этом можно
                # промолчать, поэтому не молчим.
                f"Напоминаем: рекламный бюджет ({AD_BUDGET_HINT}) вы платите напрямую "
                "Яндексу, в стоимость теста он не входит.\n\n")
    # «даже если ссылка выше потеряется» имеет смысл только там, где ссылка
    # выше есть: в письме про живой тест её нет, страницу мы ещё собираем.
    tail = ("Все ваши заказы будут там, даже если ссылка выше потеряется.\n\n"
            if link else "Все ваши заказы и проекты будут там.\n\n")
    body += (f"Личный кабинет: {base}/account\n"
             "Вход без пароля — укажите эту же почту, и мы пришлём ссылку. "
             + tail +
             "Если что-то пошло не так — просто ответьте на это письмо.\n")
    return mailer.notify_buyer(contact, subject, body)


def _mark_notified(model, order_id: int, *, field: str = "paid_notified") -> None:
    """Флаг «письмо уже ушло» отдельной короткой транзакцией: вебхук
    ЮКассы может прийти повторно, а страницу заказа можно перезагрузить.

    field разделяет владельческое и покупательское письмо: успех одного не
    имеет права погасить второе (тот же урок, что fail_notified в A2)."""
    try:
        with Session(engine) as s:
            row = s.get(model, order_id)
            if row is not None:
                setattr(row, field, True)
                s.add(row); s.commit()
    except Exception:
        logging.getLogger(__name__).warning("mark notified failed", exc_info=True)


def _record_report_failure(purchase_id: int, error: str, request: Request, check_id: int) -> None:
    """Оплата прошла, отчёт не собрался -- самый дорогой сценарий отказа для
    платного продукта: деньги списаны, услуга не оказана, и до этой правки
    единственным, кто об этом знал, был сам покупатель.

    Записываем причину (владелец видит её в /desk) и ОДИН раз пишем владельцу.
    Один -- потому что страницу можно перезагружать сколько угодно, и каждая
    перезагрузка заново дёргает генерацию. Всё внутри fail-soft: сбой записи
    или письма не имеет права уронить страницу, которую и так уже видит
    расстроенный покупатель."""
    try:
        with Session(engine) as s:
            purchase = s.get(ReportPurchase, purchase_id)
            if purchase is None:
                return
            purchase.gen_error = error[:500]
            already_notified = purchase.fail_notified
            # Заявка без оплаты (status="new") -- не денежный сбой, владелец
            # и так собирает такие вручную; письмо шлём только когда заплатили.
            paid = purchase.status == "paid"
            contact, tier, idea = purchase.contact, purchase.tier, purchase.idea
            s.add(purchase); s.commit()

        if already_notified or not paid:
            return

        base = str(request.base_url).rstrip("/")
        label = REPORT_PRICES.get(tier, {}).get("label", tier)
        sent = mailer.notify_owner(
            f"Создатель: оплачен отчёт, но он не собрался (заказ {purchase_id})",
            f"Тариф: {label}\n"
            f"Идея: {idea[:200]}\n"
            f"Покупатель: {contact}\n"
            f"Ошибка генерации: {error}\n\n"
            f"Страница отчёта: {base}/report/{check_id}\n"
            f"Заказы: {base}/desk\n\n"
            "Покупатель видит на странице сообщение об ошибке. Отчёт пересоберётся "
            "сам при следующем открытии страницы -- если причина была временной. "
            "Если нет, деньги придётся вернуть.")
        if sent:
            with Session(engine) as s:
                fresh = s.get(ReportPurchase, purchase_id)
                if fresh is not None:
                    fresh.fail_notified = True
                    s.add(fresh); s.commit()
    except Exception:
        logging.getLogger(__name__).warning("report failure notice failed", exc_info=True)


def _example_purchase(s: Session):
    """Отчёт, помеченный как публичный пример, если он собран."""
    return s.exec(select(ReportPurchase).where(
        ReportPurchase.is_example == True)).first()          # noqa: E712


def _tier_summary_html(purpose: str = "business") -> str:
    """Что входит в каждый тариф — там, где человек решает платить.

    Раньше на `/r/` стояло только «от 990 ₽», а состав тарифов открывался
    лишь на следующем экране. Для пришедшего с /social-contract это прямая
    ловушка: он идёт за обоснованием сметы, а секции «Финансовая модель» в
    дешёвом тарифе нет вовсе (C2 в PRODUCT_ROADMAP).

    Состав собирается из ALL_SECTIONS/QUICK_KEYS, а не пишется руками:
    вторая копия списка разъехалась бы с движком, как уже разъезжались цены.

    Разделов в полном тарифе стало 21 вместо восьми (E5), и перечисление
    через запятую превратилось в строчную простыню из шестнадцати фрагментов:
    блок, созданный помогать решению, решению мешал (B7). Дешёвый тариф
    так и перечисляем — пять пунктов читаются, — а полный разложен по
    группам `SECTION_GROUPS`: имя группы держит взгляд, а «Финансовая
    модель» находится в «Деньгах», а не тонет в середине списка. Для
    соцконтракта это ровно та строка, ради которой человек и платит.

    Названия разделов берём через `section_title(key, purpose)`, а не
    статический `ALL_SECTIONS` — у финансового раздела для соцконтракта
    другое имя («Смета и расчёты для комиссии»), и сам отчёт после оплаты
    покажет именно его. Показать на витрине «Финансовая модель», а после
    оплаты — другое название той же секции, значит не узнать в списке то,
    ради чего платишь (найдено кастдев-проходом по платному пути 2026-07-29).
    """
    # Заголовок «План запуска — по этапам» внутри перечисления через запятую
    # читается двусмысленно из-за тире: берём часть до него.
    def short(title: str) -> str:
        return title.split(" — ")[0].strip()

    titles = {k: section_title(k, purpose) for k, _ in ALL_SECTIONS}
    quick = [short(titles[k]) for k, _ in ALL_SECTIONS if k in QUICK_KEYS]
    extra_keys = [k for k, _ in ALL_SECTIONS if k not in QUICK_KEYS]

    group_html = ""
    for name, keys in SECTION_GROUPS:
        mine = [short(titles[k]).lower() for k in keys if k in extra_keys]
        if mine:
            group_html += (f'<li><b>{html.escape(name)}:</b> '
                           f'{html.escape(", ".join(mine))}</li>')
    # Секция, которой нет ни в одной группе, обязана всё равно попасть на
    # витрину: молча пропасть — это ровно тот разъезд движка и витрины,
    # против которого вся эта функция и написана (принцип 3).
    covered = {k for _, keys in SECTION_GROUPS for k in keys}
    loose = [short(titles[k]).lower() for k in extra_keys if k not in covered]
    if loose:
        group_html += f'<li>{html.escape(", ".join(loose))}</li>'

    quick_label = REPORT_PRICES["quick"]["label"]
    # Число разделов считаем, а не пишем: набор секций уже менялся.
    lead = (f'Всё из тарифа «{quick_label}» и ещё '
            f'{len(extra_keys)} {_plural(len(extra_keys), "раздел", "раздела", "разделов")}:')

    return (
        '<div class="tier-what-block">'
        f'<div class="tier-row">'
        f'<span class="tier-name">{html.escape(quick_label)} — '
        f'<b>{REPORT_PRICES["quick"]["price"]} ₽</b></span>'
        f'<span class="tier-what">{html.escape(", ".join(quick))}.</span>'
        f'</div>'
        f'<div class="tier-row">'
        f'<span class="tier-name">{html.escape(REPORT_PRICES["full"]["label"])} — '
        f'<b>{REPORT_PRICES["full"]["price"]} ₽</b></span>'
        f'<span class="tier-what">{html.escape(lead)}</span>'
        f'<ul class="tier-groups">{group_html}</ul>'
        f'</div>'
        '</div>')


def _example_link(text: str, purpose: str = "business") -> str:
    """Ссылка на пример — только когда пример реально существует И собран
    для ТОЙ ЖЕ аудитории, что сейчас смотрит на страницу.

    Обещать «посмотрите пример» и привести на пустую страницу хуже, чем не
    обещать вовсе (принципы 3 и 7). Поэтому витрины спрашивают об этом у
    сервера, а не носят ссылку зашитой.

    Пример ровно один на весь сервис (F2/C1) -- собран под чью-то одну
    оптику. Показать венчурный разбор фаундера как «вот что вы получите»
    получателю соцконтракта или студенту обещает не ту оптику, что мы
    реально отдадим по его заявке (принцип 4 -- одна аудитория, одна
    оптика; принцип 3 -- продаём только то, что реально отдаём).
    """
    with Session(engine) as s:
        ex = _example_purchase(s)
        if not ex:
            return ""
        example_purpose = "business"
        if ex.check_id:
            check = s.get(DemandCheck, ex.check_id)
            if check and check.purpose:
                example_purpose = check.purpose
        if example_purpose != purpose:
            return ""
    return f'<a href="/example">{html.escape(text)}</a>'


def _owner_preview(check_id: int, tier: str) -> None:
    """Заводит владельческий прогон отчёта: та же генерация, тот же промпт,
    те же данные -- просто без оплаты и с нулевой суммой, чтобы не попасть в
    выручку. Повторный вызов ничего не дублирует."""
    with Session(engine) as s:
        rec = s.get(DemandCheck, check_id)
        if not rec:
            return
        exists = s.exec(select(ReportPurchase).where(
            ReportPurchase.check_id == check_id, ReportPurchase.tier == tier,
            ReportPurchase.status == PREVIEW_STATUS)).first()
        if exists:
            return
        s.add(ReportPurchase(check_id=check_id, idea=rec.idea, tier=tier,
                             contact="", status=PREVIEW_STATUS, amount=0))
        s.commit()


def _sections_meta(purpose: str) -> list[dict]:
    """Состав отчёта для страницы: заголовок, группа и ВОПРОС раздела.

    Вопрос показывается на запертом разделе вместо общей фразы «полный разбор
    в отчёте»: «Сколько остаётся с одной продажи после всех расходов на неё?»
    продаёт лучше любого описания, потому что человек хочет знать ответ.
    """
    from app.report_engine import SECTION_SPECS, _spec
    out = []
    for s in SECTION_SPECS:
        merged = _spec(s["key"], purpose)
        out.append({"key": s["key"], "title": merged["title"],
                    "group": s["group"], "ask": merged["ask"]})
    return out


def _chosen_offer(rec: "DemandCheck") -> dict | None:
    """Заострение, выбранное на /r/. Битый JSON не имеет права уронить
    платный отчёт -- лучше собрать разбор по исходной идее, чем не собрать."""
    if not rec.chosen_offer:
        return None
    try:
        offer = json.loads(rec.chosen_offer)
    except ValueError:
        return None
    return offer if isinstance(offer, dict) else None


@app.post("/api/report/{rid}/section")
async def report_section(rid: int, request: Request, key: str):
    """Собрать ОДИН раздел отчёта и дописать его в уже сохранённый разбор.

    Разделов больше двух десятков, и каждый — свой вызов модели со своим
    бюджетом токенов (иначе объёма не будет, см. докстринг report_engine).
    Страница просит их по одному и показывает по мере готовности, вместо
    того чтобы держать человека на пустом экране, пока соберётся всё.
    """
    owner = _is_owner(request)
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec or not rec.result_json:
            return JSONResponse({"ok": False, "error": "Проверка не найдена."}, status_code=404)
        purchase = _best_report_purchase(s, rid, include_preview=owner)
        if not purchase:
            return JSONResponse({"ok": False, "error": "Раздел доступен после оплаты."},
                                status_code=403)
        # Проверяем доступ ДО генерации: иначе посторонний, набравший чужой
        # номер проверки, не просто читал бы чужой отчёт, а оплачивал бы нам
        # вызовы модели по чужой покупке.
        if not _report_access_ok(request, purchase):
            return JSONResponse({"ok": False, "error": "Этот отчёт открывается по вашей ссылке "
                                                       "или из личного кабинета."},
                                status_code=403)
        if key not in section_keys(purchase.tier):
            return JSONResponse({"ok": False, "error": "Такого раздела нет в вашем тарифе."},
                                status_code=404)
        stored = json.loads(purchase.report_json) if purchase.report_json else {"sections": []}
        for sec in stored.get("sections", []):
            if sec.get("key") == key:
                return {"ok": True, "section": sec, "cached": True}
        idea, tier, purpose = rec.idea, purchase.tier, rec.purpose
        chosen, purchase_id = _chosen_offer(rec), purchase.id
        demand_data = json.loads(rec.result_json)
        # Балл уже посчитан ядром отчёта и лежит рядом с разделами. Раздел
        # «Вердикт» без него скатывался в «доработать» на любой идее — см.
        # report_engine._verdict_anchor.
        score = stored.get("viability_score")

    try:
        section = await generate_section(key, idea, demand_data, tier,
                                         chosen_offer=chosen, purpose=purpose,
                                         viability_score=score)
    except ReportEngineError as e:
        # Сбой одного раздела не должен выглядеть как сбой всего отчёта:
        # остальные уже собраны и читаются.
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)

    with Session(engine) as s:
        fresh = s.get(ReportPurchase, purchase_id)
        data = json.loads(fresh.report_json) if fresh.report_json else {"sections": []}
        sections = data.get("sections") or []
        # Перечитываем перед записью: две вкладки могли попросить разные
        # разделы одновременно, и терять чужой результат нельзя.
        if not any(x.get("key") == key for x in sections):
            sections.append(section)
            order = section_keys(fresh.tier)
            sections.sort(key=lambda x: order.index(x["key"]) if x["key"] in order else 99)
            data["sections"] = sections
            fresh.report_json = json.dumps(data, ensure_ascii=False)
            s.add(fresh); s.commit()
    return {"ok": True, "section": section}


@app.post("/api/example/publish")
def example_publish(request: Request, check_id: str, tier: str = "full"):
    """Пометить уже собранный отчёт публичным примером (/example).

    Пример обязан быть НАСТОЯЩИМ выводом движка: написанный руками текст
    всегда получается глаже, и человек заплатит за одно, а получит другое.
    Поэтому публикуется только уже сгенерированный отчёт -- собрать его
    можно тем же владельческим прогоном, /report/{id}?key=...&preview=full.

    check_id принимает то же, что видно в адресной строке, -- public_id
    проверки спроса (буквы из /r/<...>). Числовой id тоже работает (только у
    владельца, см. _find_check), но добывать его отдельным запросом к базе
    незачем: весь остальной проект уже адресуется public_id, эта ручка была
    единственным исключением.
    """
    _check_owner(request)
    with Session(engine) as s:
        rec, _ = _find_check(s, check_id, request)
        if not rec:
            return JSONResponse({"ok": False, "error": "Проверка с таким адресом не найдена."},
                                status_code=404)
        target = s.exec(select(ReportPurchase).where(
            ReportPurchase.check_id == rec.id, ReportPurchase.tier == tier
        ).order_by(ReportPurchase.created_at.desc())).first()
        if not target:
            return JSONResponse({"ok": False, "error": "Отчёт по этой проверке ещё не собран."},
                                status_code=404)
        if not target.report_json:
            return JSONResponse({"ok": False, "error": "Отчёт пуст — сначала соберите его."},
                                status_code=409)
        # Пример ровно один: две «витрины правды» разъезжаются так же, как
        # разъезжались копии цен (B5).
        for row in s.exec(select(ReportPurchase).where(ReportPurchase.is_example == True)).all():  # noqa: E712
            row.is_example = False
            s.add(row)
        target.is_example = True
        s.add(target); s.commit()
    return {"ok": True, "url": "/example"}


@app.get("/example", response_class=HTMLResponse)
def example_page(request: Request):
    """Настоящий отчёт, собранный сервисом, — открыт целиком и бесплатно.
    Владелец называл отсутствие примера блокером доверия: человек платит
    990–2990 ₽, не видя ни строчки того, что получит."""
    with Session(engine) as s:
        ex = _example_purchase(s)
        if not ex:
            return HTMLResponse(_with_server_values("index.html"), status_code=404)
        rec = s.get(DemandCheck, ex.check_id)
        report_full = json.loads(ex.report_json)
        tier, idea = ex.tier, ex.idea
    demand_data = json.loads(rec.result_json) if rec and rec.result_json else {}

    note = (f'<div class="example-note">Это настоящий отчёт, собранный сервисом — '
            f'тариф «{html.escape(REPORT_PRICES.get(tier, {}).get("label", tier))}», '
            f'открыт целиком. Ваш будет по вашей идее и вашим цифрам спроса.</div>')
    doc_title, doc_meta = _doc_title_and_meta(ex)
    _purpose = rec.purpose if rec else PURPOSE_DEFAULT
    # Только те разделы, что реально опубликованы. Полный список тарифа
    # заставил бы страницу дозаказывать недостающее -- то есть жечь вызовы
    # модели по чужой покупке на каждого посетителя примера, а с закрытым
    # доступом (_report_access_ok) он бы ещё и упирался в 403.
    _tier_keys = [s["key"] for s in (report_full.get("sections") or []) if s.get("key")]
    tpl = _static("report.html")
    html_out = (tpl
        .replace("__DOC_TITLE__", html.escape(doc_title))
        .replace("__DOC_META__", doc_meta)
        .replace("__CHECK_ID__", str(ex.check_id or 0))
        .replace("__ACCESS_NOTE__", "")
        .replace("__OWNER_BAR__", note)
        .replace("__CHOSEN_BLOCK__", "")
        .replace("__IDEA__", html.escape(idea))
        .replace("__PREVIEW_JSON__", json.dumps(_report_preview(demand_data), ensure_ascii=False))
        .replace("__SAMPLE_JSON__", "null")
        .replace("__REPORT_JSON__", json.dumps(report_full, ensure_ascii=False))
        .replace("__UNLOCKED_TIER__", json.dumps(tier))
        .replace("__ORDER_STATUS__", json.dumps("paid"))
        .replace("__GEN_ERROR__", json.dumps(""))
        .replace("__PRICES_JSON__", json.dumps(REPORT_PRICES, ensure_ascii=False))
        .replace("__SECTIONS_JSON__", json.dumps(_sections_meta(_purpose), ensure_ascii=False))
        .replace("__TIER_KEYS_JSON__", json.dumps(_tier_keys, ensure_ascii=False))
        .replace("__PURPOSE_JSON__", json.dumps(_purpose, ensure_ascii=False))
        .replace("__QUICK_KEYS_JSON__", json.dumps(QUICK_KEYS, ensure_ascii=False)))
    return HTMLResponse(_fill_server_values(html_out))


_MONTHS_GEN = ("января", "февраля", "марта", "апреля", "мая", "июня", "июля",
               "августа", "сентября", "октября", "ноября", "декабря")


def _doc_title_and_meta(purchase) -> tuple[str, str]:
    """Заголовок листа и строка с датой — для распечатки, а не для экрана.

    Разбор для соцконтракта человек несёт в комиссию НА БУМАГЕ, и лист был
    озаглавлен «Отчёт по идее» независимо от того, что куплено, — включая
    тариф, который мы сами называем «Бизнес-план» и под этим именем продаём.
    Комиссия получала документ, чьё название не совпадает ни с чеком, ни с
    тем, зачем его принесли. Названия берём из REPORT_PRICES: третья копия
    разъехалась бы, как уже разъезжались цены (B5).

    Дата — день оплаты. Документ без даты для комиссии не документ, а
    выдумывать её неоткуда: она есть готовая.
    """
    if not purchase or purchase.status != "paid":
        return "Отчёт по идее", ""
    label = REPORT_PRICES.get(purchase.tier, {}).get("label", "Отчёт по идее")
    d = purchase.created_at
    made = f"{d.day} {_MONTHS_GEN[d.month - 1]} {d.year} г."
    return label, (f'<p class="doc-meta">{html.escape(label)} по идее · подготовлен '
                   f'{made} · projectsozdatel.ru</p>')


@app.get("/report/{rid}", response_class=HTMLResponse)
async def report_page(rid: str, request: Request):
    """Дашборд отчёта: бесплатный тизер виден всегда; полные секции --
    после оплаты, генерируются лениво при первом открытии (без воркеров,
    тот же принцип, что и во всём проекте)."""
    # Владелец может собрать любой тариф без оплаты -- иначе проверить, что
    # именно получает человек за 2990 ₽, можно только заплатив себе самому.
    # Промпты правились вслепую, а качество отчёта -- это весь платный продукт.
    owner = _is_owner(request)
    want_preview = request.query_params.get("preview") or ""

    with Session(engine) as s:
        # Тот же неугадываемый адрес, что и у страницы результата: тизер
        # показывает текст чужой идеи, и по порядковому номеру его читали
        # перебором ровно так же (E6).
        rec, redirect = _find_check(s, rid, request)
        if not rec or not rec.result_json:
            return _lost_page()
        if redirect:
            keep = str(request.url.query)
            return RedirectResponse(f"/report/{rec.public_id}" + (f"?{keep}" if keep else ""),
                                    status_code=307)
        rid = rec.id

    # Владельческий прогон заводится ПОСЛЕ того, как проверка найдена: до
    # этого сюда прилетала строка адреса, а не номер записи.
    if owner and want_preview in REPORT_PRICES:
        _owner_preview(rid, want_preview)

    with Session(engine) as s:
        purchase = _best_report_purchase(s, rid, include_preview=owner)

    # Покупка есть, но открывает её посторонний -- показываем тизер и
    # объясняем, как попасть в свой отчёт. Не 403: человек мог просто
    # открыть свою же ссылку без токена в другом браузере, и глухая ошибка
    # ему ничего не объяснит (принцип 7).
    access_note = ""
    locked_other = bool(purchase) and not _report_access_ok(request, purchase)
    if locked_other:
        access_note = (
            '<div class="status-note no-print" id="access-note">Этот отчёт уже оплачен. Откройте его '
            'по ссылке, на которую вас вернула оплата, — или войдите в '
            '<a href="/account">личный кабинет</a> с той же почтой, что указывали '
            'при заказе, и отчёт будет там.</div>')
        purchase = None
    elif (purchase and purchase.status == "paid"
            and request.query_params.get("paid") == "1"):
        # Момент возврата с оплаты -- единственный, когда человек ещё не знает,
        # что разбор собирается минутами и что вкладку можно закрыть.
        # Про письмо говорим ТОЛЬКО если оно действительно ушло: контакт мог
        # оказаться телефоном, а SMTP -- лечь (см. mailer.notify_buyer).
        # Про «можно закрыть вкладку» говорит строка сборки прямо под этой
        # плашкой -- повторять здесь незачем. Эта отвечает на другой вопрос:
        # где разбор будет лежать, когда вкладки не станет.
        mailed = ("Ссылку на разбор мы отправили вам письмом. "
                  if purchase.buyer_notified else "")
        access_note = (
            f'<div class="status-note ok no-print" id="paid-note">Оплата принята. {mailed}'
            'Он всегда доступен в <a href="/account">личном кабинете</a> — вход по той '
            'же почте, что вы указали при заказе.</div>'
            if purchase.buyer_notified else
            '<div class="status-note ok no-print" id="paid-note">Оплата принята. '
            'Разбор всегда доступен в <a href="/account">личном кабинете</a> — вход по '
            'той же почте, что вы указали при заказе.</div>')

    demand_data = json.loads(rec.result_json)
    preview = _report_preview(demand_data)
    report_full = None
    gen_error = ""
    # Бесплатный образец нужен только тем, кто ещё не купил: у покупателя
    # весь разбор и так открыт, лишний вызов модели ему ни к чему. Постороннему
    # на чужой оплаченной проверке -- тем более: образец продаёт отчёт по этой
    # идее, а он уже продан, и платить за вызов модели тут не за что.
    sample = None if (purchase or locked_other) else await _ensure_sample(rid)

    if purchase:
        if not purchase.report_json:
            try:
                # Только ЯДРО отчёта — балл, объяснение и названные риски.
                # Разделов больше двух десятков, и каждый теперь свой вызов
                # модели: собирать их все внутри HTTP-запроса значит держать
                # человека на белом экране минутами. Разделы дозаказывает сама
                # страница по одному, см. /api/report/{id}/section.
                #
                # purpose определяет оптику: для соцконтракта это обоснование
                # сметы для комиссии, а не венчурный разбор. chosen_offer --
                # заострение, выбранное человеком на /r/ (A6 в PRODUCT_ROADMAP).
                core = await generate_core(rec.idea, demand_data, purchase.tier,
                                           chosen_offer=_chosen_offer(rec),
                                           purpose=rec.purpose)
                with Session(engine) as s:
                    fresh = s.get(ReportPurchase, purchase.id)
                    fresh.report_json = json.dumps({**core, "sections": []},
                                                   ensure_ascii=False)
                    s.add(fresh); s.commit(); s.refresh(fresh)
                    purchase = fresh
            except ReportEngineError as e:
                gen_error = str(e)
                _record_report_failure(purchase.id, gen_error, request, rid)
        if purchase.report_json:
            report_full = json.loads(purchase.report_json)

    # Если человек выбрал заострённую формулировку на /r/, он должен видеть,
    # что разбор построен именно вокруг неё, а не вокруг сырой первой фразы.
    chosen = _chosen_offer(rec)
    chosen_h1 = re.sub(r"<[^>]+>", "", str(chosen.get("h1", ""))).strip() if chosen else ""
    chosen_block = (f'<div class="chosen"><span class="chosen-tag">Разбираем формулировку</span>'
                    f'<span class="chosen-h1">{html.escape(chosen_h1)}</span></div>') if chosen_h1 else ""

    # Панель владельца: собрать любой тариф без оплаты. Видна только по ключу,
    # чтобы не пришлось помнить синтаксис query-параметра.
    owner_bar = ""
    if owner:
        key = html.escape(request.query_params.get("key") or "", quote=True)
        links = " · ".join(
            f'<a href="/report/{rid}?key={key}&preview={t}">{html.escape(cfg["label"])}</a>'
            for t, cfg in REPORT_PRICES.items())
        if not purchase:
            state = "отчёт не куплен, человек видит только бесплатный тизер"
        elif purchase.status == PREVIEW_STATUS:
            state = "показан владельческий прогон, оплаты не было"
        else:
            state = "оплаченный отчёт покупателя"
        pub = ""
        if purchase and purchase.report_json:
            with Session(engine) as s:
                already = _example_purchase(s)
            pub = ('<span> · это и есть публичный пример</span>' if already and already.id == purchase.id
                   else f'<button type="button" onclick="publishExample({rid},\'{purchase.tier}\')">'
                        'Опубликовать как пример</button>')
        owner_bar = (f'<div class="owner-bar">Владелец · {html.escape(state)}. '
                     f'Собрать без оплаты: {links}{pub}</div>')

    _purpose = rec.purpose
    _tier_keys = section_keys(purchase.tier) if purchase else []
    doc_title, doc_meta = _doc_title_and_meta(purchase)
    tpl = _static("report.html")
    html_out = (tpl
        .replace("__DOC_TITLE__", html.escape(doc_title))
        .replace("__DOC_META__", doc_meta)
        .replace("__CHECK_ID__", str(rid))
        .replace("__ACCESS_NOTE__", access_note)
        .replace("__OWNER_BAR__", owner_bar)
        .replace("__CHOSEN_BLOCK__", chosen_block)
        .replace("__IDEA__", html.escape(rec.idea))
        .replace("__PREVIEW_JSON__", json.dumps(preview, ensure_ascii=False))
        .replace("__SAMPLE_JSON__", json.dumps(sample, ensure_ascii=False) if sample else "null")
        .replace("__REPORT_JSON__", json.dumps(report_full, ensure_ascii=False) if report_full else "null")
        .replace("__UNLOCKED_TIER__", json.dumps(purchase.tier if purchase else None))
        .replace("__ORDER_STATUS__", json.dumps(purchase.status if purchase else None))
        .replace("__GEN_ERROR__", json.dumps(gen_error, ensure_ascii=False))
        .replace("__PRICES_JSON__", json.dumps(REPORT_PRICES, ensure_ascii=False))
        .replace("__SECTIONS_JSON__", json.dumps(_sections_meta(_purpose), ensure_ascii=False))
        .replace("__TIER_KEYS_JSON__", json.dumps(_tier_keys, ensure_ascii=False))
        .replace("__PURPOSE_JSON__", json.dumps(_purpose, ensure_ascii=False))
        .replace("__QUICK_KEYS_JSON__", json.dumps(QUICK_KEYS, ensure_ascii=False)))
    return HTMLResponse(_fill_server_values(html_out))


@app.get("/api/orders")
def orders_list(request: Request):
    _check_owner(request)
    with Session(engine) as s:
        rows = s.exec(select(LiveTestOrder)).all()
        # Покупки отчётов раньше не были видны владельцу НИГДЕ -- ни успешные,
        # ни сорванные. Для платного продукта это значит, что оплата на 2990 ₽
        # и несостоявшаяся доставка выглядели одинаково: никак.
        # Владельческие прогоны без оплаты -- не заказы. В списке заказов они
        # выглядели бы как «ожидает оплаты» и путали бы картину продаж.
        reports = s.exec(select(ReportPurchase).where(
            ReportPurchase.status != PREVIEW_STATUS)).all()
    return {"orders": [{"id": o.id, "created_at": str(o.created_at), "idea": o.idea,
                        "contact": o.contact, "status": _effective_status(o.status, o.created_at),
                        "amount": o.amount, "idea_id": o.idea_id,
                        "project_url": f"/p/{o.idea_id}" if o.idea_id else None,
                        "chosen_offer": json.loads(o.chosen_offer) if o.chosen_offer else None}
                       for o in reversed(rows)],
            "reports": [{"id": r.id, "created_at": str(r.created_at), "idea": r.idea,
                         "contact": r.contact, "tier": r.tier,
                         "tier_label": REPORT_PRICES.get(r.tier, {}).get("label", r.tier),
                         "status": _effective_status(r.status, r.created_at),
                         "amount": r.amount,
                         "delivered": bool(r.report_json),
                         "gen_error": r.gen_error,
                         "report_url": _report_link(r)}
                        for r in reversed(reports)]}


@app.get("/api/stats")
def public_stats():
    """Живые цифры для главной. Только честные счётчики из БД."""
    with Session(engine) as s:
        ideas = len(s.exec(select(DemandCheck)).all())
        events = len(s.exec(select(SmokeEvent)).all())
    return {"ideas_checked": ideas, "events": events}


@app.get("/api/funnel")
def owner_funnel(request: Request, days: int = 0):
    """Воронка владельца из НАШЕЙ базы, без зависимости от Метрики.

    Метрика (D1) считает поведение и нужна Директу для оптимизации, но она
    настраивается руками, теряет людей на блокировщиках и не знает про
    деньги. Здесь — то, что произошло на самом деле, с разбивкой по
    аудиториям: иначе не понять, какая рекламная кампания окупается (D3).

    Каждый шаг называет, что именно он считает: число без определения —
    это приглашение сделать неверный вывод (тот же принцип, что в B3).
    """
    _check_owner(request)
    since = utcnow() - timedelta(days=days) if days > 0 else None

    with Session(engine) as s:
        checks = s.exec(select(DemandCheck)).all()
        reports = s.exec(select(ReportPurchase)).all()
        live = s.exec(select(LiveTestOrder)).all()

    if since:
        checks = [c for c in checks if c.created_at >= since]
        reports = [r for r in reports if r.created_at >= since]
        live = [o for o in live if o.created_at >= since]

    # Владельческие прогоны — не продажи и не заказы, им в воронке не место.
    reports = [r for r in reports if r.status != PREVIEW_STATUS]
    purpose_of = {c.id: (c.purpose or "business") for c in checks}

    def split(items, key=lambda x: x.purpose):
        out = {"total": len(items)}
        for p in report_purposes:
            out[p] = sum(1 for x in items if key(x) == p)
        return out

    by_check = lambda r: purpose_of.get(r.check_id, "business")   # noqa: E731

    paid_reports = [r for r in reports if r.status == "paid"]
    paid_live = [o for o in live if o.status == "paid"]

    stages = [
        ("Проверок спроса", "человек описал идею и получил результат",
         split(checks)),
        ("Заострили идею", "выбрал одну из трёх формулировок",
         split([c for c in checks if c.chosen_offer])),
        ("Сохранили в кабинет", "оставил почту, чтобы вернуться",
         split([c for c in checks if c.contact])),
        ("Дошли до витрины отчёта", "открыли страницу отчёта и увидели образец",
         split([c for c in checks if c.sample_json])),
        ("Заказали отчёт", "нажали «Получить отчёт», включая неоплаченные",
         split(reports, by_check)),
        ("Оплатили отчёт", "деньги получены",
         split(paid_reports, by_check)),
        ("Заказали тест на людях", "заявка на живой тест, включая неоплаченные",
         split(live, by_check)),
        ("Оплатили тест на людях", "деньги получены",
         split(paid_live, by_check)),
    ]

    revenue = sum(r.amount for r in paid_reports) + sum(o.amount for o in paid_live)
    return {
        "days": days,
        "purposes": list(report_purposes),
        # Русские названия аудиторий отдаёт сервер: копия в скрипте панели
        # разъехалась бы при первой же новой аудитории (F1).
        "audience_labels": audiences.labels(),
        "stages": [{"name": n, "what": w, **counts} for n, w, counts in stages],
        "revenue": revenue,
    }


@app.get("/api/diag/yandex")
async def diag_yandex(request: Request, phrase: str = "купить слона"):
    """Owner-only: сырая диагностика интеграции с Яндексом -- Вордстат
    (прокси внутри Cloud Search API), без глотания ошибок. Открыть в
    браузере с ?key=... при жалобе «нет данных», чтобы увидеть точную
    причину, а не гадать."""
    _check_owner(request)
    return await diagnose(phrase)


@app.get("/api/diag/mail")
def diag_mail(request: Request, to: str = ""):
    """Owner-only: что настроено в почте и уходит ли письмо на самом деле.

    Настройка почты — четыре переменные окружения в чужой панели, и до этой
    ручки владелец узнавал результат только по тому, пожаловался ли
    покупатель. Открыть `/api/diag/mail?key=...&to=своя@почта.ru` — придёт
    тестовое письмо либо человекочитаемая причина, почему не пришло.

    Пароль не возвращается ни при каких условиях — только факт, что он задан.
    """
    _check_owner(request)
    return mailer.diagnose(to)


# ---------------------------------------------------------------------------
# Этап ⓪→①: выбранный оффер → лендинг, сразу захощенный
# ---------------------------------------------------------------------------

_landing_tpl: Optional[str] = None


def _landing_template() -> str:
    """Шаблон проверочной страницы -- один раз за жизнь процесса, как _static()."""
    global _landing_tpl
    if _landing_tpl is None:
        _landing_tpl = (BASE_DIR / "landing_template.html").read_text(encoding="utf-8")
    return _landing_tpl


def render_landing(offer: dict) -> str:
    tpl = _landing_template()
    pains_html = "".join(
        f"<div><h2>{p['h2']}</h2><p>{p['p']}</p></div>" for p in offer["pains"]
    )
    return (tpl
            .replace("{{PRODUCT_NAME}}", offer["product_name"])
            .replace("{{EYEBROW}}", offer["eyebrow"])
            .replace("{{H1}}", offer["h1"])
            .replace("{{SUB}}", offer["sub"])
            .replace("{{DEMO_LEFT_LABEL}}", offer["demo_left_label"])
            .replace("{{DEMO_HEAD_RIGHT}}", offer.get("demo_head_right", "готово за секунды"))
            .replace("{{DEMO_LEFT_BADGE}}", offer.get("demo_left_badge", ""))
            .replace("{{DEMO_LEFT_META}}", offer.get("demo_left_meta", ""))
            .replace("{{DEMO_RIGHT_TAG}}", offer.get("demo_right_tag", "результат · черновик готов"))
            .replace("{{DEMO_LEFT_TEXT}}", offer["demo_left_text"])
            .replace("{{DEMO_RIGHT_TEXT_JSON}}", json.dumps(offer["demo_right_text"], ensure_ascii=False))
            .replace("{{PAINS_HTML}}", pains_html)
            .replace("{{IDEA_ID}}", offer["idea_id"]))


class LaunchIn(BaseModel):
    idea_text: str
    offer: dict


# Обязательные поля оффера для запуска -- общие для ручного /api/launch
# владельца и автозапуска сразу после оплаты живого теста (см. yookassa_webhook).
LAUNCH_REQUIRED_FIELDS = ("idea_id", "product_name", "h1", "sub", "pains",
                          "demo_left_label", "demo_left_text", "demo_right_text", "eyebrow")


def _template_hash() -> str:
    """Отпечаток текущего шаблона проверочной страницы.

    Сравнение с сохранённым отвечает на вопрос «эта живая страница собрана до
    правки или после». Читаем через `_static`, то есть из того же кеша, что и
    рендер -- иначе отпечаток и страница разъехались бы (A20).
    """
    return hashlib.sha256(_landing_template().encode("utf-8")).hexdigest()[:16]


def _launch_offer(s: Session, offer: dict, idea_text: str, contact: str = "") -> SmokeProject:
    """Общая логика запуска проекта -- вызывающая сторона уже проверила
    LAUNCH_REQUIRED_FIELDS. contact, если передан, привязывает проект к
    покупателю сразу (виден в его /account без ручного PATCH владельцем)."""
    html = render_landing(offer)
    existing = s.exec(select(SmokeProject).where(SmokeProject.idea_id == offer["idea_id"])).first()
    if existing:
        existing.landing_html = html
        existing.template_hash = _template_hash()
        existing.offer_json = json.dumps(offer, ensure_ascii=False)
        if contact:
            existing.contact = contact
        s.add(existing); s.commit()
        return existing
    proj = SmokeProject(
        idea_id=offer["idea_id"], product_name=offer["product_name"],
        idea_text=idea_text[:2000],
        offer_json=json.dumps(offer, ensure_ascii=False),
        landing_html=html,
        template_hash=_template_hash(),
        click_target=int(offer.get("click_target", 40)),
        lead_rate_signal=float(offer.get("lead_rate_signal", SIGNAL_RATE)),
        lead_rate_dead=float(offer.get("lead_rate_dead", DEAD_RATE)),
        contact=contact,
    )
    s.add(proj); s.commit(); s.refresh(proj)
    return proj


@app.post("/api/launch")
def launch(data: LaunchIn, request: Request):
    _check_owner(request)
    offer = data.offer
    for key in LAUNCH_REQUIRED_FIELDS:
        if not offer.get(key):
            raise HTTPException(400, f"в оффере нет поля {key}")
    with Session(engine) as s:
        proj = _launch_offer(s, offer, data.idea_text)
    return {
        "ok": True, "idea_id": proj.idea_id,
        "landing_url": f"/l/{proj.idea_id}",
        "direct_utm": (f"?utm_source=yandex_direct&utm_campaign={proj.idea_id}"
                       "&utm_content={ad_id}&utm_term={keyword}"),
        "queries": offer.get("direct_queries", []),
        "verdict_url": f"/api/verdict/{proj.idea_id}",
    }


@app.get("/l/{idea_id}", response_class=HTMLResponse)
def serve_landing(idea_id: str):
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
    if proj is None:
        raise HTTPException(404, "Лендинг не найден")
    return HTMLResponse(proj.landing_html)


# ---------------------------------------------------------------------------
# Этап ①: события и вердикт
# ---------------------------------------------------------------------------

_MAX_FIELD = 300

# Rate limit публичного endpoint'а: простое минутное окно по IP.
# In-memory достаточно: один процесс, smoke-трафик — сотни визитов/день.
_RL_WINDOW: dict[str, list[float]] = {}
_RL_LIMIT = 30          # событий с одного IP в минуту
_RL_SECONDS = 60.0


# --- Дневной предохранитель на БЕСПЛАТНЫЕ вызовы LLM ------------------------
# Поштучный rate-limit держит один адрес, но из Директа приходит МНОГО разных
# адресов, и у каждого свой лимит — общего потолка трат не было вовсе (аудит
# 2026-08-04). Каждая бесплатная проверка — два вызова модели плюс шесть
# запросов в Вордстат, всё за счёт владельца. Предохранитель считает только
# бесплатные операции; ПЛАТНАЯ генерация отчёта не ограничивается никогда:
# заплатившему нельзя ответить «приходите завтра».
# In-memory и сбрасывается рестартом — это предохранитель от разорения за
# ночь, а не бухгалтерия.
DAILY_FREE_LLM = int(os.environ.get("SOZDATEL_DAILY_FREE_LLM", "500"))
_FUSE = {"day": "", "used": 0}
_FUSE_MSG = ("Сегодня проверок больше, чем мы рассчитывали, — сервис берёт "
             "паузу до завтра. Загляните позже: проверка останется бесплатной.")


def _llm_fuse_blown() -> bool:
    """True — дневной запас бесплатных вызовов исчерпан. Иначе списывает один."""
    today = datetime.now(timezone.utc).date().isoformat()
    if _FUSE["day"] != today:
        _FUSE["day"] = today
        _FUSE["used"] = 0
    if _FUSE["used"] >= DAILY_FREE_LLM:
        return True
    _FUSE["used"] += 1
    return False


def _client_ip(request: Request) -> str:
    """IP для rate-limit -- ПОСЛЕДНИЙ узел в X-Forwarded-For, а не первый.

    Заголовок формирует клиент, и прокси лишь ДОПИСЫВАЕТ к нему свой адрес
    справа. Значит первый элемент полностью подконтролен тому, кто стучится:
    подставляя каждый раз новый «IP», он получал по свежему лимиту на каждый
    запрос — то есть лимита не было вовсе. За платным трафиком это прямой
    счёт владельцу: /api/demand и /api/sharpen дёргают LLM.

    Последний элемент дописывает наш собственный прокси (Timeweb), подделать
    его клиент не может. Если заголовка нет (локальный запуск, прямой доступ)
    -- берём адрес сокета.
    """
    chain = [p.strip() for p in (request.headers.get("x-forwarded-for") or "").split(",") if p.strip()]
    if chain:
        return chain[-1]
    return request.client.host if request.client else "?"


def _rate_limited(ip: str) -> bool:
    import time
    now = time.monotonic()
    bucket = _RL_WINDOW.setdefault(ip, [])
    while bucket and now - bucket[0] > _RL_SECONDS:
        bucket.pop(0)
    if len(bucket) >= _RL_LIMIT:
        return True
    bucket.append(now)
    if len(_RL_WINDOW) > 10000:
        # Чистим ТОЛЬКО протухшее, а не весь словарь: полный clear() сбрасывал
        # лимит всем сразу, и переполнить словарь мог кто угодно -- это была
        # вторая дверь мимо ограничения, рядом с подделкой X-Forwarded-For.
        for key in [k for k, v in _RL_WINDOW.items() if not v or now - v[-1] > _RL_SECONDS]:
            _RL_WINDOW.pop(key, None)
    return False


@app.post("/api/smoke-event")
async def smoke_event(request: Request):
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(400, "bad json")
    event = str(data.get("event", ""))[:40]
    if event not in ("page_view", "lead_submitted"):
        raise HTTPException(400, "unknown event")
    ev = SmokeEvent(
        idea=str(data.get("idea", ""))[:80],
        event=event,
        source=str(data.get("source", ""))[:_MAX_FIELD],
        campaign=str(data.get("campaign", ""))[:_MAX_FIELD],
        content=str(data.get("content", ""))[:_MAX_FIELD],
        term=str(data.get("term", ""))[:_MAX_FIELD],
        contact=str(data.get("contact", ""))[:_MAX_FIELD] if event == "lead_submitted" else "",
    )
    with Session(engine) as s:
        s.add(ev); s.commit()
    return {"ok": True}


def compute_verdict(views: int, leads: int, target: int, signal: float, dead: float) -> dict:
    """Детерминированный вердикт теста на реальных людях.

    Тексты называют пороги вслух. Голое «12% — сигнал есть» не объясняет
    ничего: человек не знает ни откуда взялось число, ни с чем его сравнили
    (B3 в PRODUCT_ROADMAP). Слова -- покупательские: этот вердикт видит и
    самозанятая из соцконтракта, а не только фаундер.
    """
    rate = (leads / views) if views else 0.0
    n_leads = f"{leads} {_plural(leads, 'заявка', 'заявки', 'заявок')}"
    n_views = f"{views} {_plural(views, 'визит', 'визита', 'визитов')}"
    # «N визитов, из них M заявок», а не «M заявок с N визитов»: предлог «с»
    # требует родительного падежа, и на числах вроде 52 фраза начинает хромать.
    got = f"{n_views}, из них {n_leads} — это {_pct(rate)}"
    if views < target:
        return {"verdict": "РАНО СУДИТЬ",
                "detail": f"Пока {n_views} из {target}, заявок {leads}. "
                          f"Меньше {target} визитов — это ещё не статистика: случайность легко "
                          "принять за результат. Ничего не меняйте, пусть наберётся."}
    if rate >= signal:
        return {"verdict": "СИГНАЛ ЕСТЬ",
                "detail": f"{got}, при пороге {_pct(signal)} и выше. Люди не просто смотрят — "
                          "оставляют контакты. Идею стоит делать."}
    if rate <= dead:
        return {"verdict": "СПРОСА НЕТ",
                "detail": f"{got}, при пороге {_pct(dead)} и ниже. Страницу видели, но не "
                          "откликнулись. Рекламу можно останавливать — это сэкономленные деньги "
                          "и месяцы работы над тем, что не купят."}
    return {"verdict": "СЕРАЯ ЗОНА",
            "detail": f"{got} — между {_pct(dead)} (интереса нет) и {_pct(signal)} (интерес есть). "
                      "Однозначного ответа цифры не дают: попробуйте другой заголовок на "
                      "той же странице."}


def _smoke_stage(views: int, click_target: int) -> int:
    """Этап проекта по шкале STAGE_NAMES. Единственное место, где это
    правило живёт: копия в `static/project.html` уже разъезжалась.

    Проект существует только потому, что человек прошёл бесплатную проверку
    спроса И оплатил тест на людях -- мы собрали ему страницу. Значит этапы
    «Идея» и «Спрос» у него ПОЗАДИ. Раньше здесь стояло `1 if views else 0`,
    и покупатель сразу после оплаты видел «Этап 1 из 7 — Идея»: продукт
    сообщал, что человек в самом начале, ровно в тот момент, когда он
    только что заплатил.

    Пока визитов меньше цели -- идёт «Тест на реальных людях»; как только их
    хватило на вердикт, разговор идёт уже про «Заявки».
    """
    return 3 if click_target and views >= click_target else 2


def _smoke_card(p: "SmokeProject", views: int, leads: int) -> dict:
    """Карточка проекта -- общая для владельца (/api/cabinet) и покупателя
    (/api/account/me), чтобы оба видели один и тот же язык: этап, прогресс,
    вердикт."""
    stage = _smoke_stage(views, p.click_target)
    v = compute_verdict(views, leads, p.click_target, p.lead_rate_signal, p.lead_rate_dead)
    rate = (leads / views) if views else 0.0
    # Эту строку читает и владелец в /desk, и покупатель в /account -- так и
    # задумано, один язык на двоих. Значит писать её надо на покупательском:
    # ему «очередь на MVP», «идею в архив» и «второй оффер на том же трафике»
    # не говорят ничего, а «оффер» с «трафиком» вдобавок запрещены (принцип 5).
    # Ссылку отдаём отдельным полем: строка уходит в разметку экранированной,
    # и путь в ней человек читал как текст, а кликнуть не мог (A17).
    next_link = None
    if views == 0:
        next_step = "Запустите рекламу в Яндекс Директе на вашу проверочную страницу."
        next_link = {"href": "/guide/direct", "text": "Пошаговая инструкция"}
    elif views < p.click_target:
        left = p.click_target - views
        next_step = (f"Идут визиты: ещё {left} "
                     f"{_plural(left, 'визит', 'визита', 'визитов')} до вывода. "
                     "Ничего не меняйте, пусть наберётся.")
    elif v["verdict"] == "СИГНАЛ ЕСТЬ":
        next_step = "Люди оставляют заявки — можно делать первую версию продукта."
    elif v["verdict"] == "СПРОСА НЕТ":
        next_step = "Заявок почти нет — остановите рекламу, чтобы не тратить бюджет зря."
    else:
        next_step = ("Результат посередине: попробуйте другое предложение "
                     "для той же аудитории.")
    return {"idea_id": p.idea_id, "name": p.product_name,
            "stage": stage, "stage_name": STAGE_NAMES[stage],
            "views": views, "leads": leads, "rate": round(rate * 100),
            "target": p.click_target, "verdict": v["verdict"],
            "next_step": next_step,
            "next_link": next_link,
            "progress": min(100, round(views / p.click_target * 100)) if p.click_target else 0,
            # Страница собрана из устаревшего шаблона. Показываем ТОЛЬКО
            # владельцу (см. /api/cabinet): покупателю это ни о чём не говорит,
            # а чинит всё равно владелец кнопкой (A20).
            "landing_stale": p.template_hash != _template_hash(),
            "landing_url": f"/l/{p.idea_id}",
            "project_url": f"/p/{p.idea_id}"}


@app.get("/api/verdict/{idea_id}")
def verdict(idea_id: str, request: Request):
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "идея не найдена")
        if not _project_access_ok(request, proj):
            raise HTTPException(401, "Нужен ключ владельца или вход в кабинет с этим проектом.")
        views = len(s.exec(select(SmokeEvent.id).where(
            SmokeEvent.idea == idea_id, SmokeEvent.event == "page_view")).all())
        leads_rows = s.exec(select(SmokeEvent.contact, SmokeEvent.created_at).where(
            SmokeEvent.idea == idea_id, SmokeEvent.event == "lead_submitted")).all()
    v = compute_verdict(views, len(leads_rows), proj.click_target,
                        proj.lead_rate_signal, proj.lead_rate_dead)
    offer = json.loads(proj.offer_json or "{}")
    return {"ok": True, "idea_id": idea_id, "product_name": proj.product_name,
            "h1": offer.get("h1", ""),
            "views": views, "leads": len(leads_rows), **v,
            # Этап считает сервер: копия правила на странице уже разъехалась
            # с ним и показывала оплатившему человеку «Идея».
            "stage": _smoke_stage(views, proj.click_target),
            "stage_names": STAGE_NAMES,
            "target": proj.click_target,
            "queries": offer.get("direct_queries", []),
            "landing_url": f"/l/{idea_id}",
            "direct_utm": (f"?utm_source=yandex_direct&utm_campaign={idea_id}"
                           "&utm_content={ad_id}&utm_term={keyword}"),
            "contacts": [c for c, _ in leads_rows]}


@app.get("/api/series/{idea_id}")
def series(idea_id: str, request: Request):
    """Визиты/заявки по дням за последние 14 дней — для графика на /p/{id}."""
    from collections import defaultdict
    from datetime import timedelta
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "идея не найдена")
        if not _project_access_ok(request, proj):
            raise HTTPException(401, "Нужен ключ владельца или вход в кабинет с этим проектом.")
        since = utcnow() - timedelta(days=14)
        rows = s.exec(select(SmokeEvent.created_at, SmokeEvent.event).where(
            SmokeEvent.idea == idea_id, SmokeEvent.created_at >= since)).all()
    days: dict[str, dict] = defaultdict(lambda: {"views": 0, "leads": 0})
    for created_at, event in rows:
        key = created_at.strftime("%d.%m")
        if event == "page_view":
            days[key]["views"] += 1
        elif event == "lead_submitted":
            days[key]["leads"] += 1
    # Полный ряд из 14 дней, включая нули — график не должен «рваться»
    out = []
    for i in range(13, -1, -1):
        d = (utcnow() - timedelta(days=i)).strftime("%d.%m")
        out.append({"date": d, **days.get(d, {"views": 0, "leads": 0})})
    return {"ok": True, "days": out}


@app.get("/api/projects")
def projects(request: Request):
    _check_owner(request)
    from collections import defaultdict
    with Session(engine) as s:
        rows = s.exec(select(SmokeProject).order_by(SmokeProject.created_at.desc())).all()
        # Все события одним запросом вместо 2×N -- тот же приём, что в /api/cabinet.
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for idea, event in s.exec(select(SmokeEvent.idea, SmokeEvent.event)).all():
            counts[(idea, event)] += 1
        out = [{"idea_id": p.idea_id, "product_name": p.product_name,
                "views": counts[(p.idea_id, "page_view")],
                "leads": counts[(p.idea_id, "lead_submitted")],
                "target": p.click_target, "landing_url": f"/l/{p.idea_id}"}
               for p in rows]
    return {"ok": True, "projects": out}


class RenameIn(BaseModel):
    name: str


@app.patch("/api/projects/{idea_id}")
def rename_project(idea_id: str, data: RenameIn, request: Request):
    """Пользовательское имя проекта: движок предлагает своё (РейтингГард),
    владелец волен переименовать (ОтзоВик). Меняется и <title> лендинга."""
    _check_owner(request)
    name = data.name.strip()[:80]
    if len(name) < 2:
        raise HTTPException(400, "имя от 2 символов")
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "проект не найден")
        old_name = proj.product_name
        proj.product_name = name
        # <title> лендинга следует за именем
        proj.landing_html = proj.landing_html.replace(
            f"<title>{old_name}</title>", f"<title>{name}</title>")
        s.add(proj); s.commit()
    return {"ok": True, "name": name}


@app.post("/api/projects/{idea_id}/refresh")
def refresh_landing(idea_id: str, request: Request):
    """Пересобрать проверочную страницу из сохранённого варианта.

    Правка шаблона не догоняет уже запущенные страницы: `landing_html` пишется
    в БД в момент запуска. Молча пересобирать нельзя -- страница может стоять
    под работающей рекламой, а менять её во время проверки значит менять то,
    что измеряешь (об этом же плейбук просит и самого покупателя). Поэтому
    решение принимает владелец, а мы лишь показываем, что страница отстала.

    Имя проекта живёт на самом проекте, а не в варианте: владелец мог
    переименовать его после запуска (PATCH /api/projects/{id}), и пересбор не
    должен возвращать машинное название.
    """
    _check_owner(request)
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "проект не найден")
        try:
            offer = json.loads(proj.offer_json)
        except ValueError:
            raise HTTPException(400, "у проекта не сохранён вариант, из которого он собран")
        if not isinstance(offer, dict) or not all(offer.get(k) for k in LAUNCH_REQUIRED_FIELDS):
            raise HTTPException(400, "у проекта не сохранён вариант, из которого он собран")
        offer = {**offer, "product_name": proj.product_name}
        proj.landing_html = render_landing(offer)
        proj.template_hash = _template_hash()
        s.add(proj); s.commit()
    return {"ok": True, "landing_url": f"/l/{idea_id}"}


class ProjectContactIn(BaseModel):
    contact: str


@app.patch("/api/projects/{idea_id}/contact")
def set_project_contact(idea_id: str, data: ProjectContactIn, request: Request):
    """Привязка проекта к покупателю -- владелец делает это вручную при
    запуске (см. «Заявки на живой тест» в кабинете), чтобы проект появился
    в личном кабинете покупателя (/account) по этой почте."""
    _check_owner(request)
    contact = data.contact.strip()[:200]
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "проект не найден")
        proj.contact = contact
        s.add(proj); s.commit()
    return {"ok": True, "contact": contact}


@app.delete("/api/projects/{idea_id}")
def delete_project(idea_id: str, request: Request):
    """Удалить заброшенный лендинг: сам проект + его события (контакты лидов
    уходят вместе с ним -- выгрузи их из /api/verdict до удаления, если нужны)."""
    _check_owner(request)
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "идея не найдена")
        for ev in s.exec(select(SmokeEvent).where(SmokeEvent.idea == idea_id)).all():
            s.delete(ev)
        # Заявка на живой тест, из которой запущен этот проект, иначе
        # осталась бы с оборванной ссылкой: /api/orders берёт project_url
        # прямо из idea_id заявки, не проверяя, жив ли ещё сам проект --
        # /desk продолжал бы звать «Проект уже запущен → открыть» на
        # страницу, которая уже отдаёт голый `{"detail": "проект не
        # найден"}` (тот же класс дефекта, что и A19: техническое сообщение
        # движка доезжает до человека). Хуже того: в /account эта же заявка
        # с непустым idea_id вообще перестала бы показываться -- ни
        # карточкой проекта (SmokeProject уже нет), ни обычной заявкой
        # (cabinet-запрос отбирает только заявки с idea_id IS NULL) --
        # человек, оплативший тест, увидел бы, что его заказ просто исчез.
        for order in s.exec(select(LiveTestOrder).where(LiveTestOrder.idea_id == idea_id)).all():
            order.idea_id = None
            s.add(order)
        s.delete(proj)
        s.commit()
    return {"ok": True, "deleted": idea_id}


class TrackedIn(BaseModel):
    name: str
    stage: int = 0
    status_note: str = ""
    external_link: str = ""


@app.post("/api/tracked")
def add_tracked(data: TrackedIn, request: Request):
    _check_owner(request)
    if not (0 <= data.stage <= 7):
        raise HTTPException(400, "stage: 0..7")
    if not data.name.strip():
        raise HTTPException(400, "нужно имя проекта")
    tp = TrackedProject(name=data.name.strip()[:80], stage=data.stage,
                        status_note=data.status_note.strip()[:200],
                        external_link=data.external_link.strip()[:300])
    with Session(engine) as s:
        s.add(tp); s.commit(); s.refresh(tp)
    return {"ok": True, "id": tp.id}


@app.patch("/api/tracked/{tp_id}")
def update_tracked(tp_id: int, data: TrackedIn, request: Request):
    _check_owner(request)
    with Session(engine) as s:
        tp = s.get(TrackedProject, tp_id)
        if tp is None:
            raise HTTPException(404, "проект не найден")
        tp.name = data.name.strip()[:80] or tp.name
        tp.stage = data.stage if 0 <= data.stage <= 7 else tp.stage
        tp.status_note = data.status_note.strip()[:200]
        tp.external_link = data.external_link.strip()[:300]
        s.add(tp); s.commit()
    return {"ok": True}


@app.delete("/api/tracked/{tp_id}")
def delete_tracked(tp_id: int, request: Request):
    _check_owner(request)
    with Session(engine) as s:
        tp = s.get(TrackedProject, tp_id)
        if tp is None:
            raise HTTPException(404, "проект не найден")
        s.delete(tp); s.commit()
    return {"ok": True}


@app.get("/api/cabinet")
def cabinet(request: Request):
    """Портфель целиком: внешние проекты + smoke-тесты Создателя.
    Smoke-этап определяется данными: есть клики -> ① Спрос, иначе ⓪ Идея."""
    _check_owner(request)
    out = {"stages": TRACKED_STAGE_NAMES, "tracked": [], "smoke": []}
    with Session(engine) as s:
        for tp in s.exec(select(TrackedProject).order_by(TrackedProject.created_at)).all():
            out["tracked"].append({"id": tp.id, "name": tp.name, "stage": tp.stage,
                                   "stage_name": TRACKED_STAGE_NAMES[tp.stage],
                                   "note": tp.status_note, "link": tp.external_link})
        # Все события одним запросом вместо 2×N (N+1 убивал время на Postgres)
        from collections import defaultdict
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for idea, event in s.exec(select(SmokeEvent.idea, SmokeEvent.event)).all():
            counts[(idea, event)] += 1

        for p in s.exec(select(SmokeProject).order_by(SmokeProject.created_at.desc())).all():
            out["smoke"].append(_smoke_card(p, counts[(p.idea_id, "page_view")],
                                            counts[(p.idea_id, "lead_submitted")]))
        wl = s.exec(select(SmokeEvent.contact).where(
            SmokeEvent.idea == "sozdatel_waitlist",
            SmokeEvent.event == "lead_submitted")).all()
        out["waitlist"] = {"count": len(wl), "contacts": list(wl)}
        logger.info("cabinet: %d tracked, %d smoke", len(out["tracked"]), len(out["smoke"]))
    return out


# ---------------------------------------------------------------------------
# Пресеты: готовые проверенные офферы mass-market идей. Запуск в один клик,
# без LLM-вызова. Контент написан вручную (Fable, 2026-07-11).
# ---------------------------------------------------------------------------

PRESET_OFFERS = [
    {
        "angle": "репутация: негатив без ответа убивает рейтинг",
        "idea_id": "otzovik_v2",
        "product_name": "ОтзоВик",
        "eyebrow": "для селлеров Wildberries и Ozon",
        "h1": "Плохой отзыв без ответа стоит тебе <em>следующих продаж</em>",
        "sub": "Сервис отвечает на каждый отзыв за секунды — по-человечески, в тоне твоего магазина, с учётом оценки. Покупатели видят: продавец не бросает клиентов.",
        "pains": [
            {"h2": "Негатив висит наверху карточки", "p": "Неотвеченный отзыв с 1★ читают все, кто зашёл в карточку — и уходят к конкуренту. Каждый день молчания — минус продажи."},
            {"h2": "Шаблонные ответы видно за версту", "p": "«Спасибо за обратную связь, нам жаль» — покупатели читают это как «нам всё равно». Ответ должен быть про их случай."},
            {"h2": "как это будет работать", "p": "Подключаешь магазин — на каждый новый отзыв готов черновик ответа в твоём тоне. Публикуешь в один клик или включаешь автоответ на 4–5★."},
        ],
        "demo_left_label": "отзыв № 4 812", "demo_left_badge": "★☆☆☆☆",
        "demo_left_text": "«Пришла кофта с затяжкой на рукаве, к празднику не успела заказать замену. Обидно!»",
        "demo_left_meta": "Марина, вчера в 23:47",
        "demo_right_tag": "ответ готов · 3 сек",
        "demo_right_text": "Марина, простите нас за затяжку — это наш брак, и к празднику мы вас подвели. Возврат уже одобрили без ожидания кофты обратно, а на следующий заказ отправили промокод в личные сообщения. Пусть праздник всё-таки удастся!",
        "demo_head_right": "готово за 3 сек",
        "direct_queries": ["ответы на отзывы вайлдберриз", "как отвечать на отзывы озон",
            "шаблоны ответов на отзывы покупателей", "ответ на негативный отзыв wildberries",
            "автоответ на отзывы маркетплейс", "сервис ответов на отзывы", "работа с отзывами wb"],
        "lead_rate_signal": 0.04, "lead_rate_dead": 0.02, "click_target": 40,
    },
    {
        "angle": "работа без договора = работа под честное слово",
        "idea_id": "dogovor_v1",
        "product_name": "ДоговорПро",
        "eyebrow": "для самозанятых и микробизнеса",
        "h1": "Договор под твою услугу — <em>за 5 минут</em>, а не за 15 тысяч",
        "sub": "Опиши, что делаешь и для кого — получи договор, составленный юристом и подогнанный ИИ под твою ситуацию. Предоплата, сроки, правки — всё зафиксировано.",
        "pains": [
            {"h2": "«Кинули на оплату» — история каждого второго", "p": "Без договора заказчик может не заплатить, а ты — ничего не докажешь. Шаблон из интернета суд читает так же скептически, как и ты его скачивал."},
            {"h2": "Юрист стоит как три твоих заказа", "p": "Составить договор у юриста — 10–20 тысяч. Для заказа на 30 тысяч это не защита, а разорение."},
            {"h2": "как это будет работать", "p": "Отвечаешь на 5 вопросов о своей услуге — получаешь готовый договор под неё: с предоплатой, этапами и лимитом правок. Основа составлена практикующим юристом."},
        ],
        "demo_left_label": "заявка № 108", "demo_left_badge": "входящий запрос",
        "demo_left_text": "«Делаю сайты на Тильде, заказчик просит начать без предоплаты, обещает заплатить по результату. Как подстраховаться?»",
        "demo_left_meta": "Денис, самозанятый, сегодня",
        "demo_right_tag": "договор готов · 12 пунктов",
        "demo_right_text": "Готов договор оказания услуг: предоплата 50%, две контрольные точки со сдачей по акту, три круга правок включены, дальше — по прайсу. Пункт 7 защищает вас, если заказчик пропадёт на согласовании.",
        "demo_head_right": "готово за 5 мин",
        "direct_queries": ["договор для самозанятого образец", "договор оказания услуг самозанятый",
            "договор с самозанятым шаблон", "как составить договор на услуги",
            "договор фрилансера с заказчиком", "договор подряда для самозанятых"],
        "lead_rate_signal": 0.035, "lead_rate_dead": 0.0175, "click_target": 40,
    },
]


@app.get("/api/presets")
def presets(request: Request):
    """Готовые офферы для запуска в один клик (владельцу)."""
    _check_owner(request)
    return {"ok": True, "presets": PRESET_OFFERS}


class WaitlistIn(BaseModel):
    contact: str


@app.post("/api/waitlist")
async def waitlist(data: WaitlistIn, request: Request):
    """Лист ожидания Создателя: контакты людей без ключа владельца.
    Создатель smoke-тестит сам себя: та же механика лидов, своя idea-метка."""
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    contact = data.contact.strip()[:_MAX_FIELD]
    if len(contact) < 4:
        raise HTTPException(400, "оставьте email или @telegram")
    with Session(engine) as s:
        s.add(SmokeEvent(idea="sozdatel_waitlist", event="lead_submitted", contact=contact))
        s.commit()
    return {"ok": True}


# ---------------------------------------------------------------------------
# Личный кабинет покупателя: вход по magic-link на почту, без пароля.
# contact уже обязателен для чека оплаты (см. payments.valid_receipt_contact) --
# письмо со ссылкой входа шлётся именно на него, отдельной учётки не нужно.
# ---------------------------------------------------------------------------

SESSION_COOKIE = "sozdatel_session"
MAGIC_LINK_TTL_MINUTES = 30
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


class AccountLinkIn(BaseModel):
    contact: str


def _send_magic_link(contact: str, request: Request, subject: str, intro: str) -> None:
    """Токен входа + письмо со ссылкой -- общая часть обычного входа в кабинет
    и сохранения бесплатной проверки под контакт. Бросает mailer.MailerError
    на стороне вызова, не глотает её сама."""
    token = secrets.token_urlsafe(32)
    with Session(engine) as s:
        s.add(MagicLinkToken(token=token, contact=contact))
        s.commit()
    base = str(request.base_url).rstrip("/")
    link = f"{base}/account/verify?token={token}"
    body = (f"{intro} (действует {MAGIC_LINK_TTL_MINUTES} минут):\n{link}\n\n"
            "Если это не вы — просто проигнорируйте это письмо.")
    mailer.send(contact, subject, body)


@app.post("/api/account/request-link")
async def account_request_link(data: AccountLinkIn, request: Request):
    # Отправляет письмо -- без лимита кто угодно мог бы забросать произвольную
    # почту письмами со ссылкой входа (чужой адрес, не только свой) и посадить
    # репутацию SMTP-аккаунта. Тот же лимит, что у остальных публичных ручек.
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    contact = data.contact.strip().lower()
    if not _EMAIL_RE.match(contact):
        return JSONResponse({"ok": False, "error": "Введите почту, на которую оформляли заказ."}, status_code=400)
    if not mailer.configured():
        return JSONResponse({"ok": False, "error": "Вход по почте пока не настроен на сервере."}, status_code=503)
    try:
        _send_magic_link(contact, request, "Вход в личный кабинет — Создатель",
                          "Ссылка для входа в личный кабинет Создателя")
    except mailer.MailerError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return {"ok": True, "message": "Если эта почта известна нам, письмо со ссылкой уже отправлено."}


def _verify_page(heading: str, lead: str, action: str, *,
                 who: str = "", fine: str = "", status: int = 200) -> HTMLResponse:
    """Страница вокруг ссылки входа -- в дизайн-системе, а не голым `<p>`.

    Раньше отказ отдавался фрагментом без doctype и стилей: человек, у
    которого перестала работать ссылка на оплаченный отчёт, видел чёрный
    Times New Roman на белом и решал, что сайт сломан.
    """
    html_out = (_static("verify.html")
                .replace("__HEADING__", heading)
                .replace("__LEAD__", lead)
                .replace("__WHO__", who)
                .replace("__ACTION__", action)
                .replace("__FINE__", fine))
    return HTMLResponse(_fill_server_values(html_out), status_code=status)


def _expired_link_page() -> HTMLResponse:
    return _verify_page(
        "Ссылка недействительна или устарела",
        f"Ссылки для входа живут {MAGIC_LINK_TTL_MINUTES} минут и открываются один раз. "
        "Запросите новую — она придёт на ту же почту.",
        '<a class="btn" href="/account">Запросить новую ссылку</a>',
        fine="Если письмо не приходит, проверьте папку «Спам».",
        status=400)


def _find_live_link(s: Session, token: str) -> Optional["MagicLinkToken"]:
    link = s.exec(select(MagicLinkToken).where(MagicLinkToken.token == token)).first()
    if (not link or link.used
            or link.created_at < utcnow() - timedelta(minutes=MAGIC_LINK_TTL_MINUTES)):
        return None
    return link


@app.get("/account/verify")
def account_verify_page(token: str):
    """Показывает страницу с кнопкой -- и НЕ гасит ссылку.

    Почтовые провайдеры и антивирусы (mail.ru, Яндекс, Kaspersky) открывают
    ссылки из писем сами, до человека, чтобы проверить их на вредоносность.
    Пока вход происходил прямо здесь, такой обход съедал одноразовый токен:
    человек кликал и получал «ссылка недействительна», просил новую -- и
    сканер съедал её тоже. Замкнутый круг, в котором покупатель не может
    открыть то, за что заплатил. Хуже того, ответ нёс `Set-Cookie` на
    180 дней, то есть кабинет открывался машине, проверявшей ссылку.

    Вход перенесён на POST: сканеры ходят GET и HEAD, форму не отправляют.
    Лишний клик здесь не потеря -- он ещё и называет почту, в чей кабинет
    человек входит.
    """
    with Session(engine) as s:
        link = _find_live_link(s, token)
        if not link:
            return _expired_link_page()
        contact = link.contact
    return _verify_page(
        "Вход в личный кабинет",
        "Остался один шаг — подтвердите, что это вы открыли письмо.",
        # Токен уходит в query действия, а не полем формы: разбор тела
        # потребовал бы python-multipart, лишней зависимости в проде.
        f'<form method="post" action="/account/verify?token='
        f'{urllib.parse.quote(token)}">'
        f'<button class="btn" type="submit">Войти в кабинет</button></form>',
        who=f'<div class="who"><span>Кабинет</span><b>{html.escape(contact)}</b></div>',
        fine=f"Ссылка действует {MAGIC_LINK_TTL_MINUTES} минут с момента отправки письма.")


@app.post("/account/verify")
def account_verify(token: str):
    with Session(engine) as s:
        link = _find_live_link(s, token)
        if not link:
            return _expired_link_page()
        link.used = True
        s.add(link)
        session_token = secrets.token_urlsafe(32)
        s.add(AccountSession(token=session_token, contact=link.contact))
        s.commit()
    resp = RedirectResponse(url="/account", status_code=303)
    resp.set_cookie(SESSION_COOKIE, session_token, max_age=180 * 24 * 3600,
                     httponly=True, samesite="lax")
    return resp


@app.post("/api/account/logout")
def account_logout():
    resp = JSONResponse({"ok": True})
    resp.delete_cookie(SESSION_COOKIE)
    return resp


def _current_contact(request: Request) -> Optional[str]:
    token = request.cookies.get(SESSION_COOKIE)
    if not token:
        return None
    with Session(engine) as s:
        sess = s.exec(select(AccountSession).where(AccountSession.token == token)).first()
        return sess.contact if sess else None


class DemandSaveIn(BaseModel):
    contact: str = ""


@app.post("/api/demand/{rid}/save")
async def demand_save(rid: int, data: DemandSaveIn, request: Request):
    """Привязать бесплатную проверку спроса к кабинету -- иначе результат,
    полученный без прямой ссылки на /account (обычный вход с посадочной),
    нигде не найти повторно. Уже вошедшему привязываем контактом сессии
    сразу; остальным -- контакт из формы + magic-link, как обычный вход."""
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec:
            return JSONResponse({"ok": False, "error": "Проверка не найдена."}, status_code=404)

        already = _current_contact(request)
        contact = already or data.contact.strip().lower()
        # check_id -- обычный автоинкремент, легко перебрать (/r/1, /r/2, ...).
        # Без этой проверки кто угодно мог бы молча переприсвоить чужую уже
        # сохранённую проверку себе, получив в своём /account идею и разбор
        # спроса постороннего человека, а у владельца она бы пропала из вида.
        if rec.contact and rec.contact != contact:
            return JSONResponse({"ok": False, "error": "Эта проверка уже сохранена в другом кабинете."}, status_code=409)

        if already:
            rec.contact = already
            s.add(rec); s.commit()
            return {"ok": True, "message": "Сохранено в кабинете."}

        if not _EMAIL_RE.match(contact):
            return JSONResponse({"ok": False, "error": "Введите почту, на которую пришлём ссылку для входа."}, status_code=400)
        rec.contact = contact
        s.add(rec); s.commit()

    if not mailer.configured():
        return {"ok": True, "message": "Сохранено. Вход по почте пока не настроен на сервере."}
    try:
        _send_magic_link(contact, request, "Проверка сохранена — вход в кабинет Создателя",
                          "Идея сохранена в кабинете. Ссылка для входа")
    except mailer.MailerError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=502)
    return {"ok": True, "message": "Сохранили. Письмо со ссылкой для входа в кабинет уже отправлено."}


class ChosenOfferIn(BaseModel):
    offer: dict


class PurposeIn(BaseModel):
    purpose: str


@app.post("/api/demand/{rid}/purpose")
def demand_purpose(rid: int, data: PurposeIn):
    """Сменить оптику разбора на уже посчитанной проверке.

    Человек мог прийти не с той витрины — это обычное дело, если он нашёл нас
    поиском, а не по объявлению. Гонять его через проверку заново незачем:
    спрос уже посчитан, меняется только то, чьими глазами читать результат
    (см. app/audiences.py). Ограничения доступа тут нет намеренно: адрес
    проверки и так неугадываемый (E6), а сменить оптику — не то же, что
    прочитать чужую идею; ничего нового смена не открывает.
    """
    if data.purpose not in audiences.keys():
        return JSONResponse({"ok": False, "error": "Неизвестная аудитория."},
                            status_code=400)
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec:
            return JSONResponse({"ok": False, "error": "Проверка не найдена."},
                                status_code=404)
        rec.purpose = data.purpose
        s.add(rec); s.commit()
    return {"ok": True, "purpose": data.purpose}


@app.post("/api/demand/{rid}/chosen")
def demand_chosen(rid: int, data: ChosenOfferIn, request: Request):
    """Запомнить, какой из трёх заострённых вариантов человек выбрал на /r/.

    Заказ отчёта идёт со страницы /report/{check_id}, где выбора уже нет на
    экране, — без этой привязки человек выбирал позиционирование, а платный
    разбор молча строился по исходной сырой формулировке идеи.
    """
    client_ip = _client_ip(request)
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec:
            return JSONResponse({"ok": False, "error": "Проверка не найдена."}, status_code=404)
        # Проверка, уже привязанная к кабинету, редактируется только своим
        # владельцем: id перебираются, а выбор влияет на платный отчёт.
        if rec.contact and rec.contact != _current_contact(request):
            return JSONResponse({"ok": False, "error": "Эта проверка принадлежит другому кабинету."}, status_code=409)
        rec.chosen_offer = json.dumps(data.offer, ensure_ascii=False)[:6000]
        s.add(rec); s.commit()
    return {"ok": True}


def _check_card(c: "DemandCheck") -> dict:
    """Проверка спроса для кабинета — с цифрами, а не одним названием.

    Человек с пятью проверенными идеями видел пять одинаковых строк и не мог
    сказать, какая сильнее, не открыв каждую (E4). Показываем то, что уже
    посчитано на бесплатной проверке: общий балл, слабое место и частотность.
    Ничего нового не считаем и не обещаем.

    Битый JSON не имеет права спрятать карточку из кабинета (принцип 7):
    без цифр — значит без цифр, но строка на месте.
    """
    score = weakest = count = None
    try:
        data = json.loads(c.result_json) if c.result_json else {}
        overall = data.get("overall") or {}
        score = overall.get("value")
        weakest = overall.get("weakest")
        known = [f.get("count") for f in (data.get("formulations") or [])
                 if isinstance(f, dict) and f.get("count") is not None]
        count = max(known) if known else c.best_count
    except (ValueError, TypeError, AttributeError):
        count = c.best_count
    return {"id": c.id, "idea": c.idea, "result_url": f"/r/{c.public_id}",
            "score": score if isinstance(score, (int, float)) else None,
            "weakest": weakest or "",
            "created_at": c.created_at.isoformat(),
            "count": count if isinstance(count, int) else None}


@app.get("/api/account/me")
def account_me(request: Request):
    contact = _current_contact(request)
    if not contact:
        return {"ok": True, "contact": None, "projects": [], "reports": [], "orders": [], "checks": []}
    with Session(engine) as s:
        projects = s.exec(select(SmokeProject).where(SmokeProject.contact == contact)
                          .order_by(SmokeProject.created_at.desc())).all()
        # Все отчёты, не только оплаченные -- начатая, но не оплаченная
        # покупка не должна пропадать из кабинета (человек мог просто
        # закрыть вкладку с оплатой и вернуться позже).
        reports = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)
                         .order_by(ReportPurchase.created_at.desc())).all()
        # Заявки на живой тест без запущенного проекта -- уже запущенные
        # (idea_id проставлен) показаны выше как карточка проекта, дублировать
        # их здесь не нужно.
        orders = s.exec(select(LiveTestOrder).where(
            LiveTestOrder.contact == contact, LiveTestOrder.idea_id.is_(None)
        ).order_by(LiveTestOrder.created_at.desc())).all()
        # Сохранённые бесплатные проверки спроса -- вход в кабинет без покупки
        # (см. POST /api/demand/{id}/save). Проверки, из которых уже выросли
        # отчёт или заявка на живой тест, не дублируем отдельной строкой --
        # они и так видны выше с более полным контекстом.
        promoted_ids = {r.check_id for r in reports} | {o.check_id for o in orders}
        checks = s.exec(select(DemandCheck).where(
            DemandCheck.contact == contact
        ).order_by(DemandCheck.created_at.desc())).all()
        # С какой стороны человек пришёл -- по самой свежей его проверке (до
        # отсева тех, что уже выросли в отчёт). Кабинет по ней решает, куда
        # вести за следующей идеей: получателя соцконтракта незачем
        # возвращать на витрину для фаундеров (принцип 4).
        purpose = checks[0].purpose if checks else "business"
        checks = [c for c in checks if c.id not in promoted_ids]
        # От самой сильной идеи к самой слабой: это и есть ответ на вопрос
        # «во что вкладываться» (E4). Проверки без балла (Вордстат молчал)
        # уходят вниз -- сравнивать их не с чем, но и прятать нечестно.
        # Сортировка устойчивая, поэтому при равном балле сохраняется
        # прежний порядок: свежие сверху.
        check_cards = [_check_card(c) for c in checks]
        check_cards.sort(key=lambda k: (k["score"] is None, -(k["score"] or 0)))
        # Адрес проверки, из которой выросла заявка. По номеру он открывался
        # бы только у хозяина проверки, а её могли и не привязывать к
        # кабинету -- тогда человек получал 404 на своей же заявке.
        order_links = {}
        for o in orders:
            if not o.check_id:
                continue
            src = s.get(DemandCheck, o.check_id)
            if src:
                order_links[o.check_id] = f"/r/{src.public_id}"
        from collections import defaultdict
        idea_ids = [p.idea_id for p in projects]
        counts: dict[tuple[str, str], int] = defaultdict(int)
        if idea_ids:
            for idea, event in s.exec(select(SmokeEvent.idea, SmokeEvent.event)
                                      .where(SmokeEvent.idea.in_(idea_ids))).all():
                counts[(idea, event)] += 1
    return {
        "ok": True, "contact": contact, "purpose": purpose,
        # Куда вести за следующей идеей -- адрес своей витрины, а не
        # хардкод в статике. Раньше это была ветка "social_contract или /"
        # прямо в account.html: третья аудитория (студент) в неё не попадала
        # и тихо утекала на витрину для фаундеров (тот же класс дефекта,
        # что и F1, только в другом файле).
        "home": audiences.for_page(purpose)["home"],
        "projects": [_smoke_card(p, counts[(p.idea_id, "page_view")],
                                 counts[(p.idea_id, "lead_submitted")]) for p in projects],
        # tier_label приходит с сервера, а не зашит в кабинете: тариф уже
        # переименовывали ("Полный отчёт" -> "Бизнес-план"), и вторая копия
        # названия в статике разъезжается с витриной незаметно.
        # Дата у каждой строки: без неё пять проверок одной идеи выглядят
        # одинаковыми, и понять, какая из них свежая, нельзя вообще никак
        # (кастдев 2026-08-02: «каша из идей»). Отдаём ISO, человеческий вид
        # собирает кабинет -- формат даты дело представления, не API.
        "reports": [{"check_id": r.check_id, "idea": r.idea, "tier": r.tier,
                     "tier_label": REPORT_PRICES.get(r.tier, {}).get("label", r.tier),
                     "status": _effective_status(r.status, r.created_at),
                     "created_at": r.created_at.isoformat(),
                     "report_url": _report_link(r)} for r in reports],
        "orders": [{"id": o.id, "idea": o.idea, "check_id": o.check_id,
                    "status": _effective_status(o.status, o.created_at),
                    "created_at": o.created_at.isoformat(),
                    "continue_url": order_links.get(o.check_id)} for o in orders],
        "checks": check_cards,
    }


@app.get("/account", response_class=HTMLResponse)
def account_page():
    return HTMLResponse(_static("account.html"))


@app.get("/legal", response_class=HTMLResponse)
def legal_page():
    return HTMLResponse(_static("legal.html"))


@app.get("/guide/direct", response_class=HTMLResponse)
def guide_direct():
    """Этап 3 из 7 — пошаговый запуск Директа, часть «Тест на реальных людях»
    (режим эксперта, только Поиск)."""
    return HTMLResponse(_with_server_values("guide-direct.html"))


def _audience_landing(key: str) -> str:
    """Витрина аудитории: одна вёрстка, тексты из реестра.

    Третья копия HTML разъехалась бы ровно так же, как разъезжались цены (B5)
    и названия тарифов (B7) — а витрин теперь три. Позиционирование главной
    при этом не меняется: она общая и о конкретных аудиториях с порога не
    говорит, ссылки живут в переключателе (F2).
    """
    a = audiences.get(key)
    c = a.copy
    promises = "".join(
        f'<div class="point"><span class="mark"></span><div>'
        f"<h3>{h3}</h3><p>{html.escape(txt)}</p></div></div>"
        for h3, txt in c.get("promises", []))
    faq = "".join(f'<div class="faq"><h3>{html.escape(q)}</h3>'
                  f"<p>{html.escape(ans)}</p></div>"
                  for q, ans in c.get("faq", []))
    # Кнопка-обгон (F10): только у аудиторий, где она задана в реестре --
    # проверка спроса всё равно считается в фоне, ведёт сразу на /report/.
    fast_plan_btn = (
        f'<button class="btn-ghost" id="plan-btn" type="button">'
        f'{html.escape(c["fast_plan_label"])}</button>'
        if c.get("fast_plan_label") else "")
    page = (_static("audience-landing.html")
            .replace("__PAGE_TITLE__", html.escape(c.get("title", "Создатель")))
            .replace("__META__", html.escape(c.get("meta", "")))
            .replace("__H1__", c.get("h1", ""))
            .replace("__SUB__", html.escape(c.get("sub", "")))
            .replace("__FIELD_LABEL__", html.escape(c.get("field_label", "Опишите идею")))
            .replace("__PLACEHOLDER__", html.escape(c.get("placeholder", "")))
            .replace("__PROMISE_TITLE__", html.escape(c.get("promise_title", "")))
            .replace("__PROMISE_SUB__", html.escape(c.get("promise_sub", "")))
            .replace("__PROMISES__", promises)
            .replace("__QUICK_NOTE__", c.get("quick_note", ""))
            .replace("__FULL_NOTE__", c.get("full_note", ""))
            .replace("__FAQ__", faq)
            .replace("__FAST_PLAN_BTN__", fast_plan_btn)
            .replace("__AUDIENCE_SWITCH__", _audience_switch_html(a.key))
            .replace("__AUDIENCE_KEY__", a.key))
    return _fill_server_values(page, a.key)


@app.get("/students", response_class=HTMLResponse)
def students_page():
    """Витрина под студенческие проекты: курсовая, диплом, конкурс, грант.

    Цены те же, что у всех (решение владельца: разные цены обидят тех, кто не
    попал в льготную группу). Отличается оптика разбора и то, что мы обещаем:
    студенту сильнее бизнес-плана работает тест на реальных людях — на защите
    он говорит не «я придумал», а «я проверил, вот цифры».
    """
    return HTMLResponse(_audience_landing("student"))


@app.get("/social-contract", response_class=HTMLResponse)
def social_contract_page():
    """Витрина под аудиторию социального контракта: те же цены и тот же
    бесплатный /api/demand -> /r/{id}, что и на главной, но своя оптика
    разбора (см. app/audiences.py).

    Главная страница по-прежнему не говорит о соцконтракте с порога — незачем
    сужать себя перед массовым посетителем. Но раньше эта витрина была
    доступна ТОЛЬКО по ссылке из объявления, и это отдельный дефект (F2):
    человек, которому нужно обоснование для комиссии, приходил на главную и
    не понимал, что мы и про него тоже. Теперь на обеих витринах есть
    переключатель.
    """
    return HTMLResponse(_audience_landing("social_contract"))


@app.get("/oferta", response_class=HTMLResponse)
def oferta_page():
    # Оферта называет порог сбора данных числом -- это условие договора, оно
    # обязано совпадать с тем, по которому реально считает движок.
    return HTMLResponse(_with_server_values("oferta.html"))


@app.get("/agreement", response_class=HTMLResponse)
def agreement_page():
    return HTMLResponse(_static("agreement.html"))


@app.get("/privacy", response_class=HTMLResponse)
def privacy_page():
    return HTMLResponse(_static("privacy.html"))


@app.get("/contacts", response_class=HTMLResponse)
def contacts_page():
    return HTMLResponse(_static("contacts.html"))


@app.get("/robots.txt")
def robots():
    from fastapi.responses import PlainTextResponse
    # Индексируем витрины (главная и страницы аудиторий -- это посадочные под
    # поиск и Директ); служебное и приватное -- нет. Личные страницы человека
    # (/r/, /report/, /account) несут ещё и meta noindex: robots.txt лишь
    # просит не ходить, а мета запрещает показывать в выдаче.
    #
    # `Disallow: /legal` убран: это публичная юридическая страница, и закрывать
    # её от индексации незачем -- ссылку на неё мы сами даём из форм согласия
    # и подвала (152-ФЗ). Прежний `Allow: /$` тоже убран: без общего
    # `Disallow: /` он ничего не разрешал сверх умолчания и только путал.
    return PlainTextResponse(
        "User-agent: *\n"
        "Disallow: /desk\nDisallow: /p/\nDisallow: /l/\nDisallow: /api/\n"
        "Disallow: /r/\nDisallow: /report/\nDisallow: /account\n"
    )


# Жёлтый квадрат с «С» -- ровно тот же значок, что стоит в <link rel="icon">
# на каждой странице. Раньше здесь отдавался фавикон СТАРОЙ дизайн-системы
# (тёмно-синий фон #11263F с оранжевой рамкой): браузер, спросивший
# /favicon.ico напрямую, получал чужой бренд рядом со ссылкой на сайт.
_FAVICON_SVG = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">'
    '<rect width="64" height="64" rx="12" fill="#FFDE59"/>'
    '<text x="32" y="44" font-family="Arial,sans-serif" font-size="36" '
    'font-weight="700" text-anchor="middle" fill="#1B1C20">С</text></svg>'
)


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    return Response(content=_FAVICON_SVG, media_type="image/svg+xml")


@app.get("/health")
def health():
    """Проверка живости ПРОЦЕССА. Намеренно НЕ трогает БД: если Postgres
    тормозит или недоступен, /health должен ответить мгновенно -- иначе
    это уже не health-check, а часть проблемы, которую он должен обнаружить."""
    return {"ok": True, "service": "sozdatel", "version": app.version}


@app.get("/health/db")
def health_db():
    """Отдельная проверка БД -- дольше и по требованию, не в общем пути."""
    import time
    t0 = time.monotonic()
    try:
        with Session(engine) as s:
            s.exec(select(SmokeProject.id).limit(1)).first()
        return {"ok": True, "db_ms": round((time.monotonic() - t0) * 1000)}
    except Exception as exc:
        return JSONResponse({"ok": False, "error": str(exc)}, status_code=503)


# Цели Метрики. Названия здесь — единственный источник правды: их же владелец
# заводит в интерфейсе Метрики, и разъехавшееся имя означает молча пустой
# отчёт по цели. Порядок соответствует пути человека.
METRIKA_GOALS = [
    ("demand_started", "Начал бесплатную проверку спроса"),
    ("demand_done", "Увидел результат проверки"),
    ("sharpen_used", "Заострил идею"),
    ("check_saved", "Сохранил проверку в кабинете"),
    ("report_viewed", "Открыл страницу отчёта"),
    ("example_viewed", "Посмотрел пример отчёта"),
    ("report_order_started", "Нажал «Получить отчёт»"),
    ("live_test_ordered", "Заказал тест на реальных людях"),
    ("live_test_paid", "Оплатил тест на реальных людях"),
    ("report_paid_quick", "Оплатил быстрый разбор"),
    ("report_paid_full", "Оплатил бизнес-план"),
]


def _inject_metrika(html: str) -> str:
    """Код счётчика — в одном месте, а не скопирован в каждый HTML-файл.
    /l/{id} (проверочные страницы конкретных проектов) сюда не попадают --
    это чужой трафик по чужой рекламе, не воронка самого Создателя.

    Вместе со счётчиком отдаём sozGoal() -- единственный способ отправить
    цель. Раньше каждая страница носила свою копию проверки
    `if (window.SOZDATEL_YM_ID && typeof ym === 'function')`, и новая
    страница просто забывала её написать. Здесь же цель получает параметр
    purpose: без него нельзя понять, какая рекламная кампания окупается
    (D3), потому что обе аудитории идут по одним и тем же шагам.
    """
    if not YANDEX_METRIKA_ID or "</head>" not in html:
        return html
    snippet = f"""<script>window.SOZDATEL_YM_ID = {YANDEX_METRIKA_ID};
window.sozGoal = function(name, params){{
  try {{
    if (!window.SOZDATEL_YM_ID || typeof ym !== 'function') return;
    ym(window.SOZDATEL_YM_ID, 'reachGoal', name, params || {{}});
  }} catch (e) {{}}   // счётчик не имеет права ломать страницу
}};</script>
<script type="text/javascript">
    (function(m,e,t,r,i,k,a){{
        m[i]=m[i]||function(){{(m[i].a=m[i].a||[]).push(arguments)}};
        m[i].l=1*new Date();
        for (var j = 0; j < document.scripts.length; j++) {{if (document.scripts[j].src === r) {{ return; }}}}
        k=e.createElement(t),a=e.getElementsByTagName(t)[0],k.async=1,k.src=r,a.parentNode.insertBefore(k,a)
    }})(window, document,'script','https://mc.webvisor.org/metrika/tag_ww.js?id={YANDEX_METRIKA_ID}', 'ym');
    ym({YANDEX_METRIKA_ID}, 'init', {{ssr:true, webvisor:true, clickmap:true, ecommerce:"dataLayer", trackLinks:true, accurateTrackBounce:true}});
</script>
<noscript><div><img src="https://mc.yandex.ru/watch/{YANDEX_METRIKA_ID}" style="position:absolute; left:-9999px;" alt="" /></div></noscript>
"""
    return html.replace("</head>", snippet + "</head>", 1)


_STATIC_CACHE: dict[str, str] = {}


def _static(name: str) -> str:
    """Читаем файл с диска один раз за жизнь процесса."""
    if name not in _STATIC_CACHE:
        _STATIC_CACHE[name] = _inject_metrika((BASE_DIR.parent / "static" / name).read_text())
    return _STATIC_CACHE[name]


@app.get("/desk", response_class=HTMLResponse)
def desk_page():
    """Рабочий стол владельца: все проекты одинаковыми карточками с цифрами,
    текущим шагом и одним действием. Гость сюда не попадает (ключ)."""
    return HTMLResponse(_static("desk.html"))


@app.get("/portfolio")
def portfolio_page():
    """Экран умер в v1.0: дублировал /desk и путал. Старые ссылки не ломаем."""
    from fastapi.responses import RedirectResponse
    return RedirectResponse("/desk", status_code=307)


@app.get("/p/{idea_id}", response_class=HTMLResponse)
def project_page(idea_id: str):
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
    if proj is None:
        raise HTTPException(404, "проект не найден")
    tpl = _static("project.html")
    return HTMLResponse(tpl.replace("{{IDEA_ID}}", idea_id)
                           .replace("{{PRODUCT_NAME}}", proj.product_name))


def _audience_switch_html(current: str) -> str:
    """Переключатель витрин — собирается из реестра, а не пишется в каждой.

    До этого `/social-contract` был доступен только по ссылке из объявления:
    с сайта на него не попасть, а пришедший по объявлению не мог уйти обратно,
    если ошибся витриной. Единственной ссылкой был логотип, который молча
    уводил на «другой» сайт.

    Витрин будет больше двух (студенты), поэтому разметка живёт здесь одна:
    копия в каждой странице разъехалась бы так же, как разъезжались цены (B5)
    и названия тарифов (B7). Текущая аудитория не ссылка — вести человека на
    страницу, где он уже стоит, значит тратить его клик.
    """
    items = []
    for a in audiences.AUDIENCES.values():
        text = html.escape(a.switch_label)
        if a.key == current:
            items.append(f'<span class="aud-cur" aria-current="page">{text}</span>')
        else:
            href = f"/{a.slug}" if a.slug else "/"
            items.append(f'<a href="{href}">{text}</a>')
    return ('<nav class="aud-switch" aria-label="Кому это нужно">'
            '<span class="aud-tag">Вы к нам зачем:</span>' + "".join(items) + "</nav>")


def _fill_server_values(html: str, purpose: str = "business") -> str:
    """Подставляет в статику всё, чему в коде есть единственный источник:
    цены, названия тарифов, пороги вердикта, рекламный бюджет.

    Зашитая в HTML копия такого значения — уже трижды пойманный источник
    вранья: кабинет звал тариф «Полный отчёт», когда витрина звала его
    «Бизнес-план»; главная обещала порог 2,5% при реальных 8%; про рекламный
    бюджет цифра жила только в плейбуке. Заметить это можно было, лишь
    сравнив две страницы с кодом глазами (B5 в PRODUCT_ROADMAP).
    """
    for slot, value in (
        # По умолчанию записки нет: её подставляет только _lost_page().
        ("__LOST_NOTE__", ""),
        # Витрина, отданная не своим обработчиком (404-страница, пример),
        # всё равно получает переключатель — с аудиторией по умолчанию.
        ("__AUDIENCE_SWITCH__", _audience_switch_html(audiences.DEFAULT.key)),
        ("__CLICK_TARGET__", str(CLICK_TARGET)),
        ("__SIGNAL_PCT__", _pct(SIGNAL_RATE)),
        ("__DEAD_PCT__", _pct(DEAD_RATE)),
        ("__AD_BUDGET__", AD_BUDGET_HINT),
        ("__LIVE_TEST_PRICE__", str(LIVE_TEST_PRICE)),
        ("__MIN_REPORT_PRICE__", str(min(t["price"] for t in REPORT_PRICES.values()))),
        ("__QUICK_LABEL__", REPORT_PRICES["quick"]["label"]),
        ("__QUICK_PRICE__", str(REPORT_PRICES["quick"]["price"])),
        ("__QUICK_WAS__", str(REPORT_PRICES["quick"]["was"])),
        ("__FULL_LABEL__", REPORT_PRICES["full"]["label"]),
        ("__FULL_PRICE__", str(REPORT_PRICES["full"]["price"])),
        ("__FULL_WAS__", str(REPORT_PRICES["full"]["was"])),
    ):
        html = html.replace(slot, value)
    # Ссылка на пример стоит денег (запрос в БД) и появляется, только когда
    # пример опубликован -- считаем её лишь если слот на странице есть.
    if "__EXAMPLE_LINK__" in html:
        html = html.replace("__EXAMPLE_LINK__", _example_link("Посмотреть пример отчёта", purpose))
    if "__TIER_SUMMARY__" in html:
        html = html.replace("__TIER_SUMMARY__", _tier_summary_html(purpose))
    return html


def _with_server_values(name: str, audience: str = "business") -> str:
    return _fill_server_values(
        _static(name).replace("__AUDIENCE_SWITCH__", _audience_switch_html(audience)))


def _lost_page() -> HTMLResponse:
    """404 по ссылке на проверку: главная + записка о том, что произошло.

    Ссылкой на результат делятся намеренно, поэтому обрезанная мессенджером
    или устаревшая ссылка -- обычное дело. Раньше здесь отдавался
    `_static("index.html")` в обход подстановки, и человек читал буквально
    «Больше __SIGNAL_PCT__ — идея живая»: первое, что он узнавал о сервисе,
    -- что сервис сломан. Молчаливая подмена страницы главной тоже плоха --
    он шёл смотреть конкретную проверку и не понимал, куда попал.
    """
    note = ('<div class="lost-note" id="lost-note" role="status">'
            '<b>Мы не нашли эту проверку.</b> Ссылка могла быть неполной — '
            'мессенджеры часто обрезают длинные адреса. Опишите идею ниже: '
            'проверим спрос заново, это бесплатно.</div>')
    html_out = _fill_server_values(_static("index.html").replace("__LOST_NOTE__", note))
    return HTMLResponse(html_out, status_code=404)


_FONTS_DIR = BASE_DIR.parent / "static" / "fonts"
# Белый список читается один раз при старте: перебирать каталог на каждый
# запрос незачем, а имя из URL к путям на диске не приклеивается.
_FONT_FILES = {p.name for p in _FONTS_DIR.glob("*") if p.suffix in (".woff2", ".css")}


@app.get("/fonts/{name}")
def font_file(name: str):
    """Шрифты лежат рядом со статикой, а не на fonts.googleapis.com: из
    России тот домен часто недоступен, а тег был рендер-блокирующим -- при
    недоступности человек видел белый экран до таймаута (принцип 8).

    Отдаём строго из static/fonts и только известные расширения: обычного
    StaticFiles-монтирования здесь нет намеренно, оно открыло бы наружу и
    HTML-шаблоны с неподставленными слотами.
    """
    if name not in _FONT_FILES:
        raise HTTPException(404, "не найдено")
    media = "text/css; charset=utf-8" if name.endswith(".css") else "font/woff2"
    # Файлы неизменяемые: имя меняется вместе с содержимым при обновлении.
    return FileResponse(_FONTS_DIR / name, media_type=media,
                        headers={"Cache-Control": "public, max-age=31536000, immutable"})


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_with_server_values("index.html"))
