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

import html
import json
import logging
import os
import re
import secrets
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
SIGNAL_RATE = 0.08      # заявок/визитов, с которых интерес считается настоящим
DEAD_RATE = 0.04        # и ниже -- интереса нет


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
    """Каждая бесплатная проверка спроса: счётчик + страница результата /r/<id>."""
    id: Optional[int] = Field(default=None, primary_key=True)
    created_at: datetime = Field(default_factory=utcnow)
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
    # Публичный пример на /example. Настоящий сгенерированный отчёт, а не
    # написанный руками: показывать более гладкий текст, чем отдаёт движок,
    # значит продавать не то, что отдаём (принцип 3). Помечает владелец.
    is_example: bool = False


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
try:  # create_all не добавляет колонки в существующие таблицы -- добиваем вручную
    from sqlalchemy import text as _sqltext
    with engine.connect() as _c:
        _c.execute(_sqltext("ALTER TABLE demandcheck ADD COLUMN IF NOT EXISTS result_json VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE livetestorder ADD COLUMN IF NOT EXISTS chosen_offer VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE smokeproject ADD COLUMN IF NOT EXISTS contact VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE livetestorder ADD COLUMN IF NOT EXISTS idea_id VARCHAR"))
        _c.execute(_sqltext("ALTER TABLE demandcheck ADD COLUMN IF NOT EXISTS contact VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE demandcheck ADD COLUMN IF NOT EXISTS purpose VARCHAR DEFAULT 'business'"))
        _c.execute(_sqltext("ALTER TABLE reportpurchase ADD COLUMN IF NOT EXISTS gen_error VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE reportpurchase ADD COLUMN IF NOT EXISTS fail_notified BOOLEAN DEFAULT FALSE"))
        _c.execute(_sqltext("ALTER TABLE reportpurchase ADD COLUMN IF NOT EXISTS paid_notified BOOLEAN DEFAULT FALSE"))
        _c.execute(_sqltext("ALTER TABLE livetestorder ADD COLUMN IF NOT EXISTS paid_notified BOOLEAN DEFAULT FALSE"))
        _c.execute(_sqltext("ALTER TABLE demandcheck ADD COLUMN IF NOT EXISTS chosen_offer VARCHAR DEFAULT ''"))
        _c.execute(_sqltext("ALTER TABLE reportpurchase ADD COLUMN IF NOT EXISTS is_example BOOLEAN DEFAULT FALSE"))
        _c.execute(_sqltext("ALTER TABLE demandcheck ADD COLUMN IF NOT EXISTS sample_json VARCHAR DEFAULT ''"))
        _c.commit()
except Exception:  # sqlite в тестах создаёт таблицу сразу с колонкой -- это норма
    pass

@asynccontextmanager
async def _lifespan(_app: FastAPI):
    """Прогрев при старте: соединение с БД и статика читаются ДО первого
    запроса пользователя. Без этого первый визит платил за всё сразу."""
    try:
        with Session(engine) as s:
            s.exec(select(SmokeProject.id).limit(1)).first()
    except Exception:
        logger.exception("warm-up db failed (non-fatal)")
    for name in ("index.html", "project.html", "guide-direct.html", "result.html", "report.html",
                 "social-contract.html", "account.html"):
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
        result = await sharpen_idea(data.idea)
        return {"ok": True, **result}
    except OfferEngineError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


# ---------------------------------------------------------------------------
# Этап ①: бесплатная проверка спроса (публичная — это вход воронки)
# ---------------------------------------------------------------------------

@app.post("/api/idea")
async def idea_suggest(request: Request):
    """«Придумать за меня» — для тех, кто пришёл без идеи (вход воронки)."""
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    try:
        return {"ok": True, "idea": await generate_idea()}
    except DemandError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.post("/api/demand")
async def demand_check(data: IdeaIn, request: Request):
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    try:
        result = await check_demand(data.idea)
    except DemandError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    check_id = None
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
            check_id = rec.id
    except Exception:
        logging.getLogger(__name__).warning("demand check not persisted", exc_info=True)
    return {"ok": True, "id": check_id, **result}


@app.post("/api/sharpen")
async def sharpen(data: IdeaIn, request: Request):
    """Бесплатное заострение идеи в 3 варианта позиционирования — по кнопке
    на странице результата, не на каждый визит (LLM-вызов тяжёлый и долгий)."""
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
    if _rate_limited(client_ip):
        raise HTTPException(429, "слишком часто")
    try:
        result = await sharpen_idea(data.idea)
        return {"ok": True, **result}
    except OfferEngineError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


LIVE_TEST_PRICE = int(os.environ.get("SOZDATEL_LIVE_TEST_PRICE", "1490"))


@app.get("/r/{rid}", response_class=HTMLResponse)
def result_page(rid: int):
    """Страница результата проверки: инструмент, а не витрина. Узкая полоска
    преемственности вместо всего пути 0->7; отсюда же -- заказ живого теста."""
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
    if not rec or not rec.result_json:
        return HTMLResponse(_static("index.html"), status_code=404)
    tpl = _static("result.html")
    safe_json = rec.result_json.replace("</", "<\\/")
    idea_json = json.dumps(rec.idea, ensure_ascii=False).replace("</", "<\\/")
    html_out = (tpl
        .replace("__CHECK_ID__", str(rec.id))
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
        .replace("__PURPOSE_JSON__", json.dumps(rec.purpose, ensure_ascii=False)))
    return HTMLResponse(_fill_server_values(html_out))


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
    try:
        with Session(engine) as s:
            order = s.get(model, int(order_id)) if order_id else None
            if order and order.status != "paid":
                order.status = "paid"; s.add(order); s.commit()
            if order is not None and not order.paid_notified:
                # Собираем данные письма ВНУТРИ сессии, а шлём после неё:
                # SMTP может отвечать секундами, держать на нём транзакцию
                # и ответ вебхуку ЮКассы незачем.
                if kind == "report":
                    label = REPORT_PRICES.get(order.tier, {}).get("label", order.tier)
                    notify = {"what": f"отчёт «{label}»", "order_id": order.id,
                              "idea": order.idea, "contact": order.contact,
                              "amount": order.amount,
                              "link": f"/report/{order.check_id}" if order.check_id else ""}
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


def _report_preview(demand_data: dict) -> dict:
    """Бесплатный тизер отчёта — из уже посчитанных данных проверки спроса,
    без новых вызовов LLM (заметки по шкалам уже сгенерированы бесплатным
    шагом check_demand). Показывается всегда, вне зависимости от оплаты.
    Текстовый разбор, а не витрина из голых цифр -- см. кастдев-фидбек
    (dimeadozen как ориентир на содержательный анализ, не «воду»)."""
    v = demand_data.get("verdict") or {}
    overall = demand_data.get("overall") or {}
    formulations = demand_data.get("formulations") or []
    known = [f["count"] for f in formulations if f.get("count") is not None]
    top = max(known) if known else None
    comp = demand_data.get("competitors") or {}
    top_names = [c.get("domain") or c.get("title") or "" for c in (comp.get("top") or [])[:3]]
    notes = {s["key"]: s.get("note", "") for s in (demand_data.get("scores") or [])}
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

    Считается ОДИН раз на проверку и кэшируется навсегда: это два вызова
    модели, и платит за них владелец. Генерируем лениво — только когда
    человек реально открыл страницу отчёта, то есть думает о покупке.
    Сбой не имеет права уронить страницу: без образца она просто остаётся
    такой, какой была (принцип 7 — деградация вместо ошибки).
    """
    with Session(engine) as s:
        rec = s.get(DemandCheck, check_id)
        if not rec or not rec.result_json:
            return None
        if rec.sample_json:
            try:
                return json.loads(rec.sample_json)
            except ValueError:
                pass
        idea, purpose = rec.idea, rec.purpose
        chosen = _chosen_offer(rec)
        demand_data = json.loads(rec.result_json)

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
        if fresh and not fresh.sample_json:
            fresh.sample_json = json.dumps(sample, ensure_ascii=False)
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
        order_id = order.id
    if not payments.configured():
        if _notify_owner_order(request, what=f"отчёт «{REPORT_PRICES[tier]['label']}»",
                               order_id=order_id, idea=idea, contact=contact,
                               amount=price, paid=False,
                               link=f"/report/{data.check_id}" if data.check_id else ""):
            _mark_notified(ReportPurchase, order_id)
        return {"ok": True, "paid": False,
                "message": "Заявка принята. Мы соберём отчёт вручную и пришлём в течение дня."}
    try:
        base = str(request.base_url).rstrip("/")
        pid, url = await payments.create_payment(
            order_id, REPORT_PRICES[tier]["price"], f"Создатель · отчёт по идее (заказ {order_id})",
            f"{base}/report/{data.check_id}?paid=1", kind="report", contact=contact)
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


def _mark_notified(model, order_id: int) -> None:
    """Флаг «владельцу уже написали» отдельной короткой транзакцией: вебхук
    ЮКассы может прийти повторно, а страницу заказа можно перезагрузить."""
    try:
        with Session(engine) as s:
            row = s.get(model, order_id)
            if row is not None:
                row.paid_notified = True
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


def _tier_summary_html() -> str:
    """Что входит в каждый тариф — там, где человек решает платить.

    Раньше на `/r/` стояло только «от 990 ₽», а состав тарифов открывался
    лишь на следующем экране. Для пришедшего с /social-contract это прямая
    ловушка: он идёт за обоснованием сметы, а секции «Финансовая модель» в
    дешёвом тарифе нет вовсе (C2 в PRODUCT_ROADMAP).

    Состав собирается из ALL_SECTIONS/QUICK_KEYS, а не пишется руками:
    вторая копия списка разъехалась бы с движком, как уже разъезжались цены.
    """
    # Заголовок «План запуска — по этапам» внутри перечисления через запятую
    # читается двусмысленно из-за тире: берём часть до него.
    def short(title: str) -> str:
        return title.split(" — ")[0].strip()

    quick = [short(t) for k, t in ALL_SECTIONS if k in QUICK_KEYS]
    extra = [short(t) for k, t in ALL_SECTIONS if k not in QUICK_KEYS]
    rows = [
        (REPORT_PRICES["quick"]["label"], REPORT_PRICES["quick"]["price"],
         ", ".join(quick) + "."),
        (REPORT_PRICES["full"]["label"], REPORT_PRICES["full"]["price"],
         "всё это и ещё " + ", ".join(t.lower() for t in extra) + "."),
    ]
    items = "".join(
        f'<div class="tier-row">'
        f'<span class="tier-name">{html.escape(label)} — <b>{price} ₽</b></span>'
        f'<span class="tier-what">{html.escape(what)}</span>'
        f'</div>'
        for label, price, what in rows)
    return f'<div class="tier-what-block">{items}</div>'


def _example_link(text: str) -> str:
    """Ссылка на пример — только когда пример реально существует.

    Обещать «посмотрите пример» и привести на пустую страницу хуже, чем не
    обещать вовсе (принципы 3 и 7). Поэтому витрины спрашивают об этом у
    сервера, а не носят ссылку зашитой.
    """
    with Session(engine) as s:
        if not _example_purchase(s):
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

    try:
        section = await generate_section(key, idea, demand_data, tier,
                                         chosen_offer=chosen, purpose=purpose)
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
def example_publish(request: Request, check_id: int, tier: str = "full"):
    """Пометить уже собранный отчёт публичным примером (/example).

    Пример обязан быть НАСТОЯЩИМ выводом движка: написанный руками текст
    всегда получается глаже, и человек заплатит за одно, а получит другое.
    Поэтому публикуется только уже сгенерированный отчёт -- собрать его
    можно тем же владельческим прогоном, /report/{id}?key=...&preview=full.
    """
    _check_owner(request)
    with Session(engine) as s:
        target = s.exec(select(ReportPurchase).where(
            ReportPurchase.check_id == check_id, ReportPurchase.tier == tier
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
    _purpose = rec.purpose if rec else PURPOSE_DEFAULT
    _tier_keys = section_keys(tier)
    tpl = _static("report.html")
    html_out = (tpl
        .replace("__CHECK_ID__", str(ex.check_id or 0))
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
        .replace("__QUICK_KEYS_JSON__", json.dumps(QUICK_KEYS, ensure_ascii=False)))
    return HTMLResponse(_fill_server_values(html_out))


@app.get("/report/{rid}", response_class=HTMLResponse)
async def report_page(rid: int, request: Request):
    """Дашборд отчёта: бесплатный тизер виден всегда; полные секции --
    после оплаты, генерируются лениво при первом открытии (без воркеров,
    тот же принцип, что и во всём проекте)."""
    # Владелец может собрать любой тариф без оплаты -- иначе проверить, что
    # именно получает человек за 2990 ₽, можно только заплатив себе самому.
    # Промпты правились вслепую, а качество отчёта -- это весь платный продукт.
    owner = _is_owner(request)
    want_preview = request.query_params.get("preview") or ""
    if owner and want_preview in REPORT_PRICES:
        _owner_preview(rid, want_preview)

    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec or not rec.result_json:
            return HTMLResponse(_static("index.html"), status_code=404)
        purchase = _best_report_purchase(s, rid, include_preview=owner)

    demand_data = json.loads(rec.result_json)
    preview = _report_preview(demand_data)
    report_full = None
    gen_error = ""
    # Бесплатный образец нужен только тем, кто ещё не купил: у покупателя
    # весь разбор и так открыт, лишний вызов модели ему ни к чему.
    sample = None if purchase else await _ensure_sample(rid)

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
    tpl = _static("report.html")
    html_out = (tpl
        .replace("__CHECK_ID__", str(rid))
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
                         "report_url": f"/report/{r.check_id}" if r.check_id else None}
                        for r in reversed(reports)]}


@app.get("/api/stats")
def public_stats():
    """Живые цифры для главной. Только честные счётчики из БД."""
    with Session(engine) as s:
        ideas = len(s.exec(select(DemandCheck)).all())
        events = len(s.exec(select(SmokeEvent)).all())
    return {"ideas_checked": ideas, "events": events}


@app.get("/api/diag/yandex")
async def diag_yandex(request: Request, phrase: str = "купить слона"):
    """Owner-only: сырая диагностика интеграции с Яндексом -- оба пути
    Вордстата (официальный OAuth API и прокси внутри Cloud Search API),
    без глотания ошибок. Открыть в браузере с ?key=... при жалобе
    «нет данных», чтобы увидеть точную причину, а не гадать."""
    _check_owner(request)
    return await diagnose(phrase)


# ---------------------------------------------------------------------------
# Этап ⓪→①: выбранный оффер → лендинг, сразу захощенный
# ---------------------------------------------------------------------------

_landing_tpl: Optional[str] = None


def render_landing(offer: dict) -> str:
    global _landing_tpl
    if _landing_tpl is None:  # читаем с диска один раз за жизнь процесса, как и _static()
        _landing_tpl = (BASE_DIR / "landing_template.html").read_text()
    tpl = _landing_tpl
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


def _launch_offer(s: Session, offer: dict, idea_text: str, contact: str = "") -> SmokeProject:
    """Общая логика запуска проекта -- вызывающая сторона уже проверила
    LAUNCH_REQUIRED_FIELDS. contact, если передан, привязывает проект к
    покупателю сразу (виден в его /account без ручного PATCH владельцем)."""
    html = render_landing(offer)
    existing = s.exec(select(SmokeProject).where(SmokeProject.idea_id == offer["idea_id"])).first()
    if existing:
        existing.landing_html = html
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
        click_target=int(offer.get("click_target", 40)),
        lead_rate_signal=float(offer.get("lead_rate_signal", 0.08)),
        lead_rate_dead=float(offer.get("lead_rate_dead", 0.04)),
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


def _rate_limited(ip: str) -> bool:
    import time
    now = time.monotonic()
    bucket = _RL_WINDOW.setdefault(ip, [])
    while bucket and now - bucket[0] > _RL_SECONDS:
        bucket.pop(0)
    if len(bucket) >= _RL_LIMIT:
        return True
    bucket.append(now)
    if len(_RL_WINDOW) > 10000:   # защита памяти от рассеянных IP
        _RL_WINDOW.clear()
    return False


@app.post("/api/smoke-event")
async def smoke_event(request: Request):
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
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


def _smoke_card(p: "SmokeProject", views: int, leads: int) -> dict:
    """Карточка проекта -- общая для владельца (/api/cabinet) и покупателя
    (/api/account/me), чтобы оба видели один и тот же язык: этап 0..7,
    прогресс, вердикт. Smoke-этап определяется данными: есть клики -> ①
    Спрос, иначе ⓪ Идея."""
    stage = 1 if views > 0 else 0
    v = compute_verdict(views, leads, p.click_target, p.lead_rate_signal, p.lead_rate_dead)
    rate = (leads / views) if views else 0.0
    if views == 0:
        next_step = "Запустить Директ на страницу — инструкция: /guide/direct"
    elif views < p.click_target:
        next_step = f"Копим клики: {p.click_target - views} до вердикта. Ничего не менять."
    elif v["verdict"] == "СИГНАЛ ЕСТЬ":
        next_step = "Сигнал есть → идея в очередь на MVP"
    elif v["verdict"] == "СПРОСА НЕТ":
        next_step = "Спроса нет → остановить кампанию, идею в архив"
    else:
        next_step = "Серая зона → второй оффер на том же трафике"
    return {"idea_id": p.idea_id, "name": p.product_name,
            "stage": stage, "stage_name": STAGE_NAMES[stage],
            "views": views, "leads": leads, "rate": round(rate * 100),
            "target": p.click_target, "verdict": v["verdict"],
            "next_step": next_step,
            "progress": min(100, round(views / p.click_target * 100)) if p.click_target else 0,
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
        "lead_rate_signal": 0.08, "lead_rate_dead": 0.04, "click_target": 40,
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
        "lead_rate_signal": 0.07, "lead_rate_dead": 0.035, "click_target": 40,
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
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
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
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
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


@app.get("/account/verify")
def account_verify(token: str):
    with Session(engine) as s:
        link = s.exec(select(MagicLinkToken).where(MagicLinkToken.token == token)).first()
        if (not link or link.used
                or link.created_at < utcnow() - timedelta(minutes=MAGIC_LINK_TTL_MINUTES)):
            return HTMLResponse(
                "<p>Ссылка недействительна или устарела. "
                '<a href="/account">Запросите новую</a>.</p>', status_code=400)
        link.used = True
        s.add(link)
        session_token = secrets.token_urlsafe(32)
        s.add(AccountSession(token=session_token, contact=link.contact))
        s.commit()
    resp = RedirectResponse(url="/account")
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
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
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


@app.post("/api/demand/{rid}/chosen")
def demand_chosen(rid: int, data: ChosenOfferIn, request: Request):
    """Запомнить, какой из трёх заострённых вариантов человек выбрал на /r/.

    Заказ отчёта идёт со страницы /report/{check_id}, где выбора уже нет на
    экране, — без этой привязки человек выбирал позиционирование, а платный
    разбор молча строился по исходной сырой формулировке идеи.
    """
    client_ip = (request.headers.get("x-forwarded-for") or "").split(",")[0].strip() \
        or (request.client.host if request.client else "?")
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
        from collections import defaultdict
        idea_ids = [p.idea_id for p in projects]
        counts: dict[tuple[str, str], int] = defaultdict(int)
        if idea_ids:
            for idea, event in s.exec(select(SmokeEvent.idea, SmokeEvent.event)
                                      .where(SmokeEvent.idea.in_(idea_ids))).all():
                counts[(idea, event)] += 1
    return {
        "ok": True, "contact": contact, "purpose": purpose,
        "projects": [_smoke_card(p, counts[(p.idea_id, "page_view")],
                                 counts[(p.idea_id, "lead_submitted")]) for p in projects],
        # tier_label приходит с сервера, а не зашит в кабинете: тариф уже
        # переименовывали ("Полный отчёт" -> "Бизнес-план"), и вторая копия
        # названия в статике разъезжается с витриной незаметно.
        "reports": [{"check_id": r.check_id, "idea": r.idea, "tier": r.tier,
                     "tier_label": REPORT_PRICES.get(r.tier, {}).get("label", r.tier),
                     "status": _effective_status(r.status, r.created_at),
                     "report_url": f"/report/{r.check_id}"} for r in reports],
        "orders": [{"id": o.id, "idea": o.idea, "check_id": o.check_id,
                    "status": _effective_status(o.status, o.created_at),
                    "continue_url": f"/r/{o.check_id}" if o.check_id else None} for o in orders],
        "checks": [{"id": c.id, "idea": c.idea, "result_url": f"/r/{c.id}"} for c in checks],
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


@app.get("/social-contract", response_class=HTMLResponse)
def social_contract_page():
    """Отдельная посадочная страница под рекламу на аудиторию социального
    контракта -- специально НЕ часть общего позиционирования сайта (см.
    CLAUDE.md), чтобы не отпугивать массового пользователя упоминанием
    грантов/соцконтракта. Ведёт в тот же бесплатный /api/demand -> /r/{id},
    что и главная страница."""
    return HTMLResponse(_with_server_values("social-contract.html"))


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
    # Индексируем витрину; служебные и проверочные страницы -- нет
    # (лендинги идей — временные, дубли по структуре: индексация вредит)
    return PlainTextResponse(
        "User-agent: *\nAllow: /$\nDisallow: /desk\nDisallow: /p/\n"
        "Disallow: /l/\nDisallow: /api/\nDisallow: /legal\n"
    )


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    # оранжевый квадрат-чертёж 1x1 svg: не 404 в каждом визите
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="%2311263F"/><rect x="3" y="3" width="10" height="10" fill="none" stroke="%23FF8A2A" stroke-width="2"/></svg>'
    return Response(content=svg, media_type="image/svg+xml")


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


def _inject_metrika(html: str) -> str:
    """Код счётчика — в одном месте, а не скопирован в каждый HTML-файл.
    /l/{id} (проверочные страницы конкретных проектов) сюда не попадают --
    это чужой трафик по чужой рекламе, не воронка самого Создателя."""
    if not YANDEX_METRIKA_ID or "</head>" not in html:
        return html
    snippet = f"""<script>window.SOZDATEL_YM_ID = {YANDEX_METRIKA_ID};</script>
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


def _fill_server_values(html: str) -> str:
    """Подставляет в статику всё, чему в коде есть единственный источник:
    цены, названия тарифов, пороги вердикта, рекламный бюджет.

    Зашитая в HTML копия такого значения — уже трижды пойманный источник
    вранья: кабинет звал тариф «Полный отчёт», когда витрина звала его
    «Бизнес-план»; главная обещала порог 2,5% при реальных 8%; про рекламный
    бюджет цифра жила только в плейбуке. Заметить это можно было, лишь
    сравнив две страницы с кодом глазами (B5 в PRODUCT_ROADMAP).
    """
    for slot, value in (
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
        html = html.replace("__EXAMPLE_LINK__", _example_link("Посмотреть пример отчёта"))
    if "__TIER_SUMMARY__" in html:
        html = html.replace("__TIER_SUMMARY__", _tier_summary_html())
    return html


def _with_server_values(name: str) -> str:
    return _fill_server_values(_static(name))


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
