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
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
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
    ReportEngineError, generate_report, ALL_SECTIONS, QUICK_KEYS,
)
from app import payments  # noqa: E402

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sozdatel")


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


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
    click_target: int = 40
    lead_rate_signal: float = 0.08
    lead_rate_dead: float = 0.04
    status: str = "running"  # running | signal | dead | gray
    created_at: datetime = Field(default_factory=utcnow)


# Единая шкала пути 0..7 -- те же названия на главной, в кабинете и в API.
STAGE_NAMES = ["Идея", "Спрос", "Проверочная страница", "Реклама",
               "Заявки", "Первые продажи", "Повторяемость", "Удержание"]


class TrackedProject(SQLModel, table=True):
    """Внешний проект в кабинете: живёт не в Создателе (например, АвтоПост
    ведёт Аналитик в Telegram), но виден на общей карте портфеля со своим
    этапом. Мост, а не переезд: ссылка ведёт в родной интерфейс проекта."""
    id: Optional[int] = Field(default=None, primary_key=True)
    name: str
    stage: int = 0                 # 0..7, индекс в STAGE_NAMES
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
    chosen_offer: str = ""    # JSON: вариант позиционирования, выбранный на /r/{id}


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
    for name in ("index.html", "portfolio.html", "project.html", "guide-direct.html", "result.html", "report.html",
                 "social-contract.html"):
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


def _check_owner(request: Request) -> None:
    if not OWNER_KEY:
        raise HTTPException(503, "Сервер не настроен: задайте SOZDATEL_OWNER_KEY в переменных окружения.")
    provided = request.headers.get("X-Owner-Key") or request.query_params.get("key") or ""
    if provided != OWNER_KEY:
        raise HTTPException(401, "Нужен ключ владельца (X-Owner-Key).")


# ---------------------------------------------------------------------------
# Этап ⓪: идея → офферы
# ---------------------------------------------------------------------------

class IdeaIn(BaseModel):
    idea: str


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
        rec = DemandCheck(idea=data.idea[:300], best_count=max(known) if known else None,
                          result_json=json.dumps(result, ensure_ascii=False))
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
        .replace("__PAY_ENABLED__", "true" if payments.configured() else "false")
        .replace("__IDEA__", html.escape(rec.idea))
        .replace("__IDEA_JSON__", idea_json)
        .replace("__RESULT_JSON__", safe_json))
    return HTMLResponse(html_out)


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
        return JSONResponse({"ok": False, "error": "Оставьте телеграм или почту — чтобы нам было куда вернуться с результатом."}, status_code=400)
    idea = ""
    chosen_offer_json = json.dumps(data.chosen_offer, ensure_ascii=False)[:2000] if data.chosen_offer else ""
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
        return {"ok": True, "paid": False,
                "message": "Заявка принята. Мы свяжемся в течение дня, запустим страницу и рекламу вашей идеи."}
    try:
        base = str(request.base_url).rstrip("/")
        # Без check_id ссылка /r/ ведёт в никуда (404) -- возвращаем на главную,
        # а не на битую страницу результата.
        return_url = f"{base}/r/{data.check_id}?paid=1" if data.check_id else f"{base}/?paid=1"
        pid, url = await payments.create_payment(
            order_id, LIVE_TEST_PRICE, f"Создатель · живой тест идеи (заказ {order_id})",
            return_url, kind="livetest")
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
    try:
        with Session(engine) as s:
            order = s.get(model, int(order_id)) if order_id else None
            if order and order.status != "paid":
                order.status = "paid"; s.add(order); s.commit()
    except Exception:
        logging.getLogger(__name__).warning("webhook order update failed", exc_info=True)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Отчёт/бизнес-план: платный разбор идеи поверх уже посчитанных данных спроса
# ---------------------------------------------------------------------------

REPORT_PRICES = {
    "quick": {"price": 990, "was": 1490, "label": "Быстрый разбор"},
    "full": {"price": 2990, "was": 3990, "label": "Полный отчёт"},
}


def _report_preview(demand_data: dict) -> dict:
    """Бесплатный тизер отчёта — из уже посчитанных данных проверки спроса,
    без LLM. Показывается всегда, вне зависимости от оплаты."""
    v = demand_data.get("verdict") or {}
    overall = demand_data.get("overall") or {}
    formulations = demand_data.get("formulations") or []
    known = [f["count"] for f in formulations if f.get("count") is not None]
    top = max(known) if known else None
    comp = demand_data.get("competitors") or {}
    return {
        "best_count": top,
        "verdict_text": v.get("text", ""),
        "verdict_level": v.get("level", "unknown"),
        "overall_value": overall.get("value"),
        "weakest": overall.get("weakest", ""),
        "competitors_count": len(comp.get("top") or []),
    }


def _best_report_purchase(s: Session, check_id: int):
    """Самая полная ОПЛАЧЕННАЯ покупка отчёта для этой проверки спроса --
    full перекрывает quick, если куплены оба."""
    rows = s.exec(select(ReportPurchase).where(
        ReportPurchase.check_id == check_id, ReportPurchase.status == "paid"
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
        return JSONResponse({"ok": False, "error": "Оставьте телеграм или почту — чтобы вернуться к отчёту."}, status_code=400)
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
        return {"ok": True, "paid": False,
                "message": "Заявка принята. Мы соберём отчёт вручную и пришлём в течение дня."}
    try:
        base = str(request.base_url).rstrip("/")
        pid, url = await payments.create_payment(
            order_id, REPORT_PRICES[tier]["price"], f"Создатель · отчёт по идее (заказ {order_id})",
            f"{base}/report/{data.check_id}?paid=1", kind="report")
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


@app.get("/report/{rid}", response_class=HTMLResponse)
async def report_page(rid: int):
    """Дашборд отчёта: бесплатный тизер виден всегда; полные секции --
    после оплаты, генерируются лениво при первом открытии (без воркеров,
    тот же принцип, что и во всём проекте)."""
    with Session(engine) as s:
        rec = s.get(DemandCheck, rid)
        if not rec or not rec.result_json:
            return HTMLResponse(_static("index.html"), status_code=404)
        purchase = _best_report_purchase(s, rid)

    demand_data = json.loads(rec.result_json)
    preview = _report_preview(demand_data)
    report_full = None
    gen_error = ""

    if purchase:
        if not purchase.report_json:
            try:
                report = await generate_report(rec.idea, demand_data, purchase.tier)
                with Session(engine) as s:
                    fresh = s.get(ReportPurchase, purchase.id)
                    fresh.report_json = json.dumps(report, ensure_ascii=False)
                    s.add(fresh); s.commit(); s.refresh(fresh)
                    purchase = fresh
            except ReportEngineError as e:
                gen_error = str(e)
        if purchase.report_json:
            report_full = json.loads(purchase.report_json)

    tpl = _static("report.html")
    html_out = (tpl
        .replace("__CHECK_ID__", str(rid))
        .replace("__IDEA__", html.escape(rec.idea))
        .replace("__PREVIEW_JSON__", json.dumps(preview, ensure_ascii=False))
        .replace("__REPORT_JSON__", json.dumps(report_full, ensure_ascii=False) if report_full else "null")
        .replace("__UNLOCKED_TIER__", json.dumps(purchase.tier if purchase else None))
        .replace("__ORDER_STATUS__", json.dumps(purchase.status if purchase else None))
        .replace("__GEN_ERROR__", json.dumps(gen_error, ensure_ascii=False))
        .replace("__PRICES_JSON__", json.dumps(REPORT_PRICES, ensure_ascii=False))
        .replace("__SECTIONS_JSON__", json.dumps([{"key": k, "title": t} for k, t in ALL_SECTIONS], ensure_ascii=False))
        .replace("__QUICK_KEYS_JSON__", json.dumps(QUICK_KEYS, ensure_ascii=False)))
    return HTMLResponse(html_out)


@app.get("/api/orders")
def orders_list(request: Request):
    _check_owner(request)
    with Session(engine) as s:
        rows = s.exec(select(LiveTestOrder)).all()
    return {"orders": [{"id": o.id, "created_at": str(o.created_at), "idea": o.idea,
                        "contact": o.contact, "status": o.status, "amount": o.amount,
                        "chosen_offer": json.loads(o.chosen_offer) if o.chosen_offer else None}
                       for o in reversed(rows)]}


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


@app.post("/api/launch")
def launch(data: LaunchIn, request: Request):
    _check_owner(request)
    offer = data.offer
    for key in ("idea_id", "product_name", "h1", "sub", "pains",
                "demo_left_label", "demo_left_text", "demo_right_text", "eyebrow"):
        if not offer.get(key):
            raise HTTPException(400, f"в оффере нет поля {key}")
    html = render_landing(offer)
    with Session(engine) as s:
        existing = s.exec(select(SmokeProject).where(SmokeProject.idea_id == offer["idea_id"])).first()
        if existing:
            existing.landing_html = html
            existing.offer_json = json.dumps(offer, ensure_ascii=False)
            s.add(existing); s.commit()
            proj = existing
        else:
            proj = SmokeProject(
                idea_id=offer["idea_id"], product_name=offer["product_name"],
                idea_text=data.idea_text[:2000],
                offer_json=json.dumps(offer, ensure_ascii=False),
                landing_html=html,
                click_target=int(offer.get("click_target", 40)),
                lead_rate_signal=float(offer.get("lead_rate_signal", 0.08)),
                lead_rate_dead=float(offer.get("lead_rate_dead", 0.04)),
            )
            s.add(proj); s.commit(); s.refresh(proj)
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
    """Детерминированный вердикт этапа ① — те же честные слова, что везде."""
    rate = (leads / views) if views else 0.0
    if views < target:
        return {"verdict": "РАНО СУДИТЬ",
                "detail": f"{views}/{target} визитов, заявок {leads}. Копим клики, ничего не менять."}
    if rate >= signal:
        return {"verdict": "СИГНАЛ ЕСТЬ",
                "detail": f"{leads} заявок с {views} визитов ({rate:.0%}). Идея — в очередь на MVP."}
    if rate <= dead:
        return {"verdict": "СПРОСА НЕТ",
                "detail": f"{rate:.0%} заявок при {views} визитах. Кампанию остановить, идею в архив — "
                          "сэкономлены месяцы разработки."}
    return {"verdict": "СЕРАЯ ЗОНА",
            "detail": f"{rate:.0%} заявок. Попробовать второй оффер (другой заголовок) на том же трафике."}


@app.get("/api/verdict/{idea_id}")
def verdict(idea_id: str, request: Request):
    _check_owner(request)
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
        if proj is None:
            raise HTTPException(404, "идея не найдена")
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
    _check_owner(request)
    from collections import defaultdict
    from datetime import timedelta
    with Session(engine) as s:
        if s.exec(select(SmokeProject.id).where(SmokeProject.idea_id == idea_id)).first() is None:
            raise HTTPException(404, "идея не найдена")
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
    out = {"stages": STAGE_NAMES, "tracked": [], "smoke": []}
    with Session(engine) as s:
        for tp in s.exec(select(TrackedProject).order_by(TrackedProject.created_at)).all():
            out["tracked"].append({"id": tp.id, "name": tp.name, "stage": tp.stage,
                                   "stage_name": STAGE_NAMES[tp.stage],
                                   "note": tp.status_note, "link": tp.external_link})
        # Все события одним запросом вместо 2×N (N+1 убивал время на Postgres)
        from collections import defaultdict
        counts: dict[tuple[str, str], int] = defaultdict(int)
        for idea, event in s.exec(select(SmokeEvent.idea, SmokeEvent.event)).all():
            counts[(idea, event)] += 1

        for p in s.exec(select(SmokeProject).order_by(SmokeProject.created_at.desc())).all():
            views = counts[(p.idea_id, "page_view")]
            leads = counts[(p.idea_id, "lead_submitted")]
            stage = 1 if views > 0 else 0
            v = compute_verdict(views, leads, p.click_target,
                                p.lead_rate_signal, p.lead_rate_dead)
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
            out["smoke"].append({"idea_id": p.idea_id, "name": p.product_name,
                                 "stage": stage, "stage_name": STAGE_NAMES[stage],
                                 "views": views, "leads": leads, "rate": round(rate * 100),
                                 "target": p.click_target, "verdict": v["verdict"],
                                 "next_step": next_step,
                                 "progress": min(100, round(views / p.click_target * 100)) if p.click_target else 0,
                                 "landing_url": f"/l/{p.idea_id}",
                                 "project_url": f"/p/{p.idea_id}"})
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


@app.get("/legal", response_class=HTMLResponse)
def legal_page():
    return HTMLResponse(_static("legal.html"))


@app.get("/guide/direct", response_class=HTMLResponse)
def guide_direct():
    """Этап 4 из 8 — пошаговый запуск Директа (режим эксперта, только Поиск)."""
    return HTMLResponse(_static("guide-direct.html"))


@app.get("/social-contract", response_class=HTMLResponse)
def social_contract_page():
    """Отдельная посадочная страница под рекламу на аудиторию социального
    контракта -- специально НЕ часть общего позиционирования сайта (см.
    CLAUDE.md), чтобы не отпугивать массового пользователя упоминанием
    грантов/соцконтракта. Ведёт в тот же бесплатный /api/demand -> /r/{id},
    что и главная страница."""
    return HTMLResponse(_static("social-contract.html"))


@app.get("/oferta", response_class=HTMLResponse)
def oferta_page():
    return HTMLResponse(_static("oferta.html"))


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


_STATIC_CACHE: dict[str, str] = {}


def _static(name: str) -> str:
    """Читаем файл с диска один раз за жизнь процесса."""
    if name not in _STATIC_CACHE:
        _STATIC_CACHE[name] = (BASE_DIR.parent / "static" / name).read_text()
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


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(_static("index.html"))
