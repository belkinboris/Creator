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

env: ANTHROPIC_API_KEY (обязателен), DATABASE_URL (по умолчанию sqlite).
"""

from __future__ import annotations

import json
import logging
import os
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
engine = create_engine(DATABASE_URL, **_engine_kwargs)

from app.offer_engine import OfferEngineError, sharpen_idea  # noqa: E402

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


STAGE_NAMES = ["Оффер", "Спрос", "Активация", "Первая ценность",
               "Мост к деньгам", "Оплата", "Масштаб", "Удержание"]


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

app = FastAPI(title="Создатель", version="0.9")

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
# Этап ⓪→①: выбранный оффер → лендинг, сразу захощенный
# ---------------------------------------------------------------------------

def render_landing(offer: dict) -> str:
    tpl = (BASE_DIR / "landing_template.html").read_text()
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


@app.get("/api/projects")
def projects(request: Request):
    _check_owner(request)
    with Session(engine) as s:
        rows = s.exec(select(SmokeProject).order_by(SmokeProject.created_at.desc())).all()
        out = []
        for p in rows:
            views = len(s.exec(select(SmokeEvent.id).where(
                SmokeEvent.idea == p.idea_id, SmokeEvent.event == "page_view")).all())
            leads = len(s.exec(select(SmokeEvent.id).where(
                SmokeEvent.idea == p.idea_id, SmokeEvent.event == "lead_submitted")).all())
            out.append({"idea_id": p.idea_id, "product_name": p.product_name,
                        "views": views, "leads": leads, "target": p.click_target,
                        "landing_url": f"/l/{p.idea_id}"})
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
    Smoke-этап определяется данными: есть клики -> ① Спрос, иначе ⓪ Оффер."""
    _check_owner(request)
    out = {"stages": STAGE_NAMES, "tracked": [], "smoke": []}
    with Session(engine) as s:
        for tp in s.exec(select(TrackedProject).order_by(TrackedProject.created_at)).all():
            out["tracked"].append({"id": tp.id, "name": tp.name, "stage": tp.stage,
                                   "stage_name": STAGE_NAMES[tp.stage],
                                   "note": tp.status_note, "link": tp.external_link})
        for p in s.exec(select(SmokeProject).order_by(SmokeProject.created_at.desc())).all():
            views = len(s.exec(select(SmokeEvent.id).where(
                SmokeEvent.idea == p.idea_id, SmokeEvent.event == "page_view")).all())
            leads = len(s.exec(select(SmokeEvent.id).where(
                SmokeEvent.idea == p.idea_id, SmokeEvent.event == "lead_submitted")).all())
            stage = 1 if views > 0 else 0
            v = compute_verdict(views, leads, p.click_target,
                                p.lead_rate_signal, p.lead_rate_dead)
            out["smoke"].append({"idea_id": p.idea_id, "name": p.product_name,
                                 "stage": stage, "stage_name": STAGE_NAMES[stage],
                                 "views": views, "leads": leads,
                                 "target": p.click_target, "verdict": v["verdict"],
                                 "landing_url": f"/l/{p.idea_id}"})
        wl = s.exec(select(SmokeEvent.contact).where(
            SmokeEvent.idea == "sozdatel_waitlist",
            SmokeEvent.event == "lead_submitted")).all()
        out["waitlist"] = {"count": len(wl), "contacts": list(wl)}
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


@app.get("/favicon.ico")
def favicon():
    from fastapi.responses import Response
    # оранжевый квадрат-чертёж 1x1 svg: не 404 в каждом визите
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16"><rect width="16" height="16" fill="%2311263F"/><rect x="3" y="3" width="10" height="10" fill="none" stroke="%23FF8A2A" stroke-width="2"/></svg>'
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/health")
def health():
    return {"ok": True, "service": "sozdatel", "version": "0.1"}


@app.get("/portfolio", response_class=HTMLResponse)
def portfolio_page():
    return HTMLResponse((BASE_DIR.parent / "static" / "portfolio.html").read_text())


@app.get("/p/{idea_id}", response_class=HTMLResponse)
def project_page(idea_id: str):
    with Session(engine) as s:
        proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == idea_id)).first()
    if proj is None:
        raise HTTPException(404, "проект не найден")
    tpl = (BASE_DIR.parent / "static" / "project.html").read_text()
    return HTMLResponse(tpl.replace("{{IDEA_ID}}", idea_id)
                           .replace("{{PRODUCT_NAME}}", proj.product_name))


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((BASE_DIR.parent / "static" / "index.html").read_text())
