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

app = FastAPI(title="Создатель", version="0.1")


# ---------------------------------------------------------------------------
# Этап ⓪: идея → офферы
# ---------------------------------------------------------------------------

class IdeaIn(BaseModel):
    idea: str


@app.post("/api/offers")
async def offers(data: IdeaIn):
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
            .replace("{{DEMO_LEFT_TEXT}}", offer["demo_left_text"])
            .replace("{{DEMO_RIGHT_TEXT_JSON}}", json.dumps(offer["demo_right_text"], ensure_ascii=False))
            .replace("{{PAINS_HTML}}", pains_html)
            .replace("{{IDEA_ID}}", offer["idea_id"]))


class LaunchIn(BaseModel):
    idea_text: str
    offer: dict


@app.post("/api/launch")
def launch(data: LaunchIn):
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


@app.post("/api/smoke-event")
async def smoke_event(request: Request):
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
def verdict(idea_id: str):
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
    return {"ok": True, "idea_id": idea_id, "product_name": proj.product_name,
            "views": views, "leads": len(leads_rows), **v,
            "contacts": [c for c, _ in leads_rows]}


@app.get("/api/projects")
def projects():
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


@app.get("/health")
def health():
    return {"ok": True, "service": "sozdatel", "version": "0.1"}


@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse((BASE_DIR.parent / "static" / "index.html").read_text())
