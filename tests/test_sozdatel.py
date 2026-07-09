"""Тесты Создателя v0.1: движок офферов, генерация лендинга, события, вердикт."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient

from app.offer_engine import OfferEngineError, sharpen_idea, _validate
from app.main import app, compute_verdict, render_landing

client = TestClient(app)

VALID_OFFER = {
    "angle": "ночной завал", "idea_id": "test_v1", "product_name": "Тест",
    "eyebrow": "для селлеров", "h1": "Отзывы отвечаются <em>сами</em>",
    "sub": "Ответ в вашем тоне за секунды.",
    "pains": [{"h2": "а", "p": "б"}, {"h2": "в", "p": "г"}, {"h2": "как это будет работать", "p": "д"}],
    "demo_left_label": "отзыв № 1", "demo_left_text": "«Плохо!»",
    "demo_right_text": "Простите нас — уже исправили и вернули деньги.",
    "direct_queries": ["q1", "q2", "q3", "q4", "q5"],
}


class TestOfferEngine:
    def test_short_idea_rejected(self):
        with pytest.raises(OfferEngineError):
            asyncio.run(sharpen_idea("коротко"))

    def test_happy_path_with_injected_llm(self):
        payload_capture = {}
        async def fake_post(payload):
            payload_capture.update(payload)
            body = {"sharpened_note": "сместил", "warning": "",
                    "offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]}
        out = asyncio.run(sharpen_idea("Сервис отвечает на отзывы за селлеров маркетплейсов", _post=fake_post))
        assert len(out["offers"]) == 3
        assert "Идея фаундера" in payload_capture["messages"][0]["content"]
        assert "РАЗНЫХ оффера" in payload_capture["system"]

    def test_validate_rejects_two_offers(self):
        with pytest.raises(OfferEngineError):
            _validate({"offers": [VALID_OFFER, VALID_OFFER]})

    def test_markdown_fences_stripped(self):
        async def fenced(payload):
            body = {"offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": "```json\n" + json.dumps(body) + "\n```"}]}
        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки", _post=fenced))
        assert out["offers"][0]["idea_id"] == "i0"


class TestLandingAndLaunch:
    def test_render_fills_all_slots(self):
        html = render_landing(VALID_OFFER)
        assert "{{" not in html, "остались незаполненные плейсхолдеры"
        assert "Отзывы отвечаются" in html
        assert 'SMOKE_IDEA = "test_v1"' in html
        assert "/api/smoke-event" in html
        assert "как это будет работать" in html

    def test_launch_hosts_landing(self):
        r = client.post("/api/launch", json={"idea_text": "тестовая идея", "offer": VALID_OFFER})
        assert r.status_code == 200
        data = r.json()
        assert data["landing_url"] == "/l/test_v1"
        page = client.get("/l/test_v1")
        assert page.status_code == 200
        assert "Отзывы отвечаются" in page.text

    def test_launch_missing_field_400(self):
        bad = dict(VALID_OFFER); bad.pop("h1")
        r = client.post("/api/launch", json={"idea_text": "x", "offer": bad})
        assert r.status_code == 400


class TestEventsAndVerdict:
    def test_event_roundtrip_and_verdict(self):
        client.post("/api/launch", json={"idea_text": "т", "offer": dict(VALID_OFFER, idea_id="verd_v1")})
        for _ in range(40):
            client.post("/api/smoke-event", json={"event": "page_view", "idea": "verd_v1",
                                                  "source": "yandex_direct"})
        for i in range(5):
            client.post("/api/smoke-event", json={"event": "lead_submitted", "idea": "verd_v1",
                                                  "contact": f"u{i}@t.ru"})
        r = client.get("/api/verdict/verd_v1").json()
        assert r["views"] == 40 and r["leads"] == 5
        assert r["verdict"] == "СИГНАЛ ЕСТЬ"      # 12.5% >= 8%
        assert len(r["contacts"]) == 5

    def test_unknown_event_rejected(self):
        r = client.post("/api/smoke-event", json={"event": "hack", "idea": "x"})
        assert r.status_code == 400

    def test_verdict_thresholds(self):
        assert compute_verdict(10, 5, 40, .08, .04)["verdict"] == "РАНО СУДИТЬ"
        assert compute_verdict(50, 1, 40, .08, .04)["verdict"] == "СПРОСА НЕТ"
        assert compute_verdict(50, 3, 40, .08, .04)["verdict"] == "СЕРАЯ ЗОНА"
        assert compute_verdict(50, 6, 40, .08, .04)["verdict"] == "СИГНАЛ ЕСТЬ"

    def test_projects_list(self):
        r = client.get("/api/projects").json()
        ids = [p["idea_id"] for p in r["projects"]]
        assert "verd_v1" in ids
