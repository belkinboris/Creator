"""Тесты Создателя v0.1: движок офферов, генерация лендинга, события, вердикт."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"

import pytest
from fastapi.testclient import TestClient

from app.offer_engine import OfferEngineError, sharpen_idea, _validate
from app.main import app, compute_verdict, render_landing

client = TestClient(app)
import app.main as main_module
main_module.OWNER_KEY = "test-owner-key"
OWNER = {"X-Owner-Key": "test-owner-key"}

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
        r = client.post("/api/launch", headers=OWNER, json={"idea_text": "тестовая идея", "offer": VALID_OFFER})
        assert r.status_code == 200
        data = r.json()
        assert data["landing_url"] == "/l/test_v1"
        page = client.get("/l/test_v1")
        assert page.status_code == 200
        assert "Отзывы отвечаются" in page.text

    def test_launch_missing_field_400(self):
        bad = dict(VALID_OFFER); bad.pop("h1")
        r = client.post("/api/launch", headers=OWNER, json={"idea_text": "x", "offer": bad})
        assert r.status_code == 400


class TestEventsAndVerdict:
    def test_event_roundtrip_and_verdict(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т", "offer": dict(VALID_OFFER, idea_id="verd_v1")})
        for _ in range(40):
            client.post("/api/smoke-event", json={"event": "page_view", "idea": "verd_v1",
                                                  "source": "yandex_direct"})
        for i in range(5):
            client.post("/api/smoke-event", json={"event": "lead_submitted", "idea": "verd_v1",
                                                  "contact": f"u{i}@t.ru"})
        r = client.get("/api/verdict/verd_v1", headers=OWNER).json()
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
        r = client.get("/api/projects", headers=OWNER).json()
        ids = [p["idea_id"] for p in r["projects"]]
        assert "verd_v1" in ids


class TestTruncationRetry:
    def test_truncated_json_retried_once_then_ok(self):
        import asyncio, json as _json
        calls = {"n": 0}
        async def flaky(payload):
            calls["n"] += 1
            assert payload["max_tokens"] >= 8000, "лимит должен быть поднят"
            if calls["n"] == 1:
                return {"content": [{"type": "text", "text": '{"offers": [{"angle": "обрыв'}]}
            body = {"offers": [dict(VALID_OFFER, idea_id=f"r{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": _json.dumps(body, ensure_ascii=False)}]}
        out = asyncio.run(sharpen_idea("Достаточно длинная идея для проверки повтора", _post=flaky))
        assert calls["n"] == 2
        assert len(out["offers"]) == 3

    def test_double_truncation_gives_human_error(self):
        import asyncio
        async def always_broken(payload):
            return {"content": [{"type": "text", "text": '{"offers": [{"angle": "обр'}]}
        with pytest.raises(OfferEngineError) as e:
            asyncio.run(sharpen_idea("Достаточно длинная идея для проверки", _post=always_broken))
        assert "Попробуйте ещё раз" in str(e.value)



class TestOwnerKey:
    def test_offers_requires_key(self):
        r = client.post("/api/offers", json={"idea": "достаточно длинная идея для проверки"})
        assert r.status_code == 401

    def test_launch_requires_key(self):
        r = client.post("/api/launch", json={"idea_text": "x", "offer": VALID_OFFER})
        assert r.status_code == 401

    def test_verdict_requires_key_but_landing_and_events_public(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т", "offer": dict(VALID_OFFER, idea_id="pub_v1")})
        assert client.get("/api/verdict/pub_v1").status_code == 401
        assert client.get("/l/pub_v1").status_code == 200                      # публично
        r = client.post("/api/smoke-event", json={"event": "page_view", "idea": "pub_v1"})
        assert r.status_code == 200                                            # публично

    def test_key_via_query_param(self):
        r = client.get("/api/verdict/pub_v1?key=test-owner-key")
        assert r.status_code == 200

    def test_delete_project_with_events(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т", "offer": dict(VALID_OFFER, idea_id="del_v1")})
        client.post("/api/smoke-event", json={"event": "page_view", "idea": "del_v1"})
        r = client.delete("/api/projects/del_v1", headers=OWNER)
        assert r.status_code == 200
        assert client.get("/l/del_v1").status_code == 404
        ids = [p["idea_id"] for p in client.get("/api/projects", headers=OWNER).json()["projects"]]
        assert "del_v1" not in ids

    def test_delete_requires_key(self):
        assert client.delete("/api/projects/whatever").status_code == 401


class TestTimeoutRetry:
    def test_timeout_retried_then_ok(self):
        import asyncio, json as _json, httpx as _httpx
        calls = {"n": 0}
        async def slow_then_ok(payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _httpx.ReadTimeout("slow")
            body = {"offers": [dict(VALID_OFFER, idea_id=f"t{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": _json.dumps(body, ensure_ascii=False)}]}
        out = asyncio.run(sharpen_idea("Достаточно длинная идея для проверки таймаута", _post=slow_then_ok))
        assert calls["n"] == 2 and len(out["offers"]) == 3

    def test_double_timeout_human_error(self):
        import asyncio, httpx as _httpx
        async def always_slow(payload):
            raise _httpx.ReadTimeout("slow")
        with pytest.raises(OfferEngineError) as e:
            asyncio.run(sharpen_idea("Достаточно длинная идея для проверки", _post=always_slow))
        assert "долго" in str(e.value)
