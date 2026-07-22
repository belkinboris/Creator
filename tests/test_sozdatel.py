"""Тесты Создателя v0.1: движок офферов, генерация лендинга, события, вердикт."""
import asyncio, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["DATABASE_URL"] = "sqlite://"
# llm_adapter читает YANDEX_* на уровне модуля — задаём тестовые значения
# до импорта, иначе payload-сборка для yandex падает на "не задан FOLDER_ID"
# даже когда сеть не используется (_post инъекция).
os.environ.setdefault("YANDEX_FOLDER_ID", "test-folder")
os.environ.setdefault("YANDEX_API_KEY", "test-yandex-key")

import pytest
from fastapi.testclient import TestClient

from app import llm_adapter
from app.offer_engine import OfferEngineError, sharpen_idea, _validate
from app.main import app, compute_verdict, render_landing

client = TestClient(app)
import app.main as main_module

import pytest as _pytest

@_pytest.fixture(autouse=True)
def _reset_rate_limit():
    """Все тесты идут с одного IP тест-клиента — сбрасываем минутное окно,
    чтобы rate limit тестировался только там, где тестируется он сам."""
    main_module._RL_WINDOW.clear()
    yield
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


def _yandex_response(text: str, *, with_reasoning: bool = False) -> dict:
    """Собирает ответ в форме Yandex Responses API (см. llm_adapter._extract_yandex_text).
    with_reasoning=True добавляет блок скрытого thinking перед message-блоком,
    чтобы проверить, что он отфильтровывается, а не попадает в текст."""
    output = []
    if with_reasoning:
        output.append({"type": "reasoning", "content": [{"type": "text", "text": "секретные мысли модели"}]})
    output.append({"type": "message", "content": [{"type": "output_text", "text": text}]})
    return {"output": output}


class TestOfferEngine:
    def test_short_idea_rejected(self):
        with pytest.raises(OfferEngineError):
            asyncio.run(sharpen_idea("коротко"))

    def test_happy_path_with_injected_llm(self):
        payload_capture = {}
        async def fake_post(provider, payload):
            assert provider == "yandex"
            payload_capture.update(payload)
            body = {"sharpened_note": "сместил", "warning": "",
                    "offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return _yandex_response(json.dumps(body, ensure_ascii=False))
        out = asyncio.run(sharpen_idea("Сервис отвечает на отзывы за селлеров маркетплейсов", _post=fake_post))
        assert len(out["offers"]) == 3
        assert "Идея фаундера" in payload_capture["input"]
        assert "РАЗНЫХ оффера" in payload_capture["instructions"]
        # DeepSeek/не-Claude жёстко просим не класть markdown внутрь JSON-полей
        assert "markdown" in payload_capture["instructions"]

    def test_thinking_budget_added_to_max_tokens(self):
        """DeepSeek thinking всегда включён -- max_output_tokens должен быть
        поднят сверх запрошенного, иначе ответ обрежется на reasoning."""
        payload_capture = {}
        async def fake_post(provider, payload):
            payload_capture.update(payload)
            body = {"offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return _yandex_response(json.dumps(body, ensure_ascii=False))
        asyncio.run(sharpen_idea("Идея достаточно длинная для проверки бюджета", _post=fake_post))
        assert payload_capture["max_output_tokens"] == 8000 + llm_adapter.YANDEX_THINKING_BUDGET

    def test_reasoning_block_filtered_out(self):
        async def fake_post(provider, payload):
            body = {"offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return _yandex_response(json.dumps(body, ensure_ascii=False), with_reasoning=True)
        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки reasoning", _post=fake_post))
        assert len(out["offers"]) == 3  # если бы reasoning не отфильтровался, JSON не распарсился бы

    def test_validate_rejects_two_offers(self):
        with pytest.raises(OfferEngineError):
            _validate({"offers": [VALID_OFFER, VALID_OFFER]})

    def test_markdown_fences_stripped(self):
        async def fenced(provider, payload):
            body = {"offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
            return _yandex_response("```json\n" + json.dumps(body) + "\n```")
        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки", _post=fenced))
        assert out["offers"][0]["idea_id"] == "i0"

    def test_anthropic_fallback_path_still_works(self, monkeypatch):
        """LLM_PROVIDER=anthropic -- путь отката, переключается без деплоя кода."""
        monkeypatch.setattr(llm_adapter, "LLM_PROVIDER", "anthropic")
        async def fake_post(provider, payload):
            assert provider == "anthropic"
            assert payload["messages"][0]["content"].startswith("Идея фаундера")
            body = {"offers": [dict(VALID_OFFER, idea_id=f"a{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]}
        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки отката", _post=fake_post))
        assert out["offers"][0]["idea_id"] == "a0"


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
        for i in range(40):
            if i % 20 == 0:
                main_module._RL_WINDOW.clear()  # 40 событий одним махом с одного IP — только в тестах
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
        proj = next(p for p in r["projects"] if p["idea_id"] == "verd_v1")
        assert proj["views"] == 40 and proj["leads"] == 5   # агрегация одним запросом, не N+1


class TestTruncationRetry:
    def test_truncated_json_retried_once_then_ok(self):
        import asyncio, json as _json
        calls = {"n": 0}
        async def flaky(provider, payload):
            calls["n"] += 1
            assert payload["max_output_tokens"] >= 8000, "лимит должен быть поднят"
            if calls["n"] == 1:
                return _yandex_response('{"offers": [{"angle": "обрыв')
            body = {"offers": [dict(VALID_OFFER, idea_id=f"r{i}") for i in range(3)]}
            return _yandex_response(_json.dumps(body, ensure_ascii=False))
        out = asyncio.run(sharpen_idea("Достаточно длинная идея для проверки повтора", _post=flaky))
        assert calls["n"] == 2
        assert len(out["offers"]) == 3

    def test_double_truncation_gives_human_error(self):
        import asyncio
        async def always_broken(provider, payload):
            return _yandex_response('{"offers": [{"angle": "обр')
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
        async def slow_then_ok(provider, payload):
            calls["n"] += 1
            if calls["n"] == 1:
                raise _httpx.ReadTimeout("slow")
            body = {"offers": [dict(VALID_OFFER, idea_id=f"t{i}") for i in range(3)]}
            return _yandex_response(_json.dumps(body, ensure_ascii=False))
        out = asyncio.run(sharpen_idea("Достаточно длинная идея для проверки таймаута", _post=slow_then_ok))
        assert calls["n"] == 2 and len(out["offers"]) == 3

    def test_double_timeout_human_error(self):
        import asyncio, httpx as _httpx
        async def always_slow(provider, payload):
            raise _httpx.ReadTimeout("slow")
        with pytest.raises(OfferEngineError) as e:
            asyncio.run(sharpen_idea("Достаточно длинная идея для проверки", _post=always_slow))
        assert "долго" in str(e.value)


class TestUniversalDemoCard:
    def test_render_with_custom_demo_fields(self):
        offer = dict(VALID_OFFER, idea_id="rob_v1",
                     demo_left_label="бриф игрока № 214",
                     demo_left_badge="входящий бриф",
                     demo_left_meta="игрок, сегодня",
                     demo_right_tag="концепт готов · 3 варианта",
                     demo_head_right="готово за 40 сек")
        html = render_landing(offer)
        assert "{{" not in html
        assert "бриф игрока № 214" in html
        assert "концепт готов · 3 варианта" in html
        assert "ответ продавца" not in html, "наследие отзывов вычищено"
        assert "★" not in html, "звёзды не появляются без запроса"

    def test_render_old_offer_gets_defaults(self):
        html = render_landing(dict(VALID_OFFER, idea_id="old_v1"))
        assert "{{" not in html
        assert "результат · черновик готов" in html
        assert "готово за секунды" in html

    def test_validator_defaults(self):
        data = _validate({"offers": [dict(VALID_OFFER, idea_id=f"d{i}") for i in range(3)]})
        for o in data["offers"]:
            assert o["demo_right_tag"] and o["demo_head_right"]


class TestCabinet:
    def test_tracked_crud_and_cabinet(self):
        r = client.post("/api/tracked", headers=OWNER, json={
            "name": "АвтоПост", "stage": 3,
            "status_note": "эксперимент первого поста, 0/10 отзывов",
            "external_link": "https://t.me/Trpst_bot"})
        assert r.status_code == 200
        tp_id = r.json()["id"]

        cab = client.get("/api/cabinet", headers=OWNER).json()
        tracked = [t for t in cab["tracked"] if t["id"] == tp_id][0]
        assert tracked["stage_name"] == "Реклама"
        assert cab["stages"][0] == "Идея" and cab["stages"][2] == "Проверочная страница" and len(cab["stages"]) == 8

        r = client.patch(f"/api/tracked/{tp_id}", headers=OWNER, json={
            "name": "АвтоПост", "stage": 4, "status_note": "мост подтверждается"})
        assert r.status_code == 200
        cab = client.get("/api/cabinet", headers=OWNER).json()
        assert [t for t in cab["tracked"] if t["id"] == tp_id][0]["stage"] == 4

        assert client.delete(f"/api/tracked/{tp_id}", headers=OWNER).status_code == 200

    def test_smoke_stage_from_data(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="cab_v1")})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        sm = [s for s in cab["smoke"] if s["idea_id"] == "cab_v1"][0]
        assert sm["stage"] == 0  # кликов нет — этап Оффер
        client.post("/api/smoke-event", json={"event": "page_view", "idea": "cab_v1"})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        sm = [s for s in cab["smoke"] if s["idea_id"] == "cab_v1"][0]
        assert sm["stage"] == 1 and sm["stage_name"] == "Спрос"

    def test_cabinet_requires_key(self):
        assert client.get("/api/cabinet").status_code == 401

    def test_tracked_validation(self):
        assert client.post("/api/tracked", headers=OWNER,
                           json={"name": "x", "stage": 9}).status_code == 400
        assert client.post("/api/tracked", headers=OWNER,
                           json={"name": "  ", "stage": 1}).status_code == 400


class TestDeskOrders:
    """Кабинет: заявки на живой тест были видны только как сырой JSON
    в /api/orders -- теперь есть страница, плюс мини-график динамики
    на карточке проекта вместо только сегодняшних цифр."""

    def test_desk_page_has_orders_section(self):
        text = client.get("/desk").text
        assert "Заявки на" in text and "живой тест" in text
        assert "/api/orders" in text
        assert "loadOrders" in text

    def test_desk_page_has_sparkline(self):
        text = client.get("/desk").text
        assert 'class="spark"' in text
        assert "drawSpark" in text and "/api/series/" in text

    def test_desk_shows_actual_waitlist_contacts_not_just_count(self):
        """Раньше кабинет показывал только число контактов -- реальные
        контакты для связи были не видны нигде в интерфейсе."""
        text = client.get("/desk").text
        assert "wl-count" in text and "wl-list" in text
        assert "d.waitlist.contacts.join" in text   # список, не только count
        assert "скопировать все" in text

    def test_desk_renders_chosen_offer_when_present(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [], "best_phrase": "", "verdict": {"level": "unknown", "text": ""},
                    "competitors": {"found": None, "top": []}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для заказа с оффером"}).json().get("id")
        finally:
            m.check_demand = orig
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "@offer_owner_view",
                        "chosen_offer": {"angle": "a", "h1": "Заголовок для владельца", "sub": "s"}})
        assert r.status_code == 200
        orders = client.get("/api/orders", headers=OWNER).json()["orders"]
        mine = next(o for o in orders if o["contact"] == "@offer_owner_view")
        assert mine["chosen_offer"]["h1"] == "Заголовок для владельца"


class TestProjectPages:
    def test_project_page_renders(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="page_v1", product_name="ОтзоВик")})
        r = client.get("/p/page_v1")
        assert r.status_code == 200
        assert "ОтзоВик" in r.text
        assert "Цель этапа" in r.text
        assert "Ключевые фразы" in r.text          # инструкция Директа на месте
        assert "НЕ менять" in r.text               # правило одной переменной
        assert 'IDEA_ID = "page_v1"' in r.text

    def test_project_page_404(self):
        assert client.get("/p/nope").status_code == 404

    def test_portfolio_page_and_clean_index(self):
        r = client.get("/portfolio")   # редирект доводит до рабочего стола
        assert r.status_code == 200 and "Кабинет" in r.text
        home = client.get("/").text
        assert "Мои проекты" not in home            # кабинет ушёл с главной
        assert "/desk" in home                      # владельцу — на рабочий стол

    def test_verdict_includes_launch_data(self):
        r = client.get("/api/verdict/page_v1", headers=OWNER).json()
        assert r["queries"] == VALID_OFFER["direct_queries"]
        assert r["landing_url"] == "/l/page_v1"
        assert "utm_source=yandex_direct" in r["direct_utm"]
        assert r["target"] == 40


class TestHardening:
    def test_rate_limit_kicks_in(self):
        import app.main as m
        m._RL_WINDOW.clear()
        codes = []
        for _ in range(35):
            r = client.post("/api/smoke-event",
                            json={"event": "page_view", "idea": "rl_v1"})
            codes.append(r.status_code)
        assert codes[:30] == [200]*30
        assert 429 in codes[30:]
        m._RL_WINDOW.clear()  # не мешаем другим тестам

    def test_favicon_not_404(self):
        r = client.get("/favicon.ico")
        assert r.status_code == 200
        assert "svg" in r.headers["content-type"]


class TestWaitlist:
    def test_waitlist_public_and_stored(self):
        r = client.post("/api/waitlist", json={"contact": "founder@test.ru"})
        assert r.status_code == 200
        cab = client.get("/api/cabinet", headers=OWNER).json()
        assert cab["waitlist"]["count"] >= 1
        assert "founder@test.ru" in cab["waitlist"]["contacts"]

    def test_waitlist_validation(self):
        assert client.post("/api/waitlist", json={"contact": "ab"}).status_code == 400

    def test_free_demand_check_in_homepage(self):
        """v2: вместо гейта «закрытого режима» -- открытая бесплатная проверка
        спроса без регистрации (вход воронки)."""
        home = client.get("/").text
        assert "Проверить спрос — бесплатно" in home
        assert "/api/demand" in home
        assert "без регистрации" in home
        assert 'prompt("Ключ владельца Создателя:")' not in home  # голого prompt по-прежнему нет


class TestPresets:
    def test_presets_require_key_and_are_valid(self):
        assert client.get("/api/presets").status_code == 401
        r = client.get("/api/presets", headers=OWNER).json()
        assert len(r["presets"]) == 2
        for o in r["presets"]:
            # каждый пресет валиден по схеме движка
            _validate({"offers": [o, dict(o, idea_id=o["idea_id"]+"b"),
                                  dict(o, idea_id=o["idea_id"]+"c")]})
            assert 5 <= len(o["direct_queries"]) <= 12

    def test_preset_launches_end_to_end(self):
        pr = client.get("/api/presets", headers=OWNER).json()["presets"][0]
        r = client.post("/api/launch", headers=OWNER,
                        json={"idea_text": "preset:"+pr["idea_id"], "offer": pr})
        assert r.status_code == 200
        page = client.get(f"/l/{pr['idea_id']}")
        assert page.status_code == 200
        assert "следующих продаж" in page.text          # h1 пресета
        assert "★☆☆☆☆" in page.text                     # демо-карточка отзыва

    def test_dogovor_preset_landing(self):
        pr = client.get("/api/presets", headers=OWNER).json()["presets"][1]
        client.post("/api/launch", headers=OWNER,
                    json={"idea_text": "preset:"+pr["idea_id"], "offer": pr})
        page = client.get(f"/l/{pr['idea_id']}").text
        assert "за 5 минут" in page
        assert "договор готов · 12 пунктов" in page
        assert "★" not in page.replace("★☆☆☆☆", "") or "★☆☆☆☆" not in page  # без звёзд отзывов


class TestHealthVersion:
    def test_health_reports_real_version(self):
        r = client.get("/health").json()
        assert r["version"] == app.version
        assert r["version"] != "0.1" or app.version == "0.1"


class TestDesk:
    def test_desk_page_and_clean_index(self):
        r = client.get("/desk")
        assert r.status_code == 200
        assert "Кабинет" in r.text and "Мои" in r.text
        assert "следующий шаг" in r.text.lower() or "next" in r.text
        home = client.get("/").text
        assert "Рабочий стол · мои проекты" not in home   # стол ушёл с главной
        assert "deskPresets" not in home                  # пресеты тоже
        assert 'id="path"' in home and "Спрос" in home    # путь 0->7 виден гостю

    def test_cabinet_has_next_step_and_progress(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="desk_fresh_v1")})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        s = [x for x in cab["smoke"] if x["idea_id"] == "desk_fresh_v1"][0]
        assert s["next_step"].startswith("Запустить Директ")   # 0 визитов
        assert s["progress"] == 0 and s["rate"] == 0
        for _ in range(5):
            client.post("/api/smoke-event", json={"event": "page_view", "idea": "desk_fresh_v1"})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        s = [x for x in cab["smoke"] if x["idea_id"] == "desk_fresh_v1"][0]
        assert "Копим клики" in s["next_step"]
        assert s["progress"] in (12, 13)  # 5/40 = 12.5%, банковское округление


class TestNightPolish:
    def test_portfolio_redirects_to_desk(self):
        r = client.get("/portfolio", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/desk"

    def test_series_endpoint(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="ser_v1")})
        for _ in range(3):
            client.post("/api/smoke-event", json={"event": "page_view", "idea": "ser_v1"})
        client.post("/api/smoke-event", json={"event": "lead_submitted", "idea": "ser_v1", "contact": "a@b.ru"})
        r = client.get("/api/series/ser_v1", headers=OWNER).json()
        assert len(r["days"]) == 14
        today = r["days"][-1]
        assert today["views"] == 3 and today["leads"] == 1
        assert r["days"][0]["views"] == 0  # полный ряд с нулями

    def test_series_requires_key_and_404(self):
        assert client.get("/api/series/ser_v1").status_code == 401
        assert client.get("/api/series/nope", headers=OWNER).status_code == 404

    def test_no_prompts_on_desk_and_manrope_everywhere(self):
        desk = client.get("/desk").text
        assert "prompt(" not in desk.replace("password", "")  # форма вместо диалогов
        # v2.4: единая система на всех страницах -- IBM Plex Sans + Mono,
        # без Manrope и декоративных дисплей-шрифтов.
        assert "IBM Plex Sans" in desk and "Manrope" not in desk and "Unbounded" not in desk
        home = client.get("/").text
        assert "Unbounded" not in home and "IBM Plex Sans" in home
        assert "prompt(" not in home

    def test_project_page_has_chart_and_autorefresh(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="chart_v1")})
        page = client.get("/p/chart_v1").text
        assert 'id="chart"' in page
        assert "setInterval" in page and "60000" in page


class TestMorningPass:
    def test_homepage_wires_demand_check(self):
        """v1-петля ?new ушла вместе со старой главной; v2-главная обязана
        уметь одно: отправить идею в /api/demand и показать цифры."""
        home = client.get("/").text
        assert "/api/demand" in home
        assert "freq-num" in home       # маркерные цифры спроса
        assert "background-image" not in home  # клетчатый фон не возвращается

    def test_no_jargon_on_pages(self):
        home = client.get("/").text
        assert "оффер" not in home.lower()
        assert "лендинг" not in home.lower()
        assert "Опишите идею" in home
        desk = client.get("/desk").text
        assert "оффер" not in desk.lower()

    def test_seo_meta(self):
        home = client.get("/").text
        assert 'name="description"' in home
        assert 'property="og:title"' in home
        r = client.get("/robots.txt")
        assert r.status_code == 200 and "Disallow: /api/" in r.text

    def test_legal_page_and_consent_on_landing(self):
        r = client.get("/legal")
        assert r.status_code == 200
        assert "152-ФЗ" in r.text and "отозвать согласие" in r.text
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="legal_v1")})
        page = client.get("/l/legal_v1").text
        assert "/legal" in page and "соглашаетесь" in page


# ---------------------------------------------------------------------------
# Ступень «Спрос» (app/demand.py)
# ---------------------------------------------------------------------------

from app.demand import (  # noqa: E402
    DemandError, check_demand, generate_formulations, _parse_search_xml, _verdict,
    wordstat_count, diagnose,
)


def _demand_post(counts=None, search_xml=None):
    """Единый фейковый _post: провайдеры yandex (LLM) / wordstat / search."""
    counts = counts or {}
    async def fake(provider, payload):
        if provider == "yandex":  # LLM: формулировки
            return _yandex_response(json.dumps(
                ["ответы на отзывы вайлдберриз", "сервис ответов на отзывы", "автоответ на отзывы озон"],
                ensure_ascii=False))
        if provider == "wordstat":
            return {"totalCount": counts.get(payload["phrase"])}
        if provider == "search":
            import base64 as _b64
            xml = search_xml or (
                '<yandexsearch><response><found priority="all">15000</found>'
                '<results><grouping><group><doc><url>https://example.ru/x</url>'
                '<title>Пример конкурента</title></doc></group></grouping></results>'
                '</response></yandexsearch>')
            return {"rawData": _b64.b64encode(xml.encode()).decode()}
        raise AssertionError(f"unexpected provider {provider}")
    return fake


class TestDemand:
    def test_short_idea_rejected(self):
        with pytest.raises(DemandError):
            asyncio.run(generate_formulations("коротко"))

    def test_full_check_happy_path(self):
        post = _demand_post(counts={
            "ответы на отзывы вайлдберриз": 5200,
            "сервис ответов на отзывы": 900,
            "автоответ на отзывы озон": 340,
        })
        out = asyncio.run(check_demand("Сервис отвечает на отзывы за селлеров маркетплейсов", _post=post))
        assert len(out["formulations"]) == 3
        assert out["best_phrase"] == "ответы на отзывы вайлдберриз"
        assert out["verdict"]["level"] == "strong"
        assert out["competitors"]["found"] == 15000
        assert out["competitors"]["top"][0]["domain"] == "example.ru"

    def test_wordstat_unavailable_degrades_not_fails(self):
        """Нет токена/квоты Вордстата -- counts=None, вердикт unknown, но ответ есть."""
        async def post(provider, payload):
            if provider == "yandex":
                return _yandex_response(json.dumps(["a b", "c d", "e f"]))
            if provider == "wordstat":
                raise RuntimeError("боевой сбой сети")
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинное описание идеи для проверки", _post=post))
        assert all(f["count"] is None for f in out["formulations"])
        assert out["verdict"]["level"] == "unknown"
        assert out["competitors"] == {"found": None, "top": []}

    def test_verdict_tiers(self):
        assert _verdict(None)["level"] == "unknown"
        assert _verdict(100)["level"] == "weak"
        assert _verdict(500)["level"] == "niche"
        assert _verdict(5000)["level"] == "strong"

    def test_parse_search_xml_limits_top3(self):
        docs = "".join(
            f"<doc><url>https://www.site{i}.ru/p</url><title>T{i}</title></doc>" for i in range(5))
        xml = f'<y><found priority="all">42</found>{docs}</y>'
        out = _parse_search_xml(xml)
        assert out["found"] == 42
        assert len(out["top"]) == 3
        assert out["top"][0]["domain"] == "site0.ru"  # www. срезан

    def test_api_demand_endpoint_public(self):
        """Роут /api/demand не требует owner-ключа (вход воронки)."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [], "best_phrase": "",
                    "verdict": {"level": "unknown", "text": ""},
                    "competitors": {"found": None, "top": []}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={"idea": "Достаточно длинная идея для эндпоинта"})
            assert r.status_code == 200 and r.json()["ok"] is True
        finally:
            m.check_demand = orig


class TestWordstatDualPath:
    """Два независимых источника частотности: официальный Wordstat API
    (Bearer OAuth) и прежний прокси внутри Cloud Search API."""

    def test_without_oauth_token_only_cloud_path_is_tried(self, monkeypatch):
        """Без YANDEX_WORDSTAT_OAUTH_TOKEN oauth-путь не трогает сеть вовсе --
        существующие тесты/прод без токена ведут себя как раньше."""
        monkeypatch.delenv("YANDEX_WORDSTAT_OAUTH_TOKEN", raising=False)
        async def post(provider, payload):
            assert provider == "wordstat"   # "wordstat_oauth" никогда не вызовется
            return {"totalCount": 4200}
        out = asyncio.run(wordstat_count("тест фраза", _post=post))
        assert out == 4200

    def test_oauth_path_tried_first_when_token_set(self, monkeypatch):
        monkeypatch.setenv("YANDEX_WORDSTAT_OAUTH_TOKEN", "test-oauth-token")
        async def post(provider, payload):
            if provider == "wordstat_oauth":
                return {"totalCount": 9000}
            raise AssertionError("cloud path не должен вызываться, если oauth уже дал ответ")
        out = asyncio.run(wordstat_count("тест фраза", _post=post))
        assert out == 9000

    def test_oauth_path_falls_back_to_cloud_on_empty_data(self, monkeypatch):
        monkeypatch.setenv("YANDEX_WORDSTAT_OAUTH_TOKEN", "test-oauth-token")
        async def post(provider, payload):
            if provider == "wordstat_oauth":
                return {}   # oauth ответил, но без totalCount -- пробуем cloud
            if provider == "wordstat":
                return {"totalCount": 700}
            raise AssertionError(f"unexpected provider {provider}")
        out = asyncio.run(wordstat_count("тест фраза", _post=post))
        assert out == 700


class TestDiagYandex:
    def test_requires_owner_key(self):
        r = client.get("/api/diag/yandex")
        assert r.status_code in (401, 403)

    def test_reports_both_paths(self, monkeypatch):
        monkeypatch.delenv("YANDEX_WORDSTAT_OAUTH_TOKEN", raising=False)
        d = asyncio.run(diagnose("тест", _post=lambda provider, payload: _diag_fake(provider)))
        assert d["env"]["wordstat_oauth_token_set"] is False
        assert d["wordstat_oauth_api"]["ok"] is False
        assert "skipped" in d["wordstat_oauth_api"]
        assert d["wordstat_cloud_api"]["ok"] is True

    def test_endpoint_returns_diagnostic_structure(self, monkeypatch):
        import app.main as m
        async def fake_diagnose(phrase):
            return {"env": {"yandex_api_key_set": True, "yandex_folder_id_set": True,
                            "wordstat_oauth_token_set": False},
                    "wordstat_oauth_api": {"ok": False, "skipped": "..."},
                    "wordstat_cloud_api": {"ok": True, "data": {"totalCount": 10}}}
        orig = m.diagnose
        m.diagnose = fake_diagnose
        try:
            r = client.get("/api/diag/yandex", headers=OWNER)
            assert r.status_code == 200
            d = r.json()
            assert "wordstat_oauth_api" in d and "wordstat_cloud_api" in d
        finally:
            m.diagnose = orig


async def _diag_fake(provider):
    if provider == "wordstat":
        return {"totalCount": 10}
    raise AssertionError(f"unexpected provider {provider}")


class TestIdeaSuggest:
    def test_generate_idea_via_llm(self):
        from app.demand import generate_idea
        async def post(provider, payload):
            assert provider == "yandex"
            return _yandex_response('"Сервис выездной заточки ножей для домашних кухонь по подписке."')
        out = asyncio.run(generate_idea(_post=post))
        assert out.startswith("Сервис выездной")   # кавычки срезаны
        assert len(out) >= 15

    def test_api_idea_endpoint_public(self):
        import app.main as m
        async def fake_gen():
            return "Достаточно длинная сгенерированная идея для теста"
        orig = m.generate_idea
        m.generate_idea = fake_gen
        try:
            r = client.post("/api/idea")
            assert r.status_code == 200
            assert r.json()["ok"] is True and "идея" in r.json()["idea"]
        finally:
            m.generate_idea = orig

    def test_homepage_has_idea_button(self):
        home = client.get("/").text
        assert "Придумать за меня" in home and "/api/idea" in home


class TestScores:
    def test_demand_score_mapping(self):
        from app.demand import _demand_score
        assert _demand_score(None) is None
        assert _demand_score(10) == 1
        assert _demand_score(400) == 4
        assert _demand_score(5000) == 8
        assert _demand_score(60000) == 10

    def test_check_demand_includes_scores(self):
        """Два разных yandex-вызова в одной проверке: формулировки и оценка."""
        score_json = json.dumps({"competition": 7, "timing": 8, "execution": 6,
            "notes": {"competition": "ниша свободна", "timing": "рынок готов", "execution": "можно за месяц"}},
            ensure_ascii=False)
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    return _yandex_response(score_json)
                return _yandex_response(json.dumps(["фразы один", "фразы два", "фразы три"], ensure_ascii=False))
            if provider == "wordstat":
                return {"totalCount": 5000}
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинная идея для проверки оценок", _post=post))
        keys = [s["key"] for s in out["scores"]]
        assert keys == ["demand", "competition", "timing", "execution"]
        assert out["scores"][0]["value"] == 8      # спрос из данных, не из LLM
        assert out["scores"][1]["note"] == "ниша свободна"

    def test_scores_degrade_without_llm_score(self):
        """LLM-оценка упала -- остаётся шкала спроса из данных, ответ живой."""
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    raise RuntimeError("боевой сбой")
                return _yandex_response(json.dumps(["a b", "c d", "e f"]))
            if provider == "wordstat":
                return {"totalCount": 700}
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинная идея для деградации оценки", _post=post))
        assert [s["key"] for s in out["scores"]] == ["demand"]
        assert out["scores"][0]["value"] == 4   # 700/мес -> диапазон 300..1000

    def test_result_page_renders_score_block(self):
        """v2.5: результат живёт на /r/<id> -- главная больше не смешивает
        витрину и инструмент."""
        home = client.get("/").text
        assert 'id="score-card"' not in home           # инлайн-результата нет
        assert "инструкцией безопаснее" not in home    # блок ушёл в плейбук этапа 4


class TestOverallAndStats:
    def test_overall_score_and_weakest(self):
        score_json = json.dumps({"competition": 3, "timing": 8, "execution": 7,
            "notes": {"competition": "рынок забит", "timing": "", "execution": ""}}, ensure_ascii=False)
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    return _yandex_response(score_json)
                return _yandex_response(json.dumps(["a b", "c d", "e f"]))
            if provider == "wordstat":
                return {"totalCount": 5000}   # спрос = 8
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинная идея для общего балла", _post=post))
        assert out["overall"]["value"] == round((8 + 3 + 8 + 7) / 4)
        assert out["overall"]["weakest"] == "Конкуренция"

    def test_demand_check_persisted_and_stats(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "a", "count": 123}], "best_phrase": "a",
                    "verdict": {"level": "weak", "text": ""},
                    "competitors": {"found": None, "top": []}, "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            before = client.get("/api/stats").json()["ideas_checked"]
            r = client.post("/api/demand", json={"idea": "Достаточно длинная идея для счётчика"})
            assert r.status_code == 200
            after = client.get("/api/stats").json()["ideas_checked"]
            assert after == before + 1
        finally:
            m.check_demand = orig

    def test_homepage_declutter_v25(self):
        home = client.get("/").text
        assert "стоит проверка спроса" not in home   # блок цифр снят с витрины
        assert 'id="press"' in home and "Мы в медиа" in home
        assert "href=" not in home.split('id="press"')[1].split("</section>")[0]  # пресса пока без ссылок

    def test_homepage_has_single_social_proof_number(self):
        """Одна честная живая цифра вместо пустоты — не выдуманный счётчик,
        подтягивается из /api/stats и не показывается при малых значениях."""
        home = client.get("/").text
        assert 'id="social-proof"' in home
        assert "/api/stats" in home
        assert "ideas_checked >= 10" in home


class TestResultPageAndOrders:
    def _make_check(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест фраза", "count": 4200}],
                    "best_phrase": "тест фраза",
                    "verdict": {"level": "strong", "text": "Спрос есть"},
                    "competitors": {"found": 100, "top": [{"title": "Т", "domain": "t.ru"}]},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 8, "note": ""}],
                    "overall": {"value": 8, "weakest": "Спрос"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={"idea": "Достаточно длинная идея для страницы результата"})
            return r.json()["id"]
        finally:
            m.check_demand = orig

    def test_demand_returns_id_and_result_page_works(self):
        rid = self._make_check()
        assert rid is not None
        page = client.get(f"/r/{rid}")
        assert page.status_code == 200
        assert "Этап 2 из 8" in page.text and "Этап 3" in page.text  # преемственность, без жаргона
        assert "Ступень" not in page.text
        assert "без ям" not in page.text
        assert "тест фраза" in page.text          # результат вшит в страницу
        assert "Путь от идеи до денег" not in page.text   # витрины здесь нет
        assert client.get("/r/999999").status_code == 404

    def test_result_page_handles_null_demand_score_gracefully(self):
        """Прочерк из 10 баллов -- явный текст вместо голого тире, когда Вордстат недоступен."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест", "count": None}],
                    "best_phrase": "тест", "verdict": {"level": "unknown", "text": ""},
                    "competitors": {"found": None, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": None, "note": ""}],
                    "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея без данных Вордстата для теста прочерка"}).json()["id"]
        finally:
            m.check_demand = orig
        text = client.get(f"/r/{rid}").text
        assert '"value": null' in text            # балл «Спрос» действительно null в вшитых данных
        assert "score-val na" in text              # шаблон умеет показать текст, а не голый дефис

    def test_live_test_order_without_payments_is_request(self):
        """Ключи ЮКассы не заданы -> заказ сохраняется как заявка, не ошибка."""
        rid = self._make_check()
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "@boris_test"})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True and d["paid"] is False and "Заявка принята" in d["message"]
        r2 = client.post("/api/live-test", json={"check_id": rid, "contact": "x"})
        assert r2.status_code == 400   # контакт слишком короткий

    def test_live_test_order_stores_chosen_offer(self):
        """Выбранный на /r/{id} вариант позиционирования уходит владельцу в /api/orders."""
        rid = self._make_check()
        offer = {"angle": "для новичков", "h1": "Быстрый старт", "sub": "Проще, чем кажется"}
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "@chosen_test", "chosen_offer": offer})
        assert r.status_code == 200
        orders = client.get("/api/orders", headers=OWNER).json()["orders"]
        mine = next(o for o in orders if o["contact"] == "@chosen_test")
        assert mine["chosen_offer"] == offer

    def test_orders_visible_to_owner_only(self):
        r = client.get("/api/orders")
        assert r.status_code in (401, 403)
        r = client.get("/api/orders", headers=OWNER)
        assert r.status_code == 200
        orders = r.json()["orders"]
        assert any(o["contact"] == "@boris_test" and o["status"] == "new" for o in orders)
        assert any(o["chosen_offer"] is None for o in orders)  # заказ без выбора оффера — поле пустое, не падает


class TestResultFunnel:
    """Лента с прогрессивным раскрытием: один фокус на экране вместо полотна."""

    def _make_check(self, **overrides):
        import app.main as m
        base = {"formulations": [{"phrase": "тест фраза", "count": 4200}],
                "best_phrase": "тест фраза",
                "verdict": {"level": "strong", "text": "Спрос есть"},
                "competitors": {"found": 100, "top": [{"title": "Т", "domain": "t.ru"}]},
                "scores": [{"key": "demand", "label": "Спрос", "value": 8, "note": ""}],
                "overall": {"value": 8, "weakest": "Спрос"}}
        base.update(overrides)
        async def fake_check(idea):
            return base
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={"idea": "Идея достаточно длинная для теста ленты"})
            return r.json()["id"]
        finally:
            m.check_demand = orig

    def test_steps_present_in_order(self):
        text = client.get(f"/r/{self._make_check()}").text
        positions = [text.index(f'data-step="{n}"') for n in (1, 2, 3, 4, 5)]
        assert positions == sorted(positions)          # шаги идут по порядку в разметке

    def test_only_first_step_active_on_load(self):
        text = client.get(f"/r/{self._make_check()}").text
        assert 'openStep(STEP_ORDER[0])' in text
        assert 'function advance(' in text and 'function reopen(' in text

    def test_score_detail_hidden_behind_toggle(self):
        """Разбор по 4 шкалам не должен идти полотном -- прячется за
        «Почему такая оценка?» и раскрывается по клику."""
        text = client.get(f"/r/{self._make_check()}").text
        assert 'id="scores" hidden' in text
        assert "Почему такая оценка?" in text
        assert "score-detail-toggle" in text

    def test_skip_link_present_for_sharpen_step(self):
        text = client.get(f"/r/{self._make_check()}").text
        assert "Пропустить" in text and "skipSharpen" in text

    def test_steps_without_data_excluded_from_order(self):
        """Пустые scores/competitors не рисуют шаг вовсе -- STEP_ORDER их не включает."""
        text = client.get(f"/r/{self._make_check(scores=[], overall=None, competitors={'found': None, 'top': []})}").text
        assert "hasScores ? 2 : null" in text            # логика исключения шага в разметке присутствует
        assert "hasComp ? 3 : null" in text


class TestPayments:
    def test_live_test_return_url_falls_back_without_check_id(self, monkeypatch):
        """/r/{check_id} без check_id — битая ссылка (404). Без check_id
        оплата должна возвращать на главную, а не на несуществующую /r/."""
        import app.main as m
        captured = {}
        async def fake_create_payment(order_id, amount, description, return_url):
            captured["return_url"] = return_url
            return "pay_x", "https://yookassa.example/pay"
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        monkeypatch.setattr(m.payments, "create_payment", fake_create_payment)
        r = client.post("/api/live-test", json={"contact": "@no_check_id"})
        assert r.status_code == 200
        assert captured["return_url"].endswith("/?paid=1")
        assert "/r/" not in captured["return_url"]

    def test_live_test_return_url_uses_check_id_when_present(self, monkeypatch):
        import app.main as m
        captured = {}
        async def fake_create_payment(order_id, amount, description, return_url):
            captured["return_url"] = return_url
            return "pay_y", "https://yookassa.example/pay"
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        monkeypatch.setattr(m.payments, "create_payment", fake_create_payment)
        r = client.post("/api/live-test", json={"check_id": 42, "contact": "@with_check_id"})
        assert r.status_code == 200
        assert captured["return_url"].endswith("/r/42?paid=1")

    def test_create_payment_via_injection(self):
        from app.payments import create_payment
        captured = {}
        async def post(kind, payload):
            assert kind == "create"
            captured.update(payload)
            return {"id": "pay_123", "confirmation": {"confirmation_url": "https://yookassa.example/pay"}}
        pid, url = asyncio.run(create_payment(7, 1490, "Создатель · живой тест", "https://x/r/1?paid=1", _post=post))
        assert pid == "pay_123" and url.startswith("https://")
        assert captured["amount"]["value"] == "1490.00"
        assert captured["metadata"]["order_id"] == "7"

    def test_webhook_marks_order_paid_only_after_verification(self, monkeypatch):
        import app.main as m
        from app.main import LiveTestOrder, Session, engine
        with Session(engine) as s:
            order = LiveTestOrder(idea="и", contact="@c", status="pending_payment",
                                  payment_id="pay_x", amount=1490)
            s.add(order); s.commit(); s.refresh(order); oid = order.id
        async def fake_fetch(pid, **kw):
            assert pid == "pay_x"
            return {"status": "succeeded", "metadata": {"order_id": str(oid)}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        r = client.post("/api/yookassa/webhook",
                        json={"event": "payment.succeeded", "object": {"id": "pay_x"}})
        assert r.status_code == 200
        with Session(engine) as s:
            assert s.get(LiveTestOrder, oid).status == "paid"

    def test_webhook_ignores_unverified(self, monkeypatch):
        import app.main as m
        async def fake_fetch(pid, **kw):
            return {}   # ЮКасса не подтвердила -- телу вебхука не верим
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        r = client.post("/api/yookassa/webhook",
                        json={"event": "payment.succeeded", "object": {"id": "fake"}})
        assert r.status_code == 200   # молча принимаем, ничего не меняем

    def test_notify_alias_matches_configured_yookassa_url(self, monkeypatch):
        """В кабинете ЮКассы указан /api/yookassa/notify -- должен работать так же, как /webhook."""
        import app.main as m
        from app.main import LiveTestOrder, Session, engine
        with Session(engine) as s:
            order = LiveTestOrder(idea="и", contact="@c2", status="pending_payment",
                                  payment_id="pay_notify", amount=1490)
            s.add(order); s.commit(); s.refresh(order); oid = order.id
        async def fake_fetch(pid, **kw):
            return {"status": "succeeded", "metadata": {"order_id": str(oid)}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        r = client.post("/api/yookassa/notify",
                        json={"event": "payment.succeeded", "object": {"id": "pay_notify"}})
        assert r.status_code == 200
        with Session(engine) as s:
            assert s.get(LiveTestOrder, oid).status == "paid"


class TestSharpenPublic:
    """Заострение идеи -- бесплатно и без ключа владельца, по кнопке на /r/{id}."""

    def test_sharpen_public_no_owner_key_required(self):
        import app.main as m
        async def fake_sharpen(idea):
            return {"sharpened_note": "сместил акценты", "warning": "",
                    "offers": [dict(VALID_OFFER, idea_id=f"pub{i}") for i in range(3)]}
        orig = m.sharpen_idea
        m.sharpen_idea = fake_sharpen
        try:
            r = client.post("/api/sharpen", json={"idea": "Идея достаточно длинная для заострения"})
            assert r.status_code == 200
            d = r.json()
            assert d["ok"] is True and len(d["offers"]) == 3
        finally:
            m.sharpen_idea = orig

    def test_sharpen_llm_failure_returns_400(self):
        import app.main as m
        async def failing(idea):
            raise OfferEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
        orig = m.sharpen_idea
        m.sharpen_idea = failing
        try:
            r = client.post("/api/sharpen", json={"idea": "Идея достаточно длинная для сбоя"})
            assert r.status_code == 400 and r.json()["ok"] is False
        finally:
            m.sharpen_idea = orig

    def test_sharpen_shown_on_result_page(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 1}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""}, "competitors": {"found": None, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для страницы заострения"}).json()["id"]
        finally:
            m.check_demand = orig
        text = client.get(f"/r/{rid}").text
        assert "/api/sharpen" in text
        assert "Заострим идею" in text


class TestGuideDirect:
    def test_guide_page_serves(self):
        r = client.get("/guide/direct")
        assert r.status_code == 200
        t = r.text
        assert "Простой старт" in t and "нельзя выключить первые 30 дней" in t
        assert "режим эксперта" in t.lower()
        assert "только Поиск" in t
        assert "Этап 4 из 8" in t
        assert "Ступень" not in t
        assert "без ям" not in t
        assert "оффер" not in t.lower() and "лендинг" not in t.lower()

    def test_result_page_links_to_guide(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 1}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""}, "competitors": {"found": None, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Достаточно длинная идея для ссылки на гайд"}).json()["id"]
        finally:
            m.check_demand = orig
        assert "/guide/direct" in client.get(f"/r/{rid}").text


class TestLegalPages:
    """Юридические страницы доступны и содержат ожидаемый контент."""

    def test_oferta_page(self):
        r = client.get("/oferta")
        assert r.status_code == 200
        assert "Публичная оферта" in r.text
        assert "ИП Белкин Борис Ильич" in r.text
        assert "1 490" in r.text or "1490" in r.text
        assert "ЮKassa" in r.text or "ЮКасса" in r.text

    def test_privacy_page(self):
        r = client.get("/privacy")
        assert r.status_code == 200
        assert "конфиденциальност" in r.text.lower()
        assert "152" in r.text
        assert "771387918350" in r.text

    def test_agreement_page(self):
        r = client.get("/agreement")
        assert r.status_code == 200
        assert "соглашение" in r.text.lower()

    def test_contacts_page(self):
        r = client.get("/contacts")
        assert r.status_code == 200
        assert "771387918350" in r.text
        assert "324774600432188" in r.text
        assert "Белкин Борис Ильич" in r.text

    def test_legal_hub_links_to_all_pages(self):
        r = client.get("/legal")
        assert r.status_code == 200
        for path in ("/oferta", "/agreement", "/privacy", "/contacts"):
            assert path in r.text, f"/legal не содержит ссылку на {path}"

    def test_legal_pages_no_jargon(self):
        for path in ("/oferta", "/agreement", "/privacy", "/contacts"):
            text = client.get(path).text
            assert "оффер" not in text.lower(), f"слово «оффер» на {path}"
            assert "лендинг" not in text.lower(), f"слово «лендинг» на {path}"


class TestFooterLinks:
    """Футер с ссылками на юридические страницы присутствует на всех публичных страницах."""

    LINKS = ["/oferta", "/agreement", "/privacy", "/contacts"]

    def _assert_footer(self, text, page_name):
        for link in self.LINKS:
            assert f'href="{link}"' in text, f"Нет ссылки {link} в футере {page_name}"

    def test_index_has_footer(self):
        self._assert_footer(client.get("/").text, "главной")

    def test_desk_has_footer(self):
        self._assert_footer(client.get("/desk").text, "кабинета")

    def test_result_has_footer(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест", "count": 100}],
                    "best_phrase": "тест", "verdict": {"level": "unknown", "text": "Нет данных"},
                    "competitors": {"found": 0, "top": []},
                    "scores": [], "overall": {"value": 0, "weakest": ""}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand",
                              json={"idea": "Достаточно длинная идея для проверки футера страницы"}).json()["id"]
        finally:
            m.check_demand = orig
        self._assert_footer(client.get(f"/r/{rid}").text, "результата")

    def test_project_has_footer(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="foot_proj_v1", product_name="ФутерПроект")})
        self._assert_footer(client.get("/p/foot_proj_v1").text, "проекта")

    def test_guide_direct_has_footer(self):
        self._assert_footer(client.get("/guide/direct").text, "гайда по Директу")


class TestProjectPage:
    """Страница /p/ переведена со старого тёмного «чертёжного» стиля на
    светлую дизайн-систему проекта."""

    def test_project_page_uses_light_design_system(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="light_proj_v1", product_name="СветлыйПроект")})
        text = client.get("/p/light_proj_v1").text
        assert text.count("Этап") >= 1
        assert "Manrope" not in text and "Onest" not in text and "JetBrains Mono" not in text
        assert "IBM Plex" in text
        assert "#FBF6EA" in text   # фон бумаги, а не --blueprint
