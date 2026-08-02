"""Тесты Создателя v0.1: движок офферов, генерация лендинга, события, вердикт."""
import asyncio, inspect, json, os, re, sys
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


def pub(rid):
    """Адрес страницы по номеру записи.

    Раньше в адресе стоял порядковый номер, и чужую идею можно было прочитать,
    набрав соседний (E6). Теперь адрес — неугадываемый `public_id`; номер
    остался только для API. Тесты создают проверки по-разному, поэтому адрес
    берём здесь одним местом.
    """
    from app.main import DemandCheck, Session, engine
    with Session(engine) as s:
        rec = s.get(DemandCheck, int(rid)) if str(rid).isdigit() else None
        return rec.public_id if rec else str(rid)



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
        assert payload_capture["input"].startswith("Идея:")
        assert "РАЗНЫХ оффера" in payload_capture["instructions"]
        assert "Проверяю идею для своего дела" in payload_capture["instructions"]   # дефолт business
        # DeepSeek/не-Claude жёстко просим не класть markdown внутрь JSON-полей
        assert "markdown" in payload_capture["instructions"]

    def test_purpose_changes_who_the_model_thinks_it_is_writing_for(self):
        """Заострение раньше всегда писалось "за фаундера" -- system-промпт
        не менялся, даже когда идею принесли с /social-contract или /students.
        Один и тот же бесплатный шаг обязан понимать, кто перед ним, как и
        платный отчёт (F1-F3)."""
        for purpose, phrase in (
            ("social_contract", "Готовлю обоснование для соцзащиты"),
            ("student", "Придумал(а) идею, хочу проверить"),
        ):
            payload_capture = {}
            async def fake_post(provider, payload):
                payload_capture.update(payload)
                body = {"offers": [dict(VALID_OFFER, idea_id=f"i{i}") for i in range(3)]}
                return _yandex_response(json.dumps(body, ensure_ascii=False))
            asyncio.run(sharpen_idea("Идея достаточно длинная для проверки аудитории",
                                     purpose=purpose, _post=fake_post))
            assert phrase in payload_capture["instructions"], purpose
            assert "Фаундер" not in payload_capture["instructions"], purpose

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
            assert payload["messages"][0]["content"].startswith("Идея:")
            body = {"offers": [dict(VALID_OFFER, idea_id=f"a{i}") for i in range(3)]}
            return {"content": [{"type": "text", "text": json.dumps(body, ensure_ascii=False)}]}
        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки отката", _post=fake_post))
        assert out["offers"][0]["idea_id"] == "a0"


def _read_static(name: str) -> str:
    """Исходник страницы с диска: часть логики живёт во встроенном скрипте,
    и проверять её по отрендеренному ответу сервера нельзя -- слоты уже
    подставлены, а сам скрипт от этого не меняется."""
    from pathlib import Path
    return (Path(__file__).resolve().parents[1] / "static" / name).read_text(encoding="utf-8")


def _slots(html: str) -> str:
    """Незаполненные слоты шаблона (`__ИМЯ__`) — то, что человек читает
    буквально, если страницу отдали в обход подстановки серверных значений."""
    import re as _re
    body = _re.sub(r"<script.*?</script>", "", html, flags=_re.S)
    return " ".join(sorted(set(_re.findall(r"__[A-Z_]+__", body))))


DEMAND_DATA_FIXTURE = {
    "formulations": [{"phrase": "ответы на отзывы вайлдберриз", "count": 5200}],
    "verdict": {"level": "strong", "text": "Спрос есть"},
    "competitors": {"found": 15000, "top": [{"title": "Т", "domain": "t.ru"}]},
    # Подобраны так, чтобы медиана A16 (см. generate_core) на модельных 62
    # давала ровно 62 -- пул [40,50,62,70,80], без этого пришлось бы
    # переписывать assert на "62" во всех остальных тестах этого класса,
    # которым сама оценка не важна.
    "scores": [
        {"key": "demand", "label": "Спрос", "value": 8, "note": ""},
        {"key": "competition", "label": "Конкуренция", "value": 7, "note": ""},
        {"key": "timing", "label": "Своевременность", "value": 5, "note": ""},
        {"key": "execution", "label": "Реализуемость", "value": 4, "note": ""},
    ],
    "overall": {"value": 6, "weakest": "Реализуемость"},
}


def _core_body(risk_count=2) -> dict:
    return {
        "viability_score": 62,
        "viability_summary": "Спрос подтверждён, но ниша уже занята двумя игроками.",
        "top_risks": [{"title": f"Риск {i}", "body": f"Объяснение риска {i}."}
                      for i in range(risk_count)],
    }


def _fake_llm(risk_count=2, body="Абзац один.\n\nАбзац два.", captured=None, table=None):
    """Один фейк на оба вида вызова: движок ходит в модель отдельно за ядром
    (балл + риски) и отдельно за каждым разделом. `table` -- для секций с
    wants_table (сейчас только "finance"), см. app/report_engine.py."""
    async def fake_post(provider, payload):
        if captured is not None:
            captured.setdefault("calls", []).append(dict(payload))
            captured.update(payload)
        if "Ты пишешь ОДИН раздел" in payload.get("instructions", ""):
            section_body = {"body": body}
            if table is not None:
                section_body["table"] = table
            return _yandex_response(json.dumps(section_body, ensure_ascii=False))
        return _yandex_response(json.dumps(_core_body(risk_count), ensure_ascii=False))
    return fake_post


class TestReportEngine:
    """Движок платного отчёта. Генерация ПОСЕКЦИОННАЯ: раньше весь отчёт был
    одним вызовом с бюджетом 12 000 токенов на всё сразу, и модель под таким
    прессом недодавала последним разделам. Владелец сравнил результат с
    dimeadozen.ai: «не стоит и 10 рублей»."""

    IDEA = "Сервис отвечает на отзывы за селлеров маркетплейсов"

    def test_short_idea_rejected(self):
        from app.report_engine import generate_report, ReportEngineError
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_report("коротко", DEMAND_DATA_FIXTURE, "quick"))

    def test_uses_dedicated_stronger_yandex_model(self):
        """Не Anthropic -- отдельная, более сильная модель внутри того же
        Yandex-провайдера, что и остальной проект."""
        from app.report_engine import generate_core, SOZDATEL_REPORT_MODEL
        cap = {}
        asyncio.run(generate_core(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                  _post=_fake_llm(captured=cap)))
        assert cap["model"] == f"gpt://test-folder/{SOZDATEL_REPORT_MODEL}"

    def test_each_section_is_its_own_call(self):
        """Суть переделки: раздел = отдельный вызов со своим бюджетом токенов.
        Одним запросом объём конкурента физически не выдаётся."""
        from app.report_engine import generate_report, QUICK_KEYS
        cap = {}
        asyncio.run(generate_report(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                    _post=_fake_llm(captured=cap)))
        section_calls = [c for c in cap["calls"]
                         if "Ты пишешь ОДИН раздел" in c["instructions"]]
        assert len(section_calls) == len(QUICK_KEYS)
        assert len(cap["calls"]) == len(QUICK_KEYS) + 1        # + ядро

    def test_section_budget_is_per_section_not_per_report(self):
        from app.report_engine import MAX_TOKENS_SECTION, section_keys
        # суммарно на порядок больше, чем прежние 12 000 на весь отчёт
        assert MAX_TOKENS_SECTION * len(section_keys("full")) > 12000 * 4

    def test_quick_tier_is_a_short_read(self):
        from app.report_engine import generate_report, QUICK_KEYS
        out = asyncio.run(generate_report(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                          _post=_fake_llm(2)))
        assert [s["key"] for s in out["sections"]] == QUICK_KEYS
        assert len(out["top_risks"]) == 2
        assert out["viability_score"] == 62

    def test_full_tier_is_much_wider_than_before(self):
        """Было 8 разделов на весь платный тариф — мало для 2990 ₽."""
        from app.report_engine import generate_report, ALL_SECTIONS
        out = asyncio.run(generate_report(self.IDEA, DEMAND_DATA_FIXTURE, "full",
                                          _post=_fake_llm(4)))
        assert len(ALL_SECTIONS) >= 20
        assert len(out["sections"]) == len(ALL_SECTIONS)
        assert len(out["top_risks"]) == 4

    def test_viability_score_is_reconciled_with_free_check_scores(self):
        """A16 (PRODUCT_ROADMAP): бесплатная проверка и платный отчёт не
        должны спорить о том, насколько идея хороша, каждый своим числом.
        Модель судит независимо (тут — 95, идея почти идеальна), но 4 шкалы
        бесплатной проверки говорят другое (спрос/конкуренция/своевременность/
        реализуемость по 8-10 из фикстуры "62"-варианта заменены на низкие
        значения) -- итог должен сесть на медиану, а не остаться сырым
        мнением модели и не рухнуть до жёсткого минимума."""
        from app.report_engine import generate_core
        demand_data = dict(DEMAND_DATA_FIXTURE, scores=[
            {"key": "demand", "label": "Спрос", "value": 2, "note": ""},
            {"key": "competition", "label": "Конкуренция", "value": 3, "note": ""},
            {"key": "timing", "label": "Своевременность", "value": 2, "note": ""},
            {"key": "execution", "label": "Реализуемость", "value": 3, "note": ""},
        ])

        async def fake_post(provider, payload):
            body = {"viability_score": 95, "viability_summary": "с", "top_risks": [
                {"title": f"Риск {i}", "body": "б"} for i in range(2)]}
            return _yandex_response(json.dumps(body, ensure_ascii=False))

        out = asyncio.run(generate_core(self.IDEA, demand_data, "quick", _post=fake_post))
        # Пул на 0-100: [20, 30, 20, 30, 95] -> медиана 30. Не 95 (модель не
        # оторвана от измеренного) и не 20 (не жёсткий минимум-потолок).
        assert out["viability_score"] == 30, out["viability_score"]

    def test_sections_are_grouped(self):
        """Плоский список из 20 разделов невозможно читать — нужны группы."""
        from app.report_engine import SECTION_GROUPS, ALL_SECTIONS
        assert len(SECTION_GROUPS) >= 4
        assert sum(len(keys) for _, keys in SECTION_GROUPS) == len(ALL_SECTIONS)

    def test_every_section_has_its_own_question(self):
        """Общий промпт «напиши про рынок» даёт пересказ идеи. Вопрос, на
        который раздел обязан ответить, — это и есть разница."""
        from app.report_engine import SECTION_SPECS
        for s in SECTION_SPECS:
            assert s["ask"].endswith("?"), s["key"]
            assert len(s["must"]) > 80, s["key"]

    def test_section_prompt_forbids_retelling_the_idea(self):
        from app.report_engine import _section_prompt
        prompt = _section_prompt("summary", "full")
        assert "Не пересказывай идею" in prompt
        assert "маркетингового жаргона" in prompt

    def test_finance_section_must_demand_concrete_numbers(self):
        """Лендинги продают «обоснование сметы» и «финансовую модель» -- промпт
        не имеет права разрешать модели отказаться считать."""
        from app.report_engine import _section_prompt, PURPOSES
        for purpose in PURPOSES:
            prompt = _section_prompt("finance", "full", purpose)
            assert "Отказ считать" in prompt and "недопустим" in prompt
            assert "допущения" in prompt

    def test_social_contract_prompt_drops_venture_optics(self):
        """Самозанятая, которой нужна выплата на пошив штор, не должна
        оцениваться критериями венчурного фонда."""
        from app.report_engine import (_section_prompt, PURPOSE_SOCIAL_CONTRACT,
                                       PURPOSE_BUSINESS)
        soc = _section_prompt("finance", "full", PURPOSE_SOCIAL_CONTRACT)
        biz = _section_prompt("finance", "full", PURPOSE_BUSINESS)
        assert "венчурного фонда" in biz
        assert "венчурного фонда" not in soc
        assert "комиссии" in soc and "350 000" in soc
        assert "смета расходов" in soc
        assert "скажи прямо" in soc

    def test_social_contract_launch_plan_is_not_creator_funnel(self):
        from app.report_engine import _section_prompt, PURPOSE_SOCIAL_CONTRACT, STAGE_NAMES
        soc = _section_prompt("launch", "full", PURPOSE_SOCIAL_CONTRACT)
        assert STAGE_NAMES[2] not in soc
        assert "самозанятого или ИП" in soc
        assert "отчитаться перед соцзащитой" in soc

    def test_business_launch_plan_references_existing_stage(self):
        """Регрессия слияния этапов: промпт велел начинать план с
        «Проверочной страницы», которой в STAGE_NAMES больше нет."""
        from app.report_engine import _section_prompt, PURPOSE_BUSINESS, STAGE_NAMES
        biz = _section_prompt("launch", "full", PURPOSE_BUSINESS)
        assert "Проверочная страница" not in biz
        assert STAGE_NAMES[2] in biz

    def test_social_contract_renames_venture_flavoured_sections(self):
        """«Что мешает скопировать» для швеи на дому — вопрос не про патенты."""
        from app.report_engine import section_title, PURPOSE_SOCIAL_CONTRACT, PURPOSE_BUSINESS
        assert section_title("moat", PURPOSE_BUSINESS) != section_title("moat", PURPOSE_SOCIAL_CONTRACT)
        assert "вернутся" in section_title("moat", PURPOSE_SOCIAL_CONTRACT)

    def test_unknown_purpose_falls_back_to_business(self):
        from app.report_engine import _section_prompt, PURPOSE_BUSINESS
        assert _section_prompt("summary", "full", "чепуха") == \
               _section_prompt("summary", "full", PURPOSE_BUSINESS)

    def test_reader_reaches_both_core_and_section_prompts(self):
        """`Audience.reader` описывал, кто на самом деле читает разбор
        («комиссия соцзащиты», «сам человек» у student), но само поле нигде
        не читалось -- модель ничего не знала о конкретном читателе, кроме
        того, что можно было косвенно вывести из persona."""
        from app.report_engine import _core_prompt, _section_prompt, PURPOSES
        from app import audiences
        for purpose in PURPOSES:
            reader = audiences.get(purpose).reader
            assert reader in _core_prompt("full", purpose)
            assert reader in _section_prompt("summary", "full", purpose)

    def test_model_knows_the_reader_already_saw_the_free_check(self):
        """Вопрос владельца 2026-08-02 после покупки за 990 ₽: «нужно
        убедиться, что быстрый разбор не показывает только информацию,
        которую пользователь уже бесплатно увидел».

        Корень был в промпте: модели давали цифры бесплатной проверки и
        велели «использовать буквально», но НЕ говорили, что читатель эти
        же цифры уже видел на /r/. Пересказ был самым естественным ответом
        на такие вводные и ничем не наказывался.
        """
        from app.report_engine import _core_prompt, _section_prompt, PURPOSES
        for purpose in PURPOSES:
            for prompt in (_core_prompt("quick", purpose),
                           _section_prompt("market", "quick", purpose)):
                low = prompt.lower()
                assert "уже видел" in low, purpose
                assert "пересказ" in low, purpose

    def test_the_rule_reaches_the_section_most_at_risk(self):
        """Раздел «Спрос и рынок» опаснее прочих: его вводные — ровно та
        таблица частотностей, которую человек уже прочитал бесплатно."""
        from app.report_engine import _section_prompt
        p = _section_prompt("market", "quick", "business")
        assert "частотностями" in p
        assert "без потери смысла" in p   # критерий, по которому резать

    def test_rule_does_not_ban_the_numbers_themselves(self):
        """Сторож от чрезмерной правки: цифры Вордстата — единственное, чем
        разбор отличается от бесплатных ИИ-генераторов (см. докстринг
        report_engine). Запретить их значило бы убить смысл продукта."""
        from app.report_engine import _section_prompt
        p = _section_prompt("market", "quick", "business")
        assert "Цифры называй" in p
        assert "буквально — не выдумывай другие" in p

    def test_purpose_reaches_the_model_prompt(self):
        from app.report_engine import generate_section, PURPOSE_SOCIAL_CONTRACT
        cap = {}
        asyncio.run(generate_section(
            "finance", "Пошив штор на заказ на дому",
            DEMAND_DATA_FIXTURE, "full", purpose=PURPOSE_SOCIAL_CONTRACT,
            _post=_fake_llm(body="Смета: 50 000 ₽.", captured=cap,
                            table={"caption": "Смета", "rows": [{"item": "Ткань", "sum": 50000}],
                                   "total": 50000})))
        flat = " ".join(cap["instructions"].split())
        assert "комиссии по социальному контракту" in flat

    def test_section_outside_the_tier_rejected(self):
        """Дешёвый тариф не должен отдавать разделы полного."""
        from app.report_engine import generate_section, ReportEngineError
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_section("finance", self.IDEA, DEMAND_DATA_FIXTURE,
                                         "quick", _post=_fake_llm()))

    def test_empty_section_body_rejected(self):
        from app.report_engine import generate_section, ReportEngineError
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_section("summary", self.IDEA, DEMAND_DATA_FIXTURE,
                                         "quick", _post=_fake_llm(body="   ")))

    def test_finance_section_without_numbers_rejected_for_estimate_required_audience(self):
        """`estimate_required` (Audience) описывал, что смета обязана быть
        посчитана «до копейки», но само поле нигде не читалось -- непустой
        ответ модели без единой суммы в рублях проходил как валидный раздел.
        Для получателя соцконтракта смета -- единственное, за чем платят
        (принцип 3): недосчитанная смета это не более скудный раздел, а
        недоставленная услуга."""
        from app.report_engine import generate_section, ReportEngineError, PURPOSE_SOCIAL_CONTRACT
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_section(
                "finance", self.IDEA, DEMAND_DATA_FIXTURE, "full",
                purpose=PURPOSE_SOCIAL_CONTRACT,
                _post=_fake_llm(body="Точных данных недостаточно для расчёта, "
                                     "но дело выглядит перспективным.")))

    def test_finance_section_with_numbers_accepted_for_estimate_required_audience(self):
        from app.report_engine import generate_section, PURPOSE_SOCIAL_CONTRACT
        out = asyncio.run(generate_section(
            "finance", self.IDEA, DEMAND_DATA_FIXTURE, "full",
            purpose=PURPOSE_SOCIAL_CONTRACT,
            _post=_fake_llm(body="Смета: аренда 15 000 ₽, материалы 10 000 ₽.",
                            table={"caption": "Смета расходов", "rows": [
                                {"item": "Аренда", "sum": 15000},
                                {"item": "Материалы", "sum": 10000}], "total": 25000})))
        assert "15 000" in out["body"]
        assert out["table"]["total"] == 25000

    def test_finance_section_without_table_rejected_for_estimate_required_audience(self):
        """F: соц-план.рф (владелец, 2026-08-02) -- смета таблицей построчно,
        не абзацем. Для соцконтракта (Audience.estimate_required) таблица —
        такая же часть услуги, за которую заплатили, как и суммы в body."""
        from app.report_engine import generate_section, ReportEngineError, PURPOSE_SOCIAL_CONTRACT
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_section(
                "finance", self.IDEA, DEMAND_DATA_FIXTURE, "full",
                purpose=PURPOSE_SOCIAL_CONTRACT,
                _post=_fake_llm(body="Смета: аренда 15 000 ₽, материалы 10 000 ₽.")))

    def test_finance_table_optional_for_business(self):
        """Для фаундера таблица — приятное дополнение, не обязательное
        условие: смета не единственное, за чем он платит (в отличие от
        соцконтракта)."""
        from app.report_engine import generate_section, PURPOSE_BUSINESS
        out = asyncio.run(generate_section(
            "finance", self.IDEA, DEMAND_DATA_FIXTURE, "full",
            purpose=PURPOSE_BUSINESS,
            _post=_fake_llm(body="Стартовые затраты около 50 000 ₽.")))
        assert "table" not in out

    def test_finance_section_without_numbers_still_accepted_for_business(self):
        """Проверка узкая для той аудитории, где смета — единственное, за чем
        платят (Audience.estimate_required). Для фаундера смета — один из
        разделов, а не весь смысл покупки, ужесточать его без решения
        владельца не за чем."""
        from app.report_engine import generate_section, PURPOSE_BUSINESS
        out = asyncio.run(generate_section(
            "finance", self.IDEA, DEMAND_DATA_FIXTURE, "full",
            purpose=PURPOSE_BUSINESS,
            _post=_fake_llm(body="Пока рано считать точно, вернёмся к этому позже.")))
        assert out["body"]

    def test_missing_viability_score_rejected(self):
        from app.report_engine import generate_core, ReportEngineError
        async def fake_post(provider, payload):
            body = _core_body(2)
            del body["viability_score"]
            return _yandex_response(json.dumps(body, ensure_ascii=False))
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_core(self.IDEA, DEMAND_DATA_FIXTURE, "quick", _post=fake_post))

    def test_too_few_risks_rejected(self):
        from app.report_engine import generate_core, ReportEngineError
        with pytest.raises(ReportEngineError):
            asyncio.run(generate_core(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                      _post=_fake_llm(1)))

    def test_truncated_json_retried_once_then_ok(self):
        from app.report_engine import generate_core
        calls = {"n": 0}
        async def fake_post(provider, payload):
            calls["n"] += 1
            if calls["n"] == 1:
                return _yandex_response('{"viability_score": 62, "viabil')   # битый JSON
            return _yandex_response(json.dumps(_core_body(2), ensure_ascii=False))
        out = asyncio.run(generate_core(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                        _post=fake_post))
        assert out["viability_score"] == 62 and calls["n"] == 2

    def test_uses_real_demand_numbers_in_context(self):
        """Отличие от дженерик-генераторов -- реальные цифры уходят в промпт."""
        from app.report_engine import generate_section
        cap = {}
        asyncio.run(generate_section("market", self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                     _post=_fake_llm(captured=cap)))
        user_content = cap["input"]
        assert "5200" in user_content or "5 200" in user_content
        assert "t.ru" in user_content

    def test_viability_label_is_deterministic_not_llm(self):
        """F: dimeadozen.ai (владелец, 2026-08-02) -- короткая метка рядом с
        баллом ("Strong execution path"), но у нас -- код, не LLM: тот же
        принцип, что у compute_verdict в demand.py, вердикт не должен звучать
        по-разному от прогона к прогону на один и тот же балл."""
        from app.report_engine import _viability_label
        assert _viability_label(96) == "Сильная позиция для запуска"
        assert _viability_label(80) == "Сильная позиция для запуска"
        assert _viability_label(79) == "Рабочий вариант, есть слабые места"
        assert _viability_label(60) == "Рабочий вариант, есть слабые места"
        assert _viability_label(59) == "Нужна доработка перед запуском"
        assert _viability_label(40) == "Нужна доработка перед запуском"
        assert _viability_label(39) == "Рискованно запускать в текущем виде"
        assert _viability_label(1) == "Рискованно запускать в текущем виде"

    def test_generate_core_includes_viability_label(self):
        from app.report_engine import generate_core
        out = asyncio.run(generate_core(self.IDEA, DEMAND_DATA_FIXTURE, "quick",
                                        _post=_fake_llm()))
        assert out["viability_label"]

    def test_parse_table_filters_malformed_rows_and_computes_total(self):
        from app.report_engine import _parse_table
        table = _parse_table({
            "caption": "Смета",
            "rows": [
                {"item": "Аренда", "sum": 15000},
                {"item": "", "sum": 5000},          # без названия -- отбросить
                {"item": "Без суммы"},               # без суммы -- отбросить
                {"item": "Реклама", "sum": "много"},  # сумма не число -- отбросить
                {"item": "Материалы", "sum": 10000},
            ],
            # total не задан -- должен посчитаться сам из валидных строк
        })
        assert table["rows"] == [{"item": "Аренда", "sum": 15000}, {"item": "Материалы", "sum": 10000}]
        assert table["total"] == 25000

    def test_parse_table_returns_none_for_garbage(self):
        from app.report_engine import _parse_table
        assert _parse_table(None) is None
        assert _parse_table("не таблица") is None
        assert _parse_table({"rows": "тоже не список"}) is None
        assert _parse_table({"rows": []}) is None
        assert _parse_table({"rows": [{"item": "", "sum": 1}]}) is None


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

    def test_delete_project_unlinks_its_live_test_order(self):
        """Найдено живым кастдев-прогоном: удаление проекта не трогало
        LiveTestOrder.idea_id, из которой этот проект был запущен.
        `/api/orders` (владелец, /desk) строит `project_url` прямо из
        `idea_id` заказа, не проверяя, жив ли ещё сам проект -- «Проект уже
        запущен → открыть» продолжал звать на страницу, которая после
        удаления отдаёт голый `{"detail": "проект не найден"}` (класс A19).
        А в /account эта же заявка вообще пропадала бы из вида: не карточка
        проекта (SmokeProject уже нет) и не обычная заявка (`/api/account/me`
        отбирает только заказы с idea_id IS NULL) -- оплативший тест человек
        решил бы, что заказ потерялся."""
        from app.main import LiveTestOrder, Session, engine
        client.post("/api/launch", headers=OWNER,
                    json={"idea_text": "т", "offer": dict(VALID_OFFER, idea_id="unlink_v1")})
        with Session(engine) as s:
            order = LiveTestOrder(idea="живой тест для проверки отвязки", contact="unlink@example.com",
                                  status="paid", amount=1490, idea_id="unlink_v1")
            s.add(order); s.commit(); s.refresh(order)
            order_id = order.id

        before = client.get("/api/orders", headers=OWNER).json()["orders"]
        before_row = next(o for o in before if o["id"] == order_id)
        assert before_row["project_url"] == "/p/unlink_v1"

        r = client.delete("/api/projects/unlink_v1", headers=OWNER)
        assert r.status_code == 200

        after = client.get("/api/orders", headers=OWNER).json()["orders"]
        after_row = next(o for o in after if o["id"] == order_id)
        assert after_row["project_url"] is None, after_row
        assert after_row["idea_id"] is None, after_row


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

    def test_smoke_stage_starts_at_the_live_test_not_at_the_idea(self):
        """Проект существует только потому, что человек прошёл проверку
        спроса И оплатил тест на людях. Значит «Идея» и «Спрос» позади.
        Раньше здесь было `1 if views else 0`, и покупатель сразу после
        оплаты видел «Этап 1 из 7 — Идея»."""
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="cab_v1")})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        sm = [s for s in cab["smoke"] if s["idea_id"] == "cab_v1"][0]
        assert sm["stage"] == 2 and sm["stage_name"] == "Тест на реальных людях"
        client.post("/api/smoke-event", json={"event": "page_view", "idea": "cab_v1"})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        sm = [s for s in cab["smoke"] if s["idea_id"] == "cab_v1"][0]
        assert sm["stage"] == 2, "пока визитов мало — всё ещё идёт тест"

    def test_smoke_stage_moves_to_leads_once_there_is_enough_data(self):
        import app.main as m
        from app.main import SmokeProject, Session, engine, select
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="cab_v2")})
        # События пишем в базу напрямую: публичная ручка ограничена 30
        # событиями в минуту на IP, и сорок подряд она не пропустит.
        from app.main import SmokeEvent
        with Session(engine) as s:
            proj = s.exec(select(SmokeProject).where(
                SmokeProject.idea_id == "cab_v2")).first()
            target = proj.click_target
            for _ in range(target):
                s.add(SmokeEvent(idea="cab_v2", event="page_view"))
            s.commit()
        cab = client.get("/api/cabinet", headers=OWNER).json()
        sm = [s for s in cab["smoke"] if s["idea_id"] == "cab_v2"][0]
        assert sm["stage"] == 3 and sm["stage_name"] == "Заявки"

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
        # Публичная навигация ведёт в кабинет ПОКУПАТЕЛЯ (/account), не владельца
        # (/desk) -- обычный посетитель не должен упираться в ключ владельца.
        assert "/account" in home
        assert "/desk" not in home

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

    def test_account_request_link_is_rate_limited(self):
        """Ручка шлёт письмо -- без лимита кто угодно мог бы забросать
        произвольную почту письмами со ссылкой входа и посадить репутацию
        SMTP-аккаунта (тот же риск, что у остальных публичных ручек)."""
        import app.main as m
        m._RL_WINDOW.clear()
        codes = [client.post("/api/account/request-link", json={"contact": "x@example.com"}).status_code
                 for _ in range(35)]
        assert 429 in codes[30:]
        m._RL_WINDOW.clear()

    def test_demand_save_is_rate_limited(self):
        import app.main as m
        m._RL_WINDOW.clear()
        codes = [client.post("/api/demand/999999/save", json={"contact": "x@example.com"}).status_code
                 for _ in range(35)]
        assert 429 in codes[30:]
        m._RL_WINDOW.clear()


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


class TestStartupMigrationOnAStaleSqliteFile:
    """Реальный баг, найденный в живом прогоне (не в тестах — они всегда
    берут свежую `sqlite://` в памяти, где ALTER вообще не нужен): миграция
    при старте использовала `ALTER TABLE ... ADD COLUMN IF NOT EXISTS` —
    это диалект Postgres, SQLite падает на нём "syntax error" (проверено
    напрямую sqlite3 3.45, дело не в старой версии движка). На БД, которая
    пережила хотя бы одно добавление колонки в прошлом запуске (а dev держит
    файл `sozdatel.db` между запусками ровно за этим), первый же ALTER падал,
    единственный `except: pass` глушил ВСЮ оставшуюся миграцию разом, и
    следующее же обращение к новой колонке ронялось 500-й."""

    def test_missing_column_gets_added_without_crashing(self, tmp_path):
        import sqlite3
        import subprocess
        import sys
        import textwrap

        db_path = tmp_path / "stale.db"
        con = sqlite3.connect(str(db_path))
        con.execute("CREATE TABLE demandcheck (id INTEGER PRIMARY KEY, idea VARCHAR)")
        con.commit()
        con.close()

        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = textwrap.dedent(f"""
            import os
            os.environ["DATABASE_URL"] = "sqlite:///{db_path}"
            os.environ.setdefault("YANDEX_FOLDER_ID", "test-folder")
            os.environ.setdefault("YANDEX_API_KEY", "test-yandex-key")
            import app.main  # выполняет миграцию на импорте
        """)
        result = subprocess.run([sys.executable, "-c", script], cwd=repo_root,
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr

        con = sqlite3.connect(str(db_path))
        cols = {row[1] for row in con.execute("PRAGMA table_info(demandcheck)")}
        con.close()
        assert "public_id" in cols
        # Колонки, добавленные ДО той, что уронила бы блок на старом коде --
        # регрессия того же класса: единственный сбой глушил и их тоже.
        assert "purpose" in cols and "sample_json" in cols


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
        # 0 визитов: зовём запустить рекламу, инструкция идёт ссылкой (A17)
        assert "Директ" in s["next_step"] and s["next_link"]
        assert s["progress"] == 0 and s["rate"] == 0
        for _ in range(5):
            client.post("/api/smoke-event", json={"event": "page_view", "idea": "desk_fresh_v1"})
        cab = client.get("/api/cabinet", headers=OWNER).json()
        s = [x for x in cab["smoke"] if x["idea_id"] == "desk_fresh_v1"][0]
        # «Копим клики» — владельческий жаргон, теперь строка покупательская
        assert "35 визитов до вывода" in s["next_step"], s["next_step"]
        assert s["progress"] in (12, 13)  # 5/40 = 12.5%, банковское округление


class TestNightPolish:
    def test_portfolio_redirects_to_desk(self):
        r = client.get("/portfolio", follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == "/desk"

    def test_dead_portfolio_file_is_gone(self):
        """Файл нёс третью шкалу этапов (Оффер · Активация · Мост · Оплата ·
        Масштаб) и путал при чтении кода. Роут-редирект живёт без него."""
        import pathlib
        root = pathlib.Path(__file__).resolve().parents[1]
        assert not (root / "static" / "portfolio.html").exists()

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
    DemandError, check_demand, competitors, generate_formulations, _parse_search_xml,
    _verdict, wordstat_count, diagnose,
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


class TestBroaderFormulations:
    """Три формулировки промахивались мимо ходового запроса.

    Живой прогон: LLM выдала максимум 941/мес, а реальный массовый запрос
    той же темы давал 3984/мес — просто в тройку он не попал. Лечится не
    угадыванием конкретного слова (у каждой ниши оно своё), а размером и
    разнообразием выборки.
    """

    def test_model_is_asked_for_six_formulations(self):
        from app.demand import FORMULATIONS_COUNT
        assert FORMULATIONS_COUNT == 6
        from app.demand import _FORMULATIONS_SYSTEM as sysmsg
        assert "6" in sysmsg

    def test_prompt_demands_different_types_not_just_more_of_the_same(self):
        """Шесть пересказов одной фразы бесполезны так же, как три: промпт
        обязан требовать РАЗНЫЕ типы запросов — услуга, широкая категория и
        описание задачи своими словами."""
        from app.demand import _FORMULATIONS_SYSTEM as sysmsg
        low = sysmsg.lower()
        assert "родовая" in low or "широк" in low
        assert "задач" in low or "проблем" in low

    def test_all_formulations_are_measured(self):
        """Шесть фраз — шесть обращений в Вордстат, ни одна не теряется."""
        asked = []
        async def post(provider, payload):
            if provider == "yandex":
                return _yandex_response(json.dumps(
                    [f"фраза {i}" for i in range(6)], ensure_ascii=False))
            if provider == "wordstat":
                asked.append(payload["phrase"])
                return {"totalCount": 100}
            return {"rawData": None}
        asyncio.run(check_demand("Достаточно длинное описание идеи для проверки", _post=post))
        assert sorted(asked) == sorted(f"фраза {i}" for i in range(6))

    def test_rows_are_sorted_by_frequency(self):
        """Порядок, в котором фразы выдала LLM, читателю не значит ничего, а
        первая строка читается как главная — наверху должна быть самая
        ходовая формулировка."""
        async def post(provider, payload):
            if provider == "yandex":
                return _yandex_response(json.dumps(["редкая", "ходовая", "средняя"],
                                                   ensure_ascii=False))
            if provider == "wordstat":
                return {"totalCount": {"редкая": 10, "ходовая": 9000, "средняя": 500}[payload["phrase"]]}
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинное описание идеи для проверки", _post=post))
        assert [f["phrase"] for f in out["formulations"]] == ["ходовая", "средняя", "редкая"]
        assert out["best_phrase"] == "ходовая"

    def test_unmeasured_phrases_sink_to_the_bottom(self):
        """Неизмеренную фразу не с чем сравнивать — она уходит вниз, но со
        страницы не пропадает."""
        async def post(provider, payload):
            if provider == "yandex":
                return _yandex_response(json.dumps(["есть", "нет данных"], ensure_ascii=False))
            if provider == "wordstat":
                if payload["phrase"] == "нет данных":
                    raise RuntimeError("сбой сети именно на этой фразе")
                return {"totalCount": 700}
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинное описание идеи для проверки", _post=post))
        assert [f["phrase"] for f in out["formulations"]] == ["есть", "нет данных"]
        assert out["formulations"][-1]["count"] is None


class TestMeasuredZeroIsNotAFailure:
    """«Вордстат ответил, запросов нет» и «Вордстат не ответил» — разные
    вещи. Раньше и то и другое давало count=None, и нормально измеренный
    ноль показывался как «нет данных у Яндекса», то есть как поломка."""

    def test_answered_without_total_count_is_zero_not_none(self):
        async def post(provider, payload):
            return {"topRequests": []}         # ответ есть, частотности нет
        assert asyncio.run(wordstat_count("фраза", _post=post)) == 0

    def test_transport_failure_is_still_none(self):
        async def post(provider, payload):
            raise RuntimeError("сети нет")
        assert asyncio.run(wordstat_count("фраза", _post=post)) is None

    def test_measured_zero_reads_as_weak_demand_not_unknown(self):
        """Ноль — это ответ «не ищут», а не «проверка не состоялась»:
        вердикт обязан быть weak, иначе страница объявит сбоем то, что
        сбоем не является."""
        async def post(provider, payload):
            if provider == "yandex":
                return _yandex_response(json.dumps(["а б", "в г"], ensure_ascii=False))
            if provider == "wordstat":
                return {"totalCount": 0}
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинное описание идеи для проверки", _post=post))
        assert out["verdict"]["level"] == "weak"
        assert all(f["count"] == 0 for f in out["formulations"])

    def test_result_page_separates_the_two_wordings(self):
        text = _static_result()
        assert "не удалось измерить" in text   # count === null
        assert "не ищут" in text               # count === 0
        assert "f.count === 0" in text         # ветка есть, а не просто слова в тексте
        assert "нет данных у Яндекса" not in text   # прежняя склейка двух случаев


def _static_result():
    from pathlib import Path
    return Path("static/result.html").read_text(encoding="utf-8")


class TestCompetitorsAreBusinesses:
    """Живой прогон показал в блоке «Конкуренты» Википедию, Ленту и женский
    журнал. Для информационного запроса это правдивая выдача, но человеку
    обещали тех, кто уже ПРОДАЁТ то же самое."""

    def test_encyclopedias_and_media_are_filtered_out(self):
        from app.demand import _is_not_competitor
        for domain in ("ru.wikipedia.org", "lenta.ru", "www.woman.ru",
                       "dzen.ru", "otvet.mail.ru", "vk.com", "youtube.com",
                       "forum-mam.ru", "habr.com", "nalog.ru"):
            assert _is_not_competitor(domain), domain

    def test_real_businesses_survive_the_filter(self):
        from app.demand import _is_not_competitor
        for domain in ("ozon.ru", "avito.ru", "my-clinic.ru", "studio22.ru",
                       "market.merch.ru", "informatika-shop.ru"):
            assert not _is_not_competitor(domain), domain

    def test_substring_match_does_not_eat_commercial_domains(self):
        """«t.me» как подстрока живёт внутри «marke(t.me)rch.ru» — фильтр по
        вхождению вырезал бы настоящего конкурента, поэтому сверяем домен
        целиком."""
        from app.demand import _is_not_competitor
        assert not _is_not_competitor("market.merch.ru")
        assert _is_not_competitor("t.me")

    def _search_post(self, domains):
        docs = "".join(f"<doc><url>https://{d}/x</url><title>Заголовок {d}</title></doc>"
                       for d in domains)
        xml = f'<y><found priority="all">15000</found>{docs}</y>'
        async def post(provider, payload):
            import base64 as _b64
            return {"rawData": _b64.b64encode(xml.encode()).decode()}
        return post

    def test_only_businesses_reach_the_page(self):
        out = asyncio.run(competitors("фраза", _post=self._search_post(
            ["ru.wikipedia.org", "lenta.ru", "shop-one.ru", "dzen.ru", "shop-two.ru"])))
        assert [c["domain"] for c in out["top"]] == ["shop-one.ru", "shop-two.ru"]
        assert out["info_only"] is False

    def test_all_informational_is_reported_as_a_finding(self):
        """Отфильтровали всё — это не пустой блок, а находка: по запросу
        читают, а не покупают. Страница обязана суметь это сказать."""
        out = asyncio.run(competitors("фраза", _post=self._search_post(
            ["ru.wikipedia.org", "lenta.ru", "www.woman.ru"])))
        assert out["top"] == []
        assert out["info_only"] is True

    def test_page_renames_the_block_away_from_competitors(self):
        text = _static_result()
        assert "Кто уже продаёт это" in text
        assert "info_only" in text


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

    def test_check_demand_surfaces_matched_phrase_honestly(self, monkeypatch):
        """Если Вордстат подсказал формулировку популярнее угаданной LLM --
        показываем ЕЁ отдельным полем, а не молча приписываем чужой счёт
        исходной фразе (иначе ручная проверка исходной фразы в Вордстате
        покажет другое число и будет выглядеть как обман/баг)."""
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    return _yandex_response(json.dumps({"competition": 5, "timing": 5, "execution": 5,
                        "notes": {"competition": "", "timing": "", "execution": ""}}, ensure_ascii=False))
                return _yandex_response(json.dumps(
                    ["создание рекламного видео", "генератор рекламных видео", "рекламное видео онлайн"],
                    ensure_ascii=False))
            if provider == "wordstat":
                if payload["phrase"] == "создание рекламного видео":
                    return {"totalCount": 157, "topRequests": [
                        {"phrase": "нейросеть для рекламы", "count": 902}]}
                return {"totalCount": 2}
            return {"rawData": None}
        out = asyncio.run(check_demand("Сервис генерирует рекламные видео через ИИ", _post=post))
        row = out["formulations"][0]
        assert row["phrase"] == "создание рекламного видео"   # исходная формулировка не подменена
        assert row["count"] == 902                             # но частотность -- реальная
        assert row["matched_phrase"] == "нейросеть для рекламы"
        assert out["best_phrase"] == "нейросеть для рекламы"   # конкурентов ищем по реальному запросу
        assert out["verdict"]["level"] == "niche"              # 902 -- между порогами niche/strong

    def test_check_demand_no_matched_phrase_when_nothing_beats_it(self):
        """Без topRequests или когда угаданная фраза и так лучшая -- поле
        matched_phrase отсутствует, чтобы не путать фронтенд лишним полем."""
        post = _demand_post(counts={
            "ответы на отзывы вайлдберриз": 5200,
            "сервис ответов на отзывы": 900,
            "автоответ на отзывы озон": 340,
        })
        out = asyncio.run(check_demand("Сервис отвечает на отзывы за селлеров маркетплейсов", _post=post))
        assert all("matched_phrase" not in f for f in out["formulations"])

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
        assert out["competitors"] == {"found": None, "top": [], "info_only": False}

    def test_verdict_has_no_internal_jargon(self):
        """A4 из PRODUCT_ROADMAP: вердикт видят обе аудитории, включая
        самозанятую, которая рекламу вообще запускать не собирается."""
        from app.demand import _verdict
        texts = [_verdict(v)["text"] for v in (None, 10, 500, 5000)]
        for t in texts:
            low = t.lower()
            for bad in ("трафик", "в холодную", "живой тест", "конверси",
                        "гипотез", "частотность", "оффер", "лендинг"):
                assert bad not in low, f"жаргон в вердикте: {bad!r} -> {t}"

    def test_verdict_states_finding_not_next_step(self):
        """Следующий шаг у аудиторий разный и подбирается CTA страницы по
        purpose -- вердикт не должен тянуть человека в чужую воронку."""
        from app.demand import _verdict
        niche = _verdict(500)["text"]
        assert "проверить на живом" not in niche.lower()
        assert "реклам" not in niche.lower()

    def test_verdict_keeps_the_number_it_reports(self):
        from app.demand import _verdict
        assert "5 000" in _verdict(5000)["text"]   # разряды через пробел

    def test_verdict_tiers(self):
        assert _verdict(None)["level"] == "unknown"
        assert _verdict(100)["level"] == "weak"
        assert _verdict(500)["level"] == "niche"
        assert _verdict(5000)["level"] == "strong"

    def test_parse_search_xml_scans_with_reserve(self):
        """Разбираем выдачу с запасом, а не ровно три: часть документов
        отсеется как информационная (_is_not_competitor), и три коммерческих
        сайта надо ещё из чего-то набрать. Обрезка до трёх — в competitors()."""
        docs = "".join(
            f"<doc><url>https://www.site{i}.ru/p</url><title>T{i}</title></doc>" for i in range(5))
        xml = f'<y><found priority="all">42</found>{docs}</y>'
        out = _parse_search_xml(xml)
        assert out["found"] == 42
        assert len(out["top"]) == 5
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


class TestWordstatFrequency:
    """Единственный источник частотности: Wordstat-прокси внутри Yandex
    Cloud Search API v2, Api-Key сервисного аккаунта (см. докстринг
    app/demand.py -- старый OAuth-путь через поддержку Директа сознательно
    не реализован, рабочая self-service альтернатива уже есть)."""

    def test_cloud_path_sends_num_phrases_in_valid_range(self):
        """Регрессия: без num_phrases Cloud Search API отвечал 400 "Value must
        be in the range of 1 to 2000" на КАЖДЫЙ запрос -- частотность никогда
        не считалась, независимо от ключей/токенов (см. живой /api/diag/yandex)."""
        captured = {}
        async def post(provider, payload):
            captured.update(payload)
            return {"totalCount": 123}
        asyncio.run(wordstat_count("тест фраза", _post=post))
        assert "num_phrases" in captured
        assert 1 <= captured["num_phrases"] <= 2000

    def test_cloud_path_sends_both_num_phrases_spellings(self):
        """Публичные примеры использования этого эндпоинта используют camelCase
        (numPhrases), офдока недоступна для проверки -- шлём оба варианта имени
        поля, чтобы не зависеть от неподтверждённой схемы."""
        captured = {}
        async def post(provider, payload):
            captured.update(payload)
            return {"totalCount": 123}
        asyncio.run(wordstat_count("тест фраза", _post=post))
        assert captured.get("num_phrases") == captured.get("numPhrases")
        assert captured["numPhrases"] > 1   # не 1 -- иначе похожие формулировки не увидим

    def test_cloud_path_prefers_higher_related_phrase_count(self):
        """Кастдев-находка: LLM угадала «создание рекламного видео» (157/мес),
        а Вордстат сам предлагает рядом реальный ходовой запрос «нейросеть для
        рекламы» (902/мес) в topRequests -- этот сигнал раньше отбрасывался,
        читался только totalCount дословно запрошенной фразы."""
        async def post(provider, payload):
            return {"totalCount": 157, "topRequests": [
                {"phrase": "нейросеть для рекламы", "count": 902},
                {"phrase": "создать рекламное видео онлайн", "count": 40},
            ]}
        out = asyncio.run(wordstat_count("создание рекламного видео", _post=post))
        assert out == 902

    def test_related_phrases_never_lower_the_count(self):
        """Похожие формулировки с меньшей частотностью не должны понижать
        totalCount дословно запрошенной фразы -- берём максимум, не среднее."""
        async def post(provider, payload):
            return {"totalCount": 5000, "topRequests": [{"phrase": "похожий запрос", "count": 10}]}
        out = asyncio.run(wordstat_count("популярная фраза", _post=post))
        assert out == 5000

    def test_malformed_related_phrases_do_not_crash(self):
        """topRequests может прийти в неожиданной форме -- деградация, не 500."""
        async def post(provider, payload):
            return {"totalCount": 200, "topRequests": ["не словарь", {"phrase": "x"}, {"count": "не число"}]}
        out = asyncio.run(wordstat_count("фраза", _post=post))
        assert out == 200


class TestDiagYandex:
    def test_requires_owner_key(self):
        r = client.get("/api/diag/yandex")
        assert r.status_code in (401, 403)

    def test_reports_wordstat_path(self, monkeypatch):
        monkeypatch.setenv("YANDEX_API_KEY", "test-key")
        d = asyncio.run(diagnose("тест", _post=lambda provider, payload: _diag_fake(provider)))
        assert d["env"]["yandex_api_key_set"] is True
        assert d["wordstat_api"]["ok"] is True
        assert d["wordstat_api"]["data"]["totalCount"] == 10

    def test_endpoint_returns_diagnostic_structure(self, monkeypatch):
        import app.main as m
        async def fake_diagnose(phrase):
            return {"env": {"yandex_api_key_set": True, "yandex_folder_id_set": True},
                    "wordstat_api": {"ok": True, "data": {"totalCount": 10}}}
        orig = m.diagnose
        m.diagnose = fake_diagnose
        try:
            r = client.get("/api/diag/yandex", headers=OWNER)
            assert r.status_code == 200
            d = r.json()
            assert "wordstat_api" in d
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

    def test_overall_capped_by_weak_demand(self):
        """Спрос -- ворота: почти нулевая частотность не должна тонуть в
        среднем с тремя хорошими LLM-шкалами и выдавать обманчиво высокий балл."""
        score_json = json.dumps({"competition": 9, "timing": 9, "execution": 9,
            "notes": {"competition": "", "timing": "", "execution": ""}}, ensure_ascii=False)
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    return _yandex_response(score_json)
                return _yandex_response(json.dumps(["a b", "c d", "e f"]))
            if provider == "wordstat":
                return {"totalCount": 1}   # спрос = 1 -- почти никто не ищет
            return {"rawData": None}
        out = asyncio.run(check_demand("Достаточно длинная идея с почти нулевым спросом", _post=post))
        assert out["scores"][0]["value"] == 1
        naive_avg = round((1 + 9 + 9 + 9) / 4)   # было бы 7 без ворот -- вводит в заблуждение
        assert naive_avg == 7
        assert out["overall"]["value"] == 1
        assert out["overall"]["weakest"] == "Спрос"

    def test_overall_hidden_when_demand_unknown_not_just_low(self):
        """Найдено живым прогоном (кастдев-проход, dev без ключей Вордстата,
        LLM-ответы подменены инъекцией): Вордстат недоступен (demand_value
        is None -- НЕ то же самое, что "спрос есть и он маленький", тогда
        было бы demand_value=1), но три LLM-шкалы отработали. Раньше это
        молча усредняло три оставшиеся шкалы и подписывало результат «среднее
        по ЧЕТЫРЁМ шкалам», хотя спрос в среднем не участвовал вообще -- рядом
        с вердиктом "unknown" ("данных нет") эта уверенная семёрка внушала
        обратное. Спрос -- ворота (см. test_overall_capped_by_weak_demand);
        если неизвестно, что за воротами, показывать число нельзя."""
        score_json = json.dumps({"competition": 7, "timing": 8, "execution": 6,
            "notes": {"competition": "", "timing": "", "execution": ""}}, ensure_ascii=False)
        async def post(provider, payload):
            if provider == "yandex":
                if "шкалам" in payload["instructions"]:
                    return _yandex_response(score_json)
                return _yandex_response(json.dumps(["a b", "c d", "e f"]))
            if provider == "wordstat":
                raise RuntimeError("Вордстат недоступен")
            return {"rawData": None}
        out = asyncio.run(check_demand("Идея с недоступным Вордстатом, но рабочими LLM-шкалами", _post=post))
        assert out["scores"][0]["value"] is None          # спрос действительно неизвестен
        assert out["overall"] is None

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
        # "Мы в медиа" без реальных ссылок ("добавим позже") читается как
        # пустое обещание, а не социальное доказательство -- убрали совсем,
        # вернём, когда появятся настоящие публикации.
        assert "Мы в медиа" not in home

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
        page = client.get(f"/r/{pub(rid)}")
        assert page.status_code == 200
        # Преемственность осталась меткой этапа и прогресс-баром. Обещание
        # «Этап 3 — …» из шапки убрано: человек встречал его раньше, чем
        # собственно результат, за которым пришёл (кастдев 2026-08-02).
        assert "Этап 2 из 7" in page.text
        assert "Результат проверки спроса" in page.text
        assert "Ступень" not in page.text
        assert "без ям" not in page.text
        assert "тест фраза" in page.text          # результат вшит в страницу
        assert "Путь от идеи до денег" not in page.text   # витрины здесь нет
        assert client.get("/r/999999").status_code == 404

    def test_result_page_shows_matched_phrase_transparency_note(self):
        """Когда Вордстат подсказал более ходовую формулировку (см.
        check_demand/wordstat_best в app/demand.py), страница должна честно
        показать, какая фраза дала эту частотность -- иначе цифра рядом с
        исходной фразой выглядит взятой с потолка при ручной перепроверке."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "создание рекламного видео", "count": 902,
                                       "matched_phrase": "нейросеть для рекламы"}],
                    "best_phrase": "нейросеть для рекламы",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": None, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 4, "note": ""}],
                    "overall": {"value": 4, "weakest": "Спрос"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={"idea": "Идея достаточно длинная для теста подсказанной фразы"})
            rid = r.json()["id"]
        finally:
            m.check_demand = orig
        page = client.get(f"/r/{pub(rid)}").text
        assert "нейросеть для рекламы" in page   # данные дошли до страницы
        assert "f.matched_phrase" in page         # фронтенд умеет её показать

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
        text = client.get(f"/r/{pub(rid)}").text
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
        text = client.get(f"/r/{pub(self._make_check())}").text
        positions = [text.index(f'data-step="{n}"') for n in (1, 2, 3, 4, 5)]
        assert positions == sorted(positions)          # шаги идут по порядку в разметке

    def test_only_first_step_active_on_load(self):
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'openStep(STEP_ORDER[0])' in text
        assert 'function advance(' in text and 'function reopen(' in text

    def test_score_detail_hidden_behind_toggle(self):
        """Разбор по 4 шкалам не должен идти полотном -- прячется за
        «Почему такая оценка?» и раскрывается по клику."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'id="scores" hidden' in text
        assert "Почему такая оценка?" in text
        assert "score-detail-toggle" in text

    def test_skip_link_present_for_sharpen_step(self):
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert "Пропустить" in text and "skipSharpen" in text

    def test_skip_link_no_longer_names_the_paid_step(self):
        """«Пропустить — сразу к живому тесту» на четвёртом шаге из пяти
        читалось странно: человек не «пропускает» проверку ради покупки, он
        просто не хочет заострять идею. Обгон переехал в липкую полоску."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert "Пропустить — сразу к живому тесту" not in text
        assert "Пропустить — сразу к бизнес-плану" not in text

    def test_paid_actions_are_reachable_from_any_step(self):
        """Оба платных действия жили только в конце ленты, и решившийся на
        втором шаге должен был доклацать до пятого (кастдев 2026-08-02)."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'id="jump"' in text
        assert 'id="jump-test"' in text and 'id="jump-report"' in text
        assert "function jumpTo(" in text

    def test_jump_bar_hides_itself_on_the_last_step(self):
        """На финальном шаге оба действия уже развёрнуты полными блоками —
        полоска дублировала бы их и закрывала форму контакта."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert "function syncJump(" in text
        assert "current !== LAST_STEP" in text

    def test_jump_marks_skipped_steps_done_instead_of_leaving_them_open(self):
        """Обгон обязан свернуть пропущенные шаги: иначе лента остаётся
        наполовину раскрытой и финальный блок теряется среди неё."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        block = text.split("function jumpTo(", 1)[1][:900]
        assert "classList.add('done')" in block

    def test_header_is_stripped_of_pre_result_clutter(self):
        """До результата, за которым человек пришёл, стояли переключатель
        оптики и обещание следующего этапа. Оба убраны."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'id="optics"' not in text
        assert 'class="path-next"' not in text
        assert "Результат проверки спроса" in text     # то, за чем пришли, на месте

    def test_share_is_an_icon_but_still_works(self):
        """Кнопку свернули в иконку, а не выбросили: делятся редко, но
        возможность нужна."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'id="share-btn"' in text and 'class="icon-btn"' in text
        assert 'aria-label="Поделиться результатом"' in text

    def test_save_to_account_survived_the_cleanup(self):
        """Единственный способ для анонимной проверки попасть в кабинет —
        эту кнопку убирать было нельзя, как бы ни чистили шапку."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert 'id="save-btn"' in text and "Сохранить в кабинете" in text

    def test_steps_without_data_excluded_from_order(self):
        """Пустые scores/competitors не рисуют шаг вовсе -- STEP_ORDER их не включает."""
        text = client.get(f"/r/{pub(self._make_check(scores=[], overall=None, competitors={'found': None, 'top': []}))}").text
        assert "hasComp ? 2 : null" in text            # логика исключения шага в разметке присутствует
        assert "hasScores ? 3 : null" in text

    def test_competitors_named_clearly_and_come_before_score(self):
        """По кастдев-фидбеку: «Кто уже отвечает на этот спрос» -- непонятное
        имя раздела, конкурентов надо смотреть раньше синтезирующей оценки,
        а не между сырым спросом и ей."""
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert "Кто уже отвечает на этот спрос" not in text
        assert "Конкуренты" in text
        assert text.index("Конкуренты") < text.index("Оценка идеи")


class TestPayments:
    def test_live_test_return_url_falls_back_without_check_id(self, monkeypatch):
        """/r/{check_id} без check_id — битая ссылка (404). Без check_id
        оплата должна возвращать на главную, а не на несуществующую /r/."""
        import app.main as m
        captured = {}
        async def fake_create_payment(order_id, amount, description, return_url, **kw):
            captured["return_url"] = return_url
            return "pay_x", "https://yookassa.example/pay"
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        monkeypatch.setattr(m.payments, "create_payment", fake_create_payment)
        r = client.post("/api/live-test", json={"contact": "no_check_id@example.com"})
        assert r.status_code == 200
        assert captured["return_url"].endswith("/?paid=1")
        assert "/r/" not in captured["return_url"]

    def test_live_test_return_url_uses_check_id_when_present(self, monkeypatch):
        import app.main as m
        captured = {}
        async def fake_create_payment(order_id, amount, description, return_url, **kw):
            captured["return_url"] = return_url
            return "pay_y", "https://yookassa.example/pay"
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        monkeypatch.setattr(m.payments, "create_payment", fake_create_payment)
        r = client.post("/api/live-test", json={"check_id": 42, "contact": "with_check_id@example.com"})
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

    def test_create_payment_includes_receipt_54fz(self):
        """Регрессия: без receipt ЮКасса отвечала 400 "Receipt is missing or
        illegal" на КАЖДЫЙ платёж -- см. живой прогон владельца."""
        from app.payments import create_payment
        captured = {}
        async def post(kind, payload):
            captured.update(payload)
            return {"id": "pay_1", "confirmation": {"confirmation_url": "https://yookassa.example/pay"}}
        asyncio.run(create_payment(7, 990, "Создатель · отчёт", "https://x/report/1?paid=1",
                                   contact="user@example.com", _post=post))
        receipt = captured["receipt"]
        assert receipt["items"][0]["amount"]["value"] == "990.00"
        assert receipt["items"][0]["vat_code"] == 1
        assert receipt["customer"]["email"] == "user@example.com"

    def test_valid_receipt_contact_accepts_email_and_phone_rejects_telegram(self):
        from app.payments import valid_receipt_contact
        assert valid_receipt_contact("user@example.com") is True
        assert valid_receipt_contact("+7 999 123-45-67") is True
        assert valid_receipt_contact("@telegram_handle") is False
        assert valid_receipt_contact("просто текст") is False
        assert valid_receipt_contact("") is False

    def test_receipt_without_email_or_phone_omits_customer(self):
        """contact = телеграм-хэндл -- чек всё равно валиден (есть items),
        просто без адресата доставки, который ЮКасса не примет как email/phone."""
        from app.payments import create_payment
        captured = {}
        async def post(kind, payload):
            captured.update(payload)
            return {"id": "pay_2", "confirmation": {"confirmation_url": "https://yookassa.example/pay"}}
        asyncio.run(create_payment(7, 990, "Создатель · отчёт", "https://x/report/1?paid=1",
                                   contact="@telegram_handle", _post=post))
        assert "customer" not in captured["receipt"]

    def test_live_test_rejects_telegram_only_contact_when_payments_configured(self, monkeypatch):
        """Регрессия: этот магазин ЮКассы отклоняет платёж без customer.email
        /customer.phone в чеке -- значит телеграм-хэндл больше не годится для
        платного заказа, и мы обязаны сказать об этом ДО похода в ЮКассу,
        а не вернуть пользователю 502 после чужого 400."""
        import app.main as m
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        async def should_not_be_called(*a, **kw):
            raise AssertionError("create_payment не должен вызываться с невалидным контактом")
        monkeypatch.setattr(m.payments, "create_payment", should_not_be_called)
        r = client.post("/api/live-test", json={"contact": "@telegram_handle"})
        assert r.status_code == 400
        assert "почта или телефон" in r.json()["error"].lower()

    def test_live_test_telegram_contact_ok_without_payments_configured(self, monkeypatch):
        """Без настроенной кассы -- заявка без оплаты, чек не создаётся,
        телеграм остаётся нормальным способом связи."""
        import app.main as m
        monkeypatch.setattr(m.payments, "configured", lambda: False)
        r = client.post("/api/live-test", json={"contact": "@telegram_handle"})
        assert r.status_code == 200 and r.json()["ok"] is True

    def test_report_rejects_telegram_only_contact_when_payments_configured(self, monkeypatch):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест", "count": 100}],
                    "verdict": {"level": "unknown", "text": ""}, "competitors": {"found": None, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для проверки контакта"}).json()["id"]
        finally:
            m.check_demand = orig
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        r = client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "@telegram_handle"})
        assert r.status_code == 400
        assert "почта или телефон" in r.json()["error"].lower()

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

    def test_webhook_auto_launches_project_when_full_offer_chosen(self, monkeypatch):
        """Полный оффер (не только angle/h1/sub) сохранён на /r/ -- при оплате
        проект должен запуститься сам, без ручного /api/launch владельцем,
        и сразу быть привязан к покупателю по contact."""
        import app.main as m
        from app.main import LiveTestOrder, SmokeProject, Session, engine, select
        chosen = dict(VALID_OFFER, idea_id="autolaunch_v1", product_name="АвтоЗапуск")
        with Session(engine) as s:
            order = LiveTestOrder(idea="и", contact="auto@example.com", status="pending_payment",
                                  payment_id="pay_auto", amount=1490,
                                  chosen_offer=json.dumps(chosen, ensure_ascii=False))
            s.add(order); s.commit(); s.refresh(order); oid = order.id
        async def fake_fetch(pid, **kw):
            return {"status": "succeeded", "metadata": {"order_id": str(oid)}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        r = client.post("/api/yookassa/webhook",
                        json={"event": "payment.succeeded", "object": {"id": "pay_auto"}})
        assert r.status_code == 200
        with Session(engine) as s:
            order = s.get(LiveTestOrder, oid)
            assert order.status == "paid"
            assert order.idea_id == "autolaunch_v1"
            proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == "autolaunch_v1")).first()
            assert proj is not None
            assert proj.contact == "auto@example.com"

    def test_webhook_does_not_autolaunch_without_chosen_offer(self, monkeypatch):
        """Пропустили заострение на /r/ -- chosen_offer пуст, проект не
        запускается сам (нет данных для лендинга), владелец делает это вручную."""
        import app.main as m
        from app.main import LiveTestOrder, SmokeProject, Session, engine, select
        with Session(engine) as s:
            order = LiveTestOrder(idea="и", contact="skip@example.com", status="pending_payment",
                                  payment_id="pay_skip", amount=1490, chosen_offer="")
            s.add(order); s.commit(); s.refresh(order); oid = order.id
        async def fake_fetch(pid, **kw):
            return {"status": "succeeded", "metadata": {"order_id": str(oid)}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        client.post("/api/yookassa/webhook", json={"event": "payment.succeeded", "object": {"id": "pay_skip"}})
        with Session(engine) as s:
            assert s.get(LiveTestOrder, oid).idea_id is None
            assert s.exec(select(SmokeProject).where(SmokeProject.contact == "skip@example.com")).first() is None

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


class TestSharpenCardsLineUp:
    """Кастдев 2026-08-02: «блоки хоть и одного размера, но текст в них не
    ровно и по-разному». Заголовок в одной карточке занимал две строки, в
    другой одну — и «Для кого», «Боль» и кнопка ехали по вертикали друг
    относительно друга."""

    def _page(self):
        from pathlib import Path
        return Path("static/result.html").read_text(encoding="utf-8")

    def test_cards_share_a_row_grid(self):
        """Само выравнивание меряет браузерный тест
        (test_sharpen_cards_line_up_row_by_row в tests/test_mobile.py) — здесь
        только сторож на механизм: строк шесть и карточка занимает все шесть."""
        text = self._page()
        assert "grid-template-rows:subgrid" in text
        assert "grid-row:span 6" in text
        assert "repeat(6, auto)" in text

    def test_meta_slots_are_always_rendered(self):
        """Subgrid выравнивает строки только если слотов в каждой карточке
        поровну. Карточка без «Для кого» или без «Боли» раньше не рисовала
        строку вовсе — и сетка разъезжалась именно на ней. Теперь пустеет
        содержимое строки, а не сама строка."""
        text = self._page()
        block = text.split("function renderSharpen(", 1)[1][:1400]
        assert block.count('<div class="sharp-meta-row">') == 2
        # Условие осталось ВНУТРИ строки, а не вокруг неё.
        assert '? `<div class="sharp-meta-row">' not in block
        assert '${o.eyebrow ? `<span class="sharp-meta-tag">' in block

    def test_empty_meta_keeps_its_place_in_the_grid(self):
        """`display:none` на пустом блоке отнял бы у карточки строку и сломал
        бы выравнивание двух остальных — прячем цветом рамки."""
        text = self._page()
        assert ".sharp-meta:empty{border-left-color:transparent}" in text
        assert ".sharp-meta:empty{display:none}" not in text

    def test_fallback_kept_where_subgrid_is_unsupported(self):
        """Старый браузер не должен получить сломанную сетку вместо
        неидеальной: subgrid включается только под @supports."""
        text = self._page()
        assert "@supports (grid-template-rows: subgrid)" in text
        assert ".sharp-card{border:1px solid var(--line)" in text  # базовый flex на месте


class TestSharpenSpeaksRussian:
    """Живой прогон: в карточке заострения оказалось английское слово. На
    русской странице это читается как недоделка."""

    def test_prompt_forbids_latin_in_visible_fields(self):
        from app.offer_engine import _system_prompt
        low = _system_prompt("business").lower()
        assert "только по-русски" in low or "только на русском" in low
        assert "латиниц" in low

    def test_service_field_is_named_as_the_exception(self):
        """idea_id — служебное поле и латиницей быть обязано (оно уходит в
        адрес проекта). Без явного исключения запрет противоречил бы схеме
        ответа, где рядом стоит «латиницей_v1»."""
        from app.offer_engine import _system_prompt
        assert "idea_id" in _system_prompt("business")

    def test_rule_reaches_every_audience(self):
        from app.offer_engine import _system_prompt
        for purpose in ("business", "social_contract", "student"):
            assert "латиниц" in _system_prompt(purpose).lower(), purpose


class TestSharpenPublic:
    """Заострение идеи -- бесплатно и без ключа владельца, по кнопке на /r/{id}."""

    def test_sharpen_public_no_owner_key_required(self):
        import app.main as m
        async def fake_sharpen(idea, *a, **kw):
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

    def test_sharpen_passes_purpose_through(self):
        """Раньше заострение всегда писалось "за фаундера" -- purpose с
        витрины/страницы результата не доезжал до движка вовсе."""
        import app.main as m
        captured = {}
        async def fake_sharpen(idea, *a, purpose="business", **kw):
            captured["purpose"] = purpose
            return {"sharpened_note": "", "warning": "",
                    "offers": [dict(VALID_OFFER, idea_id=f"pub{i}") for i in range(3)]}
        orig = m.sharpen_idea
        m.sharpen_idea = fake_sharpen
        try:
            r = client.post("/api/sharpen", json={"idea": "Идея достаточно длинная для заострения",
                                                  "purpose": "student"})
            assert r.status_code == 200
            assert captured["purpose"] == "student"
        finally:
            m.sharpen_idea = orig

    def test_sharpen_llm_failure_returns_400(self):
        import app.main as m
        async def failing(idea, *a, **kw):
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
        text = client.get(f"/r/{pub(rid)}").text
        assert "/api/sharpen" in text
        assert "Заострим идею" in text

    def test_sharpen_cards_render_audience_and_labeled_pain(self):
        """По кастдев-фидбеку: варианты заострения были неразличимы -- eyebrow
        (аудитория) уже генерируется offer_engine, но раньше не рендерился;
        боль теперь явно подписана, а не голым текстом под заголовком."""
        text = client.get(f"/r/{pub(self._make_check_for_sharpen())}").text
        assert "o.eyebrow" in text and "Для кого:" in text
        assert "Боль:" in text

    def _make_check_for_sharpen(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 1}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""}, "competitors": {"found": None, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Идея для проверки карточек заострения"}).json()["id"]
        finally:
            m.check_demand = orig


class TestReportFlow:
    """Платный отчёт/бизнес-план: заказ, оплата, ленивая генерация после
    оплаты, роутинг вебхука между LiveTestOrder и ReportPurchase."""

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
            r = client.post("/api/demand", json={"idea": "Идея достаточно длинная для отчёта"})
            return r.json()["id"]
        finally:
            m.check_demand = orig

    def test_report_order_requires_check_id(self):
        r = client.post("/api/report", json={"tier": "quick", "contact": "@x"})
        assert r.status_code == 400

    def test_report_order_requires_contact(self):
        rid = self._make_check()
        r = client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "x"})
        assert r.status_code == 400

    def test_report_order_unknown_check_id_404(self):
        r = client.post("/api/report", json={"check_id": 999999, "tier": "quick", "contact": "@no_such_check"})
        assert r.status_code == 404

    def test_report_order_without_payments_is_request(self):
        rid = self._make_check()
        r = client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "@report_x"})
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True and d["paid"] is False and "Заявка принята" in d["message"]

    def test_bad_tier_falls_back_to_quick(self):
        from app.main import ReportPurchase, Session, engine, select
        rid = self._make_check()
        r = client.post("/api/report", json={"check_id": rid, "tier": "premium!!", "contact": "@bad_tier"})
        assert r.status_code == 200
        with Session(engine) as s:
            order = s.exec(select(ReportPurchase).where(ReportPurchase.contact == "@bad_tier")).first()
            assert order.tier == "quick"

    def test_report_page_shows_free_preview_and_locked_sections(self):
        rid = self._make_check()
        text = client.get(f"/report/{pub(rid)}").text
        assert "4 200" in text or "4200" in text   # частотность в тизере, без LLM
        from app.report_engine import section_title
        assert section_title("summary") in text and section_title("verdict") in text
        assert "оффер" not in text.lower() and "лендинг" not in text.lower()

    def test_free_preview_includes_verdict_and_competitor_names(self):
        """Бесплатный тизер — не только цифры: вердикт и реальные конкуренты,
        чтобы решение о покупке не требовало долистывать весь блюр."""
        rid = self._make_check()
        text = client.get(f"/report/{pub(rid)}").text
        assert "t.ru" in text
        assert "Спрос есть" in text

    def test_free_preview_is_analysis_not_a_stat_block(self):
        """Кастдев-фидбек: крупные цифры без разбора не убеждают -- тизер
        должен читаться как текстовый анализ (заметки LLM по шкалам, уже
        посчитанные на бесплатном шаге), а не витрина из голых чисел."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест фраза", "count": 4200}],
                    "best_phrase": "тест фраза",
                    "verdict": {"level": "strong", "text": "Спрос есть"},
                    "competitors": {"found": 100, "top": [{"title": "Т", "domain": "t.ru"}]},
                    "scores": [
                        {"key": "demand", "label": "Спрос", "value": 8, "note": ""},
                        {"key": "competition", "label": "Конкуренция", "value": 6, "note": "рынок не забит"},
                        {"key": "timing", "label": "Своевременность", "value": 7, "note": "спрос растёт сейчас"},
                        {"key": "execution", "label": "Реализуемость", "value": 9, "note": "можно запустить за пару недель"},
                    ],
                    "overall": {"value": 7, "weakest": "Конкуренция"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для анализа тизера"}).json()["id"]
        finally:
            m.check_demand = orig
        text = client.get(f"/report/{pub(rid)}").text
        assert 'class="stat"' not in text          # старая витрина из цифр убрана
        assert "рынок не забит" in text
        assert "спрос растёт сейчас" in text
        assert "можно запустить за пару недель" in text

    def test_pricing_shown_near_top_not_only_at_bottom(self):
        """Цены не только в самом низу заблюренного отчёта -- дублируются
        сразу после бесплатного тизера, чтобы не заставлять листать весь блюр."""
        rid = self._make_check()
        text = client.get(f"/report/{pub(rid)}").text
        assert 'id="pricing-top"' in text
        assert text.index('id="pricing-top"') < text.index('id="sections"')

    def test_report_page_404_for_missing_check(self):
        assert client.get("/report/999999").status_code == 404

    def test_report_status_endpoint(self):
        rid = self._make_check()
        r = client.get(f"/api/report/{rid}/status")
        assert r.status_code == 200 and r.json() == {"paid": False, "tier": None}

    def test_report_unlocks_after_paid_and_generates_lazily_once(self, monkeypatch):
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        rid = self._make_check()
        client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "@unlock_test"})
        with Session(engine) as s:
            order = s.exec(select(ReportPurchase).where(ReportPurchase.contact == "@unlock_test")).first()
            order.status = "paid"; s.add(order); s.commit()
            oid, tok = order.id, order.access_token

        calls = {"core": 0, "sections": 0}
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            calls["core"] += 1
            return {"viability_score": 62, "viability_summary": "Ниша занята.",
                    "top_risks": [{"title": "Риск", "body": "Объяснение."}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            calls["sections"] += 1
            return {"key": key, "title": "Резюме", "body": "Тестовый текст отчёта."}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

        # страница отдаётся сразу: ядро есть, разделы придут отдельными запросами
        text = client.get(f"/report/{pub(rid)}?t={tok}").text
        assert "Ниша занята." in text
        assert calls["core"] == 1 and calls["sections"] == 0
        with Session(engine) as s:
            assert s.get(ReportPurchase, oid).report_json   # ядро сохранено

        r = client.post(f"/api/report/{rid}/section?key=summary&t={tok}")
        assert r.status_code == 200
        assert r.json()["section"]["body"] == "Тестовый текст отчёта."

        # повторный визит не должен звать модель снова -- всё уже сохранено
        monkeypatch.setattr(m, "generate_core", None)
        monkeypatch.setattr(m, "generate_section", None)
        assert "Тестовый текст отчёта." in client.get(f"/report/{pub(rid)}?t={tok}").text
        assert client.post(f"/api/report/{rid}/section?key=summary&t={tok}").json()["cached"] is True

    def test_report_generation_failure_shows_friendly_error(self, monkeypatch):
        import app.main as m
        from app.report_engine import ReportEngineError
        from app.main import ReportPurchase, Session, engine, select
        rid = self._make_check()
        client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "@fail_test"})
        with Session(engine) as s:
            order = s.exec(select(ReportPurchase).where(ReportPurchase.contact == "@fail_test")).first()
            order.status = "paid"; s.add(order); s.commit()

        async def failing(idea, demand_data, tier, chosen_offer=None, purpose='business'):
            raise ReportEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
        monkeypatch.setattr(m, "generate_core", failing)
        text = client.get(f"/report/{pub(rid)}").text
        assert "Не получилось собрать отчёт" in text

    def test_webhook_routes_report_kind_to_report_purchase(self, monkeypatch):
        import app.main as m
        from app.main import ReportPurchase, Session, engine
        with Session(engine) as s:
            rep = ReportPurchase(idea="и", contact="@rep", status="pending_payment",
                                payment_id="pay_rep", amount=990, tier="quick")
            s.add(rep); s.commit(); s.refresh(rep); rep_id = rep.id
        async def fake_fetch(pid, **kw):
            return {"status": "succeeded", "metadata": {"order_id": str(rep_id), "kind": "report"}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        r = client.post("/api/yookassa/webhook",
                        json={"event": "payment.succeeded", "object": {"id": "pay_rep"}})
        assert r.status_code == 200
        with Session(engine) as s:
            assert s.get(ReportPurchase, rep_id).status == "paid"

    def test_funnel_links_to_report(self):
        text = client.get(f"/r/{pub(self._make_check())}").text
        assert "/report/" in text
        assert "отчёт по идее" in text.lower()


class TestGuideDirect:
    def test_guide_page_serves(self):
        r = client.get("/guide/direct")
        assert r.status_code == 200
        t = r.text
        assert "Простой старт" in t and "нельзя выключить первые 30 дней" in t
        assert "режим эксперта" in t.lower()
        assert "только Поиск" in t
        assert "Этап 3 из 7" in t
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
        assert "/guide/direct" in client.get(f"/r/{pub(rid)}").text


class TestSocialContractPurpose:
    """Сквозная проводка purpose: /social-contract -> DemandCheck -> отчёт.
    Без неё лендинг обещает смету для комиссии, а движок отдаёт венчурный
    разбор -- ровно то, из-за чего человек чувствует, что зря заплатил."""

    def _make_check(self, purpose=None):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор на заказ", "count": 1200}],
                    "best_phrase": "пошив штор на заказ",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": [{"title": "Ш", "domain": "shtory.ru"}]},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            body = {"idea": "Пошив штор и постельного белья на заказ на дому"}
            if purpose is not None:
                body["purpose"] = purpose
            return client.post("/api/demand", json=body).json()["id"]
        finally:
            m.check_demand = orig

    def test_landing_sends_social_contract_purpose(self):
        text = client.get("/social-contract").text
        assert 'AUDIENCE = "social_contract"' in text and "purpose: AUDIENCE" in text

    def test_landing_does_not_promise_smeta_in_tier_without_finance(self):
        """Главное обещание лендинга -- обоснование сметы, но секции finance
        нет в QUICK_KEYS: за 990 ₽ сметы не будет. Об этом надо сказать на
        витрине, иначе человек покупает дешёвый тариф ровно за тем, чего в
        нём нет, и справедливо считает, что его обманули."""
        import app.main as m
        from app.report_engine import QUICK_KEYS
        assert "finance" not in QUICK_KEYS      # предпосылка теста
        # читаем отданную страницу, а не исходник: название тарифа теперь
        # подставляется из REPORT_PRICES и в статике его нет (B5)
        text = client.get("/social-contract").text
        assert "Без сметы и расчётов" in text
        assert f"в тарифе «{m.REPORT_PRICES['full']['label']}»" in text

    def test_landing_tier_name_matches_backend_label(self):
        """Витрина и страница отчёта должны звать тариф одинаково."""
        import app.main as m
        text = client.get("/social-contract").text
        assert f"<h3>{m.REPORT_PRICES['full']['label']}</h3>" in text

    def test_purpose_persisted_from_landing(self):
        from app.main import DemandCheck, Session, engine
        rid = self._make_check("social_contract")
        with Session(engine) as s:
            assert s.get(DemandCheck, rid).purpose == "social_contract"

    def test_default_purpose_is_business(self):
        """Обычная главная ничего не шлёт -- прежнее поведение сохраняется."""
        from app.main import DemandCheck, Session, engine
        rid = self._make_check()
        with Session(engine) as s:
            assert s.get(DemandCheck, rid).purpose == "business"

    def test_unknown_purpose_rejected_not_stored(self):
        from app.main import DemandCheck, Session, engine
        rid = self._make_check("../../etc/passwd")
        with Session(engine) as s:
            assert s.get(DemandCheck, rid).purpose == "business"

    def test_result_page_exposes_purpose(self):
        rid = self._make_check("social_contract")
        assert 'const PURPOSE = "social_contract";' in client.get(f"/r/{pub(rid)}").text

    def test_result_page_defaults_purpose_for_business(self):
        rid = self._make_check()
        assert 'const PURPOSE = "business";' in client.get(f"/r/{pub(rid)}").text

    def test_result_page_promotes_business_plan_for_social_contract(self):
        """Человек с /social-contract пришёл за планом для комиссии, а главной
        кнопкой ему предлагался рекламный тест за 1490 ₽, тогда как бизнес-план
        прятался в «Или...». Для рекламной кампании на эту аудиторию это прямая
        потеря конверсии: платим за клик и сразу продаём не то."""
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        # Ветка по аудитории переехала в данные с сервера (F3).
        assert "AUDIENCE.plan_first" in text
        # бизнес-план поднимается в главный блок, рекламный тест опускается
        assert "alt.className = 'next'" in text
        assert "order.className = 'alt-path'" in text
        assert "insertBefore(alt, order)" in text
        assert "бизнес-план для комиссии" in text

    def test_social_contract_copy_avoids_internal_jargon(self):
        """«Живой тест» -- наше внутреннее имя услуги, человеку из соцконтракта
        оно ничего не говорит.

        Раньше это чинилось подменой подписи ссылки «пропустить» под
        аудиторию. Теперь ссылка нейтральна для всех и платный шаг не
        называет вовсе -- гарантия та же, но без развилки, которую надо
        было помнить при каждой правке текста.
        """
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "Оставить заявку на проверку идеи" in text
        skip = text.split('id="skip-sharpen"', 1)[1].split("</a>", 1)[0]
        assert "живому тесту" not in skip and "живой тест" not in skip

    def test_swapped_blocks_keep_input_styling(self):
        """Блоки меняются ролями -- поле контакта не должно терять оформление
        из-за смены класса контейнера."""
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert ".next input,.alt-path input{" in text

    def test_report_generation_receives_stored_purpose(self, monkeypatch):
        """Главное звено: то, что сохранили при проверке спроса, реально
        доезжает до generate_report при открытии оплаченного отчёта."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        rid = self._make_check("social_contract")
        client.post("/api/report", json={"check_id": rid, "tier": "full", "contact": "@soc_purpose"})
        with Session(engine) as s:
            order = s.exec(select(ReportPurchase).where(ReportPurchase.contact == "@soc_purpose")).first()
            order.status = "paid"; s.add(order); s.commit()
            tok = order.access_token
        seen = {}
        async def fake_generate(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            seen["purpose"] = purpose
            return {"sections": [{"key": "summary", "title": "Резюме проекта", "body": "текст"}]}
        monkeypatch.setattr(m, "generate_core", fake_generate)
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert seen["purpose"] == "social_contract"


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
        self._assert_footer(client.get(f"/r/{pub(rid)}").text, "результата")

    def test_project_has_footer(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="foot_proj_v1", product_name="ФутерПроект")})
        self._assert_footer(client.get("/p/foot_proj_v1").text, "проекта")

    def test_guide_direct_has_footer(self):
        self._assert_footer(client.get("/guide/direct").text, "гайда по Директу")

    def test_social_contract_has_footer(self):
        self._assert_footer(client.get("/social-contract").text, "соцконтракт-страницы")


class TestAccountLinkEverywhere:
    """Пункт 2 сырого фидбека владельца (2026-07-31): ссылка в /account была
    только на главной -- уйдя на любую другую публичную страницу, человек не
    мог вернуться в кабинет без ручного перехода на `/`."""

    def _assert_account_link(self, text, page_name):
        assert 'href="/account"' in text, f"Нет ссылки /account на {page_name}"

    def test_index_has_account_link(self):
        self._assert_account_link(client.get("/").text, "главной")

    def test_social_contract_has_account_link(self):
        self._assert_account_link(client.get("/social-contract").text, "соцконтракт-витрины")

    def test_students_has_account_link(self):
        self._assert_account_link(client.get("/students").text, "студенческой витрины")

    def test_guide_direct_has_account_link(self):
        self._assert_account_link(client.get("/guide/direct").text, "гайда по Директу")

    def test_result_page_has_account_link(self):
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
                              json={"idea": "Достаточно длинная идея для проверки ссылки в кабинет"}).json()["id"]
        finally:
            m.check_demand = orig
        self._assert_account_link(client.get(f"/r/{pub(rid)}").text, "страницы результата")

    def test_project_page_has_account_link(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="acct_link_proj_v1", product_name="КабинетСсылкаПроект")})
        self._assert_account_link(client.get("/p/acct_link_proj_v1").text, "страницы проекта")

    def test_homepage_nav_drops_the_redundant_path_link(self):
        """Пункт 1 сырого фидбека владельца (2026-07-31): «путь 1→7» в шапке
        главной -- избыточная навигация вида "1→7", отдельная от per-проектной
        метки «Этап N из 7» на /r/, /p/ и /guide/direct. Убрана только ссылка
        в шапке -- сам раздел #path (семь ступеней) остаётся на странице как
        был, просто без дублирующего якоря в nav."""
        import re
        text = client.get("/").text
        header = text[text.index("<header>"):text.index("</header>")]
        assert "Путь 1→7" not in header
        assert 'href="#path"' not in header
        assert re.search(r'id="path"', text), "раздел «Путь от идеи до денег» пропал совсем"


class TestSocialContractPage:
    """Отдельная посадочная страница под рекламу на аудиторию социального
    контракта -- не часть общего позиционирования сайта (см. CLAUDE.md),
    доступна только по прямой ссылке /social-contract."""

    def test_page_loads_and_mentions_social_contract(self):
        r = client.get("/social-contract")
        assert r.status_code == 200
        assert "социального контракта" in r.text.lower() or "социальн" in r.text.lower()

    def test_no_jargon(self):
        text = client.get("/social-contract").text
        assert "оффер" not in text.lower()
        assert "лендинг" not in text.lower()

    def test_shares_free_demand_check_funnel(self):
        """Ведёт в тот же бесплатный /api/demand, что и главная -- не отдельный
        продукт с собственным бэкендом."""
        text = client.get("/social-contract").text
        assert "/api/demand" in text
        assert 'id="idea"' in text

    def test_homepage_positioning_does_not_start_with_social_contract(self):
        """**Прежнее решение отменено 2026-07-28, и вот почему.**

        Раньше здесь стояло «страница не должна светиться на главной вовсе» —
        чтобы не сужать себя перед массовым посетителем упоминанием
        соцконтракта. Опасение верное, но запрет оказался слишком широким:
        человек, которому нужно обоснование для комиссии, приходил на главную,
        читал про венчурную проверку идеи и уходил, не узнав, что мы и про
        него (F2). Витрина была доступна ТОЛЬКО по ссылке из объявления.

        Решение владельца: витрины публичны, но позиционирование главной не
        меняется. Поэтому проверяем не «ссылки нет», а «ссылка есть только в
        переключателе»: заголовок, подзаголовок и описание пути на главной
        по-прежнему не говорят о соцконтракте.
        """
        import re
        page = client.get("/").text
        body = page[page.index("<main>"):]
        switch = re.search(r'<nav class="aud-switch".*?</nav>', body, re.S)
        assert switch, "переключателя аудитории нет"
        without_switch = body.replace(switch.group(), "")
        assert "/social-contract" not in without_switch
        for word in ("соцконтракт", "социальн", "соцзащит", "комисси"):
            assert word not in without_switch.lower(), word

    def test_uses_light_design_system(self):
        text = client.get("/social-contract").text
        assert "IBM Plex" in text
        assert "#FBF6EA" in text
        assert "Manrope" not in text and "Onest" not in text

    def test_business_plan_is_highlighted_in_the_headline(self):
        """Пункт 13 сырого фидбека владельца (2026-07-31): «Бизнес-план» в
        заголовке витрины должен быть выделен жёлтым маркером, как остальные
        акцентные слова на сайте, а не идти обычным текстом."""
        text = client.get("/social-contract").text
        assert '<span class="hl">Бизнес-план</span>' in text

    def test_path_map_is_visible_from_this_storefront_too(self):
        """Часть пункта 14 сырого фидбека владельца (2026-07-31): карта пути
        0->6 раньше была только на главной -- пришедший через рекламу на
        /social-contract или /students не видел общую картину сервиса.
        Секция теперь общая (audience-landing.html), а не главная-only."""
        for path in ("/social-contract", "/students"):
            text = client.get(path).text
            assert 'id="path"' in text, path
            assert "Путь от идеи до денег" in text, path
            assert text.count('class="step"') == 7, path

    def test_fast_plan_button_only_on_social_contract(self):
        """F10: кнопка-обгон «Сразу сделать бизнес-план» — только у
        соцконтракта (владелец описал именно эту аудиторию), не у бизнеса
        или студента. Задаётся в реестре (app/audiences.py), не в шаблоне —
        третья копия разъехалась бы, как разъезжались цены (B5)."""
        assert 'id="plan-btn"' in client.get("/social-contract").text
        assert "Сразу сделать бизнес-план" in client.get("/social-contract").text
        assert 'id="plan-btn"' not in client.get("/students").text
        assert 'id="plan-btn"' not in client.get("/").text

    def test_fast_plan_button_still_checks_demand_but_skips_to_report(self):
        """F10: владелец согласился на «обгон» с условием -- проверка спроса
        всё равно считается в фоне (цифры Вордстата всё так же кормят платный
        отчёт), просто человек не видит промежуточный /r/. Проверяем именно
        это в JS-источнике: обе кнопки зовут один и тот же /api/demand, но
        целятся в разные страницы после."""
        text = client.get("/social-contract").text
        assert "submitIdea(btn, planBtn, '/r/')" in text
        assert "submitIdea(planBtn, btn, '/report/')" in text
        assert text.count("fetch('/api/demand'") == 1, "должен остаться ОДИН путь проверки спроса, не два разных"

    def test_fast_plan_flow_lands_on_a_working_report_page(self):
        """Сквозная проверка того, что реально произойдёт по клику: тот же
        /api/demand с purpose=social_contract, а следующая станица -- уже
        существующий /report/{id}, который и без этой кнопки умеет показывать
        тарифы до оплаты (см. app/main.py:report_page). Никакой новой
        бэкенд-логики не потребовалось, только другой адрес редиректа."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест", "count": 100}],
                    "best_phrase": "тест", "verdict": {"level": "unknown", "text": "Нет данных"},
                    "competitors": {"found": 0, "top": []},
                    "scores": [], "overall": {"value": 0, "weakest": ""}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={
                "idea": "Пошив штор и постельного белья на заказ на дому в своём районе",
                "purpose": "social_contract"})
        finally:
            m.check_demand = orig
        assert r.status_code == 200
        pub = r.json()["public_id"]
        report = client.get(f"/report/{pub}")
        assert report.status_code == 200
        assert "оффер" not in report.text.lower() and "лендинг" not in report.text.lower()


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


class TestProjectPageSpeaksToCustomer:
    """A3 из PRODUCT_ROADMAP: `/p/` писалась, когда её видел только владелец,
    и осталась на «ты» с владельческими формулировками. Теперь страница
    открыта покупателю из /account -- остальной сайт обращается на «вы»."""

    def test_no_informal_address_left(self):
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        for bad in ("не трогай", "ты знаешь", "видны только тебе", "твой целевой",
                    "Открой ", "Проверь ", "Проверь,", "Создай ", "Запусти ",
                    "Вставь ", "впиши "):
            assert bad not in text, f"«ты»-форма осталась: {bad!r}"

    def test_polite_forms_present(self):
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        for good in ("Откройте", "Проверьте", "Создайте", "Запустите", "Вставьте",
                     "впишите", "видны только вам"):
            assert good in text, f"не хватает вежливой формы: {good!r}"

    def test_internal_jargon_removed(self):
        """«smoke-тест» и «РСЯ» -- наши внутренние слова; «трафик» запрещён
        принципом 5 (пользователь не обязан знать наш язык)."""
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        assert "smoke" not in text.lower()
        assert "РСЯ" not in text
        assert "трафик" not in text.lower()


class TestOwnerKeyUrlHandoff:
    """/desk уже знает ключ владельца (sessionStorage) -- при переходе на
    /p/{id} раньше он терялся, и project.html спрашивал ключ заново нативным
    prompt() при КАЖДОМ открытии проекта. Теперь /desk кладёт ключ в ссылку,
    а project.html сначала смотрит в URL и только потом -- в prompt()."""

    def test_desk_passes_owner_key_in_project_link(self):
        text = (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()
        assert "location.href='${s.project_url}?key='+encodeURIComponent(KEY)" in text

    def test_project_page_reads_key_from_url_before_prompting(self):
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        assert "new URLSearchParams(location.search).get(\"key\")" in text
        # порядок важен: URL раньше prompt(), иначе владелец из /desk всё
        # равно увидит диалог
        url_pos = text.index("URLSearchParams(location.search)")
        prompt_pos = text.index("prompt(\"Ключ владельца:\")")
        assert url_pos < prompt_pos


class TestYandexMetrika:
    """Счётчик вставляется единой точкой в _static() (см. _inject_metrika),
    а не копипастой по каждому HTML-файлу. Цели воронки шлются из JS через
    window.SOZDATEL_YM_ID, который кладёт та же вставка."""

    def test_no_injection_without_id(self, monkeypatch):
        monkeypatch.setattr(main_module, "YANDEX_METRIKA_ID", "")
        html = "<html><head><title>т</title></head><body></body></html>"
        assert main_module._inject_metrika(html) == html

    def test_injects_snippet_with_id(self, monkeypatch):
        monkeypatch.setattr(main_module, "YANDEX_METRIKA_ID", "12345")
        html = "<html><head><title>т</title></head><body></body></html>"
        out = main_module._inject_metrika(html)
        assert "SOZDATEL_YM_ID = 12345" in out
        assert "mc.yandex.ru/watch/12345" in out
        assert out.index("SOZDATEL_YM_ID") < out.index("</head>")

    def test_noop_without_head_tag(self, monkeypatch):
        monkeypatch.setattr(main_module, "YANDEX_METRIKA_ID", "12345")
        html = "<div>нет head тега</div>"
        assert main_module._inject_metrika(html) == html

    def test_demand_started_goal_wired_in_public_entry_points(self):
        static_dir = main_module.BASE_DIR.parent / "static"
        for name in ("index.html", "audience-landing.html"):
            text = (static_dir / name).read_text()
            assert "sozGoal('demand_started'" in text, f"нет цели demand_started в {name}"

    def test_report_payment_goals_wired_and_no_reload_loop(self):
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert "report_paid_quick" in text and "report_paid_full" in text
        # старый баг: условие пускало поллер повторно после reload по quick-тарифу
        # и страница перезагружалась раз в 2с бесконечно
        assert "UNLOCKED_TIER !== 'full'" not in text


class TestPaidReportIsNotASieve:
    """Кастдев 2026-08-02, покупка за 990 ₽: «сгенерировались только резюме,
    проблема, спрос и кто отвечает на запрос, потом куча заблюренных частей».

    До оплаты заблюренные разделы — витрина, они и должны показывать, за что
    просят деньги. ПОСЛЕ оплаты те же разделы превращают купленный документ
    в решето: четыре собранных вперемешку с десятком замазанных.
    """

    def _page(self):
        from pathlib import Path
        return Path("static/report.html").read_text(encoding="utf-8")

    def test_paid_flag_separates_showcase_from_document(self):
        text = self._page()
        assert "const PAID = TIER_KEYS.length > 0;" in text

    def test_other_tier_sections_are_not_drawn_inside_a_paid_report(self):
        text = self._page()
        block = text.split("function sectionHtml(", 1)[1][:900]
        assert "if (PAID) return '';" in block

    def test_teaser_keeps_its_blurred_sections(self):
        """Сторож от чрезмерной правки: без оплаты запертые разделы обязаны
        остаться — это единственное, что объясняет, за что платят."""
        text = self._page()
        assert 'class="section locked"' in text
        assert ".section.locked p{color:var(--muted);filter:blur(3px)" in text

    def test_missing_sections_become_one_honest_line(self):
        text = self._page()
        assert "function upsellHtml()" in text
        block = text.split("function upsellHtml()", 1)[1][:700]
        assert "if (!PAID || !rest.length) return '';" in block
        assert "FULL_LABEL" in block

    def test_upsell_names_the_tier_from_the_single_source(self):
        """Название тарифа уже переименовывали; вторая копия в статике
        разъезжается с витриной незаметно (B7)."""
        text = self._page()
        assert 'const FULL_LABEL = "__FULL_LABEL__";' in text

    def test_empty_groups_disappear_with_their_sections(self):
        """Заголовок группы над пустотой ничего не сообщает, а место занимает."""
        text = self._page()
        block = text.split("function render()", 1)[1][:600]
        assert "SECTIONS.filter(s => !PAID || inTier(s.key))" in block
        assert "visible.forEach" in block

    def test_upsell_is_not_printed(self):
        """Для соцконтракта отчёт несут в комиссию на бумаге — предложение
        доплатить на этом листе неуместно."""
        text = self._page()
        assert "@media print{.upsell{display:none}}" in text

    def test_plural_rule_is_russian(self):
        """«ещё 1 разделов» в платном документе читается как небрежность."""
        text = self._page()
        assert "function plural(" in text
        assert "m10 >= 2 && m10 <= 4" in text


class TestOwnerLearnsAboutOrders:
    """A2 из PRODUCT_ROADMAP: владелец узнавал о деньгах и заявках, только
    открыв /desk глазами. Для продукта с платным рекламным трафиком это
    значит узнавать об оплате через сутки."""

    def _mail(self, monkeypatch):
        """Копит ТОЛЬКО письма владельцу: покупателю на тот же заказ уходит
        своё письмо (A10), и здесь оно только мешало бы считать. Его проверяет
        TestBuyerHearsFromUs."""
        import app.main as m
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        sent = []
        def rec(to, subject, body, **kw):
            if to == "owner@example.com":
                sent.append((subject, body))
        monkeypatch.setattr(m.mailer, "send", rec)
        return sent

    def _check(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 10}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""},
                    "competitors": {"found": None, "top": []}, "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Идея достаточно длинная для уведомления"}).json()["id"]
        finally:
            m.check_demand = orig

    def test_unpaid_live_test_request_reaches_owner(self, monkeypatch):
        sent = self._mail(monkeypatch)
        rid = self._check()
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "lead@example.com"})
        assert r.status_code == 200 and r.json()["paid"] is False
        assert len(sent) == 1
        subject, body = sent[0]
        assert "живой тест" in subject and "заявка без оплаты" in subject
        assert "lead@example.com" in body
        assert "довести вручную" in body

    def test_unpaid_report_request_reaches_owner(self, monkeypatch):
        sent = self._mail(monkeypatch)
        rid = self._check()
        r = client.post("/api/report", json={"check_id": rid, "tier": "full", "contact": "rep@example.com"})
        assert r.status_code == 200
        assert len(sent) == 1
        subject, body = sent[0]
        assert "Бизнес-план" in subject and "rep@example.com" in body
        assert f"/report/{pub(rid)}" in body

    def test_paid_webhook_notifies_owner_once(self, monkeypatch):
        """Вебхук ЮКассы может прийти повторно -- письмо должно уйти один раз."""
        import app.main as m
        from app.main import LiveTestOrder, Session, engine, select
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        sent = self._mail(monkeypatch)
        rid = self._check()
        async def fake_create(order_id, amount, desc, url, kind="livetest", contact="", _post=None):
            return ("pay_a2", "https://pay.example/x")
        monkeypatch.setattr(m.payments, "create_payment", fake_create)
        client.post("/api/live-test", json={"check_id": rid, "contact": "buyer_a2@example.com"})
        assert sent == []          # оплата ещё не подтверждена -- писать не о чем
        with Session(engine) as s:
            oid = s.exec(select(LiveTestOrder).where(LiveTestOrder.contact == "buyer_a2@example.com")).first().id
        async def fake_fetch(pid, _post=None):
            return {"status": "succeeded", "metadata": {"order_id": str(oid), "kind": "livetest"}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)

        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_a2"}})
        assert len(sent) == 1
        subject, body = sent[0]
        assert "оплачено" in subject and "1490" in body

        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_a2"}})
        assert len(sent) == 1      # повтор вебхука не шлёт второе письмо

    def test_paid_notice_does_not_swallow_failure_notice(self, monkeypatch):
        """Флаги уведомлений разведены: письмо об оплате не должно гасить
        более важное письмо о том, что оплаченная услуга не оказана."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        from app.report_engine import ReportEngineError
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        sent = self._mail(monkeypatch)
        rid = self._check()
        async def fake_create(order_id, amount, desc, url, kind="livetest", contact="", _post=None):
            return ("pay_a2b", "https://pay.example/y")
        monkeypatch.setattr(m.payments, "create_payment", fake_create)
        client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "both_a2@example.com"})
        with Session(engine) as s:
            oid = s.exec(select(ReportPurchase).where(ReportPurchase.contact == "both_a2@example.com")).first().id
        async def fake_fetch(pid, _post=None):
            return {"status": "succeeded", "metadata": {"order_id": str(oid), "kind": "report"}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_a2b"}})
        assert len(sent) == 1 and "оплачено" in sent[0][0]

        async def failing(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            raise ReportEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
        monkeypatch.setattr(m, "generate_core", failing)
        with Session(engine) as s:
            tok = s.get(ReportPurchase, oid).access_token
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert len(sent) == 2                      # письмо о сбое всё-таки ушло
        assert "не собрался" in sent[1][0]

    def test_broken_mail_does_not_break_the_order(self, monkeypatch):
        """Заявка должна приниматься, даже если почта владельца легла."""
        import app.main as m
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        def boom(*a, **kw):
            raise RuntimeError("SMTP лёг")
        monkeypatch.setattr(m.mailer, "send", boom)
        rid = self._check()
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "still@example.com"})
        assert r.status_code == 200 and r.json()["ok"] is True

    def test_no_owner_email_configured_is_silent(self, monkeypatch):
        import app.main as m
        monkeypatch.delenv("SOZDATEL_OWNER_EMAIL", raising=False)
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        sent = []
        # Письмо покупателю адресовано ему, а не владельцу, и от этой
        # переменной не зависит -- считаем только владельческие.
        monkeypatch.setattr(m.mailer, "send",
                            lambda to, *a, **kw: sent.append(to) if to == "owner@example.com" else None)
        rid = self._check()
        client.post("/api/live-test", json={"check_id": rid, "contact": "quiet@example.com"})
        assert sent == []


class TestPaidReportFailureIsNoticed:
    """A1 из PRODUCT_ROADMAP: оплата прошла, отчёт не собрался -- самый дорогой
    сценарий отказа. До этого единственным, кто знал о сбое, был покупатель:
    владельцу не приходило ничего, и покупки отчётов вообще не были видны
    в /desk -- ни успешные, ни сорванные."""

    def _paid_purchase(self, monkeypatch):
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 10}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""},
                    "competitors": {"found": None, "top": []}, "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для сбоя отчёта"}).json()["id"]
        finally:
            m.check_demand = orig
        contact = f"buyer{rid}@example.com"
        client.post("/api/report", json={"check_id": rid, "tier": "full", "contact": contact})
        with Session(engine) as s:
            o = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)).first()
            o.status = "paid"; s.add(o); s.commit()
            oid, tok = o.id, o.access_token
        return rid, oid, tok

    def _break_generation(self, monkeypatch):
        import app.main as m
        from app.report_engine import ReportEngineError
        async def failing(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            raise ReportEngineError("ИИ думал слишком долго. Подождите минуту и попробуйте ещё раз.")
        monkeypatch.setattr(m, "generate_core", failing)

    def test_failure_is_recorded_and_owner_notified_once(self, monkeypatch):
        import app.main as m
        from app.main import ReportPurchase, Session, engine
        rid, oid, tok = self._paid_purchase(monkeypatch)
        self._break_generation(monkeypatch)
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        sent = []
        monkeypatch.setattr(m.mailer, "send", lambda to, subject, body, **kw: sent.append((to, subject, body)))

        client.get(f"/report/{pub(rid)}?t={tok}")
        assert len(sent) == 1
        to, subject, body = sent[0]
        assert to == "owner@example.com"
        assert "не собрался" in subject
        assert f"buyer{rid}@example.com" in body and "ИИ думал слишком долго" in body
        with Session(engine) as s:
            p = s.get(ReportPurchase, oid)
            assert p.gen_error and p.fail_notified is True

        # покупатель перезагружает страницу -- второго письма быть не должно
        client.get(f"/report/{pub(rid)}?t={tok}")
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert len(sent) == 1

    def test_unpaid_request_does_not_alarm_owner(self, monkeypatch):
        """Заявка без оплаты -- не денежный сбой, письмо слать не за что."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 10}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""},
                    "competitors": {"found": None, "top": []}, "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея длинная для неоплаченной заявки"}).json()["id"]
        finally:
            m.check_demand = orig
        client.post("/api/report", json={"check_id": rid, "tier": "quick", "contact": "nopay@example.com"})
        self._break_generation(monkeypatch)
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        sent = []
        monkeypatch.setattr(m.mailer, "send", lambda to, subject, body, **kw: sent.append(to))
        client.get(f"/report/{pub(rid)}")
        assert sent == []

    def test_broken_mail_does_not_break_the_page(self, monkeypatch):
        """Принцип «деградация вместо ошибки»: покупатель и так видит сбой
        отчёта -- он не должен получить сверху ещё и 500 из-за почты."""
        import app.main as m
        rid, _, tok = self._paid_purchase(monkeypatch)
        self._break_generation(monkeypatch)
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        def boom(*a, **kw):
            raise RuntimeError("SMTP лёг")
        monkeypatch.setattr(m.mailer, "send", boom)
        r = client.get(f"/report/{pub(rid)}?t={tok}")
        assert r.status_code == 200
        assert "Не получилось собрать отчёт" in r.text

    def test_owner_sees_report_purchases_in_orders(self, monkeypatch):
        """Покупки отчётов не были видны владельцу нигде: оплата на 2990 ₽
        и несостоявшаяся доставка выглядели одинаково -- никак."""
        rid, oid, tok = self._paid_purchase(monkeypatch)
        self._break_generation(monkeypatch)
        monkeypatch.delenv("SOZDATEL_OWNER_EMAIL", raising=False)
        client.get(f"/report/{pub(rid)}?t={tok}")
        data = client.get("/api/orders", headers=OWNER).json()
        row = [r for r in data["reports"] if r["id"] == oid][0]
        assert row["status"] == "paid"
        assert row["delivered"] is False
        assert "ИИ думал слишком долго" in row["gen_error"]
        assert row["tier_label"] == "Бизнес-план"
        # ссылка из /desk ведёт прямо в отчёт -- вместе с токеном покупателя
        assert row["report_url"] == f"/report/{pub(rid)}?t={tok}"

    def test_desk_renders_failed_delivery(self):
        text = (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()
        assert 'id="reports"' in text
        assert "Оплачено, но отчёт не собрался" in text
        # ранний return в loadOrders прятал бы блок отчётов, пока нет живых тестов
        assert "} else {" in text

    def test_notify_owner_never_raises_and_skips_without_address(self, monkeypatch):
        from app import mailer
        monkeypatch.delenv("SOZDATEL_OWNER_EMAIL", raising=False)
        assert mailer.notify_owner("т", "т") is False
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        def boom(msg):
            raise RuntimeError("сеть легла")
        assert mailer.notify_owner("т", "т", _send=boom) is False


class TestMailer:
    """SMTP-обёртка -- тот же паттерн инъекции, что payments.py/llm_adapter.py."""

    def test_not_configured_without_env(self, monkeypatch):
        from app import mailer
        monkeypatch.delenv("SOZDATEL_SMTP_HOST", raising=False)
        assert mailer.configured() is False

    def test_configured_with_all_three_env_vars(self, monkeypatch):
        from app import mailer
        monkeypatch.setenv("SOZDATEL_SMTP_HOST", "mail.hosting.reg.ru")
        monkeypatch.setenv("SOZDATEL_SMTP_USER", "noreply@projectsozdatel.ru")
        monkeypatch.setenv("SOZDATEL_SMTP_PASSWORD", "x")
        assert mailer.configured() is True

    def test_send_without_config_and_without_injection_raises(self, monkeypatch):
        from app import mailer
        monkeypatch.delenv("SOZDATEL_SMTP_HOST", raising=False)
        with pytest.raises(mailer.MailerError):
            mailer.send("x@example.com", "тема", "текст")

    def test_send_via_injection(self):
        from app import mailer
        captured = {}
        def fake_send(msg):
            captured["to"] = msg["To"]; captured["subject"] = msg["Subject"]
            captured["body"] = msg.get_content()
        mailer.send("user@example.com", "Вход", "текст письма", _send=fake_send)
        assert captured["to"] == "user@example.com"
        assert captured["subject"] == "Вход"
        assert "текст письма" in captured["body"]


class TestReturningToAFinishedCheck:
    """Кастдев 2026-08-02: «нажимаю на идею «открыть», и мне будто заново всё
    до заострения проходить надо. Это какой-то бред».

    Лента всегда начиналась с первого шага, даже если человек эту проверку
    уже прошёл и что-то по ней купил.
    """

    def _check(self, **kw):
        import app.main as m
        from app.main import DemandCheck, Session, engine
        data = {"formulations": [{"phrase": "ф", "count": 4200}], "best_phrase": "ф",
                "verdict": {"level": "strong", "text": "Спрос есть"},
                "competitors": {"found": 10, "top": [{"title": "Т", "domain": "t.ru"}],
                                "info_only": False},
                "scores": [{"key": "demand", "label": "Спрос", "value": 8, "note": ""}],
                "overall": {"value": 8, "weakest": "Спрос", "basis": "б"}}
        with Session(engine) as s:
            rec = DemandCheck(idea="Идея, к которой человек возвращается",
                              result_json=json.dumps(data, ensure_ascii=False), **kw)
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id, rec.public_id

    def _resume_flag(self, pid):
        text = client.get(f"/r/{pid}").text
        return text.split("const RESUME = ", 1)[1].split(";", 1)[0].strip()

    def test_fresh_check_starts_from_the_first_step(self):
        """Новичку лента нужна целиком — иначе он не увидит своих же цифр."""
        _, pid = self._check()
        assert self._resume_flag(pid) == "false"

    def test_check_with_chosen_offer_resumes_at_the_end(self):
        _, pid = self._check(chosen_offer=json.dumps({"h1": "Заголовок"}, ensure_ascii=False))
        assert self._resume_flag(pid) == "true"

    def test_check_with_a_purchase_resumes_at_the_end(self):
        """Человек уже заплатил по этой идее — прокликивать бесплатные шаги
        ради возврата к покупке он точно не должен."""
        from app.main import ReportPurchase, Session, engine
        rid, pid = self._check()
        with Session(engine) as s:
            s.add(ReportPurchase(check_id=rid, idea="и", tier="quick",
                                 contact="buyer@example.com", status="paid", amount=990))
            s.commit()
        assert self._resume_flag(pid) == "true"

    def test_check_with_a_live_test_order_resumes_at_the_end(self):
        from app.main import LiveTestOrder, Session, engine
        rid, pid = self._check()
        with Session(engine) as s:
            s.add(LiveTestOrder(check_id=rid, idea="и", contact="buyer@example.com",
                                status="paid", amount=1490))
            s.commit()
        assert self._resume_flag(pid) == "true"

    def test_progress_comes_from_the_server_not_the_browser(self):
        """localStorage не пережил бы ни смену устройства, ни очистку
        браузера — а возвращаются к проверке как раз спустя время."""
        text = client.get(f"/r/{self._check()[1]}").text
        block = text.split("const RESUME = ", 1)[1][:600]
        assert "localStorage" not in block

    def test_resume_marks_earlier_steps_done_not_hidden(self):
        """Свернуть — не значит спрятать: человек должен видеть итоги шагов
        строкой и мочь развернуть любой обратно."""
        text = client.get(f"/r/{self._check()[1]}").text
        block = text.split("if (RESUME", 1)[1][:700]
        assert "classList.add('done')" in block
        assert "openStep(LAST_STEP)" in block


class TestAccountRowsCarryTheirDate:
    """Кастдев 2026-08-02: «в кабинете каша из идей, нужно написать дату
    проверки». Пять строк одной идеи выглядели одинаково."""

    def _login(self, contact="dated@example.com"):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_dated", contact=contact)); s.commit()
        client.post("/account/verify?token=tok_dated", follow_redirects=False)
        return contact

    def test_every_row_type_reports_when_it_happened(self):
        from app.main import (DemandCheck, LiveTestOrder, ReportPurchase,
                              Session, engine)
        contact = self._login()
        with Session(engine) as s:
            s.add(DemandCheck(idea="Идея с датой", contact=contact,
                              result_json=json.dumps({"overall": {"value": 5}}, ensure_ascii=False)))
            s.add(ReportPurchase(idea="Отчёт с датой", tier="quick", contact=contact,
                                 status="paid", amount=990))
            s.add(LiveTestOrder(idea="Заявка с датой", contact=contact,
                                status="new", amount=0))
            s.commit()
        d = client.get("/api/account/me").json()
        client.cookies.clear()
        for section in ("checks", "reports", "orders"):
            assert d[section], f"раздел {section} пуст — проверять нечего"
            for row in d[section]:
                assert row.get("created_at"), f"{section}: строка без даты — {row}"

    def test_cabinet_renders_the_date_in_russian(self):
        from pathlib import Path
        text = Path("static/account.html").read_text(encoding="utf-8")
        assert "toLocaleDateString('ru-RU'" in text
        assert "created_at" in text


class TestAccountCabinet:
    """Личный кабинет покупателя: magic-link на почту вместо пароля."""

    def test_request_link_rejects_bad_email(self):
        r = client.post("/api/account/request-link", json={"contact": "@telegram_handle"})
        assert r.status_code == 400

    def test_request_link_503_when_mailer_not_configured(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.mailer, "configured", lambda: False)
        r = client.post("/api/account/request-link", json={"contact": "user@example.com"})
        assert r.status_code == 503

    def test_request_link_sends_email_with_token_url(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        captured = {}
        def fake_send(to, subject, body, **kw):
            captured["to"] = to; captured["body"] = body
        monkeypatch.setattr(m.mailer, "send", fake_send)
        r = client.post("/api/account/request-link", json={"contact": "User@Example.com"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert captured["to"] == "user@example.com"   # нормализуем регистр
        assert "/account/verify?token=" in captured["body"]

    def test_request_link_surfaces_mailer_error(self, monkeypatch):
        import app.main as m
        from app.mailer import MailerError
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        def failing(to, subject, body, **kw):
            raise MailerError("Не получилось отправить письмо. Попробуйте ещё раз через минуту.")
        monkeypatch.setattr(m.mailer, "send", failing)
        r = client.post("/api/account/request-link", json={"contact": "user@example.com"})
        assert r.status_code == 502

    def _issue_session(self, monkeypatch, contact):
        """Создаёт magic-link токен напрямую в БД и проходит верификацию --
        короче, чем гонять письмо через инъекцию ради одного токена."""
        import app.main as m
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_" + contact, contact=contact))
            s.commit()
        r = client.post(f"/account/verify?token=tok_{contact}", follow_redirects=False)
        assert r.status_code in (302, 303, 307)
        return r.cookies.get("sozdatel_session")

    def test_verify_rejects_unknown_token(self):
        r = client.get("/account/verify?token=does-not-exist", follow_redirects=False)
        assert r.status_code == 400

    def test_verify_rejects_reused_token(self, monkeypatch):
        session_token = self._issue_session(monkeypatch, "reuse@example.com")
        assert session_token
        # тот же токен второй раз -- уже использован
        r = client.post("/account/verify?token=tok_reuse@example.com", follow_redirects=False)
        assert r.status_code == 400
        client.cookies.clear()

    def test_verify_rejects_expired_token(self, monkeypatch):
        import app.main as m
        from app.main import MagicLinkToken, Session, engine, utcnow
        from datetime import timedelta
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_expired", contact="old@example.com",
                                 created_at=utcnow() - timedelta(minutes=m.MAGIC_LINK_TTL_MINUTES + 1)))
            s.commit()
        r = client.get("/account/verify?token=tok_expired", follow_redirects=False)
        assert r.status_code == 400

    def test_me_without_cookie_returns_no_contact(self):
        client.cookies.clear()
        r = client.get("/api/account/me")
        assert r.status_code == 200
        d = r.json()
        assert d["contact"] is None and d["projects"] == [] and d["reports"] == []

    def test_me_lists_only_this_contacts_projects_and_paid_reports(self, monkeypatch):
        import app.main as m
        from app.main import SmokeProject, ReportPurchase, Session, engine
        contact = "cabinet_test@example.com"
        with Session(engine) as s:
            s.add(SmokeProject(idea_id="cab_proj_v1", product_name="КабинетТест",
                               idea_text="т", offer_json="{}", landing_html="<title></title>",
                               contact=contact))
            s.add(SmokeProject(idea_id="cab_proj_other_v1", product_name="ЧужойПроект",
                               idea_text="т", offer_json="{}", landing_html="<title></title>",
                               contact="other@example.com"))
            s.add(ReportPurchase(idea="идея с отчётом", tier="full", contact=contact,
                                 status="paid", check_id=None))
            s.add(ReportPurchase(idea="неоплаченный", tier="quick", contact=contact,
                                 status="pending_payment", check_id=None))
            s.commit()

        session_token = self._issue_session(monkeypatch, contact)
        client.cookies.set("sozdatel_session", session_token)
        r = client.get("/api/account/me")
        client.cookies.clear()
        assert r.status_code == 200
        d = r.json()
        assert d["contact"] == contact
        assert [p["idea_id"] for p in d["projects"]] == ["cab_proj_v1"]
        # Оба отчёта, не только оплаченный -- незавершённая покупка не должна
        # пропадать из кабинета, человек мог просто закрыть вкладку с оплатой.
        assert len(d["reports"]) == 2
        by_idea = {r["idea"]: r["status"] for r in d["reports"]}
        assert by_idea["идея с отчётом"] == "paid"
        assert by_idea["неоплаченный"] == "pending_payment"
        # Та же карточка, что видит владелец в /api/cabinet -- этап, вердикт,
        # прогресс, а не голая ссылка (по фидбеку: "какой-то ты страшненький
        # кабинет сделал", нужны те же карты, что в /desk).
        proj = d["projects"][0]
        for key in ("stage", "stage_name", "views", "leads", "rate", "target", "verdict", "next_step", "progress"):
            assert key in proj, f"в карточке проекта личного кабинета нет поля {key}"

    def test_me_shows_pending_live_test_orders_not_yet_launched(self, monkeypatch):
        """Заявка на живой тест без запущенного проекта (idea_id пуст) должна
        быть видна в кабинете -- иначе человек, начавший что-то, но не
        доведший до конца, теряет к этому доступ."""
        import app.main as m
        from app.main import DemandCheck, LiveTestOrder, Session, engine
        contact = "pending_order@example.com"
        with Session(engine) as s:
            # Проверка НЕ привязана к кабинету: человек заказал живой тест,
            # не сохраняя её. Ровно этот случай и ломался -- ссылка вела на
            # порядковый номер, который для него уже закрыт (E6), и кабинет
            # отдавал 404 на его же заявке.
            src = DemandCheck(idea="идея без запуска",
                              result_json='{"verdict": {"level": "niche", "text": "т"}}')
            s.add(src); s.commit(); s.refresh(src)
            s.add(LiveTestOrder(idea="идея без запуска", contact=contact,
                                status="pending_payment", check_id=src.id))
            s.commit()
            pid = src.public_id
        session_token = self._issue_session(monkeypatch, contact)
        client.cookies.set("sozdatel_session", session_token)
        r = client.get("/api/account/me")
        d = r.json()
        assert len(d["orders"]) == 1
        assert d["orders"][0]["idea"] == "идея без запуска"
        assert d["orders"][0]["status"] == "pending_payment"
        assert d["orders"][0]["continue_url"] == f"/r/{pid}"
        # и ссылка действительно открывается, а не ведёт в 404
        assert client.get(d["orders"][0]["continue_url"]).status_code == 200
        client.cookies.clear()

    def test_me_hides_launched_order_from_orders_list(self, monkeypatch):
        """Заявка уже стала проектом (idea_id проставлен) -- показывается
        один раз как карточка проекта, не дублируется в orders."""
        import app.main as m
        from app.main import LiveTestOrder, SmokeProject, Session, engine
        contact = "already_launched@example.com"
        with Session(engine) as s:
            s.add(SmokeProject(idea_id="already_launched_v1", product_name="Уже",
                               idea_text="т", offer_json="{}", landing_html="<title></title>",
                               contact=contact))
            s.add(LiveTestOrder(idea="и", contact=contact, status="paid", idea_id="already_launched_v1"))
            s.commit()
        session_token = self._issue_session(monkeypatch, contact)
        client.cookies.set("sozdatel_session", session_token)
        d = client.get("/api/account/me").json()
        client.cookies.clear()
        assert len(d["orders"]) == 0
        assert [p["idea_id"] for p in d["projects"]] == ["already_launched_v1"]

    def test_pending_payment_expires_after_timeout(self):
        """Брошенная оплата (закрыл вкладку, не оплатил) не должна вечно
        висеть "ожидает оплаты" -- владелец предложил таймаут."""
        import app.main as m
        from datetime import timedelta
        fresh = m._effective_status("pending_payment", m.utcnow())
        stale = m._effective_status("pending_payment", m.utcnow() - timedelta(minutes=m.PENDING_PAYMENT_TIMEOUT_MINUTES + 1))
        assert fresh == "pending_payment"
        assert stale == "expired"
        # paid/new не истекают -- таймаут применим только к незавершённой оплате
        assert m._effective_status("paid", m.utcnow() - timedelta(days=30)) == "paid"

    def test_orders_endpoint_shows_expired_and_launched_state(self, monkeypatch):
        """Владелец в /desk должен видеть, что заказ уже запущен (ссылка на
        проект) или истёк (не "ожидает оплаты" бесконечно)."""
        from app.main import LiveTestOrder, Session, engine, utcnow
        from datetime import timedelta
        with Session(engine) as s:
            s.add(LiveTestOrder(idea="запущенный", contact="a@example.com", status="paid",
                                idea_id="orders_launched_v1"))
            s.add(LiveTestOrder(idea="протух", contact="b@example.com", status="pending_payment",
                                created_at=utcnow() - timedelta(minutes=999)))
            s.commit()
        d = client.get("/api/orders", headers=OWNER).json()
        by_idea = {o["idea"]: o for o in d["orders"]}
        assert by_idea["запущенный"]["project_url"] == "/p/orders_launched_v1"
        assert by_idea["протух"]["status"] == "expired"

    def test_logout_clears_cookie(self):
        r = client.post("/api/account/logout")
        assert r.status_code == 200
        assert r.cookies.get("sozdatel_session") is None

    def test_account_page_loads(self):
        r = client.get("/account")
        assert r.status_code == 200
        assert "Личный" in r.text

    def test_owner_can_attach_contact_to_project(self):
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="contact_attach_v1", product_name="ПрикреплениеКонтакта")})
        r = client.patch("/api/projects/contact_attach_v1/contact", headers=OWNER,
                         json={"contact": "attached@example.com"})
        assert r.status_code == 200
        from app.main import SmokeProject, Session, engine, select
        with Session(engine) as s:
            proj = s.exec(select(SmokeProject).where(SmokeProject.idea_id == "contact_attach_v1")).first()
            assert proj.contact == "attached@example.com"

    def test_attach_contact_requires_owner_key(self):
        r = client.patch("/api/projects/whatever/contact", json={"contact": "x@example.com"})
        assert r.status_code == 401


class TestSaveCheckToAccount:
    """Бесплатная проверка спроса без ссылки на кабинет не должна теряться --
    можно привязать её к контакту постфактум (POST /api/demand/{id}/save),
    либо она привязывается сама, если проверку делает уже вошедший."""

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
            r = client.post("/api/demand", json={"idea": "Идея достаточно длинная для сохранения"})
            return r.json()["id"]
        finally:
            m.check_demand = orig

    def _issue_session(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_save_" + contact, contact=contact))
            s.commit()
        r = client.post(f"/account/verify?token=tok_save_{contact}", follow_redirects=False)
        assert r.status_code in (302, 303, 307)
        return r.cookies.get("sozdatel_session")

    def test_result_page_exposes_saved_flag(self):
        client.cookies.clear()
        rid = self._make_check()
        assert "const SAVED = false;" in client.get(f"/r/{pub(rid)}").text

    def test_save_requires_valid_email_when_anonymous(self):
        client.cookies.clear()
        rid = self._make_check()
        r = client.post(f"/api/demand/{rid}/save", json={"contact": "@telegram"})
        assert r.status_code == 400

    def test_save_unknown_check_404(self):
        client.cookies.clear()
        r = client.post("/api/demand/999999/save", json={"contact": "user@example.com"})
        assert r.status_code == 404

    def test_save_cannot_hijack_check_already_claimed_by_another_contact(self, monkeypatch):
        """check_id -- обычный автоинкремент, легко перебрать (/r/1, /r/2, ...).
        Без проверки владения кто угодно мог бы молча переприсвоить себе уже
        сохранённую чужую проверку и увидеть чужую идею в своём /account."""
        import app.main as m
        from app.main import DemandCheck, Session, engine
        monkeypatch.setattr(m.mailer, "configured", lambda: False)
        client.cookies.clear()
        rid = self._make_check()
        first = client.post(f"/api/demand/{rid}/save", json={"contact": "owner@example.com"})
        assert first.status_code == 200

        hijack = client.post(f"/api/demand/{rid}/save", json={"contact": "stranger@example.com"})
        assert hijack.status_code == 409

        with Session(engine) as s:
            rec = s.get(DemandCheck, rid)
            assert rec.contact == "owner@example.com"   # не перезаписалось

    def test_save_same_contact_again_is_idempotent(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.mailer, "configured", lambda: False)
        client.cookies.clear()
        rid = self._make_check()
        assert client.post(f"/api/demand/{rid}/save", json={"contact": "owner@example.com"}).status_code == 200
        again = client.post(f"/api/demand/{rid}/save", json={"contact": "Owner@Example.com"})
        assert again.status_code == 200   # тот же контакт (регистр не важен) -- не конфликт

    def test_logged_in_session_cannot_hijack_check_claimed_by_another_contact(self):
        client.cookies.clear()
        rid = self._make_check()
        assert client.post(f"/api/demand/{rid}/save", json={"contact": "owner@example.com"}).status_code == 200
        token = self._issue_session("stranger@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            r = client.post(f"/api/demand/{rid}/save", json={"contact": ""})
            assert r.status_code == 409
        finally:
            client.cookies.clear()

    def test_save_anonymous_sends_magic_link_and_persists_contact(self, monkeypatch):
        import app.main as m
        from app.main import DemandCheck, Session, engine
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        captured = {}
        def fake_send(to, subject, body, **kw):
            captured["to"] = to; captured["body"] = body
        monkeypatch.setattr(m.mailer, "send", fake_send)
        client.cookies.clear()
        rid = self._make_check()
        r = client.post(f"/api/demand/{rid}/save", json={"contact": "Saver@Example.com"})
        assert r.status_code == 200 and r.json()["ok"] is True
        assert captured["to"] == "saver@example.com"
        assert "/account/verify?token=" in captured["body"]
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid)
            assert rec.contact == "saver@example.com"

    def test_save_while_logged_in_uses_session_contact_instantly(self):
        client.cookies.clear()
        rid = self._make_check()
        token = self._issue_session("loggedin@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            r = client.post(f"/api/demand/{rid}/save", json={"contact": ""})
            assert r.status_code == 200
            assert r.json()["message"] == "Сохранено в кабинете."
        finally:
            client.cookies.clear()
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid)
            assert rec.contact == "loggedin@example.com"

    def test_demand_check_auto_attaches_contact_for_logged_in_user(self):
        client.cookies.clear()
        token = self._issue_session("autosave@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            rid = self._make_check()
        finally:
            client.cookies.clear()
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid)
            assert rec.contact == "autosave@example.com"

    def test_account_me_lists_saved_checks_not_promoted_to_report_or_order(self, monkeypatch):
        import app.main as m
        from app.main import DemandCheck, ReportPurchase, Session, engine
        client.cookies.clear()
        token = self._issue_session("checklist@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            rid_plain = self._make_check()
            rid_promoted = self._make_check()
            with Session(engine) as s:
                s.add(ReportPurchase(check_id=rid_promoted, idea="уже выросло в отчёт",
                                     tier="quick", contact="checklist@example.com",
                                     status="paid", amount=990))
                s.commit()
            d = client.get("/api/account/me").json()
        finally:
            client.cookies.clear()
        check_ids = [c["id"] for c in d["checks"]]
        assert rid_plain in check_ids
        assert rid_promoted not in check_ids   # уже виден в reports, не дублируем

    def test_result_page_has_save_button_wiring(self):
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert 'id="save-btn"' in text and "/save" in text and "trySave" in text

    def test_account_page_renders_checks_section(self):
        """Раздел собирается в JS (пустые не выводятся) -- проверяем, что
        сохранённые проверки в этой сборке участвуют."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "['Проверки спроса', d.checks, checkRow]" in text


class TestProjectPageCustomerAccess:
    """/p/{id} стало доступно из кабинета покупателя (см. _smoke_card в
    /api/account/me) -- цифры проекта не должны требовать секретный ключ
    владельца у обычного покупателя, только у самого владельца."""

    def _issue_session(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_proj_" + contact, contact=contact))
            s.commit()
        r = client.post(f"/account/verify?token=tok_proj_{contact}", follow_redirects=False)
        assert r.status_code in (302, 303, 307)
        return r.cookies.get("sozdatel_session")

    def test_owner_key_still_works(self):
        client.cookies.clear()
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="pacc_owner")})
        assert client.get("/api/verdict/pacc_owner", headers=OWNER).status_code == 200
        assert client.get("/api/series/pacc_owner", headers=OWNER).status_code == 200

    def test_customer_session_grants_access_to_own_project(self):
        client.cookies.clear()
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="pacc_mine")})
        client.patch("/api/projects/pacc_mine/contact", headers=OWNER,
                     json={"contact": "owner_of_project@example.com"})
        token = self._issue_session("owner_of_project@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            assert client.get("/api/verdict/pacc_mine").status_code == 200
            assert client.get("/api/series/pacc_mine").status_code == 200
        finally:
            client.cookies.clear()

    def test_customer_session_denied_for_someone_elses_project(self):
        client.cookies.clear()
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="pacc_other")})
        client.patch("/api/projects/pacc_other/contact", headers=OWNER,
                     json={"contact": "real_owner@example.com"})
        token = self._issue_session("stranger@example.com")
        client.cookies.set("sozdatel_session", token)
        try:
            assert client.get("/api/verdict/pacc_other").status_code == 401
        finally:
            client.cookies.clear()

    def test_no_key_no_session_still_401(self):
        client.cookies.clear()
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="pacc_anon")})
        assert client.get("/api/verdict/pacc_anon").status_code == 401
        assert client.get("/api/series/pacc_anon").status_code == 401

    def test_unknown_project_404_regardless_of_key(self):
        assert client.get("/api/series/pacc_does_not_exist", headers=OWNER).status_code == 404

    def test_project_page_tries_session_before_prompting_for_owner_key(self):
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        assert "authedFetch" in text
        assert "authedFetch(`/api/verdict/${IDEA_ID}`)" in text
        assert "authedFetch(`/api/series/${IDEA_ID}`)" in text


class TestAutoLaunchUiWiring:
    """Статическая проверка, что новые состояния заказа (запущено само /
    нужно запустить вручную) отражены в JS /desk и /account, а не только
    в API -- иначе владелец видит те же данные, что и раньше."""

    def test_desk_orders_js_handles_project_url_and_manual_fallback(self):
        text = (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()
        assert "o.project_url" in text
        assert "заострение пропустили" in text.lower()

    def test_account_page_renders_pending_orders_block(self):
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "d.orders, orderRow]" in text
        assert "continue_url" in text


class TestCustomerDevPass:
    """Правки по критическому проходу владельца как customer developer перед
    запуском рекламы: порядок финального CTA, длина плейсхолдеров на мобильном,
    /p/{id} без тупика в owner-only /desk, публичная навигация не ведёт в /desk."""

    def test_live_test_cta_comes_before_report_alt_path(self):
        """"Дальше" -- основной путь, должен идти раньше своей альтернативы
        ("Или получите отчёт"), иначе "или" читается раньше того, к чему оно
        относится."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "тест", "count": 100}],
                    "verdict": {"level": "strong", "text": "Спрос есть"},
                    "competitors": {"found": 10, "top": [{"title": "Т", "domain": "t.ru"}]},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 7, "note": ""}],
                    "overall": {"value": 7, "weakest": ""}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея достаточно длинная для проверки порядка CTA"}).json()["id"]
        finally:
            m.check_demand = orig
        text = client.get(f"/r/{pub(rid)}").text
        assert text.index('id="order"') < text.index('class="alt-path"')

    def test_contact_placeholders_short_enough_for_mobile(self):
        """Длинные плейсхолдеры («Телеграм/почта — на них пришлём чек и
        результат») визуально обрезались в узком инпуте на мобильном --
        держим короче того, что реально помещается."""
        import re
        for path, static_name in (("static/result.html", "result.html"),
                                   ("static/report.html", "report.html"),
                                   ("static/account.html", "account.html")):
            text = (main_module.BASE_DIR.parent / path).read_text()
            for m_ in re.finditer(r'placeholder="([^"]*)"', text):
                assert len(m_.group(1)) <= 30, f"плейсхолдер длиннее 30 символов в {static_name}: {m_.group(1)!r}"

    def test_project_page_has_no_dead_end_to_owner_desk(self):
        """/p/{id} публичный (не за owner-key) -- личный кабинет покупателя
        теперь на него ссылается, поэтому свои же "Кабинет →"/"← портфель"
        не должны вести в /desk, куда у покупателя нет доступа."""
        client.post("/api/launch", headers=OWNER, json={"idea_text": "т",
            "offer": dict(VALID_OFFER, idea_id="no_deadend_v1", product_name="БезТупика")})
        text = client.get("/p/no_deadend_v1").text
        assert 'href="/desk"' not in text

    def test_homepage_cabinet_link_points_to_customer_account(self):
        home = client.get("/").text
        assert 'href="/account"' in home
        assert 'href="/desk"' not in home

    def test_ad_setup_copy_does_not_overpromise_managed_service(self):
        """Владелец подтвердил: рекламу пока никому не настраивают -- клиент
        запускает Директ сам по инструкции. "Мы... сами запустим рекламу"
        обещало управляемую услугу, которой нет; текст не должен утверждать,
        что Создатель лично запускает кампанию."""
        home = client.get("/").text
        assert "Запускаем Яндекс Директ" not in home

        rid = TestReportFlow()._make_check()
        result_text = client.get(f"/r/{pub(rid)}").text
        assert "сами запустим рекламу" not in result_text

    def test_missing_frequency_is_not_called_a_finding(self):
        """Прежнее решение здесь ОТМЕНЕНО, и вот почему.

        Когда-то «нет данных» рядом с частотностью читалось как «сайт
        сломан», и по фидбеку это заменили на «почти не ищут» -- на
        формулировку-вывод о рынке. Но `count = None` означает «оба пути
        Вордстата не дали числа» (так и написано в докстринге
        `wordstat_best`), а не «спроса нет»: это вывод, которого мы не
        вправе делать (принцип 1). Хуже того, при неработающем Вордстате
        КАЖДЫЙ посетитель видел три строки «почти не ищут» и уходил
        хоронить живую идею.

        Исходная жалоба решается не враньём в другую сторону, а словами:
        подпись «не удалось измерить» говорит о нашем замере, а не о рынке,
        и вывода за Вордстат не делает. Плюс рядом есть вердикт и блок
        «Проверка не состоялась», которых тогда не было.

        Отдельно (2026-08-02): измеренный НОЛЬ больше не попадает в эту
        ветку вовсе -- он приходит как `count = 0` и подписывается «не
        ищут». Это честный ответ Вордстата, а не сбой, и раньше он
        показывался тем же текстом, что и несостоявшийся замер.
        """
        result_text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "не удалось измерить" in result_text
        # вывод про рынок на месте отсутствующего числа больше не появляется
        assert "почти не ищут" not in result_text.split("v.level === 'weak'")[0]
        report_text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        # в отчёте фраза осталась только там, где спрос ПОДТВЕРЖДЁННО слабый
        assert "verdict_level === 'weak'" in report_text

    def test_report_has_single_pricing_block_not_duplicated(self):
        """Цены дублировались сверху и снизу отчёта -- по фидбеку нижний
        блок оказался лишним после того, как верхний уже решает задачу
        "не листать до конца, чтобы увидеть цену"."""
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert text.count('id="pricing') == 1   # только pricing-top, без второго id="pricing"

    def test_report_preview_suppresses_weak_reformulate_advice(self):
        """Вердикт "почти не ищут... попробуйте переформулировать" на /r/
        уместен как шаг воронки, но на странице отчёта после того, как идею
        уже заострили, звучит как отказ от уже принятого решения купить."""
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert "p.verdict_level === 'weak'" in text

    def test_alt_path_report_button_is_ink_not_ghost(self):
        """"Посмотреть отчёт" был btn-ghost -- по фидбеку поднят до полного
        чернильного веса, отчёт не второсортная опция."""
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert '<a class="btn" href="/report/__PUBLIC_ID__">Посмотреть отчёт</a>' in text


class TestFunnelCopyClarity:
    """Кастдев-фидбек: «живой тест» на главной непонятен без контекста, шаги
    6-8 звучат как обещание того, чего продукт ещё не делает, «отчёт» не
    доносит ценность так, как «бизнес-план»."""

    def test_homepage_step_3_4_no_longer_say_bare_live_test(self):
        home = client.get("/").text
        assert "тест на реальных людях" in home.lower()
        assert '<span class="tag paid-tag">живой тест</span>' not in home
        assert "счётчиком событий" not in home   # жаргон снят

    def test_homepage_roadmap_steps_tagged_as_future(self):
        home = client.get("/").text
        assert home.count("в Создателе 2.0") == 3   # шаги 5, 6, 7 (после слияния 3+4)
        assert "скрипт разговора" not in home        # продукт этого не делает

    def test_homepage_mentions_business_plan_alt_path(self):
        home = client.get("/").text
        assert "бизнес-план" in home.lower()

    def test_full_report_tier_relabeled_business_plan(self):
        import app.main as m
        assert m.REPORT_PRICES["full"]["label"] == "Бизнес-план"


class TestStageMerge:
    """«Проверочная страница» + «Реклама» объединены в «Тест на реальных
    людях» -- один шаг вместо двух непонятных читалось запутанно (кастдев-
    фидбек). Слияние затрагивает ТОЛЬКО шкалу Создателя (STAGE_NAMES,
    SmokeProject) -- TrackedProject (внешние проекты владельца, например
    АвтоПост) использует отдельную неизменную шкалу из 8 названий, иначе
    существующие в БД записи стали бы указывать не на те этапы после деплоя
    (а stage=7 упал бы по IndexError на укороченном массиве)."""

    def test_customer_scale_has_7_stages_merged(self):
        import app.main as m
        assert len(m.STAGE_NAMES) == 7
        assert m.STAGE_NAMES == ["Идея", "Спрос", "Тест на реальных людях",
                                  "Заявки", "Первые продажи", "Повторяемость", "Удержание"]

    def test_tracked_scale_untouched_at_8_stages(self):
        import app.main as m
        assert len(m.TRACKED_STAGE_NAMES) == 8
        assert m.TRACKED_STAGE_NAMES[2] == "Проверочная страница"
        assert m.TRACKED_STAGE_NAMES[3] == "Реклама"

    def test_existing_tracked_project_at_old_stage_7_still_resolves(self):
        """До слияния «Удержание» было индексом 7 -- на укороченной customer-
        шкале это вышло бы за границы массива. Внешние проекты не должны
        сломаться после деплоя этого изменения."""
        r = client.post("/api/tracked", headers=OWNER,
                         json={"name": "Внешний проект на удержании", "stage": 7})
        assert r.status_code == 200
        tp_id = r.json()["id"]
        try:
            cab = client.get("/api/cabinet", headers=OWNER).json()
            tracked = [t for t in cab["tracked"] if t["id"] == tp_id][0]
            assert tracked["stage_name"] == "Удержание"
            assert len(cab["stages"]) == 8
        finally:
            client.delete(f"/api/tracked/{tp_id}", headers=OWNER)

    def test_project_page_uses_merged_7_stage_scale(self):
        text = (main_module.BASE_DIR.parent / "static" / "project.html").read_text()
        assert "Тест на реальных людях" in text
        # длина шкалы и сам этап приходят с сервера, а не зашиты на странице:
        # копия правила уже разъезжалась и показывала оплатившему «Идея»
        assert "names.length" in text
        assert "d.stage_names" in text
        assert "d.views > 0 ? 1 : 0" not in text
        assert "Реклама" not in text   # старое отдельное название шага ушло

    def test_desk_stage_badge_parameterized_by_scale_length(self):
        """/desk смешивает в одной сетке smoke-проекты Создателя (7 шагов) и
        внешние tracked-проекты (8 шагов) -- общий stageBadge() должен брать
        длину шкалы параметром, а не хардкодить одно число на двоих."""
        text = (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()
        assert "stageBadge(s.stage, s.stage_name, 7)" in text
        assert "stageBadge(t.stage, t.stage_name, 8)" in text

    def test_report_engine_stage_names_match_merged_scale(self):
        from app.report_engine import STAGE_NAMES as report_stage_names
        import app.main as m
        assert report_stage_names == m.STAGE_NAMES


class TestChosenOfferReachesReport:
    """A6: человек выбирает на /r/ одну из трёх заострённых формулировок, а
    платный разбор до этой правки строился по сырой первой фразе. Выбор жил
    только на заказе живого теста; отчёт заказывают с /report/{check_id},
    который про заказ ничего не знает."""

    OFFER = {"angle": "Для занятых родителей", "h1": "Шторы <em>за неделю</em>",
             "sub": "Пошив на дому", "eyebrow": "Родители 30-45",
             "pains": [{"h2": "Долго ждать", "p": "Ателье шьют месяц"}]}

    def _make_check(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig

    def _pay_report(self, rid, contact):
        from app.main import ReportPurchase, Session, engine, select
        client.post("/api/report", json={"check_id": rid, "tier": "full", "contact": contact})
        with Session(engine) as s:
            order = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)).first()
            order.status = "paid"; s.add(order); s.commit()
            return order.access_token      # ссылка покупателя на свой отчёт

    def test_choice_is_stored_on_the_check(self):
        from app.main import DemandCheck, Session, engine
        rid = self._make_check()
        r = client.post(f"/api/demand/{rid}/chosen", json={"offer": self.OFFER})
        assert r.status_code == 200 and r.json()["ok"] is True
        with Session(engine) as s:
            stored = json.loads(s.get(DemandCheck, rid).chosen_offer)
        assert stored["h1"] == self.OFFER["h1"]
        assert stored["pains"][0]["h2"] == "Долго ждать"   # оффер целиком, не только заголовок

    def test_missing_check_is_404(self):
        assert client.post("/api/demand/999777/chosen", json={"offer": self.OFFER}).status_code == 404

    def test_choice_reaches_report_generation(self, monkeypatch):
        """Главное звено всей задачи."""
        import app.main as m
        rid = self._make_check()
        client.post(f"/api/demand/{rid}/chosen", json={"offer": self.OFFER})
        tok = self._pay_report(rid, f"chosen{rid}@example.com")
        seen = {}
        async def fake_generate(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            seen["offer"] = chosen_offer
            return {"sections": [{"key": "summary", "title": "Резюме проекта", "body": "текст"}]}
        monkeypatch.setattr(m, "generate_core", fake_generate)
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert seen["offer"] is not None
        assert seen["offer"]["h1"] == self.OFFER["h1"]

    def test_report_without_choice_still_generates(self, monkeypatch):
        """Заострение необязательно -- его можно пропустить. Отчёт обязан
        собраться и без выбора, просто по исходной идее."""
        import app.main as m
        rid = self._make_check()
        tok = self._pay_report(rid, f"nochoice{rid}@example.com")
        seen = {}
        async def fake_generate(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            seen["offer"] = chosen_offer
            return {"sections": [{"key": "summary", "title": "Резюме проекта", "body": "текст"}]}
        monkeypatch.setattr(m, "generate_core", fake_generate)
        assert client.get(f"/report/{pub(rid)}?t={tok}").status_code == 200
        assert seen["offer"] is None

    def test_broken_json_does_not_break_paid_report(self, monkeypatch):
        """Битая запись не имеет права сорвать оплаченную услугу."""
        import app.main as m
        from app.main import DemandCheck, Session, engine
        rid = self._make_check()
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid); rec.chosen_offer = "{не json"; s.add(rec); s.commit()
        tok = self._pay_report(rid, f"broken{rid}@example.com")
        async def fake_generate(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            assert chosen_offer is None
            return {"sections": [{"key": "summary", "title": "Резюме проекта", "body": "текст"}]}
        monkeypatch.setattr(m, "generate_core", fake_generate)
        assert client.get(f"/report/{pub(rid)}?t={tok}").status_code == 200

    def test_saved_check_of_another_cabinet_is_protected(self):
        """id проверок перебираются, а выбор влияет на платный отчёт."""
        from app.main import DemandCheck, Session, engine
        rid = self._make_check()
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid); rec.contact = "owner@example.com"; s.add(rec); s.commit()
        r = client.post(f"/api/demand/{rid}/chosen", json={"offer": self.OFFER})
        assert r.status_code == 409

    def test_report_page_shows_the_chosen_wording(self):
        """Иначе человек не может понять, учли его выбор или нет."""
        rid = self._make_check()
        client.post(f"/api/demand/{rid}/chosen", json={"offer": self.OFFER})
        text = client.get(f"/report/{pub(rid)}").text
        assert "Разбираем формулировку" in text
        assert "Шторы за неделю" in text      # разметка <em> снята, слот не сломан
        assert "__CHOSEN_BLOCK__" not in text

    def test_report_page_without_choice_has_no_empty_block(self):
        rid = self._make_check()
        text = client.get(f"/report/{pub(rid)}").text
        assert "Разбираем формулировку" not in text
        assert "__CHOSEN_BLOCK__" not in text

    def test_result_page_sends_the_choice(self):
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "'/api/demand/' + CHECK_ID + '/chosen'" in text


class TestAccountFirstEntry:
    """B1+B2: пустой кабинет был тупиком. Человек подтверждал почту, попадал
    на экран из двух пустых заголовков подряд («Пока нет запущенных
    проектов.», «Пока нет отчётов по идее.») и не понимал, что делать."""

    def _login(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_fe_" + contact, contact=contact)); s.commit()
        r = client.post(f"/account/verify?token=tok_fe_{contact}", follow_redirects=False)
        assert r.status_code in (302, 303, 307)

    def _check(self, contact, purpose="business"):
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            rec = DemandCheck(idea="Пошив штор на заказ", contact=contact, purpose=purpose,
                              result_json='{"verdict": {"level": "niche", "text": "т"}}')
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id

    def test_empty_cabinet_has_no_dead_end_texts(self):
        """Формулировки-тупики не должны вернуться ни в каком виде."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "Пока нет запущенных проектов" not in text
        assert "Пока нет отчётов по идее" not in text

    def test_empty_cabinet_offers_one_action(self):
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert 'id="first-cta"' in text
        assert "Проверить идею — бесплатно" in text
        # пустые разделы не рендерятся вовсе
        assert ".filter(([, items]) => items.length)" in text

    def test_empty_cabinet_reports_nothing(self):
        client.cookies.clear()
        self._login("empty_fe@example.com")
        d = client.get("/api/account/me").json()
        assert d["contact"] == "empty_fe@example.com"
        assert not any([d["projects"], d["reports"], d["orders"], d["checks"]])
        client.cookies.clear()

    def test_purpose_follows_the_last_check(self):
        """Получателя соцконтракта нельзя вести за следующей идеей на витрину
        для фаундеров -- у него другая задача (принцип 4)."""
        client.cookies.clear()
        self._login("soc_fe@example.com")
        self._check("soc_fe@example.com", "social_contract")
        assert client.get("/api/account/me").json()["purpose"] == "social_contract"
        client.cookies.clear()

    def test_purpose_defaults_to_business(self):
        client.cookies.clear()
        self._login("biz_fe@example.com")
        self._check("biz_fe@example.com")
        assert client.get("/api/account/me").json()["purpose"] == "business"
        client.cookies.clear()

    def test_purpose_of_empty_cabinet_is_business(self):
        client.cookies.clear()
        self._login("noidea_fe@example.com")
        assert client.get("/api/account/me").json()["purpose"] == "business"
        client.cookies.clear()

    def test_cabinet_routes_back_to_its_own_landing_server_driven(self):
        """Раньше адрес был хардкодом в account.html ("social_contract или /"),
        и третья аудитория (студент) в него не попадала -- утекала на витрину
        для фаундеров. Сервер обязан отдавать готовый адрес, а кабинет -- не
        знать про purpose вовсе (та же дыра, что F1 закрыла в других файлах)."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "d.home" in text
        assert "social_contract" not in text

    def test_home_url_for_each_audience(self):
        client.cookies.clear()
        for purpose, expected in (("business", "/"), ("social_contract", "/social-contract"),
                                  ("student", "/students")):
            contact = f"home_{purpose}@example.com"
            self._login(contact)
            self._check(contact, purpose)
            assert client.get("/api/account/me").json()["home"] == expected, purpose
            client.cookies.clear()

    def test_tier_label_comes_from_server(self):
        """Тариф уже переименовывали («Полный отчёт» -> «Бизнес-план»), и
        вторая копия названия в статике разъехалась с витриной незаметно."""
        import app.main as m
        client.cookies.clear()
        self._login("tier_fe@example.com")
        rid = self._check("tier_fe@example.com")
        client.post("/api/report", json={"check_id": rid, "tier": "full",
                                          "contact": "tier_fe@example.com"})
        reports = client.get("/api/account/me").json()["reports"]
        assert reports[0]["tier_label"] == m.REPORT_PRICES["full"]["label"]
        client.cookies.clear()
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "Полный отчёт" not in text      # копии названия в статике больше нет
        assert "rp.tier_label" in text


    def test_buttons_are_not_underlined(self):
        """Ссылки-кнопки в кабинете были подчёркнуты, хотя на всех остальных
        страницах text-decoration снят. На экране первого входа это единственная
        кнопка, и подчёркивание сразу читается как небрежность."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        btn = text.split(".btn{")[1].split("}")[0]
        ghost = text.split(".btn-ghost{")[1].split("}")[0]
        assert "text-decoration:none" in btn
        assert "text-decoration:none" in ghost

    def test_rows_stack_on_narrow_screen(self):
        """На 390px название идеи занимает три строки и выдавливает кнопку
        «Открыть →» в столбик из двух слов."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        narrow = text.split("@media (max-width:560px){")[1].split("\n  }")[0]
        assert ".item{flex-direction:column" in narrow

    def test_login_screen_does_not_require_a_paid_order(self):
        """Вход в кабинет открыт и после бесплатной проверки -- обещать его
        только оформившим заказ значит отсечь половину вошедших."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "на которую оформляли заказ" not in text
        assert "Пароль не нужен" in text


class TestStartupPrewarmListStaysInSyncWithDisk:
    """_lifespan прогревает список статических страниц на старте, а ошибка
    поймана и только залогирована (принцип 7) -- сервис не падает, если файла
    больше нет. Из-за этого переименование/удаление файла проходит МОЛЧА: так
    и было с social-contract.html, который F2 заменила на audience-landing.html,
    а имя в списке прогрева осталось старым. Прод не падал, но каждый рестарт
    писал в лог исключение, а сама audience-landing.html (обслуживает и
    /social-contract, и /students -- обе новые витрины под рекламу) прогревом
    вообще не была покрыта."""

    def test_every_prewarmed_name_exists_on_disk(self):
        import app.main as m
        missing = [name for name in m.PREWARM_STATIC_PAGES
                  if not (main_module.BASE_DIR.parent / "static" / name).exists()]
        assert not missing, missing


class TestNumbersExplainThemselves:
    """B3: числа показывались без объяснения, а на витринах стояли ДРУГИЕ
    числа, чем считал движок. Человек с 3% читал на главной «идея живая», а
    в кабинете видел «СПРОСА НЕТ»."""

    def test_showcases_quote_the_real_thresholds(self):
        """Главное: витрина и движок не могут больше разъехаться."""
        import app.main as m
        for url in ("/", "/guide/direct"):
            text = client.get(url).text
            assert m._pct(m.SIGNAL_RATE) in text, url
            assert m._pct(m.DEAD_RATE) in text, url
            assert str(m.CLICK_TARGET) in text, url
            assert "__SIGNAL_PCT__" not in text and "__CLICK_TARGET__" not in text, url

    def test_old_wrong_numbers_are_gone(self):
        """2,5% и ~100 визитов движок не использовал никогда."""
        for url in ("/", "/guide/direct"):
            text = client.get(url).text
            assert "2,5%" not in text, url
            # «~100 визитов» осталось в оценке рекламного бюджета -- это про
            # другое; уходит именно ложное правило вердикта.
            assert "дождитесь ~100 визитов" not in text, url

    def test_pct_formats_like_russian(self):
        from app.main import _pct
        assert _pct(0.08) == "8%"
        assert _pct(0.04) == "4%"
        assert _pct(0.025) == "2,5%"
        assert _pct(0.125) == "12,5%"

    def test_verdict_names_the_threshold_it_compared_against(self):
        """Голое «12% — сигнал есть» не объясняет, с чем сравнили."""
        from app.main import compute_verdict, _pct, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE
        early = compute_verdict(10, 1, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)
        assert str(CLICK_TARGET) in early["detail"]
        signal = compute_verdict(50, 6, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)
        assert _pct(SIGNAL_RATE) in signal["detail"]
        dead = compute_verdict(50, 1, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)
        assert _pct(DEAD_RATE) in dead["detail"]
        gray = compute_verdict(100, 3, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)
        assert _pct(DEAD_RATE) in gray["detail"] and _pct(SIGNAL_RATE) in gray["detail"]

    def test_verdict_has_no_forbidden_words_or_owner_language(self):
        """Этот вердикт видит покупатель в /account и на /p/, а тест на
        запрещённые слова покрывал только demand._verdict -- вторую, очень
        похожую функцию никто не проверял, и в ней жил «оффер»."""
        from app.main import compute_verdict, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE
        cases = [(10, 1), (50, 6), (50, 1), (50, 3), (0, 0)]
        for views, leads in cases:
            low = compute_verdict(views, leads, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)["detail"].lower()
            for bad in ("оффер", "лендинг", "трафик", "mvp", "конверси", "гипотез"):
                assert bad not in low, f"жаргон в вердикте {views}/{leads}: {bad!r}"

    def test_verdict_speaks_to_the_customer_not_the_owner(self):
        """«Копим клики», «идею в архив» — язык владельца пульта, а вердикт
        читает покупатель (A3 закрыла то же самое на /p/)."""
        from app.main import compute_verdict, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE
        for views, leads in [(10, 1), (50, 6), (50, 1), (50, 3)]:
            d = compute_verdict(views, leads, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)["detail"]
            assert "Копим клики" not in d and "в архив" not in d
            assert "менять" not in d or "не меняйте" in d.lower()

    def test_zero_views_does_not_divide_by_zero(self):
        from app.main import compute_verdict
        assert compute_verdict(0, 0, 40, .08, .04)["verdict"] == "РАНО СУДИТЬ"

    def test_model_defaults_come_from_the_constants(self):
        """Иначе появится третья копия чисел -- в схеме БД."""
        import app.main as m
        p = m.SmokeProject(idea_id="x", product_name="p", idea_text="t",
                           offer_json="{}", landing_html="")
        assert p.click_target == m.CLICK_TARGET
        assert p.lead_rate_signal == m.SIGNAL_RATE
        assert p.lead_rate_dead == m.DEAD_RATE


class TestOverallScoreExplained:
    """B3: «6/10» без объяснения. Владелец на живом прогоне сам спросил, как
    при почти нулевом спросе идея получает 6/10 -- правило «спрос это ворота»
    без слов читается как ошибка счёта."""

    def _overall(self, demand_value, others):
        """Собирает overall тем же кодом, что и check_demand."""
        import asyncio, app.demand as dm
        async def fake_score(idea, rows, comp, *, _post=None):
            return [{"key": k, "label": l, "value": v, "note": ""}
                    for (k, l), v in zip([("competition", "Конкуренция"),
                                          ("timing", "Своевременность"),
                                          ("execution", "Реализуемость")], others)]
        async def fake_formulations(idea, *, _post=None):
            return ["фраза"]
        async def fake_best(phrase, *, _post=None):
            return {"phrase": phrase, "count": demand_value}
        async def fake_comp(phrase, *, _post=None):
            return {"found": 10, "top": []}
        orig = (dm.score_idea, dm.generate_formulations, dm.wordstat_best, dm.competitors)
        dm.score_idea, dm.generate_formulations = fake_score, fake_formulations
        dm.wordstat_best, dm.competitors = fake_best, fake_comp
        try:
            return asyncio.run(dm.check_demand("идея"))["overall"]
        finally:
            dm.score_idea, dm.generate_formulations, dm.wordstat_best, dm.competitors = orig

    def test_capped_by_demand_says_so(self):
        """Спрос 1 запрос/мес при трёх хороших шкалах."""
        ov = self._overall(1, [9, 9, 9])
        assert "опущен до балла спроса" in ov["basis"]
        assert ov["value"] <= 2

    def test_not_capped_says_it_is_an_average(self):
        ov = self._overall(9000, [3, 3, 3])
        assert "Среднее по четырём шкалам" in ov["basis"]
        assert "опущен" not in ov["basis"]

    def test_basis_avoids_internal_jargon(self):
        for ov in (self._overall(1, [9, 9, 9]), self._overall(9000, [3, 3, 3])):
            low = ov["basis"].lower()
            for bad in ("оффер", "лендинг", "трафик", "частотность", "конверси"):
                assert bad not in low, f"{bad!r} в объяснении балла"

    def test_result_page_renders_the_explanation(self):
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "ov.basis" in text and 'id="overall-how"' in text


    def test_numbers_agree_with_their_nouns(self):
        """«1 заявок» — мелочь, по которой сразу видно машинный текст."""
        from app.main import _plural, compute_verdict, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE
        assert _plural(1, "заявка", "заявки", "заявок") == "заявка"
        assert _plural(3, "заявка", "заявки", "заявок") == "заявки"
        assert _plural(5, "заявка", "заявки", "заявок") == "заявок"
        assert _plural(11, "заявка", "заявки", "заявок") == "заявок"   # 11, а не 1
        assert _plural(21, "заявка", "заявки", "заявок") == "заявка"
        assert _plural(112, "заявка", "заявки", "заявок") == "заявок"  # 112, а не 2
        assert "1 заявка" in compute_verdict(50, 1, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)["detail"]
        assert "1 визит " in compute_verdict(1, 0, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)["detail"]
        assert "52 визита" in compute_verdict(52, 2, CLICK_TARGET, SIGNAL_RATE, DEAD_RATE)["detail"]


class TestWeOnlyPromiseWhatWeDo:
    """A7: воронка обещала, что рекламу запустим мы, хотя оферта прямо
    говорит обратное — «запуск рекламы осуществляется Пользователем
    самостоятельно». Расхождение витрины с договором на пути к оплате — это
    возвраты и споры на холодном рекламном трафике."""

    def _check(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig

    def test_oferta_still_says_the_user_launches_ads(self):
        """Предпосылка всего теста: договор говорит именно так."""
        text = client.get("/oferta").text
        assert "Запуск рекламы осуществляется Пользователем самостоятельно" in text
        assert "Рекламный бюджет не входит в стоимость услуги" in text

    def test_funnel_does_not_promise_we_launch_the_ads(self):
        rid = self._check()
        text = client.get(f"/r/{pub(rid)}").text
        assert "запустим первую рекламу" not in text
        assert "Мы запускаем живой тест" not in text
        # и говорит, кто именно запускает
        assert "рекламу вы запустите сами" in text

    def test_order_confirmation_does_not_promise_ads(self, monkeypatch):
        """Сообщение после заявки без оплаты обещало «запустим рекламу»."""
        import app.main as m
        monkeypatch.setattr(m.payments, "configured", lambda: False)
        rid = self._check()
        r = client.post("/api/live-test", json={"check_id": rid, "contact": "a7@example.com"})
        msg = r.json()["message"]
        assert "запустим страницу и рекламу" not in msg
        assert "рекламу вы запустите сами" in msg

    def test_ad_budget_is_disclosed_before_payment(self):
        """Главное: человек узнаёт про отдельный рекламный бюджет ДО оплаты,
        а не после. Он больше самой услуги."""
        import app.main as m
        rid = self._check()
        text = client.get(f"/r/{pub(rid)}").text
        assert "Рекламный бюджет в цену не входит" in text
        assert m.AD_BUDGET_HINT in text
        assert "__AD_BUDGET__" not in text

    def test_ad_budget_figure_has_one_source(self):
        """Иначе цифра в плейбуке и цифра у кнопки разъедутся -- ровно то,
        что уже случилось с порогами вердикта (B3)."""
        import app.main as m
        guide = client.get("/guide/direct").text
        assert m.AD_BUDGET_HINT in guide
        assert "__AD_BUDGET__" not in guide
        static_src = (main_module.BASE_DIR.parent / "static" / "guide-direct.html").read_text()
        assert "3–5 тысяч" not in static_src   # зашитой копии в статике нет

    def test_paid_note_points_where_the_page_actually_appears(self):
        """«Вернёмся с первыми цифрами» -- мы никуда не возвращаемся, человек
        смотрит их сам в кабинете."""
        rid = self._check()
        text = client.get(f"/r/{pub(rid)}").text
        assert "вернёмся с первыми цифрами" not in text.lower()
        assert 'href="/account"' in text


def _flatten_digit_groups(text: str) -> str:
    """«1 490 ₽» -> «1490 ₽»: убирает пробел-разделитель разрядов между цифрами.
    Нужен и стражу цен, и его собственному тесту."""
    return re.sub(r"(?<=\d)[  ](?=\d)", "", text)


class TestNoHardcodedServerValuesInStatic:
    """B5: значение, у которого в коде есть единственный источник, не должно
    лежать второй копией в HTML. Это уже трижды оборачивалось враньём:
    кабинет звал тариф «Полный отчёт» против «Бизнес-плана» на витрине;
    главная обещала порог 2,5% при реальных 8%; цифра рекламного бюджета
    жила только в плейбуке. Каждый раз находилось глазами, а не тестом."""

    STATIC = main_module.BASE_DIR.parent / "static"

    def _sources(self):
        """Все шаблоны, которые видит человек, — включая те, что лежат НЕ в
        `static/`.

        Проверочная страница собирается из `app/landing_template.html`, и
        именно этот файл трижды подряд проваливался мимо сторожей, которые
        смотрели только `static/`: слово «лендинг» (A14), запрещённые слова в
        строке следующего шага (A17, тот же слепой угол в `app/main.py`) и
        рендер-блокирующий запрос шрифтов на чужой домен (A18) — последнее на
        единственной странице, куда идёт платный трафик.
        """
        for p in sorted(self.STATIC.glob("*.html")):
            yield p.name, p.read_text()
        landing = main_module.BASE_DIR / "landing_template.html"
        yield landing.name, landing.read_text(encoding="utf-8")

    def test_prices_are_not_hardcoded(self):
        """Цена на витрине обязана совпадать с той, что спишется."""
        import app.main as m
        amounts = {str(m.LIVE_TEST_PRICE)}
        for tier in m.REPORT_PRICES.values():
            amounts |= {str(tier["price"]), str(tier["was"])}
        bad = []
        for name, text in self._sources():
            # Сначала схлопываем разделители разрядов: «1 490 ₽» -> «1490 ₽».
            # Без этого сторож пропускал цену в оферте, записанную по-человечески.
            flat = _flatten_digit_groups(text)
            for amount in amounts:
                # «990 ₽» -- денежная запись, случайных совпадений не даёт,
                # в отличие от голого числа. Проверка (?<!\d) не даёт поймать
                # «990» внутри «2990».
                if re.search(rf"(?<!\d){amount}\s*₽", flat):
                    bad.append(f"{name}: {amount} ₽")
        assert not bad, ("суммы зашиты в статику вместо подстановки из "
                         "REPORT_PRICES/LIVE_TEST_PRICE: " + ", ".join(bad))

    def test_tier_labels_are_not_hardcoded_as_tier_references(self):
        """Название тарифа в кавычках-ёлочках -- это ссылка на тариф, а не
        обычное слово. Заголовок страницы «Бизнес-план для социального
        контракта» -- название услуги, его не трогаем."""
        import app.main as m
        labels = [t["label"] for t in m.REPORT_PRICES.values()]
        bad = []
        for name, text in self._sources():
            for label in labels:
                if f"«{label}»" in text:
                    bad.append(f"{name}: «{label}»")
        assert not bad, ("названия тарифов зашиты в статику вместо подстановки "
                         "из REPORT_PRICES: " + ", ".join(bad))

    def test_verdict_thresholds_are_not_hardcoded(self):
        """Порог сбора данных стоит в том числе в оферте -- это условие
        договора, оно обязано совпадать с движком."""
        import app.main as m
        bad = []
        for name, text in self._sources():
            for pct in (m._pct(m.SIGNAL_RATE), m._pct(m.DEAD_RATE)):
                # без границы слева «8%» ловится внутри «58%» из CSS-градиентов
                if re.search(rf"(?<![\d,]){re.escape(pct)}", text):
                    bad.append(f"{name}: {pct}")
            if re.search(rf"(?<!\d){m.CLICK_TARGET}\s+(визит|посещен)", text):
                bad.append(f"{name}: {m.CLICK_TARGET} визитов")
        assert not bad, "пороги вердикта зашиты в статику: " + ", ".join(bad)

    def test_ad_budget_is_not_hardcoded(self):
        import app.main as m
        bad = [name for name, text in self._sources() if m.AD_BUDGET_HINT in text]
        assert not bad, "рекламный бюджет зашит в статику: " + ", ".join(bad)

    def test_every_slot_used_in_static_is_filled_by_the_server(self):
        """Обратная защита: слот, который никто не подставляет, доедет до
        человека как «__FULL_LABEL__» прямо на экране."""
        import app.main as m
        used = set()
        for _, text in self._sources():
            used |= set(re.findall(r"__[A-Z][A-Z0-9_]*__", text))
        # слоты страниц подставляются в своих обработчиках, а не в _fill_server_values
        per_page = {
            "__CHECK_ID__", "__PRICE__", "__PAY_ENABLED__", "__IDEA__", "__IDEA_JSON__",
            "__RESULT_JSON__", "__SAVED__", "__PURPOSE_JSON__", "__CHOSEN_BLOCK__",
            "__PREVIEW_JSON__", "__REPORT_JSON__", "__UNLOCKED_TIER__", "__ORDER_STATUS__",
            "__GEN_ERROR__", "__PRICES_JSON__", "__SECTIONS_JSON__", "__QUICK_KEYS_JSON__",
            "__OWNER_BAR__", "__TIER_KEYS_JSON__", "__SAMPLE_JSON__", "__ACCESS_NOTE__",
            "__PUBLIC_ID__",
            "__PRODUCT_NAME__", "__IDEA_ID__", "__H1__", "__SUB__", "__EYEBROW__",
            "__PAINS__", "__CTA__", "__FORM_NOTE__",
            # витрина аудитории -- заполняет _audience_landing
            "__PAGE_TITLE__", "__META__", "__FIELD_LABEL__", "__PLACEHOLDER__",
            "__PROMISE_TITLE__", "__PROMISE_SUB__", "__PROMISES__",
            "__QUICK_NOTE__", "__FULL_NOTE__", "__FAQ__", "__AUDIENCE_KEY__",
            "__FAST_PLAN_BTN__",
            # страница результата -- audiences.for_page / состояние проверки
            "__AUDIENCE_JSON__", "__RESUME__", "__CHOSEN_H1_JSON__",
            # страница подтверждения входа -- заполняет _verify_page
            "__HEADING__", "__LEAD__", "__WHO__", "__ACTION__", "__FINE__",
            # заголовок и выходные данные листа -- _doc_title_and_meta
            "__DOC_TITLE__", "__DOC_META__",
        }
        filled = set(re.findall(r'\("(__[A-Z0-9_]+__)"',
                                inspect.getsource(m._fill_server_values)))
        unknown = used - filled - per_page
        assert not unknown, f"слоты никем не подставляются: {sorted(unknown)}"

    def test_pages_render_without_leftover_slots(self):
        """Ни один слот не должен доехать до браузера."""
        import app.main as m
        rid = None
        async def fake_check(idea):
            return {"formulations": [{"phrase": "п", "count": 10}], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"}, "competitors": {"found": 1, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig
        for url in ("/", "/social-contract", "/guide/direct", "/oferta",
                    f"/r/{pub(rid)}", f"/report/{pub(rid)}", "/account"):
            text = client.get(url).text
            assert not re.search(r"__[A-Z][A-Z0-9_]*__", text), f"незаполненный слот на {url}"


class TestOwnerReportPreview:
    """Владелец должен уметь посмотреть, что человек получает за 2990 ₽, не
    оплачивая заказ себе самому. Промпты правились вслепую (E1), а качество
    платного отчёта — это весь платный продукт."""

    def _check(self, purpose="business"):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ",
                                                    "purpose": purpose}).json()["id"]
        finally:
            m.check_demand = orig

    def _stub_generate(self, monkeypatch, seen=None):
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            if seen is not None:
                seen.update({"tier": tier, "purpose": purpose, "idea": idea})
            return {"viability_score": 62,
                    "viability_summary": "Разбор идеи по данным проверки.",
                    "top_risks": [{"title": "Риск", "body": "Объяснение."}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            if seen is not None:
                seen.update({"tier": tier, "purpose": purpose, "idea": idea})
            return {"key": key, "title": "Раздел", "body": "Разбор идеи по данным проверки."}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

    def test_owner_gets_the_full_plan_without_paying(self, monkeypatch):
        seen = {}
        self._stub_generate(monkeypatch, seen)
        rid = self._check()
        r = client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        assert r.status_code == 200
        assert seen["tier"] == "full"           # собрался именно платный тариф
        assert "Разбор идеи по данным проверки." in r.text

    def test_preview_uses_the_same_optics_as_a_real_purchase(self, monkeypatch):
        """Иначе владелец проверит не тот продукт, который продаёт."""
        seen = {}
        self._stub_generate(monkeypatch, seen)
        rid = self._check("social_contract")
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        assert seen["purpose"] == "social_contract"

    def test_preview_does_not_unlock_the_report_for_anyone_else(self, monkeypatch):
        """Главный риск: владелец прогнал чужую проверку — и её автор получил
        бизнес-план за 2990 ₽ даром."""
        self._stub_generate(monkeypatch)
        rid = self._check()
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        text = client.get(f"/report/{pub(rid)}").text          # посторонний, без ключа
        # Бесплатный образец (балл + один раздел) посторонний видит — это
        # витрина. А вот платный разбор обязан остаться запертым.
        assert "const UNLOCKED_TIER = null;" in text
        assert "const TIER_KEYS = [];" in text
        assert client.get(f"/api/report/{rid}/status").json()["paid"] is False

    def test_preview_requires_the_owner_key(self, monkeypatch):
        from app.main import ReportPurchase, Session, engine, select
        self._stub_generate(monkeypatch)
        rid = self._check()
        client.get(f"/report/{pub(rid)}?preview=full")          # без ключа
        with Session(engine) as s:
            rows = s.exec(select(ReportPurchase).where(ReportPurchase.check_id == rid)).all()
        assert rows == []

    def test_preview_is_not_counted_as_an_order(self, monkeypatch):
        """Нулевая покупка не должна выглядеть как продажа или как
        неоплаченная заявка в /desk."""
        self._stub_generate(monkeypatch)
        rid = self._check()
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        reports = client.get("/api/orders", headers=OWNER).json()["reports"]
        assert all(r["check_id"] != rid for r in reports if "check_id" in r)
        assert all(rp["report_url"] != f"/report/{pub(rid)}" for rp in reports)

    def test_preview_is_not_regenerated_on_every_open(self, monkeypatch):
        """Каждый прогон стоит вызова модели — второй заход должен брать
        уже собранное."""
        import app.main as m
        calls = []
        async def fake_generate(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            calls.append(tier)
            return {"sections": [{"key": "summary", "title": "Резюме проекта", "body": "т"}]}
        monkeypatch.setattr(m, "generate_core", fake_generate)
        rid = self._check()
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)
        assert len(calls) == 1

    def test_unknown_tier_is_ignored(self, monkeypatch):
        from app.main import ReportPurchase, Session, engine, select
        self._stub_generate(monkeypatch)
        rid = self._check()
        client.get(f"/report/{pub(rid)}?preview=../../etc/passwd", headers=OWNER)
        with Session(engine) as s:
            rows = s.exec(select(ReportPurchase).where(ReportPurchase.check_id == rid)).all()
        assert rows == []

    def test_owner_bar_is_invisible_to_everyone_else(self):
        rid = self._check()
        # проверяем отрисованный блок, а не описание класса в CSS
        assert 'class="owner-bar"' not in client.get(f"/report/{pub(rid)}").text
        assert 'class="owner-bar"' in client.get(f"/report/{pub(rid)}?key=test-owner-key").text

    def test_failed_preview_does_not_email_the_owner(self, monkeypatch):
        """Письмо про сорванную доставку — про оплаченный заказ. Свой же
        прогон владельцу писать не надо."""
        import app.main as m
        sent = []
        async def boom(idea, demand_data, tier="full", chosen_offer=None, purpose="business", **kw):
            raise m.ReportEngineError("модель недоступна")
        monkeypatch.setattr(m, "generate_core", boom)
        monkeypatch.setattr(m.mailer, "notify_owner", lambda *a, **k: sent.append(a) or True)
        rid = self._check()
        assert client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER).status_code == 200
        assert sent == []


class TestFontsAreServedFromOurOwnHost:
    """B6: шрифты грузились с fonts.googleapis.com рендер-блокирующим тегом.
    Из России домен часто недоступен, а такой тег держит отрисовку всей
    страницы: человек видел белый экран до сетевого таймаута, а не «просто
    другой шрифт». Принцип 8 — всё работает из России."""

    STATIC = main_module.BASE_DIR.parent / "static"

    def test_no_page_reaches_out_to_google(self):
        bad = [p.name for p in sorted(self.STATIC.glob("*.html"))
               if "fonts.googleapis.com" in p.read_text() or "fonts.gstatic.com" in p.read_text()]
        assert not bad, "страницы всё ещё тянут шрифты у Google: " + ", ".join(bad)

    def test_every_page_links_the_local_stylesheet(self):
        """Пропустить страницу — значит оставить её без фирменного шрифта."""
        bad = [p.name for p in sorted(self.STATIC.glob("*.html"))
               if '/fonts/fonts.css' not in p.read_text()]
        assert not bad, "страницы без локального шрифта: " + ", ".join(bad)

    def test_stylesheet_is_served_and_points_only_at_us(self):
        r = client.get("/fonts/fonts.css")
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/css")
        assert "https://" not in r.text          # ни одной внешней ссылки внутри
        assert "/fonts/IBMPlexSans-400-cyrillic.woff2" in r.text

    def test_every_font_referenced_by_the_stylesheet_exists(self):
        """Опечатка в имени файла = молча пропавший шрифт на всём сайте."""
        css = client.get("/fonts/fonts.css").text
        names = set(re.findall(r"url\(/fonts/([^)]+)\)", css))
        assert names, "в стилях не осталось ни одного шрифта"
        for name in sorted(names):
            r = client.get(f"/fonts/{name}")
            assert r.status_code == 200, f"{name} не отдаётся"
            assert r.headers["content-type"] == "font/woff2"

    def test_cyrillic_is_covered(self):
        """Сайт русский: без кириллического подмножества фирменный шрифт не
        применится вовсе, и это заметят все."""
        css = client.get("/fonts/fonts.css").text
        assert "cyrillic" in css
        assert re.search(r"unicode-range:[^;]*U\+0400-045F", css)

    def test_font_route_does_not_serve_anything_else(self):
        """Роут отдаёт файлы с диска — он не должен превращаться в способ
        читать что попало, включая HTML-шаблоны с неподставленными слотами."""
        for name in ("../index.html", "..%2Findex.html", "index.html",
                     "../../app/main.py", "fonts.css/../../index.html"):
            assert client.get(f"/fonts/{name}").status_code in (403, 404), name

    def test_fonts_are_cached_hard(self):
        """Файлы неизменяемые: имя меняется вместе с содержимым."""
        r = client.get("/fonts/fonts.css")
        assert "immutable" in r.headers.get("cache-control", "")


class TestPublicReportExample:
    """C1: примера отчёта не существовало нигде. Кастдев-находка владельца —
    «без этого доверия не будет»: человек платит 990–2990 ₽, не видя ни
    строчки того, что получит."""

    def _check(self, purpose="business"):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ",
                                                    "purpose": purpose}).json()["id"]
        finally:
            m.check_demand = orig

    def _built_report(self, monkeypatch, body="Смета расходов построчно.", purpose="business"):
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            return {"viability_score": 62, "viability_summary": "Ядро отчёта.",
                    "top_risks": [{"title": "Риск", "body": "Объяснение."}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            return {"key": key, "title": "Резюме", "body": body}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)
        rid = self._check(purpose)
        client.get(f"/report/{pub(rid)}?preview=full", headers=OWNER)   # ядро
        client.post(f"/api/report/{rid}/section?key=summary", headers=OWNER)   # раздел
        return rid

    def _clear_examples(self):
        from app.main import ReportPurchase, Session, engine, select
        with Session(engine) as s:
            for row in s.exec(select(ReportPurchase).where(ReportPurchase.is_example == True)).all():  # noqa: E712
                row.is_example = False; s.add(row)
            s.commit()

    def test_example_page_is_absent_until_published(self):
        """Пустая витрина лучше, чем ссылка в никуда."""
        self._clear_examples()
        assert client.get("/example").status_code == 404

    def test_showcases_do_not_promise_an_example_that_does_not_exist(self):
        self._clear_examples()
        rid = self._check()
        for url in (f"/r/{pub(rid)}", "/social-contract"):
            text = client.get(url).text
            assert "Посмотреть пример отчёта" not in text, url
            assert "__EXAMPLE_LINK__" not in text, url

    def test_published_example_is_open_in_full(self, monkeypatch):
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        r = client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        assert r.status_code == 200 and r.json()["url"] == "/example"
        page = client.get("/example")
        assert page.status_code == 200
        assert "Смета расходов построчно." in page.text          # текст виден без оплаты
        assert "Это настоящий отчёт, собранный сервисом" in page.text

    def test_example_page_has_no_leaked_template_placeholders(self, monkeypatch):
        """Найдено живым кастдев-прогоном: `example_page` собирала HTML из
        того же шаблона `report.html`, что и `/report/{id}`, но забыла два
        `.replace(...)` из длинной цепочки -- `__DOC_TITLE__`/`__DOC_META__`
        (заголовок и строка даты для печати, см. `_doc_title_and_meta`).
        `/report/{id}` их всегда подставляет, а `/example` -- нет, и сырые
        токены шаблона утекали прямо в `<h1>` самой важной для доверия
        публичной страницы (см. C1 в PRODUCT_ROADMAP: без примера человек
        платит 990-2990 ₽, не видя ни строчки того, что получит — а увидел
        бы буквально `__DOC_TITLE__`)."""
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        page = client.get("/example").text
        assert "__DOC_TITLE__" not in page
        assert "__DOC_META__" not in page

    def test_showcases_link_the_example_once_it_exists(self, monkeypatch):
        """Пример собран для фаундера (business) -- ссылка появляется там,
        где сейчас смотрят тем же взглядом: /r/ на такой же проверке."""
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        other = self._check()
        assert 'href="/example"' in client.get(f"/r/{pub(other)}").text

    def test_examples_do_not_leak_across_audiences(self, monkeypatch):
        """Пример собран под одну оптику (business) -- показывать его как
        «вот что вы получите» соцконтракту или студенту обещает не ту оптику,
        что мы реально отдадим по их заявке (принцип 4). Раньше пример был
        виден на любой витрине, у которой есть слот __EXAMPLE_LINK__."""
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        assert 'href="/example"' not in client.get("/social-contract").text
        assert 'href="/example"' not in client.get("/students").text
        soc_check = self._check("social_contract")
        assert 'href="/example"' not in client.get(f"/r/{pub(soc_check)}").text

    def test_example_link_appears_for_its_own_audience(self, monkeypatch):
        """Пример, собранный под соцконтракт, виден именно на витрине
        соцконтракта -- а не фаундера или студента."""
        self._clear_examples()
        rid = self._built_report(monkeypatch, purpose="social_contract")
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        assert 'href="/example"' in client.get("/social-contract").text
        assert 'href="/example"' not in client.get("/students").text
        biz_check = self._check()
        assert 'href="/example"' not in client.get(f"/r/{pub(biz_check)}").text

    def test_only_the_owner_can_publish(self, monkeypatch):
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        assert client.post(f"/api/example/publish?check_id={rid}&tier=full").status_code == 401
        assert client.get("/example").status_code == 404

    def test_cannot_publish_a_report_that_was_never_built(self):
        """Иначе на витрину уедет пустая страница."""
        self._clear_examples()
        rid = self._check()
        r = client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        assert r.status_code == 404
        assert client.get("/example").status_code == 404

    def test_example_is_exactly_one(self, monkeypatch):
        """Два «примера» разъехались бы так же, как разъезжались копии цен."""
        from app.main import ReportPurchase, Session, engine, select
        self._clear_examples()
        first = self._built_report(monkeypatch, "Первый разбор.")
        client.post(f"/api/example/publish?check_id={first}&tier=full", headers=OWNER)
        second = self._built_report(monkeypatch, "Второй разбор.")
        client.post(f"/api/example/publish?check_id={second}&tier=full", headers=OWNER)
        with Session(engine) as s:
            marked = s.exec(select(ReportPurchase).where(ReportPurchase.is_example == True)).all()  # noqa: E712
        assert len(marked) == 1
        assert "Второй разбор." in client.get("/example").text

    def test_example_says_which_tier_it_is(self, monkeypatch):
        """Иначе человек решит, что за 990 ₽ получит то же самое."""
        import app.main as m
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        assert m.REPORT_PRICES["full"]["label"] in client.get("/example").text

    def test_example_page_leaks_no_owner_controls(self, monkeypatch):
        self._clear_examples()
        rid = self._built_report(monkeypatch)
        client.post(f"/api/example/publish?check_id={rid}&tier=full", headers=OWNER)
        text = client.get("/example").text
        assert 'class="owner-bar"' not in text
        assert "preview=full" not in text


class TestTierDifferenceAtTheDecisionPoint:
    """C2: на `/r/` стояло только «от 990 ₽», а состав тарифов открывался лишь
    на следующем экране. Для пришедшего с /social-contract это ловушка: он
    идёт за обоснованием сметы, а секции «Финансовая модель» в дешёвом тарифе
    нет вовсе."""

    def _check(self, purpose="business"):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ",
                                                    "purpose": purpose}).json()["id"]
        finally:
            m.check_demand = orig

    def test_both_tiers_named_with_prices_on_the_result_page(self):
        import app.main as m
        text = client.get(f"/r/{pub(self._check())}").text
        for tier in m.REPORT_PRICES.values():
            assert tier["label"] in text
            assert f"{tier['price']} ₽" in text
        assert "__TIER_SUMMARY__" not in text

    def test_cheap_tier_lists_exactly_what_it_contains(self):
        """Состав берётся из движка, а не пишется руками."""
        from app.report_engine import ALL_SECTIONS, QUICK_KEYS
        text = client.get(f"/r/{pub(self._check())}").text
        for key, title in ALL_SECTIONS:
            if key in QUICK_KEYS:
                assert title in text, f"секция дешёвого тарифа не названа: {title}"

    def test_finance_is_visibly_absent_from_the_cheap_tier(self):
        """Главная ловушка соцконтракта: смета есть только в полном тарифе."""
        from app.report_engine import ALL_SECTIONS, QUICK_KEYS
        finance_title = dict(ALL_SECTIONS)["finance"]
        assert "finance" not in QUICK_KEYS          # предпосылка теста
        summary = main_module._tier_summary_html()
        quick_part, full_part = summary.split("</div><div class=\"tier-row\">")
        assert finance_title.lower() not in quick_part.lower()
        assert finance_title.lower() in full_part.lower()

    def test_summary_follows_the_engine_not_a_copy(self, monkeypatch):
        """Если в движке появится новая секция, витрина обязана её назвать.

        Название теперь читается через section_title() (SECTION_SPECS), а не
        напрямую из ALL_SECTIONS — иначе аудиторный заголовок секции (finance
        для соцконтракта) не попал бы на витрину. Патчим оба источника."""
        import app.main as m
        import app.report_engine as re
        new_spec = {"key": "newthing", "group": "Идея и рынок", "title": "Новая секция"}
        monkeypatch.setattr(re, "SECTION_SPECS", list(re.SECTION_SPECS) + [new_spec])
        monkeypatch.setattr(m, "ALL_SECTIONS",
                            list(m.ALL_SECTIONS) + [("newthing", "Новая секция")])
        # в полном тарифе секции перечисляются со строчной буквы
        assert "новая секция" in m._tier_summary_html().lower()

    def test_social_contract_sees_the_same_breakdown(self):
        """Блок меняет роль на главный — состав тарифов должен уехать с ним."""
        import app.main as m
        text = client.get(f"/r/{pub(self._check('social_contract'))}").text
        assert m.REPORT_PRICES["full"]["label"] in text
        assert f"{m.REPORT_PRICES['full']['price']} ₽" in text
        assert 'id="alt-report"' in text and "__TIER_SUMMARY__" not in text

    def test_tier_breakdown_names_sections_the_way_this_audience_will_see_them(self):
        """Найдено кастдев-проходом по платному пути 2026-07-29: витрина
        показывала «Финансовая модель» всем подряд, хотя у соцконтракта
        та же секция в самом отчёте после оплаты называется «Смета и расчёты
        для комиссии» (SECTION_SPECS.by_audience). Человек ищет глазами
        «смету», видит «Финансовую модель» и не узнаёт в ней то, за чем
        пришёл — витрина и отчёт должны называть секцию одинаково."""
        import app.main as m
        from app.report_engine import section_title
        biz_text = client.get(f"/r/{pub(self._check('business'))}").text.lower()
        soc_text = client.get(f"/r/{pub(self._check('social_contract'))}").text.lower()
        biz_title = section_title("finance", "business").lower()
        soc_title = section_title("finance", "social_contract").lower()
        assert biz_title != soc_title          # предпосылка теста
        assert biz_title in biz_text
        assert soc_title in soc_text
        assert soc_title not in biz_text

    def test_prices_are_still_not_hardcoded_in_static(self):
        """C2 не должна протащить обратно то, что закрыла B5."""
        src = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "990 ₽" not in src and "2990 ₽" not in src
        assert "__TIER_SUMMARY__" in src


class TestOfferCoversWhatWeSell:
    """C3: политика возврата не сформулирована. При разборе выяснилось, что
    дело шире: вся оферта описывала ТОЛЬКО живой тест за 1490 ₽. Отчёт за
    990/2990 ₽ — тот самый продукт под рекламу на соцконтракт — в договоре
    отсутствовал целиком: ни предмета, ни цены, ни сроков, ни возврата."""

    def _oferta(self):
        return client.get("/oferta").text

    def test_offer_names_every_paid_service(self):
        text = self._oferta()
        assert "Живой тест идеи" in text
        assert "Отчёт по идее" in text

    def test_offer_lists_every_price_we_charge(self):
        """Цена в договоре обязана совпадать с той, что списывается."""
        import app.main as m
        text = self._oferta()
        for price in (m.LIVE_TEST_PRICE, *(t["price"] for t in m.REPORT_PRICES.values())):
            assert f"{price} ₽" in text, f"цены {price} ₽ нет в оферте"
        for tier in m.REPORT_PRICES.values():
            assert tier["label"] in text

    def test_offer_prices_are_not_a_hardcoded_copy(self):
        """«1 490 ₽» лежало в договоре зашитой копией, и сторож B5 её
        пропускал: разряды разделены пробелом."""
        src = (main_module.BASE_DIR.parent / "static" / "oferta.html").read_text()
        assert "1 490" not in src and "1490" not in src
        assert "__LIVE_TEST_PRICE__" in src and "__FULL_PRICE__" in src

    def test_guard_now_catches_prices_with_spaced_thousands(self):
        """Регрессия на сам сторож: без этого он снова пропустит «1 490 ₽»."""
        import app.main as m
        pat = rf"(?<!\d){m.LIVE_TEST_PRICE}\s*₽"
        assert re.search(pat, _flatten_digit_groups("цена 1 490 ₽"))   # с пробелом
        assert re.search(pat, _flatten_digit_groups("цена 1490 ₽"))    # и без него
        assert not re.search(pat, _flatten_digit_groups("цена 21490 ₽"))  # не хвост числа
        # и «990» не ловится внутри «2 990»
        assert not re.search(r"(?<!\d)990\s*₽", _flatten_digit_groups("цена 2 990 ₽"))

    def test_offer_states_refund_for_the_report(self):
        text = self._oferta()
        assert "отчёт не был сформирован" in text
        assert "возвращается полностью" in text
        assert "26.1" in text

    def test_offer_states_when_the_report_appears(self):
        assert "формируется автоматически при первом открытии" in self._oferta()

    def test_offer_disclaims_guarantees_for_the_social_contract_audience(self):
        """Лендинг обещает цифры, которые «выдержат вопросы комиссии» —
        одобрение выплаты мы гарантировать не можем и не должны."""
        text = self._oferta()
        assert "не является гарантией дохода" in text
        assert "социальной защиты" in text

    def test_refund_terms_are_visible_where_people_pay(self):
        """В оферту по своей воле не заходят: условия должны стоять у кнопки."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "п", "count": 10}], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"}, "competitors": {"found": 1, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig
        for url in (f"/r/{pub(rid)}", f"/report/{pub(rid)}"):
            text = client.get(url).text
            assert "вернём деньги полностью" in text, url
            assert 'href="/oferta"' in text, url


class TestReportBuildsProgressively:
    """Разделов больше двух десятков, и каждый — свой вызов модели. Собирать
    их все внутри HTTP-запроса значит держать человека на белом экране
    минутами: страница отдаётся с ядром, разделы подтягиваются по одному."""

    def _paid_check(self, monkeypatch, tier="full"):
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        async def fake_check(idea):
            return {"formulations": [{"phrase": "п", "count": 10}], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"},
                    "competitors": {"found": 1, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig
        contact = f"prog{rid}@example.com"
        client.post("/api/report", json={"check_id": rid, "tier": tier, "contact": contact})
        with Session(engine) as s:
            o = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)).first()
            o.status = "paid"; s.add(o); s.commit()
            tok = o.access_token
        # Токен -- ключ покупателя от своего отчёта: именно с ним его
        # возвращает оплата (см. _report_access_ok).
        return rid, tok

    def _stub(self, monkeypatch, calls=None):
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            if calls is not None:
                calls.append("core")
            return {"viability_score": 55, "viability_summary": "Ядро.",
                    "top_risks": [{"title": "Риск", "body": "Объяснение."}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            if calls is not None:
                calls.append(key)
            return {"key": key, "title": f"Раздел {key}", "body": f"Текст {key}."}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

    def test_page_does_not_wait_for_all_sections(self, monkeypatch):
        """Главное: открытие страницы стоит ОДНОГО вызова модели, не двадцати."""
        calls = []
        self._stub(monkeypatch, calls)
        rid, tok = self._paid_check(monkeypatch)
        assert client.get(f"/report/{pub(rid)}?t={tok}").status_code == 200
        assert calls == ["core"]

    def test_sections_arrive_one_by_one(self, monkeypatch):
        from app.report_engine import section_keys
        calls = []
        self._stub(monkeypatch, calls)
        rid, tok = self._paid_check(monkeypatch)
        client.get(f"/report/{pub(rid)}?t={tok}")
        for key in section_keys("full")[:3]:
            r = client.post(f"/api/report/{rid}/section?key={key}&t={tok}")
            assert r.status_code == 200, key
            assert r.json()["section"]["body"] == f"Текст {key}."
        assert calls == ["core"] + section_keys("full")[:3]

    def test_finished_section_is_not_regenerated(self, monkeypatch):
        """Каждый повтор — деньги за вызов модели."""
        calls = []
        self._stub(monkeypatch, calls)
        rid, tok = self._paid_check(monkeypatch)
        client.get(f"/report/{pub(rid)}?t={tok}")
        client.post(f"/api/report/{rid}/section?key=summary&t={tok}")
        again = client.post(f"/api/report/{rid}/section?key=summary&t={tok}")
        assert again.json()["cached"] is True
        assert calls.count("summary") == 1

    def test_sections_are_stored_in_reading_order(self, monkeypatch):
        """Дозаказ идёт как попало (перезагрузили вкладку в середине) — в
        сохранённом отчёте порядок обязан остаться читаемым."""
        import json as _json
        from app.main import ReportPurchase, Session, engine, select
        from app.report_engine import section_keys
        self._stub(monkeypatch)
        rid, tok = self._paid_check(monkeypatch)
        client.get(f"/report/{pub(rid)}?t={tok}")
        order = section_keys("full")
        for key in (order[3], order[0], order[1]):
            client.post(f"/api/report/{rid}/section?key={key}&t={tok}")
        with Session(engine) as s:
            row = s.exec(select(ReportPurchase).where(
                ReportPurchase.check_id == rid)).first()
            stored = [x["key"] for x in _json.loads(row.report_json)["sections"]]
        assert stored == [order[0], order[1], order[3]]

    def test_section_needs_a_paid_report(self, monkeypatch):
        """Иначе платные разделы забираются по одному без оплаты."""
        import app.main as m
        self._stub(monkeypatch)
        async def fake_check(idea):
            return {"formulations": [], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"},
                    "competitors": {"found": 1, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея без оплаты отчёта"}).json()["id"]
        finally:
            m.check_demand = orig
        assert client.post(f"/api/report/{rid}/section?key=summary").status_code == 403

    def test_cheap_tier_cannot_pull_full_tier_sections(self, monkeypatch):
        self._stub(monkeypatch)
        rid, tok = self._paid_check(monkeypatch, tier="quick")
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert client.post(f"/api/report/{rid}/section?key=finance&t={tok}").status_code == 404
        assert client.post(f"/api/report/{rid}/section?key=summary&t={tok}").status_code == 200

    def test_one_broken_section_does_not_break_the_report(self, monkeypatch):
        """Сбой одного раздела не должен выглядеть как сбой всего отчёта."""
        import app.main as m
        self._stub(monkeypatch)
        rid, tok = self._paid_check(monkeypatch)
        client.get(f"/report/{pub(rid)}?t={tok}")
        client.post(f"/api/report/{rid}/section?key=summary&t={tok}")
        async def boom(key, *a, **kw):
            raise m.ReportEngineError("модель недоступна")
        monkeypatch.setattr(m, "generate_section", boom)
        r = client.post(f"/api/report/{rid}/section?key=market&t={tok}")
        assert r.status_code == 502 and "недоступна" in r.json()["error"]
        # уже собранный раздел на месте
        assert "Текст summary." in client.get(f"/report/{pub(rid)}?t={tok}").text

    def test_locked_section_shows_its_question(self):
        """Запертый раздел продаёт вопросом, на который отвечает, а не
        общей фразой «полный разбор в отчёте»."""
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert "s.ask" in text
        assert "const TEASER" not in text          # общих описаний больше нет

    def test_page_shows_build_progress(self):
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert 'id="build-progress"' in text
        assert "Можно читать уже готовые разделы" in text


class TestFreeSampleSellsTheReport:
    """Бесплатная часть страницы отчёта была пересказом цифр со страницы
    спроса — ни строчки сгенерированного анализа. Человек, решающий отдать
    990–2990 ₽, не мог оценить качество того, что покупает. У dimeadozen
    бесплатно открыты полное резюме, балл и названные риски — это и есть
    продающий инструмент."""

    def _check(self, purpose="business"):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand", json={"idea": "Пошив штор на заказ",
                                                    "purpose": purpose}).json()["id"]
        finally:
            m.check_demand = orig

    def _stub(self, monkeypatch, calls=None):
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            if calls is not None:
                calls.append("core")
            return {"viability_score": 61, "viability_summary": "Ниша держится на рекомендациях.",
                    "top_risks": [{"title": "Заказы не повторяются",
                                   "body": "Шторы покупают раз в несколько лет."}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            if calls is not None:
                calls.append(key)
            return {"key": key, "title": "Резюме", "body": "Настоящий текст разбора."}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

    def test_unpaid_visitor_sees_real_analysis(self, monkeypatch):
        self._stub(monkeypatch)
        rid = self._check()
        text = client.get(f"/report/{pub(rid)}").text
        assert "Настоящий текст разбора." in text          # целый раздел
        assert "Ниша держится на рекомендациях." in text   # объяснение балла
        assert "Заказы не повторяются" in text             # названный риск
        assert "Часть разбора — бесплатно" in text

    def test_sample_is_built_once_and_cached(self, monkeypatch):
        """Два вызова модели на человека — за них платит владелец."""
        from app.main import DemandCheck, Session, engine
        calls = []
        self._stub(monkeypatch, calls)
        rid = self._check()
        client.get(f"/report/{pub(rid)}")
        assert calls == ["core", "summary"]
        client.get(f"/report/{pub(rid)}")
        assert calls == ["core", "summary"]                # второй визит бесплатен
        with Session(engine) as s:
            assert s.get(DemandCheck, rid).sample_json

    def test_sample_uses_the_audience_optics(self, monkeypatch):
        """Соцконтракту нельзя показывать венчурный разбор даже в образце."""
        seen = {}
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            seen["core"] = purpose
            return {"viability_score": 61, "viability_summary": "с",
                    "top_risks": [{"title": "т", "body": "б"}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            seen["section"] = purpose
            return {"key": key, "title": "Резюме", "body": "текст"}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)
        rid = self._check("social_contract")
        client.get(f"/report/{pub(rid)}")
        assert seen == {"core": "social_contract", "section": "social_contract"}

    def test_sample_regenerates_after_a_switch_then_stays_cached_per_audience(self, monkeypatch):
        """Образец кэшировался НАВСЕГДА одной записью на проверку — человек,
        сгенерировавший его под одной оптикой и переключившийся на другую
        (POST /api/demand/{id}/purpose на /r/), видел на витрине соцконтракта
        или студента застрявший венчурный образец. Образец существует ровно
        затем, чтобы убедить купить (принцип 3) -- показывать не ту персону
        работает против этой же цели."""
        purposes_seen = []
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            purposes_seen.append(purpose)
            return {"viability_score": 61, "viability_summary": f"для {purpose}",
                    "top_risks": [{"title": "т", "body": "б"}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            return {"key": key, "title": "Резюме", "body": f"текст для {purpose}"}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

        rid = self._check("business")
        assert "для business" in client.get(f"/report/{pub(rid)}").text
        assert purposes_seen == ["business"]

        client.post(f"/api/demand/{rid}/purpose", json={"purpose": "social_contract"})
        text = client.get(f"/report/{pub(rid)}").text
        assert "для social_contract" in text          # новая оптика видна сразу
        assert "для business" not in text             # старую больше не показываем
        assert purposes_seen == ["business", "social_contract"]

        # Второй визит под той же (новой) оптикой -- из кэша, не новый вызов.
        client.get(f"/report/{pub(rid)}")
        assert purposes_seen == ["business", "social_contract"]

        # Переключение обратно -- бизнес-версия уже была посчитана, повторно
        # модель не зовём: переключаться туда-обратно не может стоить дороже
        # одной генерации на каждую из (немногих) аудиторий.
        client.post(f"/api/demand/{rid}/purpose", json={"purpose": "business"})
        assert "для business" in client.get(f"/report/{pub(rid)}").text
        assert purposes_seen == ["business", "social_contract"]

    def test_old_flat_sample_format_degrades_to_a_fresh_one_instead_of_crashing(self, monkeypatch):
        """До F8 sample_json хранил один плоский объект, а не словарь по
        аудитории. Записи, посчитанные ДО этой правки, несут именно старый
        формат — страница не имеет права упасть на разборе чужого прошлого
        формата (принцип 7), а должна просто пересчитать образец один раз
        под текущую оптику."""
        from app.main import DemandCheck, Session, engine
        purposes_seen = []
        import app.main as m
        async def fake_core(idea, demand_data, tier="full", chosen_offer=None,
                            purpose="business", **kw):
            purposes_seen.append(purpose)
            return {"viability_score": 61, "viability_summary": "новый образец",
                    "top_risks": [{"title": "т", "body": "б"}]}
        async def fake_section(key, idea, demand_data, tier="full", chosen_offer=None,
                               purpose="business", **kw):
            return {"key": key, "title": "Резюме", "body": "новый текст"}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)

        rid = self._check("business")
        with Session(engine) as s:
            rec = s.get(DemandCheck, rid)
            rec.sample_json = '{"viability_score": 40, "viability_summary": "старый плоский образец", "section": {"key": "summary", "title": "Резюме", "body": "старый текст"}}'
            s.add(rec); s.commit()

        r = client.get(f"/report/{pub(rid)}")
        assert r.status_code == 200
        assert "новый образец" in r.text
        assert "старый плоский образец" not in r.text
        assert purposes_seen == ["business"]     # пересчитано ровно один раз

        # Второй визит -- из уже перестроенного кэша, новый вызов не нужен.
        client.get(f"/report/{pub(rid)}")
        assert purposes_seen == ["business"]

    def test_buyer_does_not_pay_for_a_sample(self, monkeypatch):
        """У покупателя весь разбор открыт — лишний вызов модели ему не нужен."""
        from app.main import ReportPurchase, Session, engine, select
        calls = []
        self._stub(monkeypatch, calls)
        rid = self._check()
        contact = f"buyer_sample{rid}@example.com"
        client.post("/api/report", json={"check_id": rid, "tier": "full", "contact": contact})
        with Session(engine) as s:
            o = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)).first()
            o.status = "paid"; s.add(o); s.commit()
            tok = o.access_token
        calls.clear()
        client.get(f"/report/{pub(rid)}?t={tok}")
        assert calls == ["core"]        # только ядро платного отчёта, образца нет

    def test_page_survives_a_failed_sample(self, monkeypatch):
        """Принцип 7: не собрался образец — страница всё равно работает."""
        import app.main as m
        async def boom(*a, **kw):
            raise m.ReportEngineError("модель недоступна")
        monkeypatch.setattr(m, "generate_core", boom)
        monkeypatch.setattr(m, "generate_section", boom)
        rid = self._check()
        r = client.get(f"/report/{pub(rid)}")
        assert r.status_code == 200
        assert "const SAMPLE = null;" in r.text

    def test_sample_does_not_leak_the_paid_tier(self, monkeypatch):
        """Образец — один раздел, а не весь отчёт даром."""
        import app.main as m
        self._stub(monkeypatch)
        rid = self._check()
        text = client.get(f"/report/{pub(rid)}").text
        assert "const UNLOCKED_TIER = null;" in text
        assert "const TIER_KEYS = [];" in text
        assert m.SAMPLE_SECTION == "summary"

    def test_locked_sections_still_ask_their_questions(self, monkeypatch):
        """После образца человек должен увидеть, на что отвечает остальное."""
        from app.report_engine import SECTION_SPECS
        self._stub(monkeypatch)
        rid = self._check()
        text = client.get(f"/report/{pub(rid)}").text
        finance_ask = [s for s in SECTION_SPECS if s["key"] == "finance"][0]["ask"]
        assert finance_ask in text


class TestFunnelMiddleIsMeasured:
    """D1: между «начал проверку» и «оплатил» приборов не было. Для рекламы
    это значит, что Директ оптимизируется на клики, а не на покупателей, и
    непонятно, на каком шаге отваливается оплаченный трафик."""

    STATIC = main_module.BASE_DIR.parent / "static"

    def _all_static(self):
        return {p.name: p.read_text() for p in self.STATIC.glob("*.html")}

    def test_every_declared_goal_is_actually_sent(self):
        """Цель, заведённая в Метрике, но не отправляемая кодом, — это молча
        пустой отчёт, по которому владелец сделает неверный вывод."""
        import app.main as m
        blob = "\n".join(self._all_static().values())
        missing = [name for name, _ in m.METRIKA_GOALS if f"'{name}'" not in blob]
        assert not missing, f"цели объявлены, но не отправляются: {missing}"

    def test_no_goal_is_sent_outside_the_registry(self):
        """Обратная защита: отправляем цель, которой нет в списке — владелец
        не заведёт её в Метрике и потеряет данные."""
        import app.main as m, re
        known = {name for name, _ in m.METRIKA_GOALS}
        sent = set()
        for text in self._all_static().values():
            sent |= set(re.findall(r"sozGoal\(\s*'([a-z_]+)'", text))
        assert sent <= known, f"цели вне реестра: {sorted(sent - known)}"

    def test_middle_of_the_funnel_is_covered(self):
        """Конкретные шаги, которых не хватало."""
        pages = self._all_static()
        assert "sozGoal('demand_done'" in pages["result.html"]
        assert "sozGoal('sharpen_used'" in pages["result.html"]
        assert "sozGoal('check_saved'" in pages["result.html"]
        assert "sozGoal('live_test_ordered'" in pages["result.html"]
        assert "sozGoal('report_order_started'" in pages["report.html"]
        assert "'report_viewed'" in pages["report.html"]

    def test_goals_carry_the_audience(self):
        """Без purpose нельзя понять, какая кампания окупается: обе аудитории
        идут по одним и тем же шагам (D3)."""
        for name, text in self._all_static().items():
            pos = 0
            while (i := text.find("sozGoal(", pos)) != -1:
                # берём весь оператор до «;» -- вызов бывает многострочным
                stmt = text[i:text.find(";", i) + 1]
                assert "purpose" in stmt, f"{name}: цель без аудитории — {stmt[:70]}"
                pos = i + 1

    def test_pages_do_not_call_metrika_directly(self):
        """Каждая страница носила свою копию проверки на наличие счётчика, и
        новая страница просто забывала её написать. Единственный вход —
        sozGoal(), он же гасит ошибку, если счётчик не настроен."""
        for name, text in self._all_static().items():
            assert "reachGoal" not in text, f"{name} зовёт Метрику в обход sozGoal"

    def test_helper_is_injected_with_the_counter(self, monkeypatch):
        monkeypatch.setattr(main_module, "YANDEX_METRIKA_ID", "12345")
        out = main_module._inject_metrika("<html><head></head><body></body></html>")
        assert "window.sozGoal = function" in out
        assert "'reachGoal', name, params" in out
        assert "catch (e) {}" in out          # счётчик не ломает страницу

    def test_no_helper_no_crash_when_counter_is_off(self, monkeypatch):
        """Без SOZDATEL_YM_ID вставки нет вовсе — значит вызовы обязаны быть
        защищены проверкой на самой странице."""
        import re
        monkeypatch.setattr(main_module, "YANDEX_METRIKA_ID", "")
        assert "sozGoal" not in main_module._inject_metrika("<html><head></head></html>")
        for name, text in self._all_static().items():
            for call in re.finditer(r"window\.sozGoal\(", text):
                head = text[max(0, call.start() - 120):call.start()]
                assert "if (window.sozGoal)" in head, f"{name}: незащищённый вызов"

    def test_goal_names_are_stable_identifiers(self):
        """Имена заводит владелец руками в интерфейсе Метрики — пробел или
        кириллица там обернутся молчащей целью."""
        import app.main as m, re
        for name, title in m.METRIKA_GOALS:
            assert re.fullmatch(r"[a-z][a-z0-9_]*", name), name
            assert title and title[0].isupper(), name

    def test_paid_goals_carry_order_value_for_direct_bidding(self):
        """Перед Директом (D3, владелец 2026-08-01): автостратегии Директа
        оптимизируются по деньгам, а не по факту клика на «reachGoal» -- без
        `order_price` цель для них равнозначна пустой галочке. Проверяем
        именно на подтверждённых оплатах (не на "заказал"/"начал" — там ещё
        нет гарантии, что деньги реально пришли)."""
        pages = self._all_static()

        def stmt_for(text, goal):
            i = text.index(f"'{goal}'")
            start = text.rfind("sozGoal(", 0, i)
            return text[start:text.find(";", i) + 1]

        for goal in ("report_paid_quick", "report_paid_full"):
            stmt = stmt_for(pages["report.html"], goal)
            assert "order_price" in stmt and "currency" in stmt, f"{goal}: нет суммы для Директа"
        stmt = stmt_for(pages["result.html"], "live_test_paid")
        assert "order_price" in stmt and "currency" in stmt, "live_test_paid: нет суммы для Директа"

    def test_live_test_payment_confirmation_is_a_separate_goal_from_order_started(self):
        """До этой правки после оплаты живого теста Метрика не получала НИ
        ОДНОЙ цели о подтверждённой оплате — только "live_test_ordered" на
        старте заказа (который мог и не завершиться оплатой). Для отчёта по
        деньгам и для value-based bidding в Директе это была дыра."""
        text = self._all_static()["result.html"]
        assert "sozGoal('live_test_paid'" in text
        assert "'live_test_ordered'" in text
        assert "live_test_paid" != "live_test_ordered"

    def test_live_test_paid_goal_is_deduped(self):
        """Тот же паттерн, что у report_paid_* -- повторный визит по старой
        ссылке ?paid=1 не должен задваивать конверсию в Метрике."""
        text = self._all_static()["result.html"]
        i = text.index("live_test_paid")
        around = text[max(0, i - 400):i]
        assert "localStorage" in around, "нет дедупа через localStorage"


class TestOwnerFunnel:
    """D2 + серверная половина D3: у владельца не было вида на воронку —
    `/api/stats` отдавал два сырых счётчика. Перед тратой денег на рекламу
    нужно видеть, на каком шаге отваливается оплаченный трафик и какая из
    двух аудиторий вообще платит."""

    def _check(self, purpose="business", **fields):
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            rec = DemandCheck(idea="Пошив штор на заказ", purpose=purpose,
                              result_json='{"verdict": {"level": "niche", "text": "т"}}',
                              **fields)
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id

    def _report(self, check_id, status="new", amount=2990, tier="full"):
        from app.main import ReportPurchase, Session, engine
        with Session(engine) as s:
            s.add(ReportPurchase(check_id=check_id, idea="и", tier=tier, contact="c",
                                 status=status, amount=amount))
            s.commit()

    def _funnel(self, days=0):
        return client.get(f"/api/funnel?days={days}", headers=OWNER).json()

    def _stage(self, data, name):
        return [s for s in data["stages"] if s["name"] == name][0]

    def test_requires_owner_key(self):
        assert client.get("/api/funnel").status_code == 401

    def test_counts_every_step_of_the_path(self):
        """Пустых мест между «проверил» и «заплатил» быть не должно."""
        names = [s["name"] for s in self._funnel()["stages"]]
        for expected in ("Проверок спроса", "Заострили идею", "Сохранили в кабинет",
                         "Дошли до витрины отчёта", "Заказали отчёт", "Оплатили отчёт",
                         "Заказали тест на людях", "Оплатили тест на людях"):
            assert expected in names, expected

    def test_every_step_says_what_it_counts(self):
        """Число без определения — приглашение сделать неверный вывод (B3)."""
        for st in self._funnel()["stages"]:
            assert st["what"] and len(st["what"]) > 10, st["name"]

    def test_steps_reflect_real_rows(self):
        before = self._funnel()
        rid = self._check(chosen_offer='{"h1": "т"}', contact="a@b.ru",
                          sample_json='{"viability_score": 60}')
        self._report(rid, status="paid")
        after = self._funnel()
        for name in ("Проверок спроса", "Заострили идею", "Сохранили в кабинет",
                     "Дошли до витрины отчёта", "Заказали отчёт", "Оплатили отчёт"):
            assert self._stage(after, name)["total"] == self._stage(before, name)["total"] + 1, name

    def test_unpaid_order_counts_as_ordered_but_not_as_paid(self):
        before = self._funnel()
        rid = self._check()
        self._report(rid, status="pending_payment")
        after = self._funnel()
        assert self._stage(after, "Заказали отчёт")["total"] == \
               self._stage(before, "Заказали отчёт")["total"] + 1
        assert self._stage(after, "Оплатили отчёт")["total"] == \
               self._stage(before, "Оплатили отчёт")["total"]

    def test_split_by_audience(self):
        """Без разбивки не понять, какая рекламная кампания окупается (D3)."""
        before = self._funnel()
        self._check("social_contract")
        after = self._funnel()
        st_before, st_after = self._stage(before, "Проверок спроса"), self._stage(after, "Проверок спроса")
        assert st_after["social_contract"] == st_before["social_contract"] + 1
        assert st_after["business"] == st_before["business"]

    def test_report_orders_inherit_the_audience_of_their_check(self):
        """У покупки отчёта своего purpose нет — он берётся с проверки."""
        before = self._stage(self._funnel(), "Оплатили отчёт")
        rid = self._check("social_contract")
        self._report(rid, status="paid")
        after = self._stage(self._funnel(), "Оплатили отчёт")
        assert after["social_contract"] == before["social_contract"] + 1

    def test_owner_preview_is_not_a_sale(self):
        """Владельческий прогон бесплатен — в воронке ему не место."""
        import app.main as m
        before = self._funnel()
        rid = self._check()
        self._report(rid, status=m.PREVIEW_STATUS, amount=0)
        after = self._funnel()
        assert self._stage(after, "Заказали отчёт")["total"] == \
               self._stage(before, "Заказали отчёт")["total"]
        assert after["revenue"] == before["revenue"]

    def test_revenue_counts_only_confirmed_payments(self):
        before = self._funnel()["revenue"]
        rid = self._check()
        self._report(rid, status="pending_payment", amount=2990)
        assert self._funnel()["revenue"] == before        # ожидает оплаты — не деньги
        self._report(self._check(), status="paid", amount=2990)
        assert self._funnel()["revenue"] == before + 2990

    def test_period_filter_narrows_the_window(self):
        """Реклама оценивается за период, а не за всё время."""
        from app.main import DemandCheck, Session, engine, utcnow
        from datetime import timedelta
        old = self._check()
        with Session(engine) as s:
            rec = s.get(DemandCheck, old)
            rec.created_at = utcnow() - timedelta(days=90)
            s.add(rec); s.commit()
        assert self._stage(self._funnel(0), "Проверок спроса")["total"] > \
               self._stage(self._funnel(7), "Проверок спроса")["total"]

    def test_desk_renders_the_funnel(self):
        text = (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()
        assert 'id="funnel"' in text and "/api/funnel" in text
        assert "дошли" in text          # доля от предыдущего шага
        assert "Получено денег" in text


class TestPaidReportIsNotPublic:
    """Страница отчёта адресуется порядковым номером бесплатной проверки, и
    до этой правки оплаченный бизнес-план отдавался КАЖДОМУ, кто наберёт
    номер: 42 -> 41. Утекал не только текст за 2990 ₽ и чужая идея с чужими
    деньгами в смете -- посторонний мог через /api/report/{id}/section гонять
    генерацию по чужой покупке, и платили бы за это мы."""

    def _paid(self, monkeypatch, contact="owner_of_report@example.com", tier="full"):
        """Оплаченный отчёт: возвращает (номер проверки, токен покупателя)."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                    "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig
        client.post("/api/report", json={"check_id": rid, "tier": tier, "contact": contact})
        with Session(engine) as s:
            o = s.exec(select(ReportPurchase).where(ReportPurchase.contact == contact)).first()
            o.status = "paid"
            o.report_json = json.dumps({
                "viability_score": 62, "viability_summary": "СЕКРЕТНЫЙ ВЫВОД.",
                "top_risks": [{"title": "Риск", "body": "Объяснение."}],
                "sections": [{"key": "summary", "title": "Резюме проекта",
                              "body": "ВЫРУЧКА 400 000 РУБЛЕЙ В МЕСЯЦ"}]}, ensure_ascii=False)
            s.add(o); s.commit()
            return rid, o.access_token

    def _login(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_acc_" + contact, contact=contact)); s.commit()
        assert client.post(f"/account/verify?token=tok_acc_{contact}",
                          follow_redirects=False).status_code in (302, 303, 307)

    def _publish_example(self, contact):
        """Пример в сервисе ровно один: публикация снимает старый. Ставим
        флаг руками, поэтому чужой пример от соседнего теста снимаем сами --
        иначе /example отдаст его, а не наш."""
        from app.main import ReportPurchase, Session, engine, select
        with Session(engine) as s:
            for row in s.exec(select(ReportPurchase).where(
                    ReportPurchase.is_example == True)).all():      # noqa: E712
                row.is_example = False; s.add(row)
            mine = s.exec(select(ReportPurchase).where(
                ReportPurchase.contact == contact)).first()
            mine.is_example = True; s.add(mine); s.commit()

    def _logout(self):
        client.post("/api/account/logout")

    # --- дыра, ради которой всё это ---

    def test_stranger_cannot_read_a_paid_report_by_guessing_the_number(self, monkeypatch):
        rid, _ = self._paid(monkeypatch, contact="secret_buyer@example.com")
        self._logout()
        text = client.get(f"/report/{pub(rid)}").text
        assert "СЕКРЕТНЫЙ ВЫВОД" not in text
        assert "400 000 РУБЛЕЙ" not in text
        assert 'const UNLOCKED_TIER = null' in text or "UNLOCKED_TIER = null" in text

    def test_stranger_cannot_spend_our_money_on_someone_elses_report(self, monkeypatch):
        """Худшее в дыре было не чтение, а запись: посторонний запускал
        генерацию раздела по чужой оплаченной покупке."""
        import app.main as m
        rid, _ = self._paid(monkeypatch, contact="money_buyer@example.com")
        self._logout()
        calls = []
        async def fake_section(key, *a, **kw):
            calls.append(key)
            return {"key": key, "title": "Раздел", "body": "Текст"}
        monkeypatch.setattr(m, "generate_section", fake_section)
        r = client.post(f"/api/report/{rid}/section?key=market")
        assert r.status_code == 403
        assert calls == []                     # модель не звали вовсе

    def test_stranger_does_not_pay_for_a_sample_of_a_sold_report(self, monkeypatch):
        """Образец продаёт отчёт по этой идее, а он уже продан: звать модель
        ради постороннего не за что."""
        import app.main as m
        calls = []
        async def fake_core(*a, **kw):
            calls.append("core")
            return {"viability_score": 50, "viability_summary": "я", "top_risks": []}
        async def fake_section(key, *a, **kw):
            calls.append(key)
            return {"key": key, "title": "Раздел", "body": "Текст"}
        monkeypatch.setattr(m, "generate_core", fake_core)
        monkeypatch.setattr(m, "generate_section", fake_section)
        rid, _ = self._paid(monkeypatch, contact="sample_buyer@example.com")
        self._logout()
        calls.clear()
        assert client.get(f"/report/{pub(rid)}").status_code == 200
        assert calls == []

    def test_lost_link_is_not_offered_to_pay_a_second_time(self, monkeypatch):
        """Покупатель, открывший свою же ссылку без токена, видит «уже
        оплачен». Кнопка оплаты рядом с этой фразой — прямой путь заплатить
        за одно и то же дважды."""
        rid, _ = self._paid(monkeypatch, contact="double_pay@example.com")
        self._logout()
        text = client.get(f"/report/{pub(rid)}").text
        assert 'id="access-note"' in text
        # витрину гасит сама страница по наличию этой плашки
        assert "getElementById('access-note')" in text

    def test_stranger_is_told_how_to_reach_his_own_report(self, monkeypatch):
        """Деградация, а не 403: свою же ссылку можно открыть в другом
        браузере, и глухая ошибка человеку ничего не объяснит (принцип 7)."""
        rid, _ = self._paid(monkeypatch, contact="lost_link@example.com")
        self._logout()
        r = client.get(f"/report/{pub(rid)}")
        assert r.status_code == 200
        assert "уже оплачен" in r.text
        assert "/account" in r.text

    # --- три двери настоящего покупателя ---

    def test_buyer_opens_his_report_by_the_link_payment_returned_him(self, monkeypatch):
        rid, tok = self._paid(monkeypatch, contact="link_buyer@example.com")
        self._logout()
        text = client.get(f"/report/{pub(rid)}?t={tok}").text
        assert "СЕКРЕТНЫЙ ВЫВОД" in text and "400 000 РУБЛЕЙ" in text

    def test_buyer_who_lost_the_link_opens_it_from_his_cabinet(self, monkeypatch):
        """Токен потерялся вместе с вкладкой -- вход по своей почте обязан
        работать и без него, иначе оплаченная услуга просто пропадает."""
        rid, _ = self._paid(monkeypatch, contact="cabinet_buyer@example.com")
        self._login("cabinet_buyer@example.com")
        try:
            assert "СЕКРЕТНЫЙ ВЫВОД" in client.get(f"/report/{pub(rid)}").text
        finally:
            self._logout()

    def test_owner_still_sees_everything_by_key(self, monkeypatch):
        rid, _ = self._paid(monkeypatch, contact="key_buyer@example.com")
        self._logout()
        assert "СЕКРЕТНЫЙ ВЫВОД" in client.get(f"/report/{pub(rid)}", headers=OWNER).text

    # --- чужие ключи не подходят ---

    def test_someone_elses_cabinet_does_not_open_the_report(self, monkeypatch):
        rid, _ = self._paid(monkeypatch, contact="mine@example.com")
        self._login("notmine@example.com")
        try:
            assert "СЕКРЕТНЫЙ ВЫВОД" not in client.get(f"/report/{pub(rid)}").text
        finally:
            self._logout()

    def test_token_of_another_purchase_does_not_open_this_one(self, monkeypatch):
        rid_a, _ = self._paid(monkeypatch, contact="a_buyer@example.com")
        _, tok_b = self._paid(monkeypatch, contact="b_buyer@example.com")
        self._logout()
        assert "СЕКРЕТНЫЙ ВЫВОД" not in client.get(f"/report/{pub(rid_a)}?t={tok_b}").text

    def test_empty_token_is_not_a_master_key(self, monkeypatch):
        """Покупки, оформленные до появления токена, досыпаются при старте.
        Но если у строки токен всё-таки пуст, пустой ?t= не имеет права
        открыть отчёт."""
        from app.main import ReportPurchase, Session, engine, select
        rid, _ = self._paid(monkeypatch, contact="legacy@example.com")
        with Session(engine) as s:
            o = s.exec(select(ReportPurchase).where(
                ReportPurchase.contact == "legacy@example.com")).first()
            o.access_token = ""; s.add(o); s.commit()
        self._logout()
        assert "СЕКРЕТНЫЙ ВЫВОД" not in client.get(f"/report/{pub(rid)}?t=").text
        # но по своей почте покупатель входит и без токена
        self._login("legacy@example.com")
        try:
            assert "СЕКРЕТНЫЙ ВЫВОД" in client.get(f"/report/{pub(rid)}").text
        finally:
            self._logout()

    # --- ссылки, которые мы сами раздаём, обязаны работать ---

    def test_cabinet_link_carries_the_token(self, monkeypatch):
        """Иначе скопированная из кабинета ссылка не откроется в другом
        браузере -- ровно та ситуация, из-за которой человек и потерял её."""
        rid, tok = self._paid(monkeypatch, contact="cablink@example.com")
        self._login("cablink@example.com")
        try:
            reports = client.get("/api/account/me").json()["reports"]
        finally:
            self._logout()
        row = [r for r in reports if r["check_id"] == rid][0]
        assert row["report_url"] == f"/report/{pub(rid)}?t={tok}"

    def test_payment_returns_the_buyer_with_his_token(self, monkeypatch):
        """Вернувшись с оплаты, человек обязан увидеть отчёт сразу, ещё не
        заходя в кабинет."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine, select
        async def fake_check(idea):
            return {"formulations": [{"phrase": "п", "count": 10}], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"},
                    "competitors": {"found": 1, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Идея для возврата с оплаты"}).json()["id"]
        finally:
            m.check_demand = orig
        seen = {}
        async def fake_create(order_id, amount, desc, url, kind="livetest", contact="", _post=None):
            seen["url"] = url
            return ("pay_ret", "https://pay.example/x")
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        monkeypatch.setattr(m.payments, "create_payment", fake_create)
        client.post("/api/report", json={"check_id": rid, "tier": "full",
                                         "contact": "return@example.com"})
        with Session(engine) as s:
            tok = s.exec(select(ReportPurchase).where(
                ReportPurchase.contact == "return@example.com")).first().access_token
        assert f"/report/{pub(rid)}?t={tok}" in seen["url"]
        assert "paid=1" in seen["url"]

    def test_token_does_not_leak_through_referer(self):
        """Токен лежит в адресе страницы: ссылка наружу утащила бы его в
        Referer чужому серверу."""
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert '<meta name="referrer" content="same-origin">' in text

    def test_page_passes_the_token_when_ordering_sections(self):
        """Без этого страница покупателя упёрлась бы в собственную защиту."""
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert "qs.get('t')" in text
        assert "/section?key=" in text

    # --- публичный пример не должен пострадать ---

    def test_published_example_stays_open_to_everyone(self, monkeypatch):
        """Пример на /example -- единственный отчёт, который мы показываем
        всем намеренно."""
        rid, _ = self._paid(monkeypatch, contact="example_buyer@example.com")
        self._publish_example("example_buyer@example.com")
        self._logout()
        text = client.get("/example").text
        assert "СЕКРЕТНЫЙ ВЫВОД" in text
        # ...но по номеру проверки он по-прежнему закрыт
        assert "СЕКРЕТНЫЙ ВЫВОД" not in client.get(f"/report/{pub(rid)}").text

    def test_example_does_not_generate_on_visitors_behalf(self, monkeypatch):
        """Опубликованный пример мог оказаться неполным. Полный список
        разделов тарифа заставил бы каждую вкладку посетителя дозаказывать
        недостающее -- вызовы модели по чужой покупке на каждого зрителя."""
        rid, _ = self._paid(monkeypatch, contact="partial_example@example.com")
        self._publish_example("partial_example@example.com")
        self._logout()
        text = client.get("/example").text
        keys = json.loads(re.search(r"const TIER_KEYS = (\[.*?\]);", text).group(1))
        assert keys == ["summary"]        # ровно то, что опубликовано


class TestBuyerHearsFromUs:
    """A10: при оплате письмо уходило ВЛАДЕЛЬЦУ, а человеку, отдавшему
    990-2990 ₽, -- ничего. Только фискальный чек от ЮКассы, то есть чек, а не
    ссылка на продукт. Между тем разбор собирается по разделам минутами, и
    единственным следом покупки была вкладка в браузере: закрыл -- и ищи."""

    def _mail(self, monkeypatch):
        """Копит письма ПОКУПАТЕЛЮ: владельческие проверяет
        TestOwnerLearnsAboutOrders."""
        import app.main as m
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        sent = []
        def rec(to, subject, body, **kw):
            if to != "owner@example.com":
                sent.append((to, subject, body))
        monkeypatch.setattr(m.mailer, "send", rec)
        return sent

    def _check(self):
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "а", "count": 10}], "best_phrase": "а",
                    "verdict": {"level": "weak", "text": ""},
                    "competitors": {"found": None, "top": []}, "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            return client.post("/api/demand",
                               json={"idea": "Идея достаточно длинная для письма покупателю"}).json()["id"]
        finally:
            m.check_demand = orig

    def _pay(self, monkeypatch, kind, contact, tier="full"):
        """Проводит настоящую оплату через вебхук ЮКассы и возвращает заказ."""
        import app.main as m
        from app.main import LiveTestOrder, ReportPurchase, Session, engine, select
        monkeypatch.setattr(m.payments, "configured", lambda: True)
        async def fake_create(order_id, amount, desc, url, kind="livetest", contact="", _post=None):
            return ("pay_buyer", "https://pay.example/z")
        monkeypatch.setattr(m.payments, "create_payment", fake_create)
        rid = self._check()
        if kind == "report":
            client.post("/api/report", json={"check_id": rid, "tier": tier, "contact": contact})
            model = ReportPurchase
        else:
            client.post("/api/live-test", json={"check_id": rid, "contact": contact})
            model = LiveTestOrder
        with Session(engine) as s:
            oid = s.exec(select(model).where(model.contact == contact)).first().id
        async def fake_fetch(pid, _post=None):
            return {"status": "succeeded", "metadata": {"order_id": str(oid), "kind": kind}}
        monkeypatch.setattr(m.payments, "fetch_payment", fake_fetch)
        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_buyer"}})
        return rid, oid, model

    # --- сама дыра ---

    def test_report_buyer_gets_a_letter_with_a_link_to_his_report(self, monkeypatch):
        from app.main import ReportPurchase, Session, engine
        sent = self._mail(monkeypatch)
        rid, oid, _ = self._pay(monkeypatch, "report", "reportbuyer@example.com")
        assert len(sent) == 1
        to, subject, body = sent[0]
        assert to == "reportbuyer@example.com"
        assert "оплата принята" in subject.lower()
        with Session(engine) as s:
            tok = s.get(ReportPurchase, oid).access_token
        assert f"/report/{pub(rid)}?t={tok}" in body        # ссылка ведёт прямо в отчёт
        assert "2990 ₽" in body and "Бизнес-план" in body

    def test_letter_says_the_tab_can_be_closed(self, monkeypatch):
        """Разбор собирается минутами. Без этой фразы человек либо сидит и
        ждёт, либо закрывает вкладку и считает, что потерял покупку."""
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "report", "closetab@example.com")
        body = sent[0][2]
        assert "можно закрыть" in body
        assert "продолжится" in body

    def test_letter_shows_the_way_back_without_the_link(self, monkeypatch):
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "report", "waybackpath@example.com")
        body = sent[0][2]
        assert "/account" in body
        assert "без пароля" in body.lower()

    def test_live_test_buyer_gets_a_letter_too(self, monkeypatch):
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "livetest", "livebuyer@example.com")
        assert len(sent) == 1
        to, subject, body = sent[0]
        assert to == "livebuyer@example.com"
        assert "тест на реальных людях" in subject
        assert "1490 ₽" in body

    def test_letter_does_not_point_at_a_link_it_does_not_contain(self, monkeypatch):
        """В письме про живой тест прямой ссылки нет — страницу мы ещё
        собираем. Фраза «даже если ссылка выше потеряется» там врала бы."""
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "livetest", "nolink@example.com")
        body = sent[0][2]
        assert "ссылка выше" not in body
        sent.clear()
        self._pay(monkeypatch, "report", "haslink@example.com")
        assert "ссылка выше" in sent[0][2]      # а здесь ссылка есть

    def test_live_test_letter_repeats_the_ad_budget_warning(self, monkeypatch):
        """A7: про отдельный рекламный бюджет человек узнавал уже после
        оплаты. Письмо — последнее место, где об этом можно промолчать."""
        import app.main as m
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "livetest", "adbudget@example.com")
        body = sent[0][2]
        assert m.AD_BUDGET_HINT in body
        assert "напрямую Яндексу" in body

    # --- заявки без оплаты ---

    def test_unpaid_request_is_confirmed_to_the_buyer(self, monkeypatch):
        """Касса не настроена — заявку доводит владелец руками. Человек,
        оставивший контакт, всё равно должен получить подтверждение."""
        import app.main as m
        monkeypatch.setattr(m.payments, "configured", lambda: False)
        sent = self._mail(monkeypatch)
        rid = self._check()
        client.post("/api/report", json={"check_id": rid, "tier": "quick",
                                         "contact": "unpaidbuyer@example.com"})
        assert len(sent) == 1
        assert "заявка принята" in sent[0][1].lower()
        assert "оплата принята" not in sent[0][2].lower()

    # --- честность и надёжность ---

    def test_letter_is_sent_once_per_order(self, monkeypatch):
        """Вебхук ЮКассы приходит повторно — второе письмо о той же оплате
        выглядит как второе списание."""
        sent = self._mail(monkeypatch)
        self._pay(monkeypatch, "report", "onceonly@example.com")
        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_buyer"}})
        client.post("/api/yookassa/webhook", json={"object": {"id": "pay_buyer"}})
        assert len(sent) == 1

    def test_buyer_flag_is_separate_from_the_owner_flag(self, monkeypatch):
        """Урок A2: один флаг на двоих означал бы, что письмо владельцу
        гасит письмо покупателю."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        to_buyer = []
        def only_owner_works(to, subject, body, **kw):
            if to != "owner@example.com":
                raise RuntimeError("ящик покупателя недоступен")
            to_buyer.append(to)
        monkeypatch.setattr(m.mailer, "send", only_owner_works)
        _, oid, _ = self._pay(monkeypatch, "report", "brokenbox@example.com")
        with Session(engine) as s:
            row = s.get(ReportPurchase, oid)
            assert row.paid_notified is True      # владельцу ушло
            assert row.buyer_notified is False    # покупателю — нет, и мы это помним

    def test_phone_contact_does_not_break_the_payment(self, monkeypatch):
        """Контакт для чека 54-ФЗ может быть телефоном — письма тогда просто
        нет, но оплата обязана пройти."""
        from app.main import ReportPurchase, Session, engine
        sent = self._mail(monkeypatch)
        _, oid, _ = self._pay(monkeypatch, "report", "+79990001122")
        assert sent == []
        with Session(engine) as s:
            row = s.get(ReportPurchase, oid)
            assert row.status == "paid"
            assert row.buyer_notified is False

    def test_broken_smtp_does_not_break_the_payment(self, monkeypatch):
        """Человек уже заплатил: сбой почты не имеет права стать ошибкой
        на его экране (принцип 7)."""
        import app.main as m
        from app.main import ReportPurchase, Session, engine
        monkeypatch.setattr(m.mailer, "configured", lambda: True)
        def boom(*a, **kw):
            raise RuntimeError("SMTP лёг")
        monkeypatch.setattr(m.mailer, "send", boom)
        _, oid, _ = self._pay(monkeypatch, "report", "smtpdown@example.com")
        with Session(engine) as s:
            assert s.get(ReportPurchase, oid).status == "paid"

    def test_no_smtp_configured_is_silent(self, monkeypatch):
        import app.main as m
        monkeypatch.setattr(m.mailer, "configured", lambda: False)
        sent = []
        monkeypatch.setattr(m.mailer, "send", lambda *a, **kw: sent.append(a))
        self._pay(monkeypatch, "report", "nosmtp@example.com")
        assert sent == []

    # --- страница говорит то же самое ---

    def test_page_tells_the_buyer_the_tab_can_be_closed(self):
        text = (main_module.BASE_DIR.parent / "static" / "report.html").read_text()
        assert "Страницу можно закрыть" in text

    def test_return_from_payment_explains_where_the_report_lives(self, monkeypatch):
        from app.main import ReportPurchase, Session, engine
        self._mail(monkeypatch)
        rid, oid, _ = self._pay(monkeypatch, "report", "afterpay@example.com")
        with Session(engine) as s:
            tok = s.get(ReportPurchase, oid).access_token
        text = client.get(f"/report/{pub(rid)}?t={tok}&paid=1").text
        note = re.search(r'id="paid-note">(.*?)</div>', text, re.S).group(1)
        assert "Оплата принята" in note
        assert "личном кабинете" in note and "/account" in note
        # «вкладку можно закрыть» говорит строка сборки под плашкой -- одна
        # мысль в одном месте, а не дважды подряд
        assert "можно закрыть" not in note

    def test_page_does_not_claim_a_letter_that_never_went(self, monkeypatch):
        """Принцип честности: обещать письмо, которого не было, хуже, чем
        не обещать ничего. Контакт-телефон писем не получает."""
        from app.main import ReportPurchase, Session, engine
        self._mail(monkeypatch)
        rid, oid, _ = self._pay(monkeypatch, "report", "+79995554433")
        with Session(engine) as s:
            row = s.get(ReportPurchase, oid)
            tok, notified = row.access_token, row.buyer_notified
        assert notified is False
        text = client.get(f"/report/{pub(rid)}?t={tok}&paid=1").text
        note = re.search(r'id="paid-note">(.*?)</div>', text, re.S).group(1)
        assert "Оплата принята" in note
        assert "письмом" not in note
        assert "личном кабинете" in note      # путь назад всё равно назван

    def test_note_is_not_shown_on_every_later_visit(self, monkeypatch):
        """Плашка про оплату нужна в момент возврата, а не вечно."""
        from app.main import ReportPurchase, Session, engine
        self._mail(monkeypatch)
        rid, oid, _ = self._pay(monkeypatch, "report", "latervisit@example.com")
        with Session(engine) as s:
            tok = s.get(ReportPurchase, oid).access_token
        assert "Оплата принята" not in client.get(f"/report/{pub(rid)}?t={tok}").text


class TestMailerKnowsWhatItCanSend:
    """Контакт для чека 54-ФЗ разрешает и телефон -- почтальон обязан
    спросить, а не пытаться и падать."""

    def test_phone_is_not_an_email(self):
        from app import mailer
        for bad in ("+79990001122", "@telegram_handle", "", "не почта",
                    "a@b", "a@.ru", "a@b.", "a b@c.ru"):
            assert mailer.looks_like_email(bad) is False, bad

    def test_normal_addresses_pass(self):
        from app import mailer
        for good in ("a@b.ru", "boris.belkin+tag@mail.example.com"):
            assert mailer.looks_like_email(good) is True, good

    def test_notify_buyer_never_raises(self):
        from app import mailer
        def boom(msg):
            raise RuntimeError("SMTP лёг")
        assert mailer.notify_buyer("a@b.ru", "тема", "тело", _send=boom) is False


class TestWeakDemandStopsSelling:
    """A11: страница показывала «в поиске эту идею почти не ищут» — и вела к
    той же кнопке, что идею с хорошим спросом. Живой тест здесь особенно
    сомнителен: он гоняет рекламу по ТЕМ ЖЕ запросам, а вывод мы делаем по
    CLICK_TARGET визитам, которых при частотности ниже 300 в месяц просто
    неоткуда взять. Принцип 2: смысл сервиса в том, чтобы человек НЕ потратил
    деньги зря.

    ВАЖНО про эти тесты: разметка блока лежит на странице ВСЕГДА, а показывает
    его скрипт по уровню вердикта. Значит проверки ниже сторожат только тексты
    и подстановку серверных значений — отключи логику, и они останутся
    зелёными. Само поведение сторожит браузерный
    tests/test_mobile.py::test_weak_demand_stops_selling_in_a_real_browser."""

    def _check(self, level, count, purpose="business"):
        import app.main as m
        from app.main import DemandCheck, Session, engine
        texts = {"weak": "В поиске эту идею почти не ищут.",
                 "niche": "Спрос небольшой, но он есть.",
                 "strong": "Спрос есть."}
        data = {"formulations": [{"phrase": "фраза", "count": count},
                                 {"phrase": "вторая", "count": max(0, count // 3)}],
                "best_phrase": "фраза",
                "verdict": {"level": level, "text": texts[level]},
                "competitors": {"found": 40, "top": []},
                "scores": [{"key": "demand", "label": "Спрос", "value": 1, "note": ""}],
                "overall": {"value": 1, "weakest": "Спрос", "basis": "Опущен до спроса."}}
        with Session(engine) as s:
            rec = DemandCheck(idea="Подписка на носки по гороскопу", best_count=count,
                              purpose=purpose,
                              result_json=json.dumps(data, ensure_ascii=False))
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id

    def test_page_carries_the_honest_lead_for_weak_demand(self):
        """Главным действием становится бесплатное — переформулировать."""
        rid = self._check("weak", 30)
        text = client.get(f"/r/{pub(rid)}").text
        assert 'id="weak-lead"' in text
        assert "попробуйте другую формулировку" in text.lower()

    def test_lead_is_shown_only_when_demand_is_weak(self):
        """Проверка на самом коде: блок включает JS по уровню вердикта."""
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert "v.level === 'weak'" in text
        # и обе платные кнопки при этом перестают быть главными
        block = text.split("v.level === 'weak'")[1][:2000]
        assert "getElementById('order').className = 'alt-path'" in block
        assert "getElementById('alt-report').className = 'alt-path'" in block

    def test_live_test_carries_an_explicit_caveat(self):
        """Оговорка стоит у самой кнопки живого теста: именно этот продукт
        наши же цифры и ставят под сомнение."""
        rid = self._check("weak", 30)
        text = client.get(f"/r/{pub(rid)}").text
        assert 'id="weak-caveat"' in text
        assert "рискует не набрать" in text
        # и говорит, что деньги на рекламу всё равно уйдут
        assert "бюджет при этом всё равно тратится" in text

    def test_caveat_names_the_number_we_judge_by(self):
        """Число берётся с сервера, а не пишется руками: порог уже разъезжался
        с витриной (B5)."""
        import app.main as m
        rid = self._check("weak", 30)
        text = client.get(f"/r/{pub(rid)}").text
        assert "__CLICK_TARGET__" not in text          # слот подставлен
        assert f"{m.CLICK_TARGET} визитов" in text

    def test_header_stops_promising_the_next_stage(self):
        """Шапка обещала следующий этап так, будто вердикта не было.

        Чинилось подменой текста в строке «Дальше — …». При разгрузке шапки
        (кастдев 2026-08-02) строку убрали целиком, поэтому обещать нечем по
        построению — а совет «переформулировать» остался там, где он и
        полезен: в блоке при слабом спросе.
        """
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        assert 'id="path-next-text"' not in text
        # А совет остался там, где человек его читает: в самом блоке при
        # слабом спросе. Слово «переформулировать» жило ТОЛЬКО в удалённой
        # строке шапки, поэтому проверяем смысл, а не его.
        block = text.split("v.level === 'weak'")[1][:2000]
        assert "люди ищут то же самое, но называют иначе" in block
        assert "weak-lead" in block

    def test_free_action_leads_to_the_right_showcase(self):
        """Получателя соцконтракта нельзя возвращать на витрину для
        фаундеров (принцип 4)."""
        text = (main_module.BASE_DIR.parent / "static" / "result.html").read_text()
        block = text.split("v.level === 'weak'")[1][:2000]
        assert "AUDIENCE.home" in block

    def test_good_demand_is_untouched(self):
        """Предупреждение не должно всплывать там, где спрос есть: иначе оно
        обесценится и его перестанут читать."""
        for level, count in (("niche", 1200), ("strong", 5000)):
            rid = self._check(level, count)
            text = client.get(f"/r/{pub(rid)}").text
            # блок в разметке есть всегда, но скрыт — включает его только JS
            assert 'class="weak-lead" id="weak-lead"' in text
            assert "weak-lead').classList.add('show')" in text.replace('\n', '')

    def test_buttons_are_not_hidden(self):
        """Человек вправе купить, даже когда мы отговариваем: наше дело —
        сказать правду, а не решить за него."""
        rid = self._check("weak", 30)
        text = client.get(f"/r/{pub(rid)}").text
        assert 'id="order-btn"' in text                # заявка на живой тест
        assert f'href="/report/{pub(rid)}"' in text   # и отчёт


class TestTierListIsReadableAtTheDecisionPoint:
    """B7: C2 поставила состав тарифов к кнопке, когда разделов было восемь.
    После посекционной переработки (E5) их 21, и полный тариф на витрине
    превратился в строчную простыню из шестнадцати фрагментов через запятую.
    Блок, созданный помогать решению, решению мешал.

    Здесь нет JS-развилки: блок собирает сервер, поэтому проверки ниже
    сторожат настоящее поведение, а не текст в шаблоне."""

    def _block(self):
        import app.main as m
        return m._tier_summary_html()

    def _plain(self, html_text):
        return re.sub(r"<[^>]+>", " ", html_text)

    def test_full_tier_is_grouped_not_one_long_line(self):
        import app.main as m
        block = self._block()
        items = re.findall(r"<li>(.*?)</li>", block, re.S)
        assert len(items) >= 4, "полный тариф снова одной строкой"
        # каждая группа названа — именно имя группы держит взгляд
        names = [n for n, _ in m.SECTION_GROUPS]
        shown = [n for n in names if f"<b>{n}:</b>" in block]
        assert len(shown) >= 4, f"группы не названы: {shown}"

    def test_money_group_carries_the_estimate(self):
        """Ради этой строки человек с /social-contract и платит: смета есть
        только в полном тарифе (это и была находка C2)."""
        block = self._block()
        money = [x for x in re.findall(r"<li>(.*?)</li>", block, re.S)
                 if "Деньги" in x]
        assert money, "группы «Деньги» на витрине нет"
        assert "финансовая модель" in money[0].lower()

    def test_no_section_of_the_full_tier_is_lost(self):
        """Главное свойство блока: он обещает ровно то, что движок отдаёт."""
        import app.main as m
        plain = self._plain(self._block()).lower()
        for key, title in m.ALL_SECTIONS:
            short = title.split(" — ")[0].strip().lower()
            assert short in plain, f"раздел «{title}» пропал с витрины"

    def test_section_without_a_group_still_reaches_the_showcase(self):
        """Раскладка по группам не должна уметь ронять секцию. Молча
        пропасть — это ровно тот разъезд движка и витрины, против которого
        весь этот блок и написан (принцип 3).

        Название теперь читается через section_title() (SECTION_SPECS), а
        не напрямую из ALL_SECTIONS — патчим оба источника."""
        import app.main as m
        import app.report_engine as re_engine
        new_spec = {"key": "newthing", "group": "Идея и рынок",
                   "title": "Совершенно новая секция"}
        with_new = list(m.ALL_SECTIONS) + [("newthing", "Совершенно новая секция")]
        orig_all = m.ALL_SECTIONS
        orig_specs = re_engine.SECTION_SPECS
        m.ALL_SECTIONS = with_new
        re_engine.SECTION_SPECS = list(orig_specs) + [new_spec]
        try:
            plain = self._plain(m._tier_summary_html()).lower()
        finally:
            m.ALL_SECTIONS = orig_all
            re_engine.SECTION_SPECS = orig_specs
        assert "совершенно новая секция" in plain

    def test_count_is_computed_not_written_by_hand(self):
        import app.main as m
        extra = [k for k, _ in m.ALL_SECTIONS if k not in m.QUICK_KEYS]
        assert f"ещё {len(extra)} " in self._plain(self._block())

    def test_cheap_tier_stays_a_plain_list(self):
        """Пять пунктов читаются одной строкой — дробить их на группы значит
        сделать хуже там, где было хорошо."""
        import app.main as m
        block = self._block()
        head = block.split('<ul class="tier-groups">')[0]
        for key, title in m.ALL_SECTIONS:
            if key in m.QUICK_KEYS:
                assert title.split(" — ")[0].strip() in head, title

    def test_tier_names_and_prices_still_come_from_the_code(self):
        """B5: копия названия или цены в статике — уже трижды пойманный
        источник вранья."""
        import app.main as m
        block = self._block()
        for tier in ("quick", "full"):
            assert m.REPORT_PRICES[tier]["label"] in block
            assert f'{m.REPORT_PRICES[tier]["price"]} ₽' in block
        # Что копия названия не вернулась в статику, сторожит уже
        # TestNoHardcodedServerValuesInStatic — там проверка точнее: она
        # ищет название в кавычках-ёлочках, а не любое вхождение слова
        # (в result.html оно законно встречается в комментарии к коду).

    def test_showcase_renders_the_groups_on_the_page(self):
        """Блок должен доезжать до браузера подставленным, а не слотом."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "пошив штор", "count": 1200}],
                    "best_phrase": "пошив штор",
                    "verdict": {"level": "niche", "text": "Нишевый спрос"},
                    "competitors": {"found": 900, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            rid = client.post("/api/demand", json={"idea": "Пошив штор на заказ"}).json()["id"]
        finally:
            m.check_demand = orig
        text = client.get(f"/r/{pub(rid)}").text
        assert "__TIER_SUMMARY__" not in text
        assert 'class="tier-groups"' in text
        assert "<b>Деньги:</b>" in text


class TestMailerSpeaksBothPorts:
    """Провайдер даёт два адреса на выбор: 465 шифрует с первого байта, 587
    начинает открыто и поднимает шифрование командой STARTTLS. Код умел только
    первый — владелец, вписавший 587 (reg.ru показывает оба), получал бы
    невнятную ошибку SSL при полностью верных логине и пароле."""

    class _FakeSMTP:
        calls = []

        def __init__(self, host, port, timeout=None):
            type(self).calls.append(("connect", host, port))

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def ehlo(self):
            type(self).calls.append(("ehlo",))

        def starttls(self):
            type(self).calls.append(("starttls",))

        def login(self, user, password):
            type(self).calls.append(("login", user))

        def send_message(self, msg):
            type(self).calls.append(("send", msg["To"]))

    def _env(self, monkeypatch, port):
        from app import mailer
        monkeypatch.setenv("SOZDATEL_SMTP_HOST", "smtp.example.ru")
        monkeypatch.setenv("SOZDATEL_SMTP_PORT", str(port))
        monkeypatch.setenv("SOZDATEL_SMTP_USER", "info@example.ru")
        monkeypatch.setenv("SOZDATEL_SMTP_PASSWORD", "secret")
        ssl_cls = type("SSLFake", (self._FakeSMTP,), {"calls": []})
        plain_cls = type("PlainFake", (self._FakeSMTP,), {"calls": []})
        monkeypatch.setattr(mailer.smtplib, "SMTP_SSL", ssl_cls)
        monkeypatch.setattr(mailer.smtplib, "SMTP", plain_cls)
        return mailer, ssl_cls, plain_cls

    def test_port_465_talks_ssl_from_the_first_byte(self, monkeypatch):
        mailer, ssl_cls, plain_cls = self._env(monkeypatch, 465)
        mailer.send("to@example.ru", "тема", "тело")
        assert ("connect", "smtp.example.ru", 465) in ssl_cls.calls
        assert ("send", "to@example.ru") in ssl_cls.calls
        assert plain_cls.calls == []
        assert not any(c[0] == "starttls" for c in ssl_cls.calls)

    def test_port_587_raises_encryption_with_starttls(self, monkeypatch):
        mailer, ssl_cls, plain_cls = self._env(monkeypatch, 587)
        mailer.send("to@example.ru", "тема", "тело")
        assert ("connect", "smtp.example.ru", 587) in plain_cls.calls
        assert ("starttls",) in plain_cls.calls
        assert ("send", "to@example.ru") in plain_cls.calls
        assert ssl_cls.calls == []

    def test_starttls_happens_before_login(self, monkeypatch):
        """Логин в открытом канале отдал бы пароль в сеть как есть."""
        mailer, _, plain_cls = self._env(monkeypatch, 587)
        mailer.send("to@example.ru", "тема", "тело")
        names = [c[0] for c in plain_cls.calls]
        assert names.index("starttls") < names.index("login")


class TestOwnerCanDiagnoseMail:
    """Настройка почты — четыре переменные в чужой панели, и до этой ручки
    владелец узнавал результат только по тому, пожаловался ли покупатель.
    Тот же приём, что уже выручил с Вордстатом: показать причину, а не гадать."""

    def _clean(self, monkeypatch):
        for name in ("SOZDATEL_SMTP_HOST", "SOZDATEL_SMTP_PORT", "SOZDATEL_SMTP_USER",
                     "SOZDATEL_SMTP_PASSWORD", "SOZDATEL_OWNER_EMAIL"):
            monkeypatch.delenv(name, raising=False)

    def _configured(self, monkeypatch, port="465", user="info@example.ru"):
        monkeypatch.setenv("SOZDATEL_SMTP_HOST", "smtp.example.ru")
        monkeypatch.setenv("SOZDATEL_SMTP_PORT", port)
        monkeypatch.setenv("SOZDATEL_SMTP_USER", user)
        monkeypatch.setenv("SOZDATEL_SMTP_PASSWORD", "secret")
        monkeypatch.setenv("SOZDATEL_OWNER_EMAIL", "owner@example.com")

    def test_owner_key_is_required(self):
        """Ручка показывает настройки сервера — посторонним её знать незачем."""
        assert client.get("/api/diag/mail").status_code == 401

    def test_names_every_missing_variable(self, monkeypatch):
        self._clean(monkeypatch)
        d = client.get("/api/diag/mail", headers=OWNER).json()
        joined = " ".join(d["problems"])
        for name in ("SOZDATEL_SMTP_HOST", "SOZDATEL_SMTP_USER",
                     "SOZDATEL_SMTP_PASSWORD", "SOZDATEL_OWNER_EMAIL"):
            assert name in joined, name
        assert d["configured"] is False

    def test_password_is_never_returned(self, monkeypatch):
        """Диагностика не имеет права стать способом прочитать пароль."""
        self._configured(monkeypatch)
        raw = client.get("/api/diag/mail", headers=OWNER).text
        assert "secret" not in raw
        assert client.get("/api/diag/mail", headers=OWNER).json()["settings"]["password_set"] is True

    def test_non_numeric_port_is_called_out(self, monkeypatch):
        self._configured(monkeypatch, port="четыреста")
        d = client.get("/api/diag/mail", headers=OWNER).json()
        assert any("SOZDATEL_SMTP_PORT" in p for p in d["problems"])

    def test_login_that_is_not_an_address_is_called_out(self, monkeypatch):
        """Частая ошибка: в логин пишут имя пользователя вместо адреса ящика,
        и тогда не работают ни SPF, ни DKIM."""
        self._configured(monkeypatch, user="info")
        d = client.get("/api/diag/mail", headers=OWNER).json()
        assert any("целиком" in p for p in d["problems"])

    def test_mode_follows_the_port(self, monkeypatch):
        self._configured(monkeypatch, port="465")
        assert client.get("/api/diag/mail", headers=OWNER).json()["settings"]["mode"] == "SSL"
        self._configured(monkeypatch, port="587")
        assert client.get("/api/diag/mail", headers=OWNER).json()["settings"]["mode"] == "STARTTLS"

    def test_without_an_address_nothing_is_sent(self, monkeypatch):
        """Открыть ручку, чтобы просто посмотреть настройки, должно быть
        безопасно — письмо уходит только когда его попросили."""
        self._configured(monkeypatch)
        import app.main as m
        sent = []
        monkeypatch.setattr(m.mailer, "send", lambda *a, **kw: sent.append(a))
        d = client.get("/api/diag/mail", headers=OWNER).json()
        assert d["test_send"] is None and sent == []

    def test_successful_send_is_reported_with_a_next_step(self, monkeypatch):
        from app import mailer
        self._configured(monkeypatch)
        got = []
        d = mailer.diagnose("boris@example.com", _send=lambda msg: got.append(msg["To"]))
        assert d["test_send"]["ok"] is True
        assert got == ["boris@example.com"]
        assert "Спам" in d["test_send"]["next"]

    def test_wrong_password_is_explained_in_plain_words(self, monkeypatch):
        """Получить в ответ SMTPAuthenticationError значит остаться там же,
        где был."""
        import smtplib
        from app import mailer
        self._configured(monkeypatch)
        def boom(msg):
            raise smtplib.SMTPAuthenticationError(535, b"5.7.8 auth failed")
        d = mailer.diagnose("boris@example.com", _send=boom)
        assert d["test_send"]["ok"] is False
        assert "логин или пароль" in d["test_send"]["error"]
        assert "SMTPAuthenticationError" in d["test_send"]["technical"]

    def test_ssl_mismatch_suggests_the_other_port(self, monkeypatch):
        import ssl as _ssl
        from app import mailer
        self._configured(monkeypatch, port="465")
        def boom(msg):
            raise _ssl.SSLError("wrong version number")
        d = mailer.diagnose("boris@example.com", _send=boom)
        assert "587" in d["test_send"]["error"]

    def test_unknown_host_is_explained(self, monkeypatch):
        import socket as _socket
        from app import mailer
        self._configured(monkeypatch)
        def boom(msg):
            raise _socket.gaierror("Name or service not known")
        d = mailer.diagnose("boris@example.com", _send=boom)
        assert "SOZDATEL_SMTP_HOST" in d["test_send"]["error"]

    def test_bad_recipient_address_does_not_reach_the_server(self, monkeypatch):
        from app import mailer
        self._configured(monkeypatch)
        got = []
        d = mailer.diagnose("не почта", _send=lambda msg: got.append(msg))
        assert d["test_send"]["ok"] is False and got == []

    def test_handle_never_returns_500(self, monkeypatch):
        """Диагностика, падающая пятисотой, бесполезна ровно тогда, когда
        нужна (принцип 7)."""
        import app.main as m
        self._configured(monkeypatch)
        monkeypatch.setattr(m.mailer, "send",
                            lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("что угодно")))
        r = client.get("/api/diag/mail?to=boris@example.com", headers=OWNER)
        assert r.status_code == 200
        assert r.json()["test_send"]["ok"] is False

    def test_unconfigured_mail_says_so_instead_of_trying(self, monkeypatch):
        from app import mailer
        self._clean(monkeypatch)
        d = mailer.diagnose("boris@example.com")
        assert d["test_send"]["ok"] is False
        assert "не настроена" in d["test_send"]["error"]


class TestDeskShowsMailState:
    """Почта — единственная подсистема, которая ломается молча: не ушло письмо
    со ссылкой входа или об оплате, и об этом никто не узнаёт. JSON-ручки для
    этого мало: владелец настраивает четыре переменные в чужой панели, где
    «разбегаются глаза», и читать сырой ответ ему негде."""

    def _desk(self):
        return (main_module.BASE_DIR.parent / "static" / "desk.html").read_text()

    def test_desk_has_a_mail_block(self):
        text = self._desk()
        assert 'id="mailbox"' in text
        assert "/api/diag/mail" in text

    def test_block_offers_a_test_send(self):
        """Без кнопки владельцу пришлось бы дожидаться живого покупателя,
        чтобы узнать, работает ли почта."""
        text = self._desk()
        assert 'id="mail-to"' in text and 'id="mail-send"' in text
        assert "sendTestMail" in text

    def test_green_means_a_letter_actually_went(self):
        """«Переменные заданы» — не то же самое, что «письма уходят». Зелёная
        надпись рядом с красной ошибкой противоречила бы сама себе."""
        text = self._desk()
        assert "переменные заданы" in text
        assert "письмо уходит" in text
        # зелёный статус ставится только в ветке удачной отправки
        after = text.split("sendTestMail")[1]
        assert 'mailbox-state ok' in after

    def test_block_loads_with_the_desk(self):
        assert "loadMail()" in self._desk()

    def test_technical_line_comes_after_the_human_one(self):
        """Одна строка SMTPAuthenticationError владельцу ничего не говорит,
        но и без неё непонятно, что чинить."""
        text = self._desk()
        assert "t.error + (t.technical" in text.replace("\n", " ")


class TestIdeaIsNotReadableByGuessing:
    """E6: страница результата адресовалась порядковым номером, и чужую идею
    читали перебором: 42 -> 41. Ссылкой на результат люди делятся намеренно,
    поэтому вход требовать нельзя — но адрес обязан быть неугадываемым.
    Тот же приём, что уже закрыл платный отчёт (A9)."""

    SECRET = "Секретная идея, которую человек никому не показывал"

    def _check(self, contact="", purpose="business"):
        from app.main import DemandCheck, Session, engine
        data = {"formulations": [{"phrase": "фраза", "count": 1200}],
                "best_phrase": "фраза",
                "verdict": {"level": "niche", "text": "Нишевый спрос"},
                "competitors": {"found": 900, "top": []},
                "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": ""}],
                "overall": {"value": 6, "weakest": "Спрос", "basis": "Среднее"}}
        with Session(engine) as s:
            rec = DemandCheck(idea=self.SECRET, best_count=1200, contact=contact,
                              purpose=purpose,
                              result_json=json.dumps(data, ensure_ascii=False))
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id, rec.public_id

    def _login(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_e6_" + contact, contact=contact)); s.commit()
        assert client.post(f"/account/verify?token=tok_e6_{contact}",
                          follow_redirects=False).status_code in (302, 303, 307)

    def _logout(self):
        client.post("/api/account/logout")

    # --- сама дыра ---

    def test_stranger_cannot_read_an_idea_by_the_sequential_number(self):
        rid, _ = self._check()
        self._logout()
        r = client.get(f"/r/{rid}", follow_redirects=False)
        assert r.status_code == 404
        assert self.SECRET not in r.text

    def test_report_teaser_hides_the_idea_from_the_same_guessing(self):
        """Тизер отчёта показывает текст идеи ровно так же — дыра была на
        двух страницах, а не на одной."""
        rid, _ = self._check()
        self._logout()
        r = client.get(f"/report/{rid}", follow_redirects=False)
        assert r.status_code == 404
        assert self.SECRET not in r.text

    def test_neighbouring_numbers_reveal_nothing(self):
        """Проверка «в лоб»: пройтись по номерам, как это сделал бы любопытный."""
        rid, _ = self._check()
        self._logout()
        for n in range(max(1, rid - 3), rid + 2):
            for path in (f"/r/{n}", f"/report/{n}"):
                assert self.SECRET not in client.get(path, follow_redirects=False).text, path

    # --- и при этом ссылкой по-прежнему можно делиться ---

    def test_the_public_link_works_for_anyone_who_has_it(self):
        """Главное ограничение задачи: делиться результатом можно и дальше,
        вход мы не требуем."""
        _, pid = self._check()
        self._logout()
        r = client.get(f"/r/{pid}")
        assert r.status_code == 200 and self.SECRET in r.text

    def test_public_link_is_not_guessable(self):
        _, pid = self._check()
        assert len(pid) >= 10
        assert not pid.isdigit()

    def test_two_checks_get_different_addresses(self):
        _, a = self._check()
        _, b = self._check()
        assert a != b

    # --- старые ссылки не бьются у тех, кто и так имел право ---

    def test_owner_with_the_old_numeric_link_is_redirected_to_the_new_one(self):
        rid, pid = self._check()
        r = client.get(f"/r/{rid}", headers=OWNER, follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == f"/r/{pid}"

    def test_author_of_the_check_keeps_his_old_link(self):
        """Человек, привязавший проверку к кабинету, не должен потерять
        закладку из-за нашей правки."""
        rid, pid = self._check(contact="mine_e6@example.com")
        self._login("mine_e6@example.com")
        try:
            r = client.get(f"/r/{rid}", follow_redirects=False)
            assert r.status_code == 307 and r.headers["location"] == f"/r/{pid}"
        finally:
            self._logout()

    def test_someone_elses_session_does_not_open_the_numeric_link(self):
        rid, _ = self._check(contact="mine_e6b@example.com")
        self._login("notmine_e6@example.com")
        try:
            assert client.get(f"/r/{rid}", follow_redirects=False).status_code == 404
        finally:
            self._logout()

    def test_report_redirect_keeps_the_query(self):
        """Иначе владельческий прогон и токен покупателя терялись бы на
        редиректе."""
        rid, pid = self._check()
        r = client.get(f"/report/{rid}?preview=quick", headers=OWNER, follow_redirects=False)
        assert r.status_code == 307
        assert r.headers["location"] == f"/report/{pid}?preview=quick"

    # --- ссылки, которые раздаём мы сами ---

    def test_demand_api_returns_the_public_address(self):
        """Витрина строит адрес из ответа сервера — иначе она увела бы
        человека на номер, который для него же и закрыт."""
        import app.main as m
        async def fake_check(idea):
            return {"formulations": [{"phrase": "п", "count": 10}], "best_phrase": "п",
                    "verdict": {"level": "niche", "text": "т"},
                    "competitors": {"found": 1, "top": []},
                    "scores": [], "overall": None}
        orig = m.check_demand
        m.check_demand = fake_check
        try:
            d = client.post("/api/demand", json={"idea": "Идея для проверки адреса"}).json()
        finally:
            m.check_demand = orig
        assert d["public_id"] and not str(d["public_id"]).isdigit()
        assert client.get(f"/r/{d['public_id']}").status_code == 200

    def test_showcases_send_people_to_the_public_address(self):
        for name in ("index.html", "audience-landing.html"):
            src = (main_module.BASE_DIR.parent / "static" / name).read_text()
            assert "data.public_id" in src, name

    def test_buyer_link_uses_the_public_address(self):
        """Ссылка из письма и из кабинета не должна вести на закрытый номер."""
        from app.main import ReportPurchase, Session, engine
        import app.main as m
        rid, pid = self._check(contact="buyer_e6@example.com")
        with Session(engine) as s:
            p = ReportPurchase(check_id=rid, idea=self.SECRET, tier="full", amount=2990,
                               status="paid", contact="buyer_e6@example.com")
            s.add(p); s.commit(); s.refresh(p)
            link = m._report_link(p)
        assert link.startswith(f"/report/{pid}?t=")
        assert f"/report/{rid}" not in link


class TestIdeasCanBeCompared:
    """E4: человек с пятью проверенными идеями видел пять одинаковых строк и
    не мог сказать, какая сильнее, не открыв каждую. Цифры уже посчитаны на
    бесплатной проверке — вопрос был только в том, чтобы их показать и
    расставить строки по силе."""

    def _login(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_e4_" + contact, contact=contact)); s.commit()
        assert client.post(f"/account/verify?token=tok_e4_{contact}",
                          follow_redirects=False).status_code in (302, 303, 307)

    def _logout(self):
        client.post("/api/account/logout")

    def _check(self, contact, idea, score, count, weakest="Спрос", broken=False):
        from app.main import DemandCheck, Session, engine
        if broken:
            raw = "{не json"
        else:
            raw = json.dumps({
                "formulations": [{"phrase": "ф", "count": count}],
                "best_phrase": "ф",
                "verdict": {"level": "niche", "text": "т"},
                "competitors": {"found": 10, "top": []},
                "scores": [],
                "overall": ({"value": score, "weakest": weakest, "basis": "б"}
                            if score is not None else None)}, ensure_ascii=False)
        with Session(engine) as s:
            rec = DemandCheck(idea=idea, contact=contact, best_count=count, result_json=raw)
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id

    def _cards(self, contact):
        self._login(contact)
        try:
            return client.get("/api/account/me").json()["checks"]
        finally:
            self._logout()

    def test_check_carries_the_numbers_it_was_judged_by(self):
        c = "cmp1@example.com"
        self._check(c, "Пошив штор", 7, 1200, weakest="Конкуренция")
        card = self._cards(c)[0]
        assert card["score"] == 7
        assert card["count"] == 1200
        assert card["weakest"] == "Конкуренция"

    def test_strongest_idea_comes_first(self):
        """Это и есть ответ на вопрос «во что вкладываться»: порядок строк."""
        c = "cmp2@example.com"
        self._check(c, "Слабая", 2, 30)
        self._check(c, "Сильная", 8, 5000)
        self._check(c, "Средняя", 5, 900)
        assert [x["idea"] for x in self._cards(c)] == ["Сильная", "Средняя", "Слабая"]

    def test_checks_without_a_score_go_last_but_stay_visible(self):
        """Вордстат мог промолчать. Сравнивать такую проверку не с чем, но
        прятать её из кабинета — значит терять работу человека."""
        c = "cmp3@example.com"
        self._check(c, "Без балла", None, None)
        self._check(c, "С баллом", 4, 400)
        cards = self._cards(c)
        assert [x["idea"] for x in cards] == ["С баллом", "Без балла"]
        assert cards[-1]["score"] is None

    def test_broken_json_does_not_hide_the_card(self):
        """Принцип 7: без цифр — значит без цифр, но строка на месте."""
        c = "cmp4@example.com"
        self._check(c, "Битая запись", 9, 999, broken=True)
        cards = self._cards(c)
        assert len(cards) == 1 and cards[0]["idea"] == "Битая запись"
        assert cards[0]["score"] is None

    def test_cabinet_renders_the_figures(self):
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "figures" in text and "c.score" in text and "c.count" in text
        assert "запросов/мес" in text

    def test_note_appears_only_when_there_is_something_to_compare(self):
        """Над одной строкой подпись про сортировку — шум."""
        text = (main_module.BASE_DIR.parent / "static" / "account.html").read_text()
        assert "items.length > 1" in text
        assert "От самой сильной идеи к самой слабой" in text

    def test_link_from_the_cabinet_opens(self):
        """Ссылка ведёт на неугадываемый адрес, а не на закрытый номер (E6)."""
        from app.main import DemandCheck, Session, engine
        c = "cmp5@example.com"
        rid = self._check(c, "Идея из кабинета", 6, 800)
        with Session(engine) as s:
            pid = s.get(DemandCheck, rid).public_id
        card = self._cards(c)[0]
        assert card["result_url"] == f"/r/{pid}"
        assert client.get(card["result_url"]).status_code == 200


class TestCabinetLinksSurvivedTheAddressChange:
    """Регрессия, которую внесла E6: ссылки в кабинете остались с порядковым
    номером. Он открывается только у хозяина ПРОВЕРКИ, а у заявки на живой
    тест проверка к кабинету может быть не привязана вовсе — и человек
    получал 404 на своей же заявке."""

    def _login(self, contact):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token="tok_lnk_" + contact, contact=contact)); s.commit()
        assert client.post(f"/account/verify?token=tok_lnk_{contact}",
                          follow_redirects=False).status_code in (302, 303, 307)

    def test_live_test_order_link_opens_for_its_buyer(self):
        from app.main import DemandCheck, LiveTestOrder, Session, engine
        contact = "lnk1@example.com"
        with Session(engine) as s:
            src = DemandCheck(idea="Идея заявки",     # контакт НЕ проставлен
                              result_json='{"verdict": {"level": "niche", "text": "т"}}')
            s.add(src); s.commit(); s.refresh(src)
            s.add(LiveTestOrder(idea="Идея заявки", contact=contact,
                                status="pending_payment", check_id=src.id))
            s.commit()
        self._login(contact)
        try:
            url = client.get("/api/account/me").json()["orders"][0]["continue_url"]
            assert not url.rstrip("/").split("/")[-1].isdigit(), url
            assert client.get(url).status_code == 200
        finally:
            client.post("/api/account/logout")

    def test_link_is_absent_when_the_check_is_gone(self):
        """Ссылка в никуда хуже отсутствия ссылки."""
        from app.main import LiveTestOrder, Session, engine
        contact = "lnk2@example.com"
        with Session(engine) as s:
            s.add(LiveTestOrder(idea="Заявка без проверки", contact=contact,
                                status="pending_payment", check_id=999777))
            s.commit()
        self._login(contact)
        try:
            assert client.get("/api/account/me").json()["orders"][0]["continue_url"] is None
        finally:
            client.post("/api/account/logout")


class TestUnmeasuredDemandIsNotSoldAsMeasured:
    """A12: когда Вордстат не дал числа, страница вела себя так, будто дала.
    Напротив каждой фразы стояло «почти не ищут» — вывод о рынке на месте
    отсутствующего числа, — а финал продавал тест и разбор как обычно.

    Это самое вероятное состояние прода сегодня: у владельца ещё нет
    OAuth-токена Вордстата. То есть каждый посетитель видел три строки
    «почти не ищут» и уходил хоронить живую идею.

    Разметка блока лежит на странице всегда, показывает его скрипт по уровню
    вердикта — поэтому поведение сторожит браузерный тест
    tests/test_mobile.py::test_unmeasured_demand_stops_selling_in_a_real_browser,
    а проверки ниже отвечают за тексты."""

    def _check(self):
        from app.main import DemandCheck, Session, engine
        data = {"formulations": [{"phrase": "пошив штор", "count": None},
                                 {"phrase": "сшить шторы", "count": None}],
                "best_phrase": "пошив штор",
                "verdict": {"level": "unknown",
                            "text": "Данные Яндекса о числе запросов сейчас недоступны."},
                "competitors": {"found": None, "top": []},
                "scores": [], "overall": None}
        with Session(engine) as s:
            rec = DemandCheck(idea="Идея без цифр",
                              result_json=json.dumps(data, ensure_ascii=False))
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.public_id

    def test_absent_number_is_not_called_a_market_finding(self):
        """`count = None` означает «Вордстат не дал числа» (см. докстринг
        wordstat_best), а не «спроса нет». Ноль теперь читается отдельной
        подписью «не ищут» — это ответ, а не сбой, и путать их нельзя."""
        text = client.get(f"/r/{self._check()}").text
        assert "не удалось измерить" in text
        assert "почти не ищут" not in text.split("v.level === 'weak'")[0]

    def test_page_says_the_check_did_not_happen(self):
        text = client.get(f"/r/{self._check()}").text
        assert "v.level === 'unknown'" in text
        block = text.split("v.level === 'unknown'")[1][:2200]
        assert "не состоялась" in block
        assert "не значит, что спроса нет" in block

    def test_free_retry_becomes_the_main_action(self):
        """Единственное честное действие здесь бесплатное — повторить."""
        text = client.get(f"/r/{self._check()}").text
        block = text.split("v.level === 'unknown'")[1][:2200]
        assert "Проверить ещё раз" in block
        assert "getElementById('order').className = 'alt-path'" in block
        assert "getElementById('alt-report').className = 'alt-path'" in block

    def test_buttons_are_still_there(self):
        """Отговариваем, но не решаем за человека."""
        text = client.get(f"/r/{self._check()}").text
        assert 'id="order-btn"' in text and 'id="alt-report"' in text

    def test_report_block_stops_claiming_we_counted_something(self):
        """«Разбор на данных, которые мы уже честно посчитали» на экране, где
        ничего не посчитано, — то же враньё, только ниже."""
        text = client.get(f"/r/{self._check()}").text
        block = text.split("v.level === 'unknown'")[1][:2200]
        assert "alt-p" in block
        assert "цифр спроса" in block

    def test_header_stops_promising_the_next_stage(self):
        """Строку «Дальше — …» убрали из шапки целиком (кастдев 2026-08-02),
        поэтому обещать следующий этап поверх честного «проверка не
        состоялась» больше нечем. Совет повторить проверку остался в самом
        блоке — там, где человек его и читает."""
        text = client.get(f"/r/{self._check()}").text
        assert 'id="path-next-text"' not in text
        block = text.split("v.level === 'unknown'")[1][:2200]
        assert "Проверить ещё раз" in block

    def test_retry_leads_to_the_right_showcase(self):
        """Получателя соцконтракта нельзя возвращать на витрину фаундеров."""
        text = client.get(f"/r/{self._check()}").text
        block = text.split("v.level === 'unknown'")[1][:2200]
        assert "AUDIENCE.home" in block


class TestLandingPromisesNothingOnTheOwnersBehalf:
    """A14: страница, за которую платят 1490 ₽ нам и 3–5 тысяч Яндексу,
    несла зашитые обещания, которых владелец идеи не давал.

    Найдено кастдев-проходом по `/l/`. Три вещи сразу:
      · «Ранним — 50% на первый месяц» — скидку мы придумали за него и
        показывали живым людям;
      · «Запускаемся скоро» / «продукт готовится к запуску» — предполагало,
        что продукта ещё нет. Для услуги (пошив штор, груминг, пекарня)
        человек ищет исполнителя СЕГОДНЯ, и такая рамка его отпугивает —
        а мы потом по этой конверсии выносим вердикт «спроса нет»;
      · слово «лендинг» в подвале — запрещено во всех текстах для
        пользователя, но тесты сканировали только `static/`, а шаблон
        лежит в `app/`.
    """

    def _text(self):
        import re
        html = main_module.render_landing(VALID_OFFER)
        body = html[html.index('<div class="wrap">'):html.index("<script>")]
        return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", body))

    def test_no_invented_discount(self):
        """Скидка на чужой продукт — обещание, которое владелец не давал и
        может не сдержать."""
        t = self._text().lower()
        assert "50%" not in t
        assert "скидк" not in t
        assert "первый месяц" not in t

    def test_does_not_assume_the_product_is_unlaunched(self):
        """Услугу заказывают сегодня, а не «когда откроем доступ»."""
        t = self._text().lower()
        for bad in ("запускаемся скоро", "готовится к запуску",
                    "ранний доступ", "ранним", "откроем доступ"):
            assert bad not in t, bad

    def test_forbidden_words_are_absent_from_the_landing_too(self):
        """Файл лежит в app/, поэтому под сторожа по static/ не попадал —
        а это единственная страница, которую видят посторонние люди."""
        t = self._text().lower()
        for bad in ("лендинг", "оффер", "смоук", "smoke", "гипотез", "конверси"):
            assert bad not in t, bad

    def test_it_still_says_what_will_happen_next(self):
        """Убрать обещание не значит оставить человека без ответа."""
        t = self._text()
        assert "Оставить заявку" in t
        assert "напишем вам" in t.lower()
        assert "Заявка принята" in t

    def test_privacy_consent_survived(self):
        """152-ФЗ и модерация Директа: без ссылки на политику страница не
        имеет права собирать контакты."""
        html = main_module.render_landing(VALID_OFFER)
        assert 'href="/legal"' in html
        assert "политикой персональных данных" in html

    def test_the_offer_itself_still_reaches_the_page(self):
        """Правка текстов вокруг формы не должна съесть саму идею."""
        html = main_module.render_landing(VALID_OFFER)
        assert VALID_OFFER["h1"] in html
        assert VALID_OFFER["sub"] in html
        assert VALID_OFFER["pains"][0]["h2"] in html


class TestBrokenLinkLandsOnAWorkingPage:
    """E7: битая ссылка на результат отдавала главную с СЫРЫМИ слотами.

    Найдено кастдев-проходом «что видит человек, когда всё сломалось».
    Оба 404-пути (`/r/` и `/report/`) возвращали `_static("index.html")`
    в обход `_fill_server_values` — и человек читал на главной буквально
    «Больше __SIGNAL_PCT__ — идея живая, меньше __DEAD_PCT__». Первое, что
    он видит о сервисе, — что сервис сломан.

    Ссылками на результат делятся намеренно (см. CLAUDE.md про public_id),
    поэтому обрезанная в мессенджере или устаревшая ссылка — обычное дело,
    а не экзотика. Второе следствие того же места: человек приходил по
    ссылке «посмотри мою проверку», молча попадал на главную и не понимал,
    что произошло. Поэтому 404 не просто чинится, а объясняется — и
    оставляет форму проверки под рукой (принцип 7: деградация вместо
    ошибки, но не молчаливая).
    """

    def test_result_404_has_no_raw_slots(self):
        r = client.get("/r/nosuchcheck")
        assert r.status_code == 404
        assert "__" not in _slots(r.text), _slots(r.text)

    def test_report_404_has_no_raw_slots(self):
        r = client.get("/report/nosuchcheck")
        assert r.status_code == 404
        assert "__" not in _slots(r.text), _slots(r.text)

    def test_404_explains_what_happened(self):
        """Молча подменить запрошенную страницу главной — значит соврать,
        что человек попал куда хотел."""
        t = client.get("/r/nosuchcheck").text
        assert "не нашли" in t.lower()
        assert "ссылк" in t.lower()

    def test_404_keeps_the_free_check_at_hand(self):
        """Тупик из битой ссылки делать незачем: форма проверки уже здесь."""
        t = client.get("/r/nosuchcheck").text
        assert 'id="idea"' in t
        assert 'id="check-btn"' in t

    def test_normal_home_page_says_nothing_about_a_broken_link(self):
        """Записка появляется только по 404, иначе она пугает всех подряд."""
        r = client.get("/")
        assert r.status_code == 200
        assert "не нашли" not in r.text.lower()

    def test_normal_home_page_has_no_raw_slots_either(self):
        """Сторож на будущее: новый слот в index.html без подстановки
        сломает главную ровно так же."""
        r = client.get("/")
        assert "__" not in _slots(r.text), _slots(r.text)


class TestMagicLinkSurvivesMailScanners:
    """A15: почтовый сканер съедал ссылку входа раньше человека.

    Найдено кастдев-проходом по пути «покупатель вернулся за своим отчётом».
    Токен гасился прямо в GET-обработчике, а почтовые провайдеры и антивирусы
    (mail.ru, Яндекс, Kaspersky) открывают ссылки из писем сами, до человека,
    чтобы проверить их на вредоносность. Последствий было два, и оба тяжёлые:

      · **человек не мог войти вообще.** Он кликает — «Ссылка недействительна»,
        просит новую — сканер съедает и её. Замкнутый круг, из которого нет
        выхода: заплатил 2990 ₽ и не может открыть купленное;
      · **сессия выдавалась сканеру.** GET отвечал 307 с `Set-Cookie` на
        180 дней — то есть кабинет покупателя открывался машине, которая
        просто проверяла ссылку.

    Лечится тем, что вход происходит на POST, а GET только показывает
    страницу с кнопкой: сканеры ходят GET и HEAD, форму не отправляют.
    Заодно страница отказа перестала быть голым `<p>` без стилей — человек,
    потерявший доступ к оплаченному, видел чёрный Times New Roman на белом,
    и это выглядит как сломанный сайт, а не как объяснение.
    """

    def _token(self, name, contact="scan@example.com", **kw):
        from app.main import MagicLinkToken, Session, engine
        with Session(engine) as s:
            s.add(MagicLinkToken(token=name, contact=contact, **kw))
            s.commit()
        return name

    def test_scanner_get_does_not_consume_the_link(self):
        """Главное: после автоматического открытия ссылка ещё жива."""
        self._token("scan_alive")
        client.get("/account/verify?token=scan_alive", follow_redirects=False)
        r = client.post("/account/verify?token=scan_alive", follow_redirects=False)
        assert r.status_code in (302, 303, 307), r.status_code
        client.cookies.clear()

    def test_scanner_get_gets_no_session(self):
        """Сессия на 180 дней не должна достаться машине, проверявшей ссылку."""
        client.cookies.clear()
        self._token("scan_nocookie")
        r = client.get("/account/verify?token=scan_nocookie", follow_redirects=False)
        assert r.status_code == 200
        assert "sozdatel_session" not in r.cookies
        client.cookies.clear()

    def test_get_shows_a_page_with_a_button(self):
        self._token("scan_page")
        t = client.get("/account/verify?token=scan_page").text
        assert "<!doctype" in t.lower()
        assert 'method="post"' in t.lower()
        assert "Войти" in t

    def test_get_names_the_account_being_entered(self):
        """Человек должен видеть, в чей кабинет входит: почт бывает две."""
        self._token("scan_named", contact="buyer@example.com")
        assert "buyer@example.com" in client.get("/account/verify?token=scan_named").text

    def test_post_logs_in_and_burns_the_token(self):
        self._token("scan_burn")
        r = client.post("/account/verify?token=scan_burn", follow_redirects=False)
        assert r.status_code in (302, 303, 307)
        assert r.cookies.get("sozdatel_session")
        client.cookies.clear()
        again = client.post("/account/verify?token=scan_burn", follow_redirects=False)
        assert again.status_code == 400
        client.cookies.clear()

    def test_expired_link_is_refused_on_both_verbs(self):
        from app.main import utcnow
        from datetime import timedelta
        import app.main as m
        self._token("scan_old",
                    created_at=utcnow() - timedelta(minutes=m.MAGIC_LINK_TTL_MINUTES + 1))
        assert client.get("/account/verify?token=scan_old").status_code == 400
        assert client.post("/account/verify?token=scan_old",
                           follow_redirects=False).status_code == 400

    def test_refusal_is_a_real_page_not_a_bare_fragment(self):
        """Потерявший доступ к оплаченному не должен решить, что сайт умер."""
        t = client.get("/account/verify?token=no-such-token").text
        assert "<!doctype" in t.lower()
        assert "IBM Plex" in t
        assert "устарела" in t or "недействительна" in t

    def test_refusal_offers_the_way_back(self):
        t = client.get("/account/verify?token=no-such-token").text
        assert 'href="/account"' in t

    def test_verify_page_has_no_raw_slots(self):
        self._token("scan_slots")
        for t in (client.get("/account/verify?token=scan_slots").text,
                  client.get("/account/verify?token=nope").text):
            assert "__" not in _slots(t), _slots(t)


class TestFrequencyIsAttributedWithoutPointing:
    """B8: подпись про подсказанную Вордстатом формулировку указывала «справа».

    Найдено кастдев-проходом 2026-07-27. Когда Вордстат подсказывает более
    ходовую формулировку, чем угадала LLM, число в строке относится к ЕЙ, а не
    к тому, что человек читает слева (`matched_phrase`, см. `_best_related`).
    Подпись существует ровно для того, чтобы это назвать -- принцип 1, число
    не приписывается чужой фразе молча.

    Но написана она была через направление на экране: «цифра справа про неё».
    На узком экране `.freq-row` переключается в столбик, число уходит ВНИЗ, и
    подпись показывает не туда. Трафик из Директа преимущественно мобильный,
    то есть неверный вариант видело большинство.

    Лечится не вёрсткой, а порядком и словами: атрибуция идёт ПОСЛЕ числа и
    не называет сторон вообще — тогда она верна при любой раскладке.
    """

    def _script(self):
        """Без комментариев: объяснение «почему» само называет старый текст,
        и сторож ловил бы собственную документацию."""
        import re as _re
        t = _read_static("result.html")
        t = _re.sub(r"/\*.*?\*/", "", t, flags=_re.S)
        return "\n".join(l for l in t.splitlines()
                          if not l.lstrip().startswith("//"))

    def test_attribution_names_no_direction(self):
        """Любое слово о стороне экрана снова разъедется с вёрсткой."""
        t = self._script()
        for bad in ("цифра справа", "число справа", "справа про неё",
                    "цифра слева", "цифра выше"):
            assert bad not in t, bad

    def test_attribution_still_says_the_number_is_not_about_the_asked_phrase(self):
        """Убрать направление не значит убрать саму оговорку: без неё ручная
        проверка исходной фразы в Вордстате покажет другое число."""
        t = self._script()
        assert "matched_phrase" in t
        assert "ходов" in t          # «более ходовая формулировка»
        assert "Вордстат" in t

    def test_weak_branch_does_not_call_a_prompted_phrase_the_users_own(self):
        """Тот же обман на другом экране: в ветке слабого спроса цифра могла
        быть посчитана по подсказке Вордстата, а текст звал её «вашей
        формулировкой»."""
        t = self._script()
        assert "Самую популярную из ваших формулировок" not in t


class TestScaleCaptionsLookAlike:
    """B9 (шаг 2): подписи под шкалами оценки выглядели набором обрывков.

    Найдено на живой форме данных 2026-07-28, после жалобы владельца «сделай,
    чтобы всё было единообразно». Два дефекта в одном месте:

      · **«Спрос» — единственная из четырёх ячеек без подписи вообще.**
        `check_demand` кладёт ей `note: ""` (три остальные пишет модель), и
        человек видит три объяснённых числа и одно голое. Причём голым
        оказывается самое важное: спрос — единственная шкала на реальных
        данных Яндекса и одновременно потолок общего балла.
        Фикстуры это прятали — в них у «Спроса» подпись была, чего реальный
        движок не делает (тот же урок, что в A13: заглушка обязана совпадать
        с настоящим поведением).
      · **пунктуация вразнобой.** Модель пишет фрагменты как придётся:
        «Рынок растёт.» с точкой, «Начать можно одной» без. Рядом в сетке это
        читается как небрежность.

    Правится на выдаче, а не при записи: тогда чинятся и уже сохранённые
    проверки, и правило живёт в одном месте на одном языке. В ячейке подпись
    работает как caption — точка в конце не нужна; в тизере отчёта те же
    заметки идут отдельными абзацами, и там точка обязательна.
    """

    def _served(self, rec_id):
        html_out = client.get(f"/r/{pub(rec_id)}").text
        raw = html_out.split("const DATA = ", 1)[1].split(";\n", 1)[0]
        return json.loads(raw)

    def _make(self, notes, count=480):
        import app.main as m
        from app.main import DemandCheck, Session, engine
        data = {
            "formulations": [{"phrase": "груминг с выездом", "count": count}],
            "best_phrase": "груминг с выездом",
            "verdict": {"level": "niche", "text": "Спрос небольшой."},
            "competitors": {"found": 900, "top": [{"title": "Г", "domain": "g.ru"}]},
            "scores": [{"key": "demand", "label": "Спрос", "value": 4, "note": notes[0]},
                       {"key": "competition", "label": "Конкуренция", "value": 5, "note": notes[1]},
                       {"key": "timing", "label": "Своевременность", "value": 8, "note": notes[2]},
                       {"key": "execution", "label": "Реализуемость", "value": 8, "note": notes[3]}],
            "overall": {"value": 4, "weakest": "Спрос", "basis": "б"},
        }
        with Session(engine) as s:
            rec = DemandCheck(idea="Груминг с выездом на дом", best_count=count,
                              result_json=json.dumps(data, ensure_ascii=False))
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id

    def test_demand_scale_gets_a_caption_like_its_neighbours(self):
        """Настоящий check_demand кладёт сюда пустую строку — значит подпись
        обязан дописать тот, кто отдаёт страницу."""
        rid = self._make(["", "Много мастеров", "Рынок растёт.", "Начать можно одной"])
        notes = [s["note"] for s in self._served(rid)["scores"]]
        assert all(notes), notes

    def test_demand_caption_is_built_from_the_number_not_invented(self):
        """Принцип 1: подпись к шкале спроса — это её же частотность."""
        rid = self._make(["", "а", "б", "в"], count=2400)
        demand = self._served(rid)["scores"][0]
        # Неразрывный пробел -- тот же разделитель, что даёт toLocaleString('ru-RU')
        # для частотностей выше по странице: число не должно рваться переносом.
        assert "2\u00a0400" in demand["note"], demand["note"]

    def test_missing_frequency_does_not_invent_a_caption(self):
        """Нет данных — так и говорим, а не пишем «0 запросов»."""
        rid = self._make(["", "а", "б", "в"], count=None)
        demand = self._served(rid)["scores"][0]
        assert "0" not in demand["note"]
        assert "недоступн" in demand["note"].lower(), demand["note"]

    def test_captions_do_not_end_with_a_period(self):
        """В ячейке это подпись под числом, а не предложение: одни с точкой,
        другие без — та самая «каша», на которую жаловался владелец."""
        rid = self._make(["", "Много мастеров", "Рынок растёт.", "Начать можно одной"])
        for s in self._served(rid)["scores"]:
            assert not s["note"].endswith("."), s["note"]

    def test_caption_keeps_its_meaning(self):
        """Нормализация не должна съедать текст."""
        rid = self._make(["", "Много частных мастеров.", "б", "в"])
        assert self._served(rid)["scores"][1]["note"] == "Много частных мастеров"

    def test_teaser_paragraphs_all_end_as_sentences(self):
        """Те же заметки в тизере отчёта идут отдельными абзацами — там точка
        обязательна, иначе абзац выглядит оборванным."""
        import app.main as m
        from app.main import Session, engine
        rid = self._make(["", "Много мастеров", "Рынок растёт.", "Начать можно одной"])
        with Session(engine) as s:
            data = json.loads(s.get(m.DemandCheck, rid).result_json)
        prev = m._report_preview(data)
        for key in ("competition_note", "timing_note", "execution_note"):
            assert prev[key].endswith("."), (key, prev[key])


class TestSharpenCardDoesNotGlueTwoFields:
    """B9 (шаг 3): карточка заострения склеивала два поля в кривое предложение.

    Найдено кастдев-проходом за фаундера 2026-07-28. Шаг заострения в прежних
    проходах всегда пропускался кнопкой «Пропустить», поэтому дефект дожил.

    Движок отдаёт боль двумя полями: `h2` — название боли, `p` — объяснение,
    каждое самостоятельное предложение (на посадочной `/l/` они и рисуются
    отдельными элементами). Карточка же склеивала их через « — » в одну
    строку, и объяснение приносило с собой свою заглавную букву:

        Ателье срывают сроки — Обещают три недели, шьют полтора месяца

    Заглавная после тире посреди строки — не опечатка модели, а наш шаблон:
    так склеит ЛЮБУЮ пару. Плюс точка в конце то есть, то нет — модель пишет
    как придётся. Это последний бесплатный экран перед решением платить.

    Лечится тем, что поля перестают притворяться одним предложением: название
    боли и объяснение — разными строками, как «Для кого» рядом. Пунктуацию
    приводит к виду сервер, там же, где живёт остальная нормализация (B9 шаг 2).
    """

    def _script(self):
        return _read_static("result.html")

    def test_card_does_not_join_pain_fields_with_a_dash(self):
        t = self._script()
        assert "pains[0].h2 || '')} — ${" not in t
        assert "sharp-pain" in t, "название боли и объяснение должны быть разными узлами"

    def test_explanation_is_normalised_to_a_sentence_on_the_server(self, monkeypatch):
        """Точку дописывает сервер, а не шаблон: правило уже живёт в Python
        (_as_sentence), второй копии в JS быть не должно."""
        import app.main as m
        offers = [dict(VALID_OFFER, pains=[{"h2": "Ателье срывают сроки",
                                            "p": "Обещают три недели, шьют полтора месяца"}])]

        async def fake_sharpen(idea, *a, **kw):
            return {"sharpened_note": "", "warning": "", "offers": offers}

        monkeypatch.setattr(m, "sharpen_idea", fake_sharpen)
        r = client.post("/api/sharpen", json={"idea": "Идея достаточной длины для проверки"})
        assert r.status_code == 200, r.text
        pain = r.json()["offers"][0]["pains"][0]
        assert pain["p"].endswith("."), pain
        assert pain["h2"] == "Ателье срывают сроки", pain     # название боли не трогаем

    def test_normalisation_does_not_double_existing_punctuation(self, monkeypatch):
        import app.main as m
        offers = [dict(VALID_OFFER, pains=[{"h2": "Срывают сроки", "p": "Шьют полтора месяца."}])]

        async def fake_sharpen(idea, *a, **kw):
            return {"sharpened_note": "", "warning": "", "offers": offers}

        monkeypatch.setattr(m, "sharpen_idea", fake_sharpen)
        r = client.post("/api/sharpen", json={"idea": "Идея достаточной длины для проверки"})
        assert r.json()["offers"][0]["pains"][0]["p"] == "Шьют полтора месяца."

    def test_offer_without_pains_still_passes_through(self, monkeypatch):
        """Нормализация не должна ронять вариант с пустым или битым полем."""
        import app.main as m
        offers = [dict(VALID_OFFER, pains=[]), dict(VALID_OFFER, pains=[{"h2": "х"}])]

        async def fake_sharpen(idea, *a, **kw):
            return {"sharpened_note": "", "warning": "", "offers": offers}

        monkeypatch.setattr(m, "sharpen_idea", fake_sharpen)
        r = client.post("/api/sharpen", json={"idea": "Идея достаточной длины для проверки"})
        assert r.status_code == 200
        assert len(r.json()["offers"]) == 2


class TestNextStepSpeaksToTheBuyer:
    """A17: карточка проекта диктовала покупателю следующий шаг по-владельчески.

    Найдено кастдев-проходом по ПЛАТНОМУ пути 2026-07-28: оплатил тест на
    реальных людях, вошёл в кабинет — и в карточке своего проекта прочитал
    строку «следующий шаг», написанную не для него.

    `_smoke_card` собирает `next_step` один раз для обеих панелей — и
    владельческой `/desk`, и покупательской `/account` (так и задумано: один
    язык на двоих). Но написан он был владельческим:

      · «Запустить Директ на страницу — инструкция: **/guide/direct**» —
        человеку показывали путь в адресной строке вместо ссылки. Кликнуть
        нельзя, надо перепечатывать руками;
      · «Сигнал есть → идея в очередь на **MVP**» — очередь владельца, не его;
      · «Спроса нет → идею **в архив**» — архив опять же владельческий;
      · «Серая зона → второй **оффер** на том же **трафике**» — «оффер»
        запрещён во всех текстах для пользователя (принцип 5), «трафик» тоже.

    **Сторож на запрещённые слова этого не ловил:** он сканирует `static/`, а
    строка живёт в `app/main.py`. Ровно та же дыра, что в A14, где «лендинг»
    отсиделся в `app/landing_template.html`. Поэтому проверяем не файл, а
    вывод `_smoke_card` по всем ветвям вердикта.
    """

    def _cards(self):
        """Карточка во всех состояниях, через которые проходит проект."""
        import app.main as m
        p = m.SmokeProject(idea_id="x", product_name="П", idea_text="и",
                           offer_json="{}", landing_html="<h1>т</h1>",
                           click_target=40, lead_rate_signal=0.08, lead_rate_dead=0.04)
        return {
            "нет визитов": m._smoke_card(p, 0, 0),
            "копим": m._smoke_card(p, 12, 1),
            "сигнал есть": m._smoke_card(p, 40, 8),
            "спроса нет": m._smoke_card(p, 40, 0),
            "серая зона": m._smoke_card(p, 40, 2),
        }

    def test_no_forbidden_words_anywhere_in_the_next_step(self):
        for label, card in self._cards().items():
            t = card["next_step"].lower()
            for bad in ("оффер", "лендинг", "трафик", "mvp", "архив", "конверси"):
                assert bad not in t, f"{label}: {card['next_step']}"

    def test_the_grey_zone_covers_all_verdicts(self):
        """Сторож бесполезен, если ветви на самом деле не разошлись."""
        steps = {c["next_step"] for c in self._cards().values()}
        assert len(steps) == 5, steps

    def test_no_raw_path_is_shown_as_text(self):
        """Путь в адресной строке — не текст для человека, это ссылка."""
        for label, card in self._cards().items():
            assert "/guide" not in card["next_step"], f"{label}: {card['next_step']}"

    def test_first_step_offers_the_instruction_as_a_link(self):
        card = self._cards()["нет визитов"]
        assert card["next_link"] == {"href": "/guide/direct",
                                     "text": "Пошаговая инструкция"}, card

    def test_other_states_carry_no_link(self):
        for label, card in self._cards().items():
            if label != "нет визитов":
                assert card["next_link"] is None, f"{label}: {card['next_link']}"

    def test_cabinet_renders_the_link_not_the_path(self):
        assert "next_link" in _read_static("account.html")
        assert "next_link" in _read_static("desk.html")


class TestPrintedPlanLooksLikeADocument:
    """C5: распечатанный бизнес-план не выглядел документом.

    Найдено кастдев-проходом 2026-07-28 по пути соцконтракта: человек несёт
    разбор в соцзащиту, и комиссии он нужен НА БУМАГЕ — кнопка «Скачать PDF»
    для этой аудитории не украшение, а способ доставки.

    Печать сама по себе работает (`@media print` прячет шапку, оглавление и
    витрину тарифов, все разделы уходят на лист). Но лист был озаглавлен
    **«Отчёт по идее»** независимо от того, что человек купил, — включая тариф,
    который мы сами называем «Бизнес-план» и под этим именем продаём. Комиссия
    получала документ, название которого не совпадает ни с чеком, ни с тем,
    зачем его принесли.

    И на листе **не было даты**. Документ без даты для комиссии — не документ;
    дата у нас есть готовая, это день оплаты разбора.
    """

    def _buy(self, purpose="social_contract", tier="full"):
        import app.main as m
        from app.main import DemandCheck, ReportPurchase, Session, engine
        data = {"formulations": [{"phrase": "ф", "count": 480}],
                "verdict": {"level": "niche", "text": "т"},
                "competitors": {"found": 9, "top": []},
                "scores": [{"key": "demand", "label": "Спрос", "value": 4, "note": ""}],
                "overall": {"value": 4, "weakest": "Спрос"}}
        with Session(engine) as s:
            c = DemandCheck(idea="Груминг с выездом", purpose=purpose,
                            result_json=json.dumps(data, ensure_ascii=False))
            s.add(c); s.commit(); s.refresh(c)
            p = ReportPurchase(check_id=c.id, idea=c.idea, tier=tier, status="paid",
                               contact="m@example.com", amount=2990,
                               report_json=json.dumps({"viability_score": 60,
                                                       "viability_summary": "с",
                                                       "top_risks": [], "sections": []},
                                                      ensure_ascii=False))
            s.add(p); s.commit(); s.refresh(p)
            return c.public_id, p.access_token, p.created_at

    def test_full_tier_is_titled_as_the_business_plan_that_was_bought(self):
        pid, tok, _ = self._buy(tier="full")
        t = client.get(f"/report/{pid}?t={tok}").text
        assert "<h1>Бизнес-план</h1>" in t, t[t.find("<h1"):t.find("<h1") + 120]

    def test_quick_tier_keeps_its_own_name(self):
        pid, tok, _ = self._buy(tier="quick")
        t = client.get(f"/report/{pid}?t={tok}").text
        assert "<h1>Быстрый разбор</h1>" in t

    def test_names_come_from_the_price_list_not_from_the_template(self):
        """Третья копия названий тарифов разъехалась бы, как уже разъезжались
        цены (B5): заголовок берётся из REPORT_PRICES."""
        import app.main as m
        for tier, cfg in m.REPORT_PRICES.items():
            pid, tok, _ = self._buy(tier=tier)
            assert f"<h1>{cfg['label']}</h1>" in client.get(f"/report/{pid}?t={tok}").text

    def test_unpaid_teaser_is_not_called_a_business_plan(self):
        """До оплаты человек видит тизер — называть его купленным тарифом
        значит обещать то, чего он ещё не получил (принцип 3)."""
        import app.main as m
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            c = DemandCheck(idea="Идея без покупки",
                            result_json=json.dumps({"formulations": [], "verdict": {},
                                                    "competitors": {}, "scores": [],
                                                    "overall": {}}, ensure_ascii=False))
            s.add(c); s.commit(); s.refresh(c)
            pid = c.public_id
        t = client.get(f"/r/{pid}").text and client.get(f"/report/{pid}").text
        assert "<h1>Бизнес-план</h1>" not in t
        assert "<h1>Отчёт по идее</h1>" in t

    def test_printed_page_carries_the_date_it_was_bought(self):
        pid, tok, made = self._buy()
        t = client.get(f"/report/{pid}?t={tok}").text
        assert "doc-meta" in t, "на листе нет строки с датой"
        assert f"{made.day} " in t or f"{made.day:02d}" in t, made

    def test_date_line_is_hidden_on_screen_and_shown_in_print(self):
        """На экране дата лишняя — там и так видно, что отчёт открыт. Нужна
        она ровно на листе, который уходит в комиссию."""
        t = _read_static("report.html")
        assert ".doc-meta{display:none}" in t.replace(" ", "")
        assert ".doc-meta" in t.split("@media print")[1]

    def test_unpaid_teaser_has_no_date_line(self):
        """Тизер не документ, и датировать его нечем."""
        import app.main as m
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            c = DemandCheck(idea="Ещё идея без покупки",
                            result_json=json.dumps({"formulations": [], "verdict": {},
                                                    "competitors": {}, "scores": [],
                                                    "overall": {}}, ensure_ascii=False))
            s.add(c); s.commit(); s.refresh(c)
            pid = c.public_id
        assert 'class="doc-meta"' not in client.get(f"/report/{pid}").text


class TestLandingDoesNotWaitForGoogle:
    """A18: проверочная страница грузила шрифты с fonts.googleapis.com.

    Найдено кастдев-проходом 2026-07-28. B6 убрала эту зависимость со ВСЕХ
    страниц сайта — и промахнулась мимо одной: шаблон проверочной страницы
    лежит в `app/`, а сторож смотрел `static/`. Ровно тот же слепой угол, что
    в A14 («лендинг» в этом же файле) и A17 (запрещённые слова в `app/main.py`).
    Третий раз подряд.

    Промах пришёлся на единственную страницу, куда идёт ПЛАТНЫЙ трафик.
    Замер (2026-07-28, Chromium, домен «молчит» 8 с): заголовок появляется
    через **8,07 с**, `first-contentful-paint` не наступает вовсе — экран
    белый всё это время. `display=swap` тут не спасает: он про подмену
    шрифта, а рендер блокирует сам `<link rel=stylesheet>` в `<head>`.

    Цена ошибки не косметическая. Человек платит нам 1490 ₽ и ещё 3–5 тысяч
    Яндексу, посетитель уходит с белого экрана — а мы по этой конверсии
    выносим вердикт «спроса нет» и говорим это заказчику как факт о рынке
    (принципы 1 и 8).
    """

    def _html(self):
        import app.main as m
        return m.render_landing(VALID_OFFER)

    def test_no_external_hosts_at_all(self):
        html_out = self._html()
        for bad in ("fonts.googleapis.com", "fonts.gstatic.com", "//cdn.",
                    "https://", "http://"):
            assert bad not in html_out, bad

    def test_fonts_come_from_our_own_route(self):
        assert '/fonts/fonts.css' in self._html()

    def test_decorative_families_are_gone(self):
        """Manrope/Onest/JetBrains у нас не лежат — оставить их в CSS значит
        оставить страницу без заявленного шрифта и с чужим запросом."""
        html_out = self._html()
        for bad in ("Manrope", "Onest", "JetBrains"):
            assert bad not in html_out, bad

    def test_families_used_are_the_ones_we_actually_serve(self):
        import re
        from pathlib import Path
        served = (Path(__file__).resolve().parents[1] / "static" / "fonts" / "fonts.css"
                  ).read_text(encoding="utf-8")
        have = set(re.findall(r"font-family:\s*['\"]([^'\"]+)['\"]", served))
        used = set(re.findall(r"['\"]([A-Z][A-Za-z ]+)['\"]\s*,\s*(?:system-ui|sans-serif|monospace)",
                              self._html()))
        assert used, "в шаблоне не осталось ни одного именованного шрифта"
        assert used <= have, f"шрифты, которых мы не отдаём: {used - have}"

    def test_the_page_still_renders_its_content(self):
        """Смена шрифтов не должна съесть саму страницу."""
        html_out = self._html()
        assert VALID_OFFER["h1"] in html_out
        assert VALID_OFFER["pains"][0]["h2"] in html_out
        assert 'href="/legal"' in html_out


class TestEngineErrorsAreHumanReadable:
    """A19: технические сообщения движков доезжали прямо до человека.

    Найдено 2026-07-28 сплошным просмотром строк в `app/*.py` — там, куда
    сторожа по `static/` никогда не заглядывали. Четвёртый дефект из того же
    слепого угла (A14, A17, A18).

    Обе ошибки движков попадают на экран дословно: `/api/sharpen` отдаёт
    `str(e)`, а `result.html` показывает `data.error`; у отчёта тот же путь
    через `__GEN_ERROR__`. Докстринг `sharpen_idea` прямо обещает «бросает
    OfferEngineError с человеческим текстом» — но проверки структуры бросали
    сырьё:

      · «нужно ровно 3 **оффера**» — запрещённое слово (принцип 5) на
        бесплатном шаге, прямо перед решением платить;
      · «в **оффере** нет поля demo_left_label», «pains должен содержать
        3 блока», «direct_queries: 5-12 фраз» — имена полей из нашего JSON;
      · у отчёта — «недостаточно top_risks», «нет корректного
        viability_score», «пустой раздел finance». Это читает человек,
        который уже заплатил 990–2990 ₽.

    Техническая причина нужна нам, а не посетителю: она уходит в лог
    (`.tech`), а на экран идёт одна человеческая фраза. Заодно неполный ответ
    модели стал поводом ПОВТОРИТЬ запрос — ровно как испорченный JSON и
    таймаут, которые движок уже переспрашивает: пропущенное поле такая же
    случайная осечка, а человек до этой правки упирался в тупик с первого раза.
    """

    def _sharpen_error(self, body):
        """Форма ответа — та же, что у настоящего провайдера по умолчанию
        (`_yandex_response`). С анthropic-образной заглушкой движок падал в
        ветку «битый JSON» и до проверок структуры вообще не доходил: тест
        зеленел, ничего не проверив."""
        import asyncio, json as _json
        from app.offer_engine import sharpen_idea, OfferEngineError

        async def fake_post(provider, payload):
            return _yandex_response(_json.dumps(body, ensure_ascii=False))
        try:
            asyncio.run(sharpen_idea("Идея достаточно длинная для проверки", _post=fake_post))
        except OfferEngineError as e:
            return e
        raise AssertionError("движок не пожаловался, хотя ответ битый")

    def test_missing_field_does_not_leak_json_names(self):
        broken = dict(VALID_OFFER); broken.pop("h1")
        e = self._sharpen_error({"offers": [broken, dict(VALID_OFFER), dict(VALID_OFFER)]})
        assert "оффер" not in str(e).lower(), str(e)
        assert "h1" not in str(e), str(e)

    def test_wrong_offer_count_does_not_say_offer(self):
        e = self._sharpen_error({"offers": [dict(VALID_OFFER)]})
        assert "оффер" not in str(e).lower(), str(e)

    def test_message_tells_the_person_what_to_do(self):
        e = self._sharpen_error({"offers": [dict(VALID_OFFER)]})
        assert "ещё раз" in str(e).lower(), str(e)

    def test_technical_reason_survives_for_the_log(self):
        """Человеческий текст не должен стоить нам возможности починить."""
        broken = dict(VALID_OFFER); broken.pop("h1")
        e = self._sharpen_error({"offers": [broken, dict(VALID_OFFER), dict(VALID_OFFER)]})
        assert "h1" in getattr(e, "tech", ""), getattr(e, "tech", None)

    def test_broken_structure_is_retried_like_broken_json(self):
        """Пропущенное поле — такая же случайная осечка модели, как испорченный
        JSON, который движок уже переспрашивает."""
        import asyncio, json as _json
        from app.offer_engine import sharpen_idea
        calls = {"n": 0}

        async def fake_post(provider, payload):
            calls["n"] += 1
            body = ({"offers": [dict(VALID_OFFER)]} if calls["n"] == 1
                    else {"offers": [dict(VALID_OFFER) for _ in range(3)]})
            return _yandex_response(_json.dumps(body, ensure_ascii=False))

        out = asyncio.run(sharpen_idea("Идея достаточно длинная для проверки", _post=fake_post))
        assert calls["n"] == 2, calls
        assert len(out["offers"]) == 3

    def test_report_engine_hides_its_field_names_too(self):
        """Эти строки читает человек, который уже заплатил."""
        import asyncio, json as _json
        from app.report_engine import generate_core, ReportEngineError

        async def fake_post(provider, payload):
            body = {"viability_score": 55, "viability_summary": "с", "top_risks": []}
            return _yandex_response(_json.dumps(body, ensure_ascii=False))
        try:
            asyncio.run(generate_core("Груминг собак с выездом на дом клиента",
                                      DEMAND_DATA_FIXTURE, "full", _post=fake_post))
        except ReportEngineError as e:
            assert "top_risks" not in str(e), str(e)
            assert "top_risks" in getattr(e, "tech", ""), getattr(e, "tech", None)
            # У отчёта своё действие: страница пересобирает разделы по перезагрузке,
            # поэтому зовём обновить, а не «попробовать ещё раз» как на /r/.
            assert "обновите" in str(e).lower(), str(e)
        else:
            raise AssertionError("движок не пожаловался на пустые риски")

    def test_no_engine_message_carries_a_forbidden_word(self):
        """Сплошной сторож: ни одна строка, которая может доехать до экрана,
        не содержит запрещённых слов."""
        import ast, pathlib
        bad = []
        for name in ("offer_engine.py", "report_engine.py"):
            f = pathlib.Path(main_module.BASE_DIR) / name
            for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id.endswith("EngineError")):
                    continue
                for arg in node.args:
                    txt = ast.unparse(arg)
                    for w in ("оффер", "лендинг", "top_risks", "viability_",
                              "direct_queries", "pains", "offers"):
                        if w in txt:
                            bad.append(f"{name}:{node.lineno} → {txt}")
        assert not bad, "\n".join(bad)


class TestOwnerCanSeeAndFixStaleLandings:
    """A20: правка шаблона не доходила до уже запущенных страниц, и владелец
    об этом не знал.

    `landing_html` рендерится и кладётся в БД **в момент запуска** проекта.
    Значит любая правка шаблона действует только на будущие запуски, а живые
    страницы остаются какими были. Само по себе это защитимо: менять страницу
    под работающей рекламой значит менять то, что измеряешь, — плейбук прямо
    просит ничего не трогать во время проверки.

    Не защитимо другое: **владелец не мог ни узнать, что страница устарела, ни
    обновить её**. Перезапуск требовал вручную собрать POST `/api/launch` с
    полным JSON варианта.

    Цена вопроса не теоретическая. A18 убрала со страницы рендер-блокирующий
    запрос шрифтов на чужой домен: замер показал 8,07 с белого экрана, когда
    хост молчит. Исправление лежит в шаблоне — и **не доходит** до тех, за
    чей трафик уже платят. То же с пилюлей-бейджем (C4) и любой будущей
    правкой.

    Теперь у проекта хранится отпечаток шаблона, из которого он собран,
    `/api/cabinet` показывает `landing_stale`, а `POST
    /api/projects/{id}/refresh` пересобирает страницу из сохранённого варианта.
    Обновление — ДЕЙСТВИЕ ВЛАДЕЛЬЦА, а не тихий автопересбор: он должен видеть,
    что вмешивается в идущую проверку.
    """

    def _launch(self, idea_id="stale1"):
        r = client.post("/api/launch", headers=OWNER, json={
            "idea_text": "Идея для проверки устаревания",
            "offer": dict(VALID_OFFER, idea_id=idea_id)})
        assert r.status_code == 200, r.text
        return idea_id

    def _make_stale(self, idea_id):
        """Так выглядит страница, собранная до правки шаблона."""
        from app.main import SmokeProject, Session, engine, select
        with Session(engine) as s:
            proj = s.exec(select(SmokeProject).where(
                SmokeProject.idea_id == idea_id)).first()
            proj.landing_html = proj.landing_html.replace(
                '<link rel="stylesheet" href="/fonts/fonts.css">',
                '<link href="https://fonts.googleapis.com/css2?family=Manrope" rel="stylesheet">')
            proj.template_hash = "старый-отпечаток"
            s.add(proj); s.commit()

    def _card(self, idea_id):
        cab = client.get("/api/cabinet", headers=OWNER).json()
        return [x for x in cab["smoke"] if x["idea_id"] == idea_id][0]

    def test_fresh_launch_is_not_marked_stale(self):
        assert self._card(self._launch("stale_fresh"))["landing_stale"] is False

    def test_page_built_before_the_fix_is_marked_stale(self):
        idea_id = self._launch("stale_old")
        self._make_stale(idea_id)
        assert self._card(idea_id)["landing_stale"] is True

    def test_projects_launched_before_this_feature_count_as_stale(self):
        """У старых записей отпечатка нет вовсе — молчать о них нельзя."""
        from app.main import SmokeProject, Session, engine, select
        idea_id = self._launch("stale_nohash")
        with Session(engine) as s:
            proj = s.exec(select(SmokeProject).where(
                SmokeProject.idea_id == idea_id)).first()
            proj.template_hash = ""
            s.add(proj); s.commit()
        assert self._card(idea_id)["landing_stale"] is True

    def test_refresh_rebuilds_the_page_from_the_saved_offer(self):
        idea_id = self._launch("stale_fix")
        self._make_stale(idea_id)
        assert "fonts.googleapis.com" in client.get(f"/l/{idea_id}").text
        r = client.post(f"/api/projects/{idea_id}/refresh", headers=OWNER)
        assert r.status_code == 200, r.text
        page = client.get(f"/l/{idea_id}").text
        assert "fonts.googleapis.com" not in page
        assert "/fonts/fonts.css" in page
        assert self._card(idea_id)["landing_stale"] is False

    def test_refresh_keeps_the_owners_own_name(self):
        """Переименование живёт на проекте, а не в варианте — пересбор не
        должен возвращать машинное имя."""
        idea_id = self._launch("stale_named")
        client.patch(f"/api/projects/{idea_id}", headers=OWNER, json={"name": "ОтзоВик"})
        self._make_stale(idea_id)
        client.post(f"/api/projects/{idea_id}/refresh", headers=OWNER)
        assert "<title>ОтзоВик</title>" in client.get(f"/l/{idea_id}").text

    def test_refresh_is_owner_only(self):
        idea_id = self._launch("stale_guard")
        assert client.post(f"/api/projects/{idea_id}/refresh").status_code == 401

    def test_refresh_of_unknown_project_is_404(self):
        assert client.post("/api/projects/no-such/refresh", headers=OWNER).status_code == 404

    def test_desk_shows_the_warning_and_the_button(self):
        text = _read_static("desk.html")
        assert "landing_stale" in text
        assert "refresh" in text


class TestAudienceLivesInOnePlace:
    """F1: аудитория была размазана по четырём файлам.

    `purpose` протянут через весь продукт — проверку спроса, оптику отчёта,
    воронку владельца, цели Метрики, — но ОПИСАНА аудитория была четырьмя
    разными способами в четырёх местах: персона и критерий балла в
    `report_engine`, переопределения разделов через ключ `"social"` внутри
    `SECTION_SPECS`, ветка `IS_SOCIAL_CONTRACT` в скрипте страницы результата
    и русская метка в `PURPOSE_LABEL` прямо в `desk.html`.

    На двух аудиториях это ещё держалось. На трёх и больше — нет: это тот же
    дефект-класс, что уже дал расхождения в ценах (B5), названиях тарифов
    (B7), правиле этапа (A13) и адресе отчёта (E6). Каждый раз копия правила
    в двух местах молча разъезжалась.

    Поэтому аудитория стала первоклассной сущностью в `app/audiences.py`:
    одна запись описывает адрес витрины, метку, кто читает результат, персону
    для модели, что означает балл и обязательна ли смета. Всё остальное
    спрашивает у реестра.
    """

    def test_registry_lists_the_audiences_we_actually_have(self):
        from app.audiences import AUDIENCES
        assert set(AUDIENCES) == {"business", "social_contract", "student"}

    def test_every_audience_is_described_completely(self):
        """Полупустая запись — это молчаливый откат к оптике фаундера."""
        from app.audiences import AUDIENCES
        for key, a in AUDIENCES.items():
            for field in ("key", "label", "reader", "persona", "viability"):
                assert getattr(a, field), f"{key}: пустое поле {field}"
            assert a.key == key

    def test_slugs_are_unique_and_home_belongs_to_the_founder(self):
        from app.audiences import AUDIENCES, by_slug
        slugs = [a.slug for a in AUDIENCES.values()]
        assert len(slugs) == len(set(slugs)), slugs
        assert by_slug("").key == "business"
        assert by_slug("social-contract").key == "social_contract"

    def test_unknown_audience_falls_back_to_the_founder(self):
        """Мусор в `purpose` не должен ронять страницу — но и притворяться
        соцконтрактом тоже не должен (принцип 7)."""
        from app.audiences import get
        assert get("нет такой").key == "business"
        assert get("").key == "business"
        assert get(None).key == "business"

    def test_report_engine_takes_the_persona_from_the_registry(self):
        """Вторая копия персоны — это разъехавшаяся оптика платного продукта."""
        import app.report_engine as re_mod
        from app.audiences import get
        assert not hasattr(re_mod, "_PERSONA"), "персона осталась копией в движке"
        assert not hasattr(re_mod, "_VIABILITY_SPEC"), "критерий балла остался копией"
        for key in ("business", "social_contract"):
            assert get(key).persona[:40] in re_mod._core_prompt("full", key)
            assert get(key).viability[:40] in re_mod._core_prompt("full", key)

    def test_section_overrides_are_keyed_by_audience(self):
        """Ключ `"social"` внутри разделов — это «аудиторий ровно две».
        Третья потребовала бы `"student"` и ещё одного `if`."""
        from app.report_engine import SECTION_SPECS
        for s in SECTION_SPECS:
            assert "social" not in s, f"{s['key']}: переопределения не по ключу аудитории"
            from app.audiences import AUDIENCES
            for aud in (s.get("by_audience") or {}):
                assert aud in AUDIENCES, f"{s['key']}: {aud}"

    def test_section_titles_still_differ_where_they_did(self):
        """Переезд не должен потерять то, ради чего оптика заводилась."""
        from app.report_engine import section_title
        assert section_title("finance", "social_contract") == "Смета и расчёты для комиссии"
        assert section_title("finance", "business") == "Финансовая модель"

    def test_desk_labels_come_from_the_server(self):
        """Русские названия аудиторий жили копией в скрипте панели."""
        text = _read_static("desk.html")
        assert "PURPOSE_LABEL = {business:" not in text
        r = client.get("/api/funnel", headers=OWNER)
        assert r.status_code == 200, r.text
        labels = r.json()["audience_labels"]
        assert labels["business"] and labels["social_contract"]
        assert labels["social_contract"] != labels["business"]


class TestVisitorCanFindHisOwnEntrance:
    """F2: `/social-contract` был доступен только по ссылке из объявления.

    С самого сайта на него не попасть: ни с главной, ни из подвала, ни из
    результата проверки. Человек, которому нужно обоснование для комиссии
    соцзащиты, приходит на главную и читает про венчурную проверку идеи —
    и не понимает, что это про него тоже.

    Обратное так же верно: пришедший по объявлению соцконтрактник не может
    уйти на витрину фаундера, если ошибся. Единственная ссылка — логотип,
    который молча уводит на «другой» сайт.

    Витрин будет больше двух (F3, студенты), поэтому переключатель собирается
    из реестра аудиторий на сервере и вставляется в страницы одним слотом.
    Копия в каждой витрине разъехалась бы ровно так же, как разъезжались цены
    (B5) и названия тарифов (B7).
    """

    def _switch(self, path):
        import re
        t = client.get(path).text
        m = re.search(r'<nav class="aud-switch".*?</nav>', t, re.S)
        assert m, f"на {path} нет переключателя аудитории"
        return m.group()

    def test_home_offers_the_other_entrances(self):
        block = self._switch("/")
        assert 'href="/social-contract"' in block

    def test_social_contract_page_offers_the_way_back(self):
        block = self._switch("/social-contract")
        assert 'href="/"' in block

    def test_current_audience_is_shown_but_not_a_link(self):
        """Ссылка на страницу, где человек уже стоит, — шум и лишний клик."""
        import re
        block = self._switch("/social-contract")
        assert 'href="/social-contract"' not in block
        assert re.search(r'aria-current="page"', block), block

    def test_every_audience_from_the_registry_is_offered(self):
        """Новая аудитория обязана появиться в переключателе сама."""
        from app.audiences import AUDIENCES
        block = self._switch("/")
        for a in AUDIENCES.values():
            assert a.switch_label in block, a.key

    def test_labels_speak_from_the_visitors_side(self):
        """«business» и «social_contract» — наши слова. Человек про себя
        говорит иначе (принцип 5)."""
        from app.audiences import AUDIENCES
        for a in AUDIENCES.values():
            low = a.switch_label.lower()
            assert a.key not in low
            for bad in ("оффер", "лендинг", "аудитория", "purpose"):
                assert bad not in low, (a.key, a.switch_label)

    def test_switch_is_built_in_one_place(self):
        """Разметка переключателя не должна лежать копией в витринах."""
        for name in ("index.html", "audience-landing.html"):
            t = _read_static(name)
            assert "__AUDIENCE_SWITCH__" in t, name
            assert 'class="aud-switch"' not in t, name

    def test_result_page_lets_you_switch_optics_without_rechecking(self):
        """Спрос уже посчитан — гонять человека через проверку заново, чтобы
        сменить оптику разбора, незачем."""
        import app.main as m
        rid = None

        async def fake_check(idea):
            return {"formulations": [{"phrase": "ф", "count": 480}], "best_phrase": "ф",
                    "verdict": {"level": "niche", "text": "т"},
                    "competitors": {"found": 9, "top": []},
                    "scores": [{"key": "demand", "label": "Спрос", "value": 4, "note": ""}],
                    "overall": {"value": 4, "weakest": "Спрос", "basis": "б"}}

        import app.main
        old = app.main.check_demand
        app.main.check_demand = fake_check
        try:
            r = client.post("/api/demand", json={"idea": "Груминг с выездом на дом",
                                                 "purpose": "business"})
            rid = r.json()["id"]
        finally:
            app.main.check_demand = old
        r = client.post(f"/api/demand/{rid}/purpose", json={"purpose": "social_contract"})
        assert r.status_code == 200, r.text
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            assert s.get(DemandCheck, rid).purpose == "social_contract"

    def test_switching_to_an_unknown_audience_is_refused(self):
        assert client.post("/api/demand/999999/purpose",
                           json={"purpose": "нет такой"}).status_code in (400, 404)


class TestStudentAudience:
    """F3: третья аудитория — студенты, и общий шаблон витрины.

    **Позиционирование переписано 2026-08-02** (владелец): первая версия
    строилась на неверном предположении — курсовая/диплом/защита. На деле
    замысел был другой: молодой человек уверен, что его идея сделает его
    богатым, и готов заплатить за честную проверку, пока сам не потратил на
    неё время и деньги. Курсовая тут почти ни при чём. Реальная нужда та же,
    что у business (проверить идею честно, без поблажек), просто аудитория
    моложе — поэтому персона и критерий баллов у student теперь намеренно
    близки к business, не смягчённая академическая версия. Разница — в
    маркетинге/тоне витрины, не в том, как считается балл. Цены единые для
    всех аудиторий — решение владельца.

    **Нулевой поисковый спрос — тоже общий с business ответ.** У идеи из этой
    аудитории спроса в Яндексе может не быть вовсе, а воронка на слабом
    спросе намеренно перестаёт продавать (A11/A12). Прятать вердикт нельзя —
    принцип 1 не обсуждается, и принцип 2 тоже: вердикт имеет право сказать
    «нет».
    """

    def test_student_is_in_the_registry_with_its_own_optics(self):
        from app.audiences import AUDIENCES, get
        assert "student" in AUDIENCES
        a = get("student")
        assert a.slug == "students"
        for field in ("label", "reader", "persona", "viability", "switch_label"):
            assert getattr(a, field), field
        assert a.persona != get("business").persona
        assert a.viability != get("business").viability

    def test_student_optics_deliberately_match_business_honesty(self):
        """Перевёрнуто 2026-08-02: раньше тест требовал ОТКАЗ от венчурных
        критериев для студента (мерить курсовую венчурной линейкой — вред).
        Теперь наоборот — реальная аудитория хочет ту же честную, без
        поблажек, оценку, что и фаундер (владелец: «куча студентов думают,
        что разбогатеют», и наша ценность именно в честной проверке этой
        веры). Персона student намеренно венчурная, как и у business —
        не смягчённая версия."""
        from app.audiences import get
        low = get("student").persona.lower()
        assert "венчурного фонда" in low
        assert "снисходительности" in low or "не поблажка" in get("student").viability.lower()
        assert "студенческ" not in low, "старая академическая рамка не должна была вернуться"

    def test_student_landing_drops_coursework_framing(self):
        """Прежняя витрина обещала помощь с курсовой/защитой — владелец
        объяснил (2026-08-02), что это не тот человек, который к нам придёт:
        реальная аудитория верит, что разбогатеет на идее, курсовая почти
        ни при чём. Проверяем, что старая рамка не вернулась.

        Слово «защит» само по себе не годится: страница легитимно ссылается
        на витрину соцконтракта («...обоснование для соцзащиты») в общем
        переключателе аудиторий (F2) — проверяем именно академическую защиту
        курсовой/диплома, не эту ссылку."""
        text = client.get("/students").text
        for bad in ("курсов", "диплом", "кафедр", "жюри", "защиту курсовой",
                    "на защите"):
            assert bad not in text.lower(), bad

    def test_the_page_opens_and_carries_the_switch(self):
        r = client.get("/students")
        assert r.status_code == 200
        assert 'class="aud-switch"' in r.text
        assert 'href="/social-contract"' in r.text and 'href="/"' in r.text
        assert 'href="/students"' not in r.text.split("</nav>")[0]

    def test_the_page_sends_its_own_audience_with_the_check(self):
        t = client.get("/students").text
        assert 'AUDIENCE = "student"' in t, "витрина не знает свою аудиторию"
        assert "purpose: AUDIENCE" in t, "проверка уходит без аудитории"

    def test_both_audience_pages_come_from_one_template(self):
        """Третья копия витрины разъехалась бы, как разъезжались цены (B5)."""
        import pathlib
        static = pathlib.Path(main_module.BASE_DIR).parent / "static"
        assert (static / "audience-landing.html").exists()
        assert not (static / "students.html").exists()
        assert not (static / "social-contract.html").exists()

    def test_prices_are_the_same_for_everyone(self):
        """Решение владельца: разные цены обидят тех, кто не попал в льготную
        группу. Витрины обязаны называть одни и те же суммы."""
        import re
        nums = [set(re.findall(r"(\d{3,4}) ₽", client.get(p).text))
                for p in ("/social-contract", "/students")]
        assert nums[0] == nums[1], nums

    def test_students_pricing_note_does_not_promise_tailoring_the_report_lacks(self):
        """D4-находка (2026-08-01): витрина обещала «аудитория и когда идею
        стоит закрыть», а SECTION_SPECS для student не имеет ни одного
        by_audience-переопределения (в отличие от social_contract) -- те же
        разделы, что у бизнес-аудитории. Пока это не поменялось, витрина не
        должна обещать подстройку, которой раздел отчёта не делает."""
        text = client.get("/students").text
        assert "когда идею стоит закрыть" not in text
        assert "рынок, конкуренты, финансы, риски и план запуска" in text

    def test_audience_pages_are_tinted_but_share_the_one_accent(self):
        """Пункт 12 (Борис): вкладки аудиторий должны различаться, но
        `CLAUDE.md` запрещает больше одного акцентного цвета -- решение
        владельца: единственный акцент (жёлтый маркер) остаётся, различается
        только тон рамки/фона карточек в пределах бумажной палитры. CSS живёт
        в общем шаблоне (обе вкладки получают оба правила) -- меняется только
        `data-audience` на `<body>`, от него и зависит, какое правило сработает."""
        import re
        social_text = client.get("/social-contract").text
        student_text = client.get("/students").text
        for text in (social_text, student_text):
            assert "__AUDIENCE_KEY__" not in text
            assert "#FFDE59" in text          # маркер остаётся везде
        assert 'data-audience="social_contract"' in social_text
        assert 'data-audience="student"' in student_text

        def rule(text, audience):
            m = re.search(r'\[data-audience="%s"\]\{([^}]+)\}' % audience, text)
            assert m, f"нет правила для {audience}"
            return m.group(1)
        social_rule, student_rule = rule(social_text, "social_contract"), rule(student_text, "student")
        assert social_rule != student_rule

    def test_weak_demand_answer_is_worded_separately_but_says_the_same_thing(self):
        """Перевёрнуто 2026-08-02: раньше студенту при нулевом спросе
        предлагался другой следующий шаг ("законный материал для работы").
        Теперь нужда та же, что у business — не тратить время и деньги на
        идею без спроса, реформулировать. Формулировка своя (у каждой
        аудитории свой текст в реестре, F1), но совет один и тот же."""
        from app.audiences import get
        founder, student = get("business").weak_demand, get("student").weak_demand
        assert founder and student and founder != student
        for text in (founder, student):
            assert "переформул" in text.lower()
            assert "не тратить" in text.lower() or "не тратьте" in text.lower()

    def test_result_page_takes_the_weak_answer_from_the_server(self):
        """Ветка `IS_SOCIAL_CONTRACT` в скрипте — это «аудиторий ровно две»."""
        t = _read_static("result.html")
        assert "IS_SOCIAL_CONTRACT" not in t
        assert "AUDIENCE.plan_first" in t
        assert "AUDIENCE.weak_demand" in t, "ответ на слабый спрос всё ещё один на всех"

    def test_social_contract_page_did_not_change_its_promise(self):
        """Переезд на общий шаблон — не повод потерять то, ради чего витрина
        заводилась."""
        t = client.get("/social-contract").text
        assert "комисси" in t.lower()
        assert "смет" in t.lower()
        assert 'id="idea"' in t and "/api/demand" in t


class TestOpticsCanBeSwitchedOnTheResultPage:
    """F3 (остаток): ручка смены оптики была, кнопки не было.

    Прошлый цикл добавил `POST /api/demand/{id}/purpose` — сменить аудиторию
    на уже посчитанной проверке, не проходя её заново. Но в интерфейсе этой
    возможности не существовало: человек, попавший не на ту витрину, видел
    результат чужими глазами и мог только начать всё сначала.

    Случай не редкий. Витрин три, находят нас и поиском, и по ссылке от
    знакомого, а на самой `/r/` до этой правки вообще ничего не говорило, чьими
    глазами он читает разбор. Между тем на выбор аудитории завязано многое:
    что стоит главным действием (`plan_first`), что мы отвечаем при слабом
    спросе и какой персоной модель напишет платный разбор.

    Спрос при смене не пересчитывается — он от аудитории не зависит, это
    цифры Яндекса. Меняется только оптика.
    """

    def _check(self, purpose="business"):
        import app.main as m
        from app.main import DemandCheck, Session, engine
        data = {"formulations": [{"phrase": "ф", "count": 480}], "best_phrase": "ф",
                "verdict": {"level": "niche", "text": "т"},
                "competitors": {"found": 9, "top": []},
                "scores": [{"key": "demand", "label": "Спрос", "value": 4, "note": ""}],
                "overall": {"value": 4, "weakest": "Спрос", "basis": "б"}}
        with Session(engine) as s:
            rec = DemandCheck(idea="Груминг собак с выездом на дом", purpose=purpose,
                              result_json=json.dumps(data, ensure_ascii=False))
            s.add(rec); s.commit(); s.refresh(rec)
            return rec.id, rec.public_id

    def test_optics_switcher_is_gone_from_the_result_page(self):
        """Кастдев 2026-08-02: переключатель оптики стоял НАД заголовком, то
        есть человек упирался в вопрос «под какую вы задачу?» раньше, чем
        видел результат, за которым пришёл. Витрина, с которой он пришёл, уже
        ответила на этот вопрос. Ручка POST /api/demand/{id}/purpose осталась
        — убран только блок со страницы результата."""
        _, pid = self._check("student")
        t = client.get(f"/r/{pid}").text
        assert 'id="optics"' not in t
        assert "Разбор под задачу" not in t

    def test_switching_keeps_the_numbers(self):
        """Спрос от аудитории не зависит — это цифры Яндекса, а не мнение."""
        import app.main as m
        rid, pid = self._check("business")
        before = client.get(f"/r/{pid}").text
        assert client.post(f"/api/demand/{rid}/purpose",
                           json={"purpose": "student"}).status_code == 200
        after = client.get(f"/r/{pid}").text

        def payload(html_out):
            return html_out.split("const DATA = ", 1)[1].split(";\n", 1)[0]
        assert json.loads(payload(before))["formulations"] == \
               json.loads(payload(after))["formulations"]

    def test_switching_changes_the_optics(self):
        rid, pid = self._check("business")
        client.post(f"/api/demand/{rid}/purpose", json={"purpose": "social_contract"})
        t = client.get(f"/r/{pid}").text
        aud = json.loads(t.split("const AUDIENCE = ", 1)[1].split(";\n", 1)[0])
        assert aud["key"] == "social_contract"
        assert aud["plan_first"] is True

    def test_report_built_after_the_switch_uses_the_new_optics(self):
        """Смена оптики бессмысленна, если платный разбор её не увидит."""
        import app.main as m
        from app.audiences import get
        rid, _ = self._check("business")
        client.post(f"/api/demand/{rid}/purpose", json={"purpose": "student"})
        from app.main import DemandCheck, Session, engine
        with Session(engine) as s:
            purpose = s.get(DemandCheck, rid).purpose
        from app.report_engine import _core_prompt
        prompt = _core_prompt("full", purpose)
        assert get("student").persona[:40] in prompt

    def test_audience_still_reaches_the_page_after_the_switcher_is_gone(self):
        """Блок убран, но САМА оптика по-прежнему разворачивает финальный шаг:
        это то, ради чего аудитории и разведены. Проверяем, что настройка
        доезжает до страницы, а не ушла вместе с переключателем."""
        _, pid = self._check("social_contract")
        t = client.get(f"/r/{pid}").text
        aud = json.loads(t.split("const AUDIENCE = ", 1)[1].split(";\n", 1)[0])
        assert aud["plan_first"] is True
