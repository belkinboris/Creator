"""Системный прогон ключевых страниц на узком экране (B4 в PRODUCT_ROADMAP).

До этого мобильная вёрстка проверялась точечно: сессия делала скриншот той
страницы, которую как раз правила, и смотрела глазами. Так уже пропускались
дефекты, которые ловятся механически:

* обрезанный плейсхолдер «Телеграм или почта — куда вернуться с отчё» —
  нашёл владелец руками;
* «Открыть →», выдавленная в столбик из двух слов в `/account`.

Здесь один и тот же набор проверок гоняется по всем страницам сразу:

1. страница не скроллится вбок (правило дизайн-системы);
2. ни один видимый элемент не шире экрана;
3. ни в одном поле ввода не обрезан плейсхолдер.

Тесты требуют Playwright и локальный Chromium. Там, где их нет (чужая машина,
CI без браузеров), модуль целиком пропускается -- основной набор тестов не
должен зависеть от наличия браузера.
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

pytest.importorskip("playwright.sync_api", reason="Playwright не установлен")
from playwright.sync_api import sync_playwright  # noqa: E402

CHROMIUM = os.environ.get("SOZDATEL_CHROMIUM", "/opt/pw-browsers/chromium")
if not Path(CHROMIUM).exists():
    pytest.skip(f"Chromium не найден: {CHROMIUM}", allow_module_level=True)

ROOT = Path(__file__).resolve().parents[1]
OWNER_KEY = "mobile-sweep-key"
NARROW = 390          # iPhone 12/13 mini -- самый узкий экран, который реально встречается
TALL = 1400


# ---------------------------------------------------------------------------
# Поднимаем настоящий сервер: Playwright не умеет ходить в TestClient
# ---------------------------------------------------------------------------

def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _seed(db_url: str) -> dict:
    """Наполняет временную базу так, чтобы у каждой страницы были данные.
    Пустая страница мобильную вёрстку не проверяет -- ломается она как раз
    на длинных названиях и плотных цифрах."""
    env = dict(os.environ, DATABASE_URL=db_url, SOZDATEL_OWNER_KEY=OWNER_KEY)
    code = r'''
import json, sys
from app.main import (DemandCheck, ReportPurchase, SmokeProject, SmokeEvent,
                      MagicLinkToken, Session, engine)
data = {
 "formulations": [{"phrase": "пошив штор и постельного белья на заказ", "count": 1200,
                   "matched_phrase": "пошив штор на дому недорого"}],
 "best_phrase": "пошив штор и постельного белья на заказ",
 "verdict": {"level": "niche", "text": "Спрос небольшой, но он есть: людей мало, и каждый клиент будет на счету."},
 "competitors": {"found": 15000, "top": [
    {"title": "Пошив штор на заказ в Москве — большой каталог тканей и бесплатный замер",
     "domain": "shtory-na-zakaz-v-moskve.example.ru"},
    {"title": "Ателье", "domain": "atelier.ru"}]},
 "scores": [{"key": "demand", "label": "Спрос", "value": 6, "note": "1 200 запросов в месяц"},
            {"key": "competition", "label": "Конкуренция", "value": 6, "note": "Есть сильные игроки, но ниша не забита"},
            {"key": "timing", "label": "Своевременность", "value": 7, "note": "Рынок готов сейчас"},
            {"key": "execution", "label": "Реализуемость", "value": 8, "note": "Можно начать одному, без вложений в производство"}],
 "overall": {"value": 6, "weakest": "Спрос",
             "basis": "Среднее по четырём шкалам: спрос, конкуренция, своевременность, реализуемость."}}
idea = "Пошив штор и постельного белья на заказ на дому с бесплатным замером и доставкой"
out = {}
with Session(engine) as s:
    for purpose in ("business", "social_contract"):
        rec = DemandCheck(idea=idea, best_count=1200, purpose=purpose,
                          contact="sweep@example.com",
                          result_json=json.dumps(data, ensure_ascii=False))
        s.add(rec); s.commit(); s.refresh(rec)
        out[purpose] = rec.id
    s.add(ReportPurchase(check_id=out["business"], idea=idea, tier="full",
                         contact="sweep@example.com", status="paid", amount=2990,
                         is_example=True,
                         report_json=json.dumps({"sections": [
                             {"key": "summary", "title": "Резюме проекта",
                              "body": "Спрос подтверждён цифрами Вордстата: " + "текст разбора. " * 40}]},
                             ensure_ascii=False)))
    p = SmokeProject(idea_id="sweep1", product_name="Шторы за неделю, а не за месяц",
                     idea_text=idea, offer_json="{}", landing_html="<h1>тест</h1>",
                     contact="sweep@example.com")
    s.add(p); s.commit()
    for _ in range(52):
        s.add(SmokeEvent(idea="sweep1", event="page_view"))
    for _ in range(3):
        s.add(SmokeEvent(idea="sweep1", event="lead_submitted", contact="lead@example.com"))
    s.add(MagicLinkToken(token="sweep_token", contact="sweep@example.com"))
    # Идея, которую почти не ищут: вердикт имеет право сказать «нет», и
    # страница обязана перестать продавать так, будто ничего не случилось.
    weak = dict(data)
    weak["formulations"] = [{"phrase": "подписка на носки по гороскопу", "count": 30}]
    weak["verdict"] = {"level": "weak", "text": "В поиске эту идею почти не ищут."}
    weak["overall"] = {"value": 1, "weakest": "Спрос", "basis": "Опущен до балла спроса."}
    rec = DemandCheck(idea="Подписка на носки по гороскопу с доставкой", best_count=30,
                      purpose="business", result_json=json.dumps(weak, ensure_ascii=False))
    s.add(rec); s.commit(); s.refresh(rec)
    out["weak"] = rec.id
    s.commit()
print(json.dumps(out))
'''
    r = subprocess.run([sys.executable, "-c", code], cwd=ROOT, env=env,
                       capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError(f"не удалось наполнить базу: {r.stderr[-2000:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    db = tmp_path_factory.mktemp("mobile") / "sweep.db"
    db_url = f"sqlite:///{db}"
    ids = _seed(db_url)
    port = _free_port()
    env = dict(os.environ, DATABASE_URL=db_url, SOZDATEL_OWNER_KEY=OWNER_KEY)
    proc = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.main:app",
         "--host", "127.0.0.1", "--port", str(port), "--log-level", "warning"],
        cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    break
            except OSError:
                if proc.poll() is not None:
                    raise RuntimeError("uvicorn не поднялся")
                time.sleep(0.1)
        else:
            raise RuntimeError("uvicorn не ответил за 10 секунд")
        yield {"base": base, "ids": ids}
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()


@pytest.fixture(scope="module")
def browser():
    with sync_playwright() as pw:
        b = pw.chromium.launch(executable_path=CHROMIUM)
        try:
            yield b
        finally:
            b.close()


# ---------------------------------------------------------------------------
# Сами проверки -- выполняются в браузере, возвращают список нарушителей
# ---------------------------------------------------------------------------

_OVERFLOWING_ELEMENTS = """
(width) => {
  const bad = [];
  for (const el of document.querySelectorAll('body *')) {
    const r = el.getBoundingClientRect();
    if (r.width === 0 && r.height === 0) continue;          // скрытые не считаем
    const st = getComputedStyle(el);
    if (st.display === 'none' || st.visibility === 'hidden') continue;
    // Содержимое, которому автор сам разрешил горизонтальный скролл внутри
    // своего контейнера, -- не дефект, а нормальный приём для широких блоков
    // (так устроена лента разделов в отчёте). Дефект -- когда вбок уезжает
    // сама страница. Поэтому пропускаем всё, что лежит внутри скроллера.
    let inScroller = false;
    for (let p = el; p && p !== document.body; p = p.parentElement) {
      const ps = getComputedStyle(p);
      if (ps.overflowX === 'auto' || ps.overflowX === 'scroll') { inScroller = true; break; }
    }
    if (inScroller) continue;
    if (r.right > width + 1 || r.left < -1) {
      bad.push(el.tagName.toLowerCase() + (el.id ? '#' + el.id : '') +
               (el.className && typeof el.className === 'string'
                 ? '.' + el.className.trim().split(/\\s+/).join('.') : '') +
               ` [${Math.round(r.left)}..${Math.round(r.right)}]`);
    }
  }
  return bad.slice(0, 8);
}
"""

# input и textarea обрезают плейсхолдер по-разному: в input он всегда одной
# строкой и обрезается по ширине (ровно так владелец поймал «Телеграм или
# почта — куда вернуться с отчё»), в textarea переносится и обрезается по
# высоте. Меряем каждый своим способом, иначе любой длинный текст в textarea
# выглядит как дефект, хотя он просто переносится.
_CLIPPED_PLACEHOLDERS = """
() => {
  const bad = [];
  const probe = document.createElement('div');
  document.body.appendChild(probe);
  const measure = (el, wrap) => {
    const st = getComputedStyle(el);
    const inner = el.clientWidth - parseFloat(st.paddingLeft) - parseFloat(st.paddingRight);
    probe.style.cssText = 'position:absolute;visibility:hidden;top:-9999px;left:0';
    probe.style.font = st.font;
    probe.style.letterSpacing = st.letterSpacing;
    probe.style.lineHeight = st.lineHeight;
    probe.style.whiteSpace = wrap ? 'pre-wrap' : 'pre';
    probe.style.width = wrap ? inner + 'px' : 'auto';
    probe.textContent = el.placeholder;
    const r = probe.getBoundingClientRect();
    return {w: r.width, h: r.height, inner};
  };
  for (const el of document.querySelectorAll('input[placeholder]')) {
    if (el.getBoundingClientRect().width === 0) continue;
    const m = measure(el, false);
    if (m.w > m.inner + 1) {
      bad.push(`${el.id || el.name || 'input'}: «${el.placeholder}» — ${Math.round(m.w)}px не влезает в ${Math.round(m.inner)}px`);
    }
  }
  for (const el of document.querySelectorAll('textarea[placeholder]')) {
    if (el.getBoundingClientRect().width === 0) continue;
    const st = getComputedStyle(el);
    const m = measure(el, true);
    const innerH = el.clientHeight - parseFloat(st.paddingTop) - parseFloat(st.paddingBottom);
    if (m.h > innerH + 1) {
      bad.push(`${el.id || el.name || 'textarea'}: «${el.placeholder}» — ${Math.round(m.h)}px по высоте не влезает в ${Math.round(innerH)}px`);
    }
  }
  probe.remove();
  return bad;
}
"""


def _goto(page, url):
    """Ждём domcontentloaded, а не networkidle: на страницах стоит счётчик
    Яндекс.Метрики, его запросы за прокси не завершаются никогда -- прогон
    вставал на таймаут вместо того, чтобы что-то проверить. Разметку рисует
    свой скрипт, ему внешние запросы не нужны."""
    page.goto(url, wait_until="domcontentloaded", timeout=15000)
    page.wait_for_timeout(500)


def _context(browser, width=NARROW):
    ctx = browser.new_context(viewport={"width": width, "height": TALL})
    ctx.set_default_timeout(15000)
    # Режем всё, что уходит наружу. Страницы тянут CSS шрифтов с
    # fonts.googleapis.com рендер-блокирующим тегом в <head>: без сети каждая
    # загрузка стоила ~13 секунд ожидания таймаута. Мы проверяем свою вёрстку,
    # а не чужую доступность.
    #
    # Важно понимать, чем меряем: без внешнего CSS браузер берёт запасной
    # шрифт, метрики текста немного другие. На находках «не влезает на пару
    # пикселей» это стоит перепроверить глазами. Зато у страниц стоит
    # display=swap -- первые мгновения реальный человек тоже видит запасной
    # шрифт, так что случай скорее строгий, чем поблажливый.
    ctx.route("**/*", lambda route: (
        route.continue_() if "127.0.0.1" in route.request.url or route.request.url.startswith("data:")
        else route.abort()))
    return ctx


def _open(browser, url, *, width=NARROW, cookies=None):
    ctx = _context(browser, width)
    if cookies:
        ctx.add_cookies(cookies)
    page = ctx.new_page()
    _goto(page, url)
    return ctx, page


def _audit(page, width=NARROW):
    return {
        "page_scrolls_sideways": page.evaluate(
            "(w) => document.documentElement.scrollWidth > w + 1", width),
        "wide_elements": page.evaluate(_OVERFLOWING_ELEMENTS, width),
        "clipped_placeholders": page.evaluate(_CLIPPED_PLACEHOLDERS),
    }


def _problems(page, label, width=NARROW):
    a = _audit(page, width)
    out = []
    if a["page_scrolls_sideways"]:
        out.append(f"{label} на {width}px: страница скроллится вбок")
    for el in a["wide_elements"]:
        out.append(f"{label} на {width}px: шире экрана — {el}")
    for ph in a["clipped_placeholders"]:
        out.append(f"{label} на {width}px: обрезан плейсхолдер — {ph}")
    return out


def _assert_clean(page, label, width=NARROW):
    problems = _problems(page, label, width)
    assert not problems, "\n".join(problems)


# ---------------------------------------------------------------------------
# Страницы
# ---------------------------------------------------------------------------

def test_public_pages_fit_narrow_screen(site, browser):
    """Один прогон по всем страницам сразу, с общим отчётом: если поехало
    несколько страниц, чинить их надо вместе, а не по одной за прогон."""
    ids = site["ids"]
    pages = [
        ("главная", "/"),
        ("соцконтракт", "/social-contract"),
        ("плейбук Директа", "/guide/direct"),
        ("результат проверки", f"/r/{ids['business']}"),
        ("результат проверки, соцконтракт", f"/r/{ids['social_contract']}"),
        ("отчёт", f"/report/{ids['business']}"),
        ("публичный пример отчёта", "/example"),
        ("кабинет до входа", "/account"),
        ("оферта", "/oferta"),
        ("соглашение", "/agreement"),
        ("конфиденциальность", "/privacy"),
        ("реквизиты", "/contacts"),
    ]
    broken = []
    for label, path in pages:
        ctx, page = _open(browser, site["base"] + path)
        try:
            broken += _problems(page, label)
        finally:
            ctx.close()
    assert not broken, "\n".join(broken)


def test_result_page_fits_with_every_step_open(site, browser):
    """Лента раскрывается по шагам -- в свёрнутом виде половина вёрстки просто
    не участвует, и точечный скриншот её не проверяет."""
    ctx, page = _open(browser, f"{site['base']}/r/{site['ids']['business']}")
    try:
        for _ in range(6):
            btns = page.locator(".step-next .btn:visible")
            if btns.count() == 0:
                break
            btns.first.click()
            page.wait_for_timeout(350)
            _assert_clean(page, "результат проверки, раскрытая лента")
        toggle = page.locator("#score-detail-toggle")
        if toggle.count() and toggle.is_visible():
            toggle.click()
            page.wait_for_timeout(250)
            _assert_clean(page, "результат проверки, разбор оценки раскрыт")
    finally:
        ctx.close()


def test_project_page_fits_narrow_screen(site, browser):
    """/p/ -- самая плотная страница: график, цифры, инструкция запуска."""
    url = f"{site['base']}/p/sweep1?key={OWNER_KEY}"
    ctx, page = _open(browser, url)
    try:
        page.wait_for_timeout(600)          # дорисовывается canvas и цифры
        _assert_clean(page, "страница проекта")
    finally:
        ctx.close()


def test_owner_desk_fits_narrow_screen(site, browser):
    ctx, page = _open(browser, f"{site['base']}/desk?key={OWNER_KEY}")
    try:
        page.wait_for_timeout(600)
        _assert_clean(page, "рабочий стол владельца")
    finally:
        ctx.close()


def test_cabinet_fits_narrow_screen_when_logged_in(site, browser):
    """Кабинет с содержимым: длинное название идеи, карточка проекта, отчёт."""
    ctx = _context(browser)
    page = ctx.new_page()
    try:
        _goto(page, f"{site['base']}/account/verify?token=sweep_token")
        _goto(page, f"{site['base']}/account")
        assert page.locator("#known").is_visible(), "вход в кабинет не сработал"
        _assert_clean(page, "кабинет покупателя")
    finally:
        ctx.close()


def test_weak_demand_stops_selling_in_a_real_browser(site, browser):
    """A11. Проверять это подстроками в HTML бесполезно: разметка блока лежит
    на странице всегда, а включает его скрипт по уровню вердикта. Отключи
    логику — текстовые проверки останутся зелёными. Поэтому смотрим глазами
    браузера: что видно и какими классами разведены блоки."""
    ctx, page = _open(browser, f"{site['base']}/r/{site['ids']['weak']}")
    try:
        # доходим до последнего шага ленты
        for _ in range(6):
            btns = page.locator(".step-next .btn:visible, #skip-sharpen:visible")
            if btns.count() == 0:
                break
            btns.first.click()
            page.wait_for_timeout(300)

        assert page.locator("#weak-lead").is_visible(), \
            "при почти нулевом спросе бесплатный шаг обязан стать главным"
        assert page.locator("#weak-caveat").is_visible(), \
            "оговорка обязана стоять у кнопки живого теста"

        # оба платных блока перестают быть главными
        cls = page.evaluate("""() => ({
            order: document.getElementById('order').className,
            report: document.getElementById('alt-report').className})""")
        assert cls["order"] == "alt-path", cls
        assert cls["report"] == "alt-path", cls

        # но купить по-прежнему можно: наше дело сказать правду, не решить за человека
        assert page.locator("#order-btn").is_visible()
        assert page.locator("#alt-report .btn").is_visible()

        # и шапка больше не обещает следующий этап
        assert "переформулировать" in page.locator("#path-next-text").inner_text()
        _assert_clean(page, "результат со слабым спросом")
    finally:
        ctx.close()


def test_good_demand_keeps_the_live_test_as_the_main_action(site, browser):
    """Обратная сторона: предупреждение не должно всплывать там, где спрос
    есть, иначе оно обесценится и его перестанут читать."""
    ctx, page = _open(browser, f"{site['base']}/r/{site['ids']['business']}")
    try:
        for _ in range(6):
            btns = page.locator(".step-next .btn:visible, #skip-sharpen:visible")
            if btns.count() == 0:
                break
            btns.first.click()
            page.wait_for_timeout(300)
        assert not page.locator("#weak-lead").is_visible()
        assert not page.locator("#weak-caveat").is_visible()
        assert page.evaluate(
            "() => document.getElementById('order').className") == "next"
    finally:
        ctx.close()


def test_owner_sees_mail_state_in_the_desk(site, browser):
    """Блок «Почта» рисует скрипт по ответу /api/diag/mail — подстроками в
    HTML такое не проверить (урок A11). Смотрим глазами браузера."""
    ctx = _context(browser)
    ctx.add_init_script(f"sessionStorage.setItem('sozdatel_key','{OWNER_KEY}')")
    page = ctx.new_page()
    try:
        _goto(page, f"{site['base']}/desk")
        page.wait_for_selector("#mailbox", state="visible", timeout=10000)
        # в прогоне SMTP не настроен — блок обязан сказать это прямо
        state = page.inner_text("#mail-state")
        assert "не настроена" in state, state
        problems = page.eval_on_selector_all(".mail-problems li", "e => e.map(x => x.innerText)")
        assert any("SOZDATEL_SMTP_HOST" in p for p in problems), problems

        # кнопка проверки работает и объясняет отказ, а не молчит
        page.fill("#mail-to", "boris@example.com")
        page.click("#mail-send")
        page.wait_for_timeout(1200)
        assert "не настроена" in page.inner_text("#mail-result").lower()
        _assert_clean(page, "кабинет владельца, блок почты")
    finally:
        ctx.close()
