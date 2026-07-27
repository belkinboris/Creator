from playwright.sync_api import sync_playwright
B = "http://127.0.0.1:8572"
with sync_playwright() as pw:
    b = pw.chromium.launch(executable_path="/opt/pw-browsers/chromium")
    for rid, who in [(1, "biz"), (2, "soc")]:
        for w, name in [(1000, "desktop"), (390, "mobile")]:
            ctx = b.new_context(viewport={"width": w, "height": 950})
            ctx.route("**/*", lambda r: r.continue_() if "127.0.0.1" in r.request.url or r.request.url.startswith("data:") else r.abort())
            p = ctx.new_page()
            p.goto(f"{B}/r/{rid}", wait_until="domcontentloaded"); p.wait_for_timeout(400)
            for _ in range(5):
                btns = p.locator(".step-next .btn:visible")
                if btns.count() == 0: break
                btns.first.click(); p.wait_for_timeout(400)
            sk = p.locator("#skip-sharpen")
            if sk.is_visible(): sk.click(); p.wait_for_timeout(500)
            p.locator(".tier-what-block").first.scroll_into_view_if_needed(); p.wait_for_timeout(250)
            assert not p.evaluate("() => document.documentElement.scrollWidth > innerWidth + 1"), f"вбок {who}/{name}"
            p.screenshot(path=f"/tmp/claude-0/-home-user-Creator/0945360a-8d71-55e7-b700-f957b5f18a81/scratchpad/c2-{who}-{name}.png")
            ctx.close()
    b.close()
print("ok, вбок не едет")
