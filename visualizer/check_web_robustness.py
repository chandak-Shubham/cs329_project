"""Browser regression checks for enforced donations, slow playback, and recovery.

Run: python visualizer/check_web_robustness.py
Requires Playwright and installed Chrome/Edge. Uses the local page only.
"""
import csv
import json
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

from check_web_demo import click_steps, serve, set_range, wait_for_model

ROOT = Path(__file__).resolve().parents[1]


def check(p, channel):
    browser = p.chromium.launch(channel=channel, headless=True)
    page = browser.new_page(viewport={"width": 1366, "height": 768})
    errors = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    server, url = serve()
    page.goto(url)
    wait_for_model(page)
    play = page.locator("#play").bounding_box()
    grid = page.locator("#world").bounding_box()
    assert play["y"] + play["height"] < 768
    assert grid["y"] + grid["height"] < 768

    for invalid in ["", "-1", "1.5", "4294967296"]:
        page.locator("#seed").fill(invalid)
        before = page.evaluate("JSON.stringify(kinshipApp.engine.agents)")
        page.locator("#reset").click()
        assert page.evaluate("JSON.stringify(kinshipApp.engine.agents)") == before
    page.locator("#seed").fill("42")
    page.locator("#reset").click()

    page.locator('#scenario').select_option('cooperation')
    page.locator("#enforce-hamilton").check()
    set_range(page, "benefit", 2)
    expect(page.locator("#scenario")).to_have_value("custom")
    expect(page.locator("#condition-status")).to_contain_text("condition failed")
    click_steps(page, 1)
    expect(page.locator("#donations")).to_have_text("0")
    assert page.evaluate("kinshipApp.engine.hamiltonBlocked") > 0
    page.locator("#check-details").evaluate("el => el.open = true")
    expect(page.locator("#check-rows")).to_contain_text("Failed: rB <= C")
    # Historical check records must retain the settings used at decision time.
    set_range(page, "benefit", 10)
    assert page.evaluate("kinshipApp.engine.lastChecks.every(c=>c.B===2)")
    click_steps(page, 1)
    assert page.evaluate("kinshipApp.engine.lastChecks.every(c=>c.B===10)")
    with page.expect_download() as event:
        page.locator("#export").click()
    out = ROOT / f".browser_checks/enforced-{channel}.csv"
    event.value.save_as(out)
    with out.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["enforceHamilton"] == "true" for row in rows)
    for row in rows:
        for c in json.loads(row["checks"]):
            assert c["allowed"] == (c["r"] * c["B"] > c["C"] and c["donorEnergy"] > c["C"] and c["mimicPass"])

    # Sustained real playback with births exercises the formerly negative ripple radius.
    page.locator("#scenario").select_option("kin")
    set_range(page, "benefit", 10)
    set_range(page, "cost", 0.5)
    set_range(page, "drain", 0)
    page.locator("#reset").click()
    set_range(page, "speed", 30)
    cdp = page.context.new_cdp_session(page)
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": 6})
    page.locator("#play").click()
    page.wait_for_function("kinshipApp.engine.stepCount >= 30", timeout=30000)
    before = page.evaluate("kinshipApp.engine.stepCount")
    page.wait_for_function("n => kinshipApp.engine.stepCount >= n + 15", arg=before, timeout=30000)
    assert page.evaluate("kinshipApp.engine.births") > 0
    expect(page.locator("#runtime-error")).to_be_hidden()
    page.locator("#play").click()
    cdp.send("Emulation.setCPUThrottlingRate", {"rate": 1})

    # An unexpected draw exception must stop visibly and be recoverable with Reset.
    page.evaluate("""() => {
      const ctx=document.getElementById('world').getContext('2d');
      window.originalArc=ctx.arc;
      ctx.arc=()=>{throw new Error('deliberate rendering fault for recovery test')};
    }""")
    expect(page.locator("#runtime-error")).to_be_visible()
    expect(page.locator("#play")).to_be_disabled()
    assert not page.evaluate("kinshipApp.running")
    page.evaluate("() => {document.getElementById('world').getContext('2d').arc=window.originalArc;}")
    page.locator("#reset").click()
    expect(page.locator("#runtime-error")).to_be_hidden()
    page.locator("#play").click()
    page.wait_for_function("kinshipApp.engine.stepCount >= 5")
    page.locator("#play").click()
    assert not errors, errors
    page.screenshot(path=str(ROOT / f".browser_checks/enforced-{channel}-desktop.png"), full_page=True)

    page.set_viewport_size({"width": 390, "height": 844})
    page.reload()
    wait_for_model(page)
    assert not page.locator("#settings-panel").evaluate("el=>el.open")
    assert page.locator("#world").bounding_box()["y"] < 844
    page.locator("#step").click()
    expect(page.locator("#step-count")).to_have_text("0001")
    page.locator("#check-details").evaluate("el=>el.open=true")
    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
    page.screenshot(path=str(ROOT / f".browser_checks/enforced-{channel}-mobile.png"), full_page=True)
    browser.close()
    server.shutdown()
    print(f"{channel}: strict inequality, failure details, decision CSV, seeds, visible controls, slow playback, error recovery, mobile: PASS")


if __name__ == "__main__":
    (ROOT / ".browser_checks").mkdir(exist_ok=True)
    with sync_playwright() as p:
        for channel in ["chrome", "msedge"]:
            check(p, channel)
