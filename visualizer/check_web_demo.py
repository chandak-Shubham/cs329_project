"""Optional browser acceptance checks. Requires Playwright and installed Chrome/Edge.

python visualizer/check_web_demo.py --browsers chrome msedge
Screenshots and downloaded CSVs are saved to the ignored .review_tools directory.
"""
import argparse
import csv
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

ROOT = Path(__file__).resolve().parents[1]


def set_range(page, control, value):
    page.locator(f"#{control}").evaluate(
        "(el, value) => {el.value=value;el.dispatchEvent(new Event('input',{bubbles:true}));}", str(value)
    )


def click_steps(page, count):
    # The actual DOM handlers are used; no direct engine stepping.
    page.evaluate("count => {for(let i=0;i<count;i++)document.getElementById('step').click();}", count)


def check_browser(playwright, channel, firefox_executable=None):
    if channel == "firefox":
        options = {"executable_path": firefox_executable} if firefox_executable else {}
        browser = playwright.firefox.launch(headless=True, **options)
    else:
        browser = playwright.chromium.launch(channel=channel, headless=True)
    context = browser.new_context(viewport={"width": 1440, "height": 1100}, device_scale_factor=1)
    page = context.new_page()
    errors, requests = [], []
    page.on("pageerror", lambda error: errors.append(str(error)))
    page.on("request", lambda request: requests.append(request.url))
    page.goto((ROOT / "index.html").as_uri())
    expect(page.locator('#strategy')).to_have_value('neural')
    assert page.evaluate('kinshipApp.engine.strategy') == 'neural'
    expect(page.locator('.research-verdict')).to_contain_text('not yet supported')
    expect(page.locator('#research-results')).to_contain_text('Historical regression')
    assert page.locator('#research-results tbody tr').count() >= 35
    page.locator('#research-results img').evaluate_all("images=>images.forEach(img=>img.loading='eager')")
    page.wait_for_function("Array.from(document.querySelectorAll('#research-results img')).every(img=>img.complete&&img.naturalWidth>0)")
    page.locator('#compare-strategies').click()
    expect(page.locator('#comparison-table tbody tr')).to_have_count(5, timeout=30000)
    expect(page.locator('#comparison-status')).to_contain_text('Hamilton filter off')
    assert page.evaluate('kinshipApp.engine.stepCount') == 0
    page.locator('#scenario').select_option('cooperation')  # explicitly test a rule baseline below
    expect(page.locator("#population")).to_have_text("24")
    sidebar = page.locator(".sidebar").bounding_box()
    results = page.locator(".results").bounding_box()
    assert sidebar["x"] + sidebar["width"] <= results["x"] + 1
    expect(page.locator("#above-count")).to_have_text("0")
    initial = page.evaluate("JSON.stringify(kinshipApp.engine.agents)")
    page.locator("#step").click()
    expect(page.locator("#step-count")).to_have_text("0001")
    assert int(page.locator("#donations").inner_text()) > 0
    assert page.evaluate("kinshipApp.engine.hamiltonAbove + kinshipApp.engine.hamiltonBelow === kinshipApp.engine.totalDonations")
    expect(page.locator("#run-summary")).to_contain_text("After 1 step,")
    assert not page.evaluate("kinshipApp.engine.enforceHamilton")
    assert page.evaluate("kinshipApp.engine.hamiltonBelow") > 0
    assert page.locator("#check-rows tr").count() > 0
    page.locator("#reset").click()
    assert initial == page.evaluate("JSON.stringify(kinshipApp.engine.agents)")
    page.locator("#play").click()
    page.wait_for_function("kinshipApp.engine.stepCount >= 2")
    page.locator("#play").click()
    paused_step = page.evaluate("kinshipApp.engine.stepCount")
    page.wait_for_timeout(250)
    assert page.evaluate("kinshipApp.engine.stepCount") == paused_step
    page.locator("#world").focus()
    page.keyboard.press("ArrowRight")
    assert page.evaluate("kinshipApp.engine.stepCount") == paused_step + 1
    page.keyboard.press("Space")
    assert page.evaluate("kinshipApp.running")
    page.keyboard.press("Space")
    assert not page.evaluate("kinshipApp.running")
    for strategy in ["fixed", "kin", "selfish", "neural", "reputation"]:
        page.locator("#strategy").select_option(strategy)
        page.locator("#reset").click()
        click_steps(page, 5)
        assert page.evaluate("kinshipApp.engine.strategy") == strategy
        expect(page.locator("#step-count")).to_have_text("0005")
    page.locator("#seed").fill("101")
    page.locator("#reset").click()
    assert initial != page.evaluate("JSON.stringify(kinshipApp.engine.agents)")
    page.locator("#seed").fill("42")
    page.locator("#reset").click()
    page.locator("#enforce-hamilton").check()
    set_range(page, "benefit", 0)
    expect(page.locator("#threshold")).to_have_text("r* = ∞")
    assert "warning" in page.locator("#hamilton").get_attribute("class")
    expect(page.locator("#condition-status")).to_contain_text("condition failed")
    page.locator('#step').click()
    expect(page.locator('#donations')).to_have_text('0')
    set_range(page, "cost", 3)
    set_range(page, "drain", 0.75)
    assert page.evaluate("[kinshipApp.engine.B,kinshipApp.engine.C,kinshipApp.engine.cBase]") == [0, 3, 0.75]
    page.locator('#scenario').select_option('cooperation')
    page.locator('#enforce-hamilton').uncheck()
    click_steps(page, 3)
    page.locator("#inject").click()
    assert page.evaluate("kinshipApp.engine.metrics().mimics") == 12
    expect(page.locator("#intervention-status")).to_contain_text("12 genetic mimics")
    page.locator("#agent-select").select_option("0")
    expect(page.locator("#agent-details")).to_contain_text("Reputation")
    box = page.locator("#world").bounding_box()
    agent = page.evaluate("kinshipApp.engine.agents[1]")
    page.mouse.click(box["x"] + (agent["x"] + 0.5) / page.evaluate("kinshipApp.engine.L") * box["width"],
                     box["y"] + (1 - (agent["y"] + 0.5) / page.evaluate("kinshipApp.engine.L")) * box["height"])
    expect(page.locator("#agent-select")).to_have_value(str(agent["id"]))
    page.locator("#shock").click()
    expect(page.locator("#shock")).to_be_disabled()
    click_steps(page, 30)
    assert page.evaluate("kinshipApp.engine.shockRemaining") == 0
    expect(page.locator("#intervention-status")).to_contain_text("ended")
    expect(page.locator("#shock")).to_be_enabled()
    with page.expect_download() as download_event:
        page.locator("#export").click()
    destination = ROOT / f".review_tools/web-{channel}.csv"
    download_event.value.save_as(destination)
    with destination.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == page.evaluate("kinshipApp.engine.stepCount") + 1
    assert any("Injected" in row["events"] for row in rows)
    assert any("doubled" in row["events"] for row in rows)
    assert rows[-1]["population"] == page.locator("#population").inner_text()
    page.locator('#scenario').select_option('invasion')
    click_steps(page, 40)
    assert page.evaluate("kinshipApp.engine.totalInjected") == 0
    click_steps(page, 1)
    if page.evaluate("kinshipApp.engine.stepCount") == 41:
        assert page.evaluate("kinshipApp.engine.events.find(e=>e.type==='injection').step") == 40
        expect(page.locator('#intervention-status')).to_contain_text('mimics')
    else:
        # Enforcement can cause extinction before the scheduled attack. Do not fabricate it.
        assert page.evaluate("kinshipApp.engine.finished && kinshipApp.engine.agents.length === 0")
        assert page.evaluate("kinshipApp.engine.totalInjected") == 0
    page.locator('#scenario').select_option('stress')
    assert page.evaluate("[kinshipApp.engine.L,kinshipApp.engine.B,kinshipApp.engine.C,kinshipApp.engine.cBase,kinshipApp.engine.cDensity]") == [20, 3, 2, 0.8, 0.1]
    page.locator('#scenario').select_option('kin')
    assert page.evaluate("kinshipApp.engine.strategy") == 'kin'
    page.locator('[id="scenario"]').select_option('selfish')
    click_steps(page, 100)
    expect(page.locator("#population")).to_have_text("0")
    expect(page.locator("#play")).to_be_disabled()
    expect(page.locator("#world-overlay")).to_contain_text("extinct")
    page.locator('#scenario').select_option('cooperation')
    page.locator('#enforce-hamilton').uncheck()
    click_steps(page, 3)
    page.locator("#inject").click()
    click_steps(page, 4)
    for width in [1440, 1024, 768, 620, 390, 320]:
        page.set_viewport_size({"width": width, "height": 1000})
        page.wait_for_timeout(120)
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth"), f"Horizontal overflow at {width}px"
        assert page.locator("#world").bounding_box()["width"] > 200
        page.screenshot(path=str(ROOT / f".review_tools/web-{channel}-{width}.png"), full_page=True)
    assert not errors, errors
    assert not [url for url in requests if url.startswith(("http:", "https:"))], requests
    context.close()
    browser.close()
    print(f"{channel}: sidebar, Hamilton counts, controls, five strategies, seeded reset, keyboard, interventions, inspection, CSV, extinction, six responsive widths, offline requests: PASS")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--browsers", nargs="+", choices=["chrome", "msedge", "firefox"], default=["chrome", "msedge"])
    parser.add_argument("--firefox-executable", help="Optional path to an existing Playwright Firefox executable")
    args = parser.parse_args()
    (ROOT / ".review_tools").mkdir(exist_ok=True)
    with sync_playwright() as playwright:
        for channel in args.browsers:
            check_browser(playwright, channel, args.firefox_executable)


if __name__ == "__main__":
    main()
