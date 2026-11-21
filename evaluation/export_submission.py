"""Optional printable submission export using installed Mistune and Edge.

python evaluation/export_submission.py
Scientific experiments do not depend on these presentation dependencies.
"""
from html import escape
import hashlib
from importlib.metadata import version
import json
from pathlib import Path


def main():
    import mistune
    from playwright.sync_api import sync_playwright
    root = Path(__file__).resolve().parents[1]
    source = root/'docs/final_report.md'
    markdown = mistune.create_markdown(escape=True, plugins=['table'])
    body = markdown(source.read_text(encoding='utf-8'))
    css = """
    @page { size: A4; margin: 17mm 16mm 19mm; }
    * { box-sizing: border-box; }
    body { color: #172333; font-family: Arial, sans-serif; font-size: 10pt; line-height: 1.5;
           max-width: 980px; margin: 36px auto; padding: 0 24px; }
    h1 { font-size: 24pt; line-height: 1.15; color: #183d50; }
    h2 { font-size: 16pt; color: #183d50; border-bottom: 1px solid #aebfc8; padding-bottom: 5px; margin-top: 28px; }
    h3 { font-size: 12pt; color: #183d50; margin-top: 20px; }
    h1,h2,h3 { break-after: avoid; }
    p { margin: 9px 0; orphans: 3; widows: 3; }
    a { color: #075985; text-decoration: underline; overflow-wrap: anywhere; }
    code { font-family: Consolas, monospace; font-size: 8.5pt; overflow-wrap: anywhere; }
    pre { padding: 12px; background: #f1f5f8; border: 1px solid #cbd5df; white-space: pre-wrap; }
    table { border-collapse: collapse; width: 100%; font-size: 8.2pt; line-height: 1.35; table-layout: fixed; margin: 12px 0; }
    th,td { border: 1px solid #c0cbd2; padding: 6px; vertical-align: top; overflow-wrap: anywhere; }
    th { text-align: left; background: #e9f0f4; }
    tr { break-inside: avoid; }
    thead { display: table-header-group; }
    img { width: 100%; height: auto; break-inside: avoid; }
    @media print { body { margin: 0; padding: 0; max-width: none; } a { color: #172333; } }
    """
    html = '<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
    html += '<title>'+escape('CS329 Group 21 Final Report')+'</title><style>'+css+'</style></head><body>'+body+'</body></html>'
    target = root/'docs/final_report.html'
    target.write_text(html,encoding='utf-8')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge',headless=True)
        page = browser.new_page(viewport={'width':1100,'height':1300})
        page.goto(target.as_uri(), wait_until='load')
        page.wait_for_function('Array.from(document.images).every(img=>img.complete && img.naturalWidth>0)')
        assert page.locator('h1').count() == 1
        assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
        page.pdf(path=str(root/'docs/final_report.pdf'),format='A4',print_background=True,
                 prefer_css_page_size=True,display_header_footer=True,
                 header_template='<span></span>',
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#64748b">CS329 · Group 21 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        preview = root/'.review_tools/final_report_preview.png'
        preview.parent.mkdir(exist_ok=True)
        page.screenshot(path=str(preview))
        browser_version = browser.version
        browser.close()
    exports = dict(source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   exporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   browser=browser_version, packages={name:version(name) for name in ('mistune','playwright')},
                   files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (target,target.with_suffix('.pdf'))})
    source.with_name('final_report_exports.json').write_text(json.dumps(exports,indent=2),encoding='utf-8')
    print(f'Exported {target} and {target.with_suffix(".pdf")}')


if __name__ == '__main__':
    main()
