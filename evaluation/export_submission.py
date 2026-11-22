"""Printable PDF of the final report, with typeset equations.

python -m pip install -r requirements-report.txt
python evaluation/export_submission.py

Converts docs/final_report.md with Mistune, typesets $...$ and $$...$$ with
KaTeX in Microsoft Edge (headless, via Playwright) and prints docs/final_report.pdf.
It also writes docs/final_report.html, the same report for the project web page.
Scientific experiments do not depend on these presentation dependencies.
"""
from html import escape
from pathlib import Path
import re

KATEX = 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist'
CSS = """
@page { size: A4; margin: 16mm 15mm 18mm; }
* { box-sizing: border-box; }
body { color: #172333; font-family: Georgia, 'Times New Roman', serif; font-size: 10.2pt; line-height: 1.48;
       max-width: 980px; margin: 36px auto; padding: 0 24px; }
h1 { font-size: 21pt; line-height: 1.15; color: #183d50; margin-bottom: 6px; }
h2 { font-size: 14.5pt; color: #183d50; border-bottom: 1px solid #aebfc8; padding-bottom: 4px; margin-top: 24px; }
h3 { font-size: 11.5pt; color: #183d50; margin-top: 18px; }
h1, h2, h3 { break-after: avoid; font-family: Arial, sans-serif; }
p, li { orphans: 3; widows: 3; }
a { color: #075985; overflow-wrap: anywhere; }
code { font-family: Consolas, monospace; font-size: 8.3pt; overflow-wrap: anywhere; }
pre { padding: 10px; background: #f1f5f8; border: 1px solid #cbd5df; white-space: pre-wrap; font-size: 8pt; }
table { border-collapse: collapse; width: 100%; font-size: 8pt; line-height: 1.3; margin: 10px 0; font-family: Arial, sans-serif; }
th, td { border: 1px solid #c0cbd2; padding: 4px 5px; vertical-align: top; overflow-wrap: anywhere; }
th { text-align: left; background: #e9f0f4; }
tr, img, pre, .katex-display { break-inside: avoid; }
thead { display: table-header-group; }
img { width: 100%; height: auto; }
.katex { font-size: 1.02em; }
@media print { body { margin: 0; padding: 0; max-width: none; } a { color: #172333; text-decoration: none; } }
"""


def protect_math(text):
    """Swap math for placeholders so Markdown cannot reinterpret _, * or |."""
    spans = []
    def keep(display):
        def sub(m):
            spans.append((display, m.group(1)))
            return f'MATHSPAN{len(spans)-1}X'
        return sub
    text = re.sub(r'\$\$(.+?)\$\$', keep(True), text, flags=re.S)
    text = re.sub(r'(?<![\\$])\$(?!\$)(.+?)(?<!\\)\$', keep(False), text)
    return text, spans


def restore_math(html, spans):
    def put(m):
        display, tex = spans[int(m.group(1))]
        tex = escape(tex.strip(), quote=False)
        return f'<span class="math-display">\\[{tex}\\]</span>' if display else f'<span class="math-inline">\\({tex}\\)</span>'
    return re.sub(r'MATHSPAN(\d+)X', put, html)


def main():
    import mistune
    from playwright.sync_api import sync_playwright
    root = Path(__file__).resolve().parents[1]
    source = root/'docs/final_report.md'
    text, spans = protect_math(source.read_text(encoding='utf-8'))
    body = restore_math(mistune.create_markdown(escape=True, plugins=['table'])(text), spans)
    def page_html(content):
        return ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                f'<title>CS329 Group 21 Final Report</title><link rel="stylesheet" href="{KATEX}/katex.min.css">'
                f'<style>{CSS}</style></head><body>{content}'
                f'<script src="{KATEX}/katex.min.js"></script><script src="{KATEX}/contrib/auto-render.min.js"></script>'
                '<script>renderMathInElement(document.body,{delimiters:[{left:"\\\\[",right:"\\\\]",display:true},'
                '{left:"\\\\(",right:"\\\\)",display:false}],throwOnError:true});window.mathDone=true;</script>'
                '</body></html>')
    # Web version for the project page; its links stay relative to docs/.
    (root/'docs/final_report.html').write_text(page_html(body), encoding='utf-8', newline='\n')
    work = root/'.browser_checks'
    work.mkdir(exist_ok=True)
    # For printing from the work folder, point relative links at the repository.
    printed = body.replace('src="../', f'src="{(root).as_uri()}/').replace('href="../', f'href="{(root).as_uri()}/')
    target = work/'final_report.html'
    target.write_text(page_html(printed), encoding='utf-8')
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel='msedge', headless=True)
        page = browser.new_page(viewport={'width': 1100, 'height': 1300})
        errors = []
        page.on('pageerror', lambda e: errors.append(str(e)))
        page.goto(target.as_uri(), wait_until='networkidle')
        page.wait_for_function('window.mathDone === true', timeout=30000)
        page.wait_for_function('Array.from(document.images).every(img => img.complete && img.naturalWidth > 0)')
        if errors:
            raise RuntimeError(f'KaTeX/page errors: {errors}')
        rendered = page.locator('.katex').count()
        unrendered = page.evaluate("document.body.innerText.includes('MATHSPAN')")
        if unrendered or rendered < len(spans):
            raise RuntimeError(f'Only {rendered} of {len(spans)} equations rendered')
        page.pdf(path=str(root/'docs/final_report.pdf'), format='A4', print_background=True,
                 prefer_css_page_size=True, display_header_footer=True, header_template='<span></span>',
                 footer_template='<div style="font-size:8px;width:100%;text-align:center;color:#64748b">'
                                 'CS329 · Group 21 · <span class="pageNumber"></span> / <span class="totalPages"></span></div>')
        browser.close()
    print(f'Rendered {len(spans)} equations; wrote docs/final_report.pdf')


if __name__ == '__main__':
    main()
