"""Assemble the GitHub Pages site (used by .github/workflows/pages.yml).

python visualizer/build_site.py _site

The whole repository is larger than the 1 GB a Pages site may use, so the site
gets the page, the report, every results table, figure and evaluation log, and
the trained checkpoints the page offers (read from the page's own data). The
raw decision-row archives (*.npz) and other checkpoints stay in the repository.
Markdown files are rendered to HTML, because Pages serves .md as plain text.
"""
from html import escape
import json
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
STYLE = ('body{font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:#20262e;max-width:1100px;margin:32px auto;padding:0 16px;background:#fff}'
         'table{border-collapse:collapse;font-size:13px;display:block;overflow-x:auto}th,td{border:1px solid #d5dbe1;padding:5px 8px;text-align:left}'
         'th{background:#f2f5f8}code{font-size:13px}pre{background:#f5f7f9;padding:10px;overflow-x:auto}img{max-width:100%}a{color:#2563eb}')
MD_LINK = re.compile(r'(href="(?!https?:|#|mailto:)[^"#]+?)\.md(#[^"]*)?"')


def markdown_page(source: Path) -> str:
    import mistune
    body = mistune.create_markdown(escape=True, plugins=['table'])(source.read_text(encoding='utf-8'))
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">'
            f'<title>{escape(source.stem)}</title><style>{STYLE}</style></head><body>{body}</body></html>')


def main(out: Path) -> None:
    if out.exists():
        shutil.rmtree(out)
    html = (ROOT/'index.html').read_text(encoding='utf-8')
    data = json.loads(re.search(r'<script id="research-data" type="application/json">(.*?)</script>', html, re.S)[1])
    offered = {e['path'] for e in data['models']['checkpoints'].values()}

    def wanted(path: Path) -> bool:
        rel = path.relative_to(ROOT).as_posix()
        if rel.startswith('results/'):
            if '/checkpoints/' in rel:
                return rel in offered
            return path.suffix not in ('.npz', '.log', '.lock')
        return rel in ('index.html', 'README.md', 'requirements.txt') or rel.startswith('docs/')

    count = size = 0
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and '.git' not in path.parts and wanted(path):
            target = out/path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            count += 1; size += path.stat().st_size
    missing = [p for p in offered if not (out/p).exists()]
    if missing:
        raise SystemExit(f'Checkpoints offered by the page are missing: {missing[:3]}')
    # Render Markdown (the report already has a typeset HTML version) and point links at .html.
    for md in out.rglob('*.md'):
        html_path = md.with_suffix('.html')
        if not html_path.exists():
            html_path.write_text(markdown_page(md), encoding='utf-8')
    for page in out.rglob('*.html'):
        text = page.read_text(encoding='utf-8')
        fixed = MD_LINK.sub(lambda m: f'{m[1]}.html{m[2] or ""}"', text)
        if fixed != text:
            page.write_text(fixed, encoding='utf-8')
    (out/'.nojekyll').write_text('', encoding='utf-8')
    print(f'Site: {count} files, {size/2**20:.0f} MB, {len(offered)} checkpoints -> {out}')
    if size > 950 * 2**20:
        raise SystemExit('Site exceeds the GitHub Pages size limit')


if __name__ == '__main__':
    main(Path(sys.argv[1] if len(sys.argv) > 1 else '_site').resolve())
