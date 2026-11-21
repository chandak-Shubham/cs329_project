"""Verify the delivered experiment and report artifacts without retraining.

python evaluation/verify_submission.py
"""
import json
from pathlib import Path
import re
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from evaluation.complete_study import digest,verify_checkpoint
from evaluation.submission_report import validate_analysis


def main():
    diagnostics=json.loads((ROOT/'results/mechanism_study/diagnostics.json').read_text(encoding='utf-8'))
    count=worlds=0
    for name,expected in (('submission_study',95),('mechanism_study',65)):
        root=ROOT/'results'/name
        if digest(root/'analysis.json') != diagnostics['integrity'][name]['analysis_sha256']:
            raise ValueError(f'Stale analysis: {name}')
        analysis=json.loads((root/'analysis.json').read_text(encoding='utf-8'))
        validate_analysis(analysis)
        paths=list((root/'checkpoints').glob('*.pt'))
        if len(paths) != expected:
            raise ValueError(f'Expected {expected} checkpoints in {name}')
        for path in paths:
            verify_checkpoint(path)
        manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8'))
        for name_in_manifest,expected_hash in manifest.items():
            path=root/name_in_manifest
            if not path.resolve().is_relative_to(root.resolve()):
                raise ValueError('Manifest path escapes study directory')
            if digest(path) != expected_hash:
                raise ValueError(f'Artifact hash mismatch: {path}')
        count += len(paths)
        worlds += len(analysis['runs'])
    for source in (ROOT/'README.md',ROOT/'docs/final_report.md',ROOT/'docs/final_study_protocol.md'):
        text=source.read_text(encoding='utf-8')
        for target in re.findall(r'!?\[[^\]]*\]\(([^)]+)\)',text):
            if '://' in target or target.startswith(('#','mailto:')):
                continue
            target=target.split('#',1)[0]
            if target and not (source.parent/target).exists():
                raise ValueError(f'Broken local link in {source.name}: {target}')
    exports=json.loads((ROOT/'docs/final_report_exports.json').read_text(encoding='utf-8'))
    if exports['source_sha256'] != digest(ROOT/'docs/final_report.md'):
        raise ValueError('HTML/PDF exports use an outdated Markdown report')
    for name,expected_hash in exports['files'].items():
        if digest(ROOT/'docs'/name) != expected_hash:
            raise ValueError(f'Export hash mismatch: {name}')
    print(f'Verified {count} checkpoints, {worlds} evaluation worlds, both artifact manifests, report exports and submission links.')


if __name__ == '__main__':
    main()
