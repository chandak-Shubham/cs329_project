import hashlib
from visualizer.export_web_data import text_artifact_matches


def test_historical_text_hash_accepts_only_recorded_content_across_line_endings(tmp_path):
    path=tmp_path/'analysis.json'
    original=b'{\r\n  "value": 7\r\n}\r\n'
    expected=hashlib.sha256(original).hexdigest()
    path.write_bytes(original.replace(b'\r\n',b'\n'))
    assert text_artifact_matches(path,expected)
    path.write_bytes(original)
    assert text_artifact_matches(path,expected)
    path.write_bytes(original.replace(b'7',b'8'))
    assert not text_artifact_matches(path,expected)
