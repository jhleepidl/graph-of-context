from pathlib import Path

from goc_logger.export import export_dot


def test_exporter_meta_v0(tmp_path: Path) -> None:
    jsonl = tmp_path / "graph.jsonl"
    jsonl.write_text(
        '{"task_id":"T0001","event_type":"SNAPSHOT","payload":{"nodes":[{"id":"doc:C1","type":"doc_ref","clause_id":"C1","kind":"rule","slot":"export_logs","published_at":"2025-01-01"}],"edges":[]}}\n'
    )
    dot = tmp_path / "out.dot"
    export_dot(jsonl, "T0001", dot)
    text = dot.read_text(encoding="utf-8")
    assert "C1" in text
    assert "rule/export_logs" in text
    assert "2025-01-01" in text


def test_exporter_meta_v1_attrs(tmp_path: Path) -> None:
    jsonl = tmp_path / "graph.jsonl"
    jsonl.write_text(
        '{"task_id":"T0002","event_type":"SNAPSHOT","payload":{"nodes":[{"id":"doc:C2","type":"doc_ref","clause_id":"C2","attrs":{"kind":"update","slot":"export_identifiers","published_at":"2025-02-01"}}],"edges":[]}}\n'
    )
    dot = tmp_path / "out.dot"
    export_dot(jsonl, "T0002", dot)
    text = dot.read_text(encoding="utf-8")
    assert "C2" in text
    assert "update/export_identifiers" in text
    assert "2025-02-01" in text
