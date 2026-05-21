from __future__ import annotations

from unittest.mock import Mock

from msu_research_radar.cli import main


def test_cli_collect_openalex_command(monkeypatch, capsys) -> None:
    fake_result = {
        "records_collected": 1000,
        "requested_limit": 1000,
        "pages_collected": 5,
        "chunks_total": 2,
        "chunks_ok": 2,
        "chunks_failed": 0,
        "raw_run_dir": "data/raw/openalex/openalex_20260521T000000Z",
        "manifest_path": "data/raw/openalex/openalex_20260521T000000Z/manifest.json",
        "csv_path": "data/interim/openalex_works.csv",
        "parquet_path": "data/interim/openalex_works.parquet",
        "records": [],
    }
    mock_fn = Mock(return_value=fake_result)
    monkeypatch.setattr("msu_research_radar.cli.collect_openalex_works", mock_fn)

    code = main(
        [
            "collect",
            "openalex",
            "--from-date",
            "2022-01-01",
            "--to-date",
            "2026-05-20",
            "--limit",
            "1000",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "records_collected" in output
    assert "openalex_works.parquet" in output
    mock_fn.assert_called_once_with(
        from_date="2022-01-01",
        to_date="2026-05-20",
        limit=1000,
        per_page=200,
        chunk_by="year",
        connect_timeout=10.0,
        read_timeout=60.0,
        max_retries=5,
        request_delay_seconds=0.3,
    )


def test_cli_collect_openalex_no_limit_mode(monkeypatch, capsys) -> None:
    fake_result = {
        "records_collected": 44623,
        "requested_limit": None,
        "pages_collected": 224,
        "chunks_total": 5,
        "chunks_ok": 5,
        "chunks_failed": 0,
        "raw_run_dir": "data/raw/openalex/openalex_20260521T000000Z",
        "manifest_path": "data/raw/openalex/openalex_20260521T000000Z/manifest.json",
        "csv_path": "data/interim/openalex_works.csv",
        "parquet_path": "data/interim/openalex_works.parquet",
        "records": [],
    }
    mock_fn = Mock(return_value=fake_result)
    monkeypatch.setattr("msu_research_radar.cli.collect_openalex_works", mock_fn)

    code = main(
        [
            "collect",
            "openalex",
            "--from-date",
            "2022-01-01",
            "--to-date",
            "2026-05-20",
            "--no-limit",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "requested_limit" in output
    assert "null" in output
    mock_fn.assert_called_once_with(
        from_date="2022-01-01",
        to_date="2026-05-20",
        limit=None,
        per_page=200,
        chunk_by="year",
        connect_timeout=10.0,
        read_timeout=60.0,
        max_retries=5,
        request_delay_seconds=0.3,
    )
