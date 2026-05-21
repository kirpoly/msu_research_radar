from __future__ import annotations

from unittest.mock import Mock

from msu_research_radar.cli import main


def test_cli_authors_extract_msu(monkeypatch, capsys) -> None:
    fake = {"authors_extracted": 12, "output_path": "data/interim/msu_authors.csv"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.extract_msu_authors", mock_fn)

    code = main(
        [
            "authors",
            "extract-msu",
            "--input",
            "data/interim/openalex_works.parquet",
            "--output",
            "data/interim/msu_authors.csv",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "authors_extracted" in output
    mock_fn.assert_called_once()


def test_cli_istina_resolve_batch(monkeypatch, capsys) -> None:
    fake = {"accepted": 10, "rejected": 5, "resolved_profiles_path": "data/interim/istina_batch/resolved_profiles.jsonl"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.resolve_batch_istina_authors", mock_fn)

    code = main(
        [
            "istina",
            "resolve-batch",
            "--input",
            "data/interim/msu_authors.csv",
            "--output-dir",
            "data/interim/istina_batch",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "accepted" in output
    mock_fn.assert_called_once()


def test_cli_export_build_master(monkeypatch, capsys) -> None:
    fake = {"rows": 100, "output_path": "data/processed/master_msu_author_graph.parquet"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.build_master_author_graph", mock_fn)

    code = main(
        [
            "export",
            "build-master",
            "--works",
            "data/interim/openalex_works.parquet",
            "--authors",
            "data/interim/msu_authors.csv",
            "--resolved",
            "data/interim/istina_batch/resolved_profiles.jsonl",
            "--output",
            "data/processed/master_msu_author_graph.parquet",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "rows" in output
    mock_fn.assert_called_once()


def test_cli_score_labs(monkeypatch, capsys) -> None:
    fake = {"rows": 30, "output_path": "exports/lab_scores.xlsx"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.export_lab_scores", mock_fn)

    code = main(
        [
            "score",
            "labs",
            "--input",
            "data/processed/master_msu_author_graph.parquet",
            "--output",
            "exports/lab_scores.xlsx",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "lab_scores.xlsx" in output
    mock_fn.assert_called_once_with(
        input_master="data/processed/master_msu_author_graph.parquet",
        output_path="exports/lab_scores.xlsx",
        config_path="config/scoring_weights.yaml",
    )


def test_cli_score_authors(monkeypatch, capsys) -> None:
    fake = {"rows": 50, "output_path": "exports/author_scores.xlsx"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.export_author_scores", mock_fn)

    code = main(
        [
            "score",
            "authors",
            "--input",
            "data/processed/master_msu_author_graph.parquet",
            "--output",
            "exports/author_scores.xlsx",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "author_scores.xlsx" in output
    mock_fn.assert_called_once_with(
        input_master="data/processed/master_msu_author_graph.parquet",
        output_path="exports/author_scores.xlsx",
        config_path="config/scoring_weights.yaml",
    )


def test_cli_radar_inspect_affiliations(monkeypatch, capsys) -> None:
    fake = {"rows": 5, "output_path": "exports/affiliations.csv"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.inspect_affiliations", mock_fn)

    code = main(
        [
            "radar",
            "inspect-affiliations",
            "--input",
            "data/interim/openalex_works.parquet",
            "--output",
            "exports/affiliations.csv",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "affiliations.csv" in output
    mock_fn.assert_called_once_with(
        input_path="data/interim/openalex_works.parquet",
        output_path="exports/affiliations.csv",
        limit=500,
    )


def test_cli_radar_shortlist(monkeypatch, capsys) -> None:
    fake = {"supervisors": 3, "output_path": "exports/supervisor_radar.xlsx"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.export_shortlist", mock_fn)

    code = main(
        [
            "radar",
            "shortlist",
            "--works",
            "data/interim/openalex_works.parquet",
            "--resolved",
            "data/interim/istina_batch/resolved_profiles.jsonl",
            "--profile-name",
            "bio_msu",
            "--output",
            "exports/supervisor_radar.xlsx",
        ]
    )
    output = capsys.readouterr().out
    assert code == 0
    assert "supervisor_radar.xlsx" in output
    mock_fn.assert_called_once_with(
        works_path="data/interim/openalex_works.parquet",
        output_path="exports/supervisor_radar.xlsx",
        profile_config="config/research_profiles.yaml",
        profile_name="bio_msu",
        resolved_path="data/interim/istina_batch/resolved_profiles.jsonl",
    )
