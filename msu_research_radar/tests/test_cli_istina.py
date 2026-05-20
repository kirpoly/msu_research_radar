from __future__ import annotations

from unittest.mock import Mock

from msu_research_radar.cli import main


def test_cli_search_person_command(monkeypatch, capsys) -> None:
    fake = [{"name": "Александрушкина Наталья Андреевна", "profile_url": "https://istina.msu.ru/profile/N.Alexandrushkina/"}]
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.search_istina_employees", mock_fn)

    code = main(["istina", "search-person", "Александрушкина Н"])
    output = capsys.readouterr().out

    assert code == 0
    assert "Александрушкина Наталья Андреевна" in output
    mock_fn.assert_called_once()


def test_cli_resolve_person_command(monkeypatch, capsys) -> None:
    fake = {"chosen_candidate": {"name": "Александрушкина Наталья Андреевна"}, "output_json_path": "data/interim/istina_profiles/test.json"}
    mock_fn = Mock(return_value=fake)
    monkeypatch.setattr("msu_research_radar.cli.resolve_istina_person", mock_fn)

    code = main(
        [
            "istina",
            "resolve-person",
            "Александрушкина Н",
            "--article-title",
            "Target Article",
            "--coauthor",
            "Makarevich P. I.",
        ]
    )
    output = capsys.readouterr().out

    assert code == 0
    assert "Александрушкина Наталья Андреевна" in output
    mock_fn.assert_called_once()

