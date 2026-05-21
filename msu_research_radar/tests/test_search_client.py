from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock

import pytest
import requests

from msu_research_radar.istina import search_client


SEARCH_HTML = """
<html>
  <body>
    <div class="worker_short span-24 last">
      <div class="span-2">
        <a href="/profile/N.Alexandrushkina/"></a>
      </div>
      <div class="span-22 last">
        <h3><a href="/workers/50264918/">Александрушкина Наталья Андреевна</a></h3>
      </div>
      <div class="span-22 last">
        <div>МГУ имени М.В. Ломоносова, Лаборатория медицинской биоинженерии, научный сотрудник</div>
      </div>
    </div>
  </body>
</html>
"""


class _Response:
    def __init__(self, text: str, url: str) -> None:
        self.text = text
        self.url = url
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None


def test_search_istina_employees_parses_cards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(search_client, "SEARCH_CACHE_DIR", tmp_path / "search_cache")
    response = _Response(
        SEARCH_HTML,
        "https://istina.msu.ru/workers/worker_search/?s=Alexandrushkina+N",
    )
    get_mock = Mock(return_value=response)
    monkeypatch.setattr(requests, "get", get_mock)

    candidates = search_client.search_istina_employees("Alexandrushkina N", refresh=True)

    assert len(candidates) == 1
    assert candidates[0]["name"] == "Александрушкина Наталья Андреевна"
    assert candidates[0]["profile_url"] == "https://istina.msu.ru/profile/N.Alexandrushkina/"
    assert candidates[0]["worker_url"] == "https://istina.msu.ru/workers/50264918/"
    assert "Лаборатория медицинской биоинженерии" in (candidates[0]["affiliation_hint"] or "")
    get_mock.assert_called_once()


def test_search_uses_cache_when_refresh_false(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_dir = tmp_path / "search_cache"
    monkeypatch.setattr(search_client, "SEARCH_CACHE_DIR", cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{search_client._query_key('Alexandrushkina N')}.html"
    cache_path.write_text(SEARCH_HTML, encoding="utf-8")

    get_mock = Mock(side_effect=AssertionError("network call should not happen"))
    monkeypatch.setattr(requests, "get", get_mock)

    candidates = search_client.search_istina_employees("Alexandrushkina N", refresh=False)
    assert len(candidates) == 1


def test_search_prints_diagnostics_when_structure_unexpected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(search_client, "SEARCH_CACHE_DIR", tmp_path / "search_cache")
    response = _Response("<html><body><h2>Результаты поиска:</h2></body></html>", "https://istina.msu.ru/workers/worker_search/?s=none")
    monkeypatch.setattr(requests, "get", Mock(return_value=response))

    candidates = search_client.search_istina_employees("none", refresh=True)

    assert candidates == []
    output = capsys.readouterr().out
    assert "[istina-search] diagnostics" in output
    assert "cards=0" in output


def test_build_effective_istina_queries_strips_raw_full_name() -> None:
    queries = search_client.build_effective_istina_queries("Natalia A. Alexandrushkina")
    assert queries
    assert "Alexandrushkina N" in queries
    assert "Natalia A. Alexandrushkina" not in queries


def test_search_istina_employees_normalized_uses_effective_queries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mock_search = Mock(
        side_effect=[
            [{"name": "Person 1", "worker_url": "https://istina.msu.ru/workers/1/", "profile_url": None}],
            [{"name": "Person 1", "worker_url": "https://istina.msu.ru/workers/1/", "profile_url": None}],
        ]
    )
    monkeypatch.setattr(search_client, "build_effective_istina_queries", Mock(return_value=["A N", "A N."]))
    monkeypatch.setattr(search_client, "search_istina_employees", mock_search)

    payload = search_client.search_istina_employees_normalized("Any Input")

    assert payload["input_name"] == "Any Input"
    assert payload["effective_queries"] == ["A N", "A N."]
    assert len(payload["candidates"]) == 1
    assert payload["candidates"][0]["matched_query"] == "A N"
