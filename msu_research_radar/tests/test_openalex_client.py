from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from msu_research_radar.publications.openalex_client import (
    MSU_ROR,
    collect_openalex_works,
)


class _Response:
    def __init__(self, payload: dict[str, Any]) -> None:
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict[str, Any]:
        return self._payload


def _make_work(
    work_id: str,
    title: str,
    *,
    publication_date: str = "2026-05-09",
    year: int = 2026,
    institutions: list[dict[str, Any]] | None = None,
    authors: list[str] | None = None,
) -> dict[str, Any]:
    author_names = authors or ["Author One", "Author Two"]
    author_institutions = institutions or [{"display_name": "MSU", "ror": MSU_ROR}]
    authorships = []
    for name in author_names:
        authorships.append(
            {
                "author": {"display_name": name},
                "raw_affiliation_strings": ["Lomonosov Moscow State University"],
                "institutions": author_institutions,
            }
        )

    return {
        "id": work_id,
        "doi": "https://doi.org/10.1234/example",
        "display_name": title,
        "abstract_inverted_index": {"Hello": [0], "world": [1]},
        "publication_date": publication_date,
        "publication_year": year,
        "cited_by_count": 5,
        "authorships": authorships,
        "primary_location": {"source": {"display_name": "Test Journal"}},
        "topics": [{"display_name": "Stem Cells"}],
    }


def test_collect_openalex_works_pagination_and_persistence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = {
        "meta": {"next_cursor": "abc"},
        "results": [
            _make_work("https://openalex.org/W1", "Title 1"),
            _make_work("https://openalex.org/W2", "Title 2"),
        ],
    }
    page2 = {
        "meta": {"next_cursor": None},
        "results": [_make_work("https://openalex.org/W3", "Title 3")],
    }
    calls: list[dict[str, Any]] = []

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> _Response:
        calls.append({"url": url, "params": dict(params)})
        assert MSU_ROR in params["filter"]
        assert "from_publication_date:2022-01-01" in params["filter"]
        assert "to_publication_date:2026-05-20" in params["filter"]
        assert params["sort"] == "publication_date:desc"
        cursor = params.get("cursor")
        if cursor == "*":
            return _Response(page1)
        if cursor == "abc":
            return _Response(page2)
        raise AssertionError(f"Unexpected cursor {cursor}")

    monkeypatch.setattr("msu_research_radar.publications.openalex_client.requests.get", fake_get)

    result = collect_openalex_works(
        from_date="2022-01-01",
        to_date="2026-05-20",
        limit=3,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    assert result["records_collected"] == 3
    assert result["pages_collected"] == 2
    assert len(calls) == 2

    run_dir = Path(result["raw_run_dir"])
    assert run_dir.exists()
    assert (run_dir / "page_0001.json").exists()
    assert (run_dir / "page_0002.json").exists()
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["records_collected"] == 3
    assert manifest["pages_collected"] == 2

    parquet_df = pd.read_parquet(result["parquet_path"])
    csv_df = pd.read_csv(result["csv_path"])
    assert len(parquet_df) == 3
    assert len(csv_df) == 3
    assert set(
        [
            "openalex_id",
            "doi",
            "title",
            "abstract",
            "publication_date",
            "publication_year",
            "journal",
            "cited_by_count",
            "authors",
            "raw_affiliation_strings",
            "institutions",
            "topics",
        ]
    ).issubset(parquet_df.columns)
    assert parquet_df.loc[0, "abstract"] == "Hello world"


def test_collect_openalex_contains_alexandrushkina_2026_article(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alex_work = _make_work(
        "https://openalex.org/W7160719354",
        "Reconceptualizing Mesenchymal Stromal Cell Sheets: From Delivery Tool to Models of Morphogenesis",
        publication_date="2026-05-09",
        year=2026,
        institutions=[{"display_name": "Lomonosov Moscow State University", "ror": MSU_ROR}],
        authors=[
            "Alexandrushkina Natalia A.",
            "Glazieva Valentina S.",
            "Vodopetova Maria A.",
            "Eremichev Roman Yu",
            "Hu Yu-Chen",
            "Makarevich Pavel I.",
        ],
    )
    payload = {"meta": {"next_cursor": None}, "results": [alex_work]}

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> _Response:
        return _Response(payload)

    monkeypatch.setattr("msu_research_radar.publications.openalex_client.requests.get", fake_get)

    result = collect_openalex_works(
        from_date="2026-01-01",
        to_date="2026-12-31",
        limit=10,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    records = result["records"]
    matched = [record for record in records if record["openalex_id"] == "https://openalex.org/W7160719354"]
    assert matched, "Alexandrushkina 2026 article should be present in filtered MSU set."
    record = matched[0]
    assert record["publication_year"] == 2026
    assert record["publication_date"] == "2026-05-09"
    assert (
        record["title"]
        == "Reconceptualizing Mesenchymal Stromal Cell Sheets: From Delivery Tool to Models of Morphogenesis"
    )
    assert any(inst.get("ror") == MSU_ROR for inst in record["institutions"])


def test_collect_openalex_works_no_limit_collects_all_pages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    page1 = {
        "meta": {"next_cursor": "cursor2"},
        "results": [_make_work("https://openalex.org/W10", "A1"), _make_work("https://openalex.org/W11", "A2")],
    }
    page2 = {
        "meta": {"next_cursor": None},
        "results": [_make_work("https://openalex.org/W12", "A3")],
    }
    call_count = {"n": 0}

    def fake_get(url: str, params: dict[str, Any], timeout: int) -> _Response:
        call_count["n"] += 1
        if params.get("cursor") == "*":
            return _Response(page1)
        if params.get("cursor") == "cursor2":
            return _Response(page2)
        raise AssertionError(f"Unexpected cursor {params.get('cursor')}")

    monkeypatch.setattr("msu_research_radar.publications.openalex_client.requests.get", fake_get)

    result = collect_openalex_works(
        from_date="2022-01-01",
        to_date="2026-05-20",
        limit=None,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )

    assert result["requested_limit"] is None
    assert result["records_collected"] == 3
    assert result["pages_collected"] == 2
    assert call_count["n"] == 2


@pytest.mark.live_openalex
def test_live_openalex_contains_alexandrushkina_article(tmp_path: Path) -> None:
    if os.getenv("RUN_LIVE_OPENALEX") != "1":
        pytest.skip("Set RUN_LIVE_OPENALEX=1 to run live OpenAlex test.")

    result = collect_openalex_works(
        from_date="2026-01-01",
        to_date="2026-12-31",
        limit=200,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
    )
    assert any(record["openalex_id"] == "https://openalex.org/W7160719354" for record in result["records"])
