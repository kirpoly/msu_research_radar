from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import requests

from msu_research_radar.publications import openalex_client
from msu_research_radar.publications.openalex_client import (
    MSU_ROR,
    collect_openalex_works,
)


class _Response:
    def __init__(self, payload: dict[str, Any], status_code: int = 200, headers: dict[str, str] | None = None) -> None:
        self._payload = payload
        self.status_code = status_code
        self.headers = headers or {}

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")

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
    topic: str = "Mesenchymal stem cell research",
    subfield: str = "Genetics",
    field: str = "Medicine",
    domain: str = "Health Sciences",
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
                "is_corresponding": False,
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
        "primary_topic": {
            "display_name": topic,
            "subfield": {"display_name": subfield},
            "field": {"display_name": field},
            "domain": {"display_name": domain},
        },
    }


def test_collect_openalex_works_chunked_by_year_and_dedupes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, Any]] = []
    data_map = {
        ("2022-01-01", "2022-12-31", "*"): _Response(
            {
                "meta": {"next_cursor": None},
                "results": [
                    _make_work("https://openalex.org/W1", "T1", publication_date="2022-01-01", year=2022),
                    _make_work("https://openalex.org/W2", "T2", publication_date="2022-02-01", year=2022),
                ],
            }
        ),
        ("2023-01-01", "2023-12-31", "*"): _Response(
            {
                "meta": {"next_cursor": None},
                "results": [
                    _make_work("https://openalex.org/W2", "T2 duplicate", publication_date="2023-01-03", year=2023),
                    _make_work("https://openalex.org/W3", "T3", publication_date="2023-03-01", year=2023),
                ],
            }
        ),
    }

    def fake_get(self: requests.Session, url: str, params: dict[str, Any], timeout: tuple[float, float]) -> _Response:
        calls.append({"url": url, "params": dict(params), "timeout": timeout})
        filter_parts = {
            item.split(":", 1)[0]: item.split(":", 1)[1]
            for item in str(params["filter"]).split(",")
            if ":" in item
        }
        key = (
            filter_parts["from_publication_date"],
            filter_parts["to_publication_date"],
            str(params.get("cursor")),
        )
        return data_map[key]

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _x: None)
    monkeypatch.setattr(openalex_client.random, "uniform", lambda _a, _b: 0.0)

    result = collect_openalex_works(
        from_date="2022-01-01",
        to_date="2023-12-31",
        limit=None,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        chunk_by="year",
    )

    assert result["records_collected"] == 3
    assert result["chunks_total"] == 2
    assert result["chunks_ok"] == 2
    assert result["chunks_failed"] == 0
    assert len(calls) == 2

    df = pd.read_parquet(result["parquet_path"])
    assert sorted(df["openalex_id"].tolist()) == [
        "https://openalex.org/W1",
        "https://openalex.org/W2",
        "https://openalex.org/W3",
    ]
    assert "topic" in df.columns
    assert "subfield" in df.columns
    assert "field" in df.columns
    assert "domain" in df.columns
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["chunks_ok"] == 2
    assert manifest["chunks_failed"] == 0
    assert len(manifest["chunk_summaries"]) == 2


def test_collect_openalex_retries_on_429_then_succeeds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"calls": 0}

    def fake_get(self: requests.Session, url: str, params: dict[str, Any], timeout: tuple[float, float]) -> _Response:
        state["calls"] += 1
        if state["calls"] == 1:
            return _Response({"meta": {"next_cursor": "*"}, "results": []}, status_code=429, headers={"Retry-After": "0"})
        return _Response(
            {
                "meta": {"next_cursor": None},
                "results": [_make_work("https://openalex.org/W1", "T1", publication_date="2022-01-01", year=2022)],
            }
        )

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _x: None)
    monkeypatch.setattr(openalex_client.random, "uniform", lambda _a, _b: 0.0)

    result = collect_openalex_works(
        from_date="2022-01-01",
        to_date="2022-12-31",
        limit=None,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        max_retries=2,
    )

    assert state["calls"] == 2
    assert result["records_collected"] == 1
    manifest = json.loads(Path(result["manifest_path"]).read_text(encoding="utf-8"))
    assert manifest["chunk_summaries"][0]["retry_events_used"] >= 1


def test_collect_openalex_partial_failure_does_not_overwrite_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    interim_dir = tmp_path / "interim"
    interim_dir.mkdir(parents=True, exist_ok=True)
    existing_csv = interim_dir / "openalex_works.csv"
    existing_parquet = interim_dir / "openalex_works.parquet"
    pd.DataFrame([{"openalex_id": "old"}]).to_csv(existing_csv, index=False)
    pd.DataFrame([{"openalex_id": "old"}]).to_parquet(existing_parquet, index=False, engine="pyarrow")

    def fake_get(self: requests.Session, url: str, params: dict[str, Any], timeout: tuple[float, float]) -> _Response:
        filter_parts = {
            item.split(":", 1)[0]: item.split(":", 1)[1]
            for item in str(params["filter"]).split(",")
            if ":" in item
        }
        if filter_parts["from_publication_date"] == "2022-01-01":
            return _Response(
                {
                    "meta": {"next_cursor": None},
                    "results": [_make_work("https://openalex.org/W1", "T1", publication_date="2022-01-01", year=2022)],
                }
            )
        raise requests.SSLError("record layer failure")

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _x: None)
    monkeypatch.setattr(openalex_client.random, "uniform", lambda _a, _b: 0.0)

    with pytest.raises(RuntimeError):
        collect_openalex_works(
            from_date="2022-01-01",
            to_date="2023-12-31",
            limit=None,
            raw_base_dir=tmp_path / "raw",
            interim_dir=interim_dir,
            chunk_by="year",
            max_retries=0,
        )

    csv_df = pd.read_csv(existing_csv)
    parquet_df = pd.read_parquet(existing_parquet)
    assert csv_df.iloc[0]["openalex_id"] == "old"
    assert parquet_df.iloc[0]["openalex_id"] == "old"

    manifests = sorted((tmp_path / "raw").glob("openalex_*/manifest.json"))
    assert manifests
    manifest = json.loads(manifests[-1].read_text(encoding="utf-8"))
    assert manifest["chunks_failed"] == 1


def test_collect_openalex_chunk_none_keeps_single_range(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    call_filters: list[str] = []

    def fake_get(self: requests.Session, url: str, params: dict[str, Any], timeout: tuple[float, float]) -> _Response:
        call_filters.append(str(params["filter"]))
        return _Response(
            {
                "meta": {"next_cursor": None},
                "results": [_make_work("https://openalex.org/W1", "T1", publication_date="2022-01-01", year=2022)],
            }
        )

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _x: None)
    monkeypatch.setattr(openalex_client.random, "uniform", lambda _a, _b: 0.0)

    result = collect_openalex_works(
        from_date="2022-01-01",
        to_date="2024-05-20",
        limit=None,
        raw_base_dir=tmp_path / "raw",
        interim_dir=tmp_path / "interim",
        chunk_by="none",
    )
    assert result["chunks_total"] == 1
    assert len(call_filters) == 1
    assert "from_publication_date:2022-01-01" in call_filters[0]
    assert "to_publication_date:2024-05-20" in call_filters[0]


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

    def fake_get(self: requests.Session, url: str, params: dict[str, Any], timeout: tuple[float, float]) -> _Response:
        return _Response({"meta": {"next_cursor": None}, "results": [alex_work]})

    monkeypatch.setattr(requests.Session, "get", fake_get)
    monkeypatch.setattr(openalex_client.time, "sleep", lambda _x: None)
    monkeypatch.setattr(openalex_client.random, "uniform", lambda _a, _b: 0.0)

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
    assert record["topic"] == "Mesenchymal stem cell research"
    assert record["subfield"] == "Genetics"
    assert record["field"] == "Medicine"
    assert record["domain"] == "Health Sciences"


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
