"""OpenAlex works collection for MSU-affiliated publications."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
MSU_ROR = "https://ror.org/010pmpe69"

RAW_OPENALEX_DIR = Path("data/raw/openalex")
INTERIM_DIR = Path("data/interim")


def _reconstruct_abstract(abstract_inverted_index: dict[str, list[int]] | None) -> str | None:
    if not abstract_inverted_index:
        return None
    max_position = -1
    for positions in abstract_inverted_index.values():
        if positions:
            max_position = max(max_position, max(positions))
    if max_position < 0:
        return None

    tokens = [""] * (max_position + 1)
    for token, positions in abstract_inverted_index.items():
        for position in positions:
            if 0 <= position < len(tokens):
                tokens[position] = token

    reconstructed = " ".join(value for value in tokens if value)
    return reconstructed or None


def _normalize_work(work: dict[str, Any]) -> dict[str, Any]:
    authorships = work.get("authorships") or []
    authors: list[str] = []
    raw_affiliation_strings: list[str] = []
    institutions: list[dict[str, str | None]] = []

    seen_authors: set[str] = set()
    seen_affiliations: set[str] = set()
    seen_institutions: set[tuple[str | None, str | None]] = set()

    for authorship in authorships:
        author_name = (authorship.get("author") or {}).get("display_name")
        if author_name and author_name.lower() not in seen_authors:
            seen_authors.add(author_name.lower())
            authors.append(author_name)

        for raw_affiliation in authorship.get("raw_affiliation_strings") or []:
            if raw_affiliation and raw_affiliation not in seen_affiliations:
                seen_affiliations.add(raw_affiliation)
                raw_affiliation_strings.append(raw_affiliation)

        for institution in authorship.get("institutions") or []:
            display_name = institution.get("display_name")
            ror = institution.get("ror")
            key = (display_name, ror)
            if key in seen_institutions:
                continue
            seen_institutions.add(key)
            institutions.append({"display_name": display_name, "ror": ror})

    topics = [topic.get("display_name") for topic in (work.get("topics") or []) if topic.get("display_name")]
    primary_location = work.get("primary_location") or {}
    source = primary_location.get("source") or {}

    return {
        "openalex_id": work.get("id"),
        "doi": work.get("doi"),
        "title": work.get("display_name"),
        "abstract": _reconstruct_abstract(work.get("abstract_inverted_index")),
        "publication_date": work.get("publication_date"),
        "publication_year": work.get("publication_year"),
        "journal": source.get("display_name"),
        "cited_by_count": work.get("cited_by_count"),
        "authors": authors,
        "raw_affiliation_strings": raw_affiliation_strings,
        "institutions": institutions,
        "topics": topics,
    }


def _serialize_for_table(records: list[dict[str, Any]]) -> pd.DataFrame:
    serialized: list[dict[str, Any]] = []
    for record in records:
        serialized.append(
            {
                **record,
                "authors": json.dumps(record.get("authors") or [], ensure_ascii=False),
                "raw_affiliation_strings": json.dumps(
                    record.get("raw_affiliation_strings") or [],
                    ensure_ascii=False,
                ),
                "institutions": json.dumps(record.get("institutions") or [], ensure_ascii=False),
                "topics": json.dumps(record.get("topics") or [], ensure_ascii=False),
            }
        )
    return pd.DataFrame(serialized)


def collect_openalex_works(
    from_date: str,
    to_date: str,
    limit: int | None,
    per_page: int = 200,
    *,
    raw_base_dir: Path = RAW_OPENALEX_DIR,
    interim_dir: Path = INTERIM_DIR,
) -> dict[str, Any]:
    """Collect OpenAlex works for MSU ROR in a date range."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")

    run_id = datetime.now(timezone.utc).strftime("openalex_%Y%m%dT%H%M%SZ")
    run_raw_dir = raw_base_dir / run_id
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    filter_value = (
        f"institutions.ror:{MSU_ROR},"
        f"from_publication_date:{from_date},"
        f"to_publication_date:{to_date}"
    )
    params = {
        "filter": filter_value,
        "per-page": min(per_page, 200),
        "cursor": "*",
        "sort": "publication_date:desc",
        "select": (
            "id,doi,display_name,abstract_inverted_index,publication_date,publication_year,"
            "cited_by_count,authorships,primary_location,topics"
        ),
    }

    all_works: list[dict[str, Any]] = []
    page_count = 0
    cursors: list[str] = []

    while True:
        if limit is not None and len(all_works) >= limit:
            break
        response = requests.get(OPENALEX_WORKS_URL, params=params, timeout=45)
        response.raise_for_status()
        payload = response.json()
        page_count += 1
        page_path = run_raw_dir / f"page_{page_count:04d}.json"
        page_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        results = payload.get("results") or []
        if not results:
            break
        all_works.extend(results)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        cursors.append(next_cursor)
        params["cursor"] = next_cursor

    collected_raw = all_works if limit is None else all_works[:limit]
    normalized_records = [_normalize_work(work) for work in collected_raw]
    table = _serialize_for_table(normalized_records)

    parquet_path = interim_dir / "openalex_works.parquet"
    csv_path = interim_dir / "openalex_works.csv"
    table.to_parquet(parquet_path, index=False, engine="pyarrow")
    table.to_csv(csv_path, index=False)

    manifest = {
        "run_id": run_id,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenAlex",
        "endpoint": OPENALEX_WORKS_URL,
        "filter": filter_value,
        "from_date": from_date,
        "to_date": to_date,
        "limit": limit,
        "collection_mode": "no_limit" if limit is None else "limited",
        "per_page": params["per-page"],
        "sort": params["sort"],
        "pages_collected": page_count,
        "records_collected": len(collected_raw),
        "next_cursors_seen": cursors,
        "output_csv": str(csv_path),
        "output_parquet": str(parquet_path),
    }
    manifest_path = run_raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "run_id": run_id,
        "records_collected": len(collected_raw),
        "requested_limit": limit,
        "pages_collected": page_count,
        "raw_run_dir": str(run_raw_dir),
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "records": normalized_records,
    }
