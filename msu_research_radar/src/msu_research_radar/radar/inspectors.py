"""Inspection helpers for configuring radar profiles from real data."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

from msu_research_radar.radar.tagging import clean_text, json_list


def _load_table(input_path: str | Path) -> pd.DataFrame:
    path = Path(input_path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input extension for {path}")


def _write_table(df: pd.DataFrame, output_path: str | Path, sheet_name: str) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".xlsx":
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, index=False, sheet_name=sheet_name)
    else:
        df.to_csv(path, index=False)


def _append_sample(samples: list[str], value: str, limit: int) -> None:
    if value and value not in samples and len(samples) < limit:
        samples.append(value)


def inspect_affiliations(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 500,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Export frequent affiliation-like strings with examples."""
    df = _load_table(input_path)
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "sample_authors": [], "sample_titles": []}
    )

    for _, row in df.iterrows():
        values: list[str] = []
        for key in ("lab", "department", "institute"):
            value = clean_text(row.get(key))
            if value:
                values.append(value)
        for key in ("raw_affiliation_strings", "known_msu_affiliations", "current_affiliations"):
            values.extend(clean_text(item) for item in json_list(row.get(key)) if clean_text(item))

        for value in sorted(set(values)):
            bucket = buckets[value]
            bucket["count"] += 1
            _append_sample(bucket["sample_authors"], clean_text(row.get("author_name")), sample_limit)
            _append_sample(bucket["sample_titles"], clean_text(row.get("title")), sample_limit)

    out_rows = [
        {
            "affiliation": affiliation,
            "count": payload["count"],
            "sample_authors": " | ".join(payload["sample_authors"]),
            "sample_titles": " | ".join(payload["sample_titles"]),
        }
        for affiliation, payload in buckets.items()
    ]
    result = pd.DataFrame(out_rows)
    if not result.empty:
        result = result.sort_values(["count", "affiliation"], ascending=[False, True]).head(limit).reset_index(drop=True)
    _write_table(result, output_path, "affiliations")
    return {"input_path": str(input_path), "output_path": str(output_path), "rows": int(len(result))}


def inspect_topics(
    input_path: str | Path,
    output_path: str | Path,
    *,
    limit: int = 500,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Export frequent topic-like strings with examples."""
    df = _load_table(input_path)
    buckets: dict[tuple[str, str], dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "sample_titles": [], "sample_journals": []}
    )

    for _, row in df.iterrows():
        values: list[tuple[str, str]] = []
        for key in ("topic", "subfield", "field", "domain"):
            value = clean_text(row.get(key))
            if value:
                values.append((key, value))
        for item in json_list(row.get("topics")):
            value = clean_text(item)
            if value:
                values.append(("topics", value))

        for key, value in sorted(set(values)):
            bucket = buckets[(key, value)]
            bucket["count"] += 1
            _append_sample(bucket["sample_titles"], clean_text(row.get("title")), sample_limit)
            _append_sample(bucket["sample_journals"], clean_text(row.get("journal")), sample_limit)

    out_rows = [
        {
            "source_field": key,
            "topic_value": value,
            "count": payload["count"],
            "sample_titles": " | ".join(payload["sample_titles"]),
            "sample_journals": " | ".join(payload["sample_journals"]),
        }
        for (key, value), payload in buckets.items()
    ]
    result = pd.DataFrame(out_rows)
    if not result.empty:
        result = result.sort_values(["count", "source_field", "topic_value"], ascending=[False, True, True])
        result = result.head(limit).reset_index(drop=True)
    _write_table(result, output_path, "topics")
    return {"input_path": str(input_path), "output_path": str(output_path), "rows": int(len(result))}
