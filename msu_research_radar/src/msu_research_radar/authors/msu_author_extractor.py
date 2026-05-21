"""Extract and aggregate MSU-affiliated authors from OpenAlex works."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd

MSU_ROR = "https://ror.org/010pmpe69"


def _loads_json(value: Any) -> Any:
    if pd.isna(value):
        return []
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return json.loads(value)
    if value is None:
        return []
    return value


def _clean(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return " ".join(str(value).split())


def _author_key(authorship: dict[str, Any]) -> str:
    author = authorship.get("author") or {}
    openalex_id = _clean(author.get("id"))
    if openalex_id:
        return f"openalex:{openalex_id}"
    orcid = _clean(author.get("orcid") or authorship.get("raw_orcid"))
    if orcid:
        return f"orcid:{orcid}"
    return f"name:{_clean(author.get('display_name'))}"


def _extract_msu_authors(authorships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    msu_authors: list[dict[str, Any]] = []
    for authorship in authorships:
        institutions = authorship.get("institutions") or []
        has_msu = any((institution.get("ror") or "").lower() == MSU_ROR for institution in institutions)
        if not has_msu:
            continue

        author = authorship.get("author") or {}
        name = _clean(author.get("display_name"))
        if not name:
            continue
        msu_authors.append(authorship)
    return msu_authors


def extract_msu_authors(input_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Extract MSU-affiliated authors and aggregate author-level statistics."""
    in_path = Path(input_path)
    out_path = Path(output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    works_df = pd.read_parquet(in_path)
    stats: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "author_name": "",
            "openalex_author_id": "",
            "orcid": "",
            "raw_author_names": set(),
            "paper_count": 0,
            "first_author_count": 0,
            "last_author_count": 0,
            "known_msu_affiliations": set(),
            "sample_titles": [],
            "sample_dois": [],
            "coauthors_msu": set(),
        }
    )

    total_works = len(works_df)
    works_with_msu_authors = 0

    for _, row in works_df.iterrows():
        authorships = _loads_json(row.get("authorships"))
        if not authorships:
            continue

        msu_authorships = _extract_msu_authors(authorships)
        if not msu_authorships:
            continue
        works_with_msu_authors += 1

        msu_names = []
        for authorship in msu_authorships:
            author_name = _clean((authorship.get("author") or {}).get("display_name"))
            if author_name:
                msu_names.append(author_name)

        for authorship in msu_authorships:
            author = authorship.get("author") or {}
            author_name = _clean(author.get("display_name"))
            if not author_name:
                continue

            bucket = stats[_author_key(authorship)]
            if not bucket["author_name"]:
                bucket["author_name"] = author_name
            if not bucket["openalex_author_id"]:
                bucket["openalex_author_id"] = _clean(author.get("id"))
            if not bucket["orcid"]:
                bucket["orcid"] = _clean(author.get("orcid") or authorship.get("raw_orcid"))
            raw_author_name = _clean(authorship.get("raw_author_name"))
            if raw_author_name:
                bucket["raw_author_names"].add(raw_author_name)
            bucket["paper_count"] += 1

            position = (authorship.get("author_position") or "").lower()
            if position == "first":
                bucket["first_author_count"] += 1
            if position == "last":
                bucket["last_author_count"] += 1

            for affiliation in authorship.get("raw_affiliation_strings") or []:
                if affiliation:
                    bucket["known_msu_affiliations"].add(_clean(affiliation))

            title = _clean(row.get("title"))
            if title and len(bucket["sample_titles"]) < 10 and title not in bucket["sample_titles"]:
                bucket["sample_titles"].append(title)

            doi = _clean(row.get("doi"))
            if doi and len(bucket["sample_dois"]) < 10 and doi not in bucket["sample_dois"]:
                bucket["sample_dois"].append(doi)

            for coauthor in msu_names:
                if coauthor and coauthor != author_name:
                    bucket["coauthors_msu"].add(coauthor)

    rows: list[dict[str, Any]] = []
    for _, bucket in stats.items():
        rows.append(
            {
                "author_name": bucket["author_name"],
                "openalex_author_id": bucket["openalex_author_id"] or None,
                "orcid": bucket["orcid"] or None,
                "raw_author_names": json.dumps(sorted(bucket["raw_author_names"]), ensure_ascii=False),
                "paper_count": bucket["paper_count"],
                "first_author_count": bucket["first_author_count"],
                "last_author_count": bucket["last_author_count"],
                "known_msu_affiliations": json.dumps(
                    sorted(bucket["known_msu_affiliations"]),
                    ensure_ascii=False,
                ),
                "sample_titles": json.dumps(bucket["sample_titles"], ensure_ascii=False),
                "sample_dois": json.dumps(bucket["sample_dois"], ensure_ascii=False),
                "coauthors_msu": json.dumps(sorted(bucket["coauthors_msu"]), ensure_ascii=False),
            }
        )

    authors_df = pd.DataFrame(rows)
    if not authors_df.empty:
        authors_df = authors_df.sort_values(["paper_count", "author_name"], ascending=[False, True]).reset_index(
            drop=True
        )
    authors_df.to_csv(out_path, index=False)

    return {
        "input_path": str(in_path),
        "output_path": str(out_path),
        "works_total": total_works,
        "works_with_msu_authors": works_with_msu_authors,
        "authors_extracted": len(authors_df),
    }
