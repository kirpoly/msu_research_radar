"""Build master MSU author-paper graph dataset."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

from msu_research_radar.radar.tagging import apply_profile_tags, json_dumps_list, json_list, load_research_profile

MSU_ROR = "https://ror.org/010pmpe69"


def _loads_json(value: Any) -> Any:
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        return json.loads(text)
    if value is None:
        return []
    return value


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _norm_name(value: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", " ", value.lower(), flags=re.UNICODE).split())


def _name_signature(value: str) -> str:
    norm = _norm_name(value)
    if not norm:
        return ""
    parts = [item for item in norm.split() if item]
    if not parts:
        return ""
    if len(parts) == 1:
        surname = parts[0]
        first_initial = ""
    else:
        surname = parts[-1]
        first_initial = parts[0][0] if parts[0] else ""
    return f"{surname}|{first_initial}"


def _extract_msu_authorship_rows(works_df: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for _, work in works_df.iterrows():
        authorships = _loads_json(work.get("authorships"))
        if not authorships:
            continue

        msu_authorships = []
        for authorship in authorships:
            institutions = authorship.get("institutions") or []
            if any((institution.get("ror") or "").lower() == MSU_ROR for institution in institutions):
                msu_authorships.append(authorship)

        if not msu_authorships:
            continue

        msu_names = [_clean((item.get("author") or {}).get("display_name")) for item in msu_authorships]
        msu_names = [name for name in msu_names if name]

        for authorship in msu_authorships:
            author = authorship.get("author") or {}
            author_name = _clean(author.get("display_name"))
            if not author_name:
                continue
            position = _clean(authorship.get("author_position"))
            coauthors = sorted({name for name in msu_names if name and name != author_name})
            rows.append(
                {
                    "openalex_id": work.get("openalex_id"),
                    "doi": work.get("doi"),
                    "title": work.get("title"),
                    "publication_date": work.get("publication_date"),
                    "publication_year": work.get("publication_year"),
                    "journal": work.get("journal"),
                    "cited_by_count": work.get("cited_by_count"),
                    "abstract": work.get("abstract"),
                    "topics": work.get("topics"),
                    "topic": work.get("topic"),
                    "subfield": work.get("subfield"),
                    "field": work.get("field"),
                    "domain": work.get("domain"),
                    "author_name": author_name,
                    "raw_author_name": _clean(authorship.get("raw_author_name")) or None,
                    "openalex_author_id": author.get("id"),
                    "raw_orcid": author.get("orcid") or authorship.get("raw_orcid"),
                    "raw_affiliation_strings": authorship.get("raw_affiliation_strings") or [],
                    "author_position": position or None,
                    "is_first_author": position.lower() == "first",
                    "is_last_author": position.lower() == "last",
                    "is_corresponding": bool(authorship.get("is_corresponding")),
                    "msu_coauthors": coauthors,
                    "msu_coauthor_count": len(coauthors),
                }
            )
    return rows


def _read_resolved_jsonl(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    exact_mapping: dict[str, dict[str, Any]] = {}
    signature_mapping: dict[str, list[dict[str, Any]]] = {}
    if not path.exists():
        return exact_mapping, signature_mapping
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            key = _norm_name(_clean(payload.get("author_name")))
            if key:
                exact_mapping[key] = payload
                signature = _name_signature(key)
                if signature:
                    signature_mapping.setdefault(signature, []).append(payload)
    return exact_mapping, signature_mapping


def _pick_unique_profile(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    by_profile: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        parsed = candidate.get("parsed_profile") or {}
        profile_key = (
            _clean(candidate.get("profile_url"))
            or _clean(parsed.get("profile_url"))
            or _clean(parsed.get("irid"))
            or _norm_name(_clean(candidate.get("author_name")))
        )
        by_profile[profile_key] = candidate
    if len(by_profile) == 1:
        return next(iter(by_profile.values()))
    return None


def build_master_author_graph(
    works_path: str | Path,
    authors_path: str | Path,
    resolved_path: str | Path,
    output_path: str | Path,
    *,
    profile_config: str | Path | None = None,
    profile_name: str | None = None,
) -> dict[str, Any]:
    """Build author-paper master dataset for accepted Istina matches."""
    works_df = pd.read_parquet(works_path)
    _ = pd.read_csv(authors_path)  # reserved for schema guard/traceability
    resolved_map, resolved_signature_map = _read_resolved_jsonl(Path(resolved_path))
    research_profile = load_research_profile(profile_config, profile_name) if profile_config else None

    authorship_rows = _extract_msu_authorship_rows(works_df)
    master_rows: list[dict[str, Any]] = []
    for row in authorship_rows:
        normalized_author = _norm_name(_clean(row["author_name"]))
        resolved = resolved_map.get(normalized_author)
        if not resolved:
            signature = _name_signature(normalized_author)
            candidates = resolved_signature_map.get(signature) or []
            if len(candidates) == 1:
                resolved = candidates[0]
            else:
                resolved = _pick_unique_profile(candidates)
        if not resolved:
            continue

        parsed_profile = resolved.get("parsed_profile") or {}
        merged = dict(row)
        merged.update(
            {
                "istina_profile_url": resolved.get("profile_url") or parsed_profile.get("profile_url"),
                "irid": parsed_profile.get("irid"),
                "orcid": parsed_profile.get("orcid") or row.get("raw_orcid"),
                "spin": parsed_profile.get("spin"),
                "researcher_id": parsed_profile.get("researcher_id"),
                "lab": parsed_profile.get("lab"),
                "department": parsed_profile.get("department"),
                "institute": parsed_profile.get("institute"),
                "position": parsed_profile.get("position"),
                "matched_query": resolved.get("matched_query"),
                "resolution_score": resolved.get("resolution_score"),
                "matched_title": resolved.get("matched_title"),
                "matched_coauthors": resolved.get("matched_coauthors") or [],
            }
        )
        if profile_config:
            merged.update(apply_profile_tags(merged, research_profile or {}))
        master_rows.append(merged)

    master_df = pd.DataFrame(master_rows)
    if not master_df.empty:
        for column in ("msu_coauthors", "matched_coauthors", "raw_affiliation_strings", "topics"):
            if column in master_df.columns:
                master_df[column] = master_df[column].apply(lambda x: json_dumps_list(json_list(x)))
        for column in ("matched_affiliation_tags", "matched_topic_tags", "matched_role_tags", "match_reasons"):
            if column in master_df.columns:
                master_df[column] = master_df[column].apply(lambda x: json_dumps_list(json_list(x)))
        master_df = master_df.sort_values(["publication_date", "author_name"], ascending=[False, True]).reset_index(
            drop=True
        )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    master_df.to_parquet(output, index=False, engine="pyarrow")

    return {
        "works_path": str(works_path),
        "authors_path": str(authors_path),
        "resolved_path": str(resolved_path),
        "profile_config": str(profile_config) if profile_config else None,
        "profile_name": profile_name,
        "output_path": str(output),
        "rows": int(len(master_df)),
        "resolved_authors_used": int(master_df["author_name"].nunique()) if not master_df.empty else 0,
    }
