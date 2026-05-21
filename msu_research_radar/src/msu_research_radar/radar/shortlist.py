"""Excel shortlist export for configurable supervisor/contact discovery."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from msu_research_radar.radar.tagging import (
    apply_profile_tags,
    clean_text,
    json_dumps_list,
    json_list,
    load_research_profile,
    normalize_name,
)

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


def _name_signature(value: str) -> str:
    norm = normalize_name(value)
    if not norm:
        return ""
    parts = [part for part in norm.split() if part]
    if not parts:
        return ""
    surname = parts[-1] if len(parts) > 1 else parts[0]
    first_initial = parts[0][0] if len(parts) > 1 and parts[0] else ""
    return f"{surname}|{first_initial}"


def _read_resolved_jsonl(path: Path | None) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    exact: dict[str, dict[str, Any]] = {}
    signatures: dict[str, list[dict[str, Any]]] = {}
    if path is None or not path.exists():
        return exact, signatures
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            key = normalize_name(clean_text(payload.get("author_name")))
            if not key:
                continue
            exact[key] = payload
            signature = _name_signature(key)
            if signature:
                signatures.setdefault(signature, []).append(payload)
    return exact, signatures


def _pick_unique_profile(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    by_profile: dict[str, dict[str, Any]] = {}
    for candidate in candidates:
        parsed = candidate.get("parsed_profile") or {}
        key = (
            clean_text(candidate.get("profile_url"))
            or clean_text(parsed.get("profile_url"))
            or clean_text(parsed.get("irid"))
            or normalize_name(clean_text(candidate.get("author_name")))
        )
        if key:
            by_profile[key] = candidate
    return next(iter(by_profile.values())) if len(by_profile) == 1 else None


def _resolve_author(name: str, exact: dict[str, dict[str, Any]], signatures: dict[str, list[dict[str, Any]]]) -> dict[str, Any] | None:
    normalized = normalize_name(name)
    resolved = exact.get(normalized)
    if resolved:
        return resolved
    candidates = signatures.get(_name_signature(normalized)) or []
    if len(candidates) == 1:
        return candidates[0]
    return _pick_unique_profile(candidates)


def _msu_authorships(authorships: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for authorship in authorships:
        institutions = authorship.get("institutions") or []
        if any((institution.get("ror") or "").lower() == MSU_ROR for institution in institutions):
            rows.append(authorship)
    return rows


def _author_identity(row: dict[str, Any]) -> str:
    for key in ("istina_profile_url", "irid", "openalex_author_id", "orcid", "author_name"):
        value = clean_text(row.get(key))
        if value:
            return f"{key}:{normalize_name(value) if key == 'author_name' else value}"
    return "author:unknown"


def _unit_key(row: dict[str, Any]) -> str:
    for key in ("lab", "department", "institute"):
        value = clean_text(row.get(key))
        if value:
            return value
    affiliations = json_list(row.get("raw_affiliation_strings"))
    return clean_text(affiliations[0]) if affiliations else "Unknown unit"


def _rows_from_works(works_path: str | Path, resolved_path: str | Path | None, profile: dict[str, Any]) -> pd.DataFrame:
    works_df = pd.read_parquet(works_path)
    exact, signatures = _read_resolved_jsonl(Path(resolved_path) if resolved_path else None)
    rows: list[dict[str, Any]] = []

    for _, work in works_df.iterrows():
        authorships = _loads_json(work.get("authorships"))
        msu_authors = _msu_authorships(authorships)
        if not msu_authors:
            continue
        msu_names = [
            clean_text((authorship.get("author") or {}).get("display_name"))
            for authorship in msu_authors
            if clean_text((authorship.get("author") or {}).get("display_name"))
        ]

        for authorship in msu_authors:
            author = authorship.get("author") or {}
            author_name = clean_text(author.get("display_name"))
            if not author_name:
                continue
            resolved = _resolve_author(author_name, exact, signatures)
            parsed_profile = (resolved or {}).get("parsed_profile") or {}
            position = clean_text(authorship.get("author_position"))
            row = {
                "openalex_id": work.get("openalex_id"),
                "doi": work.get("doi"),
                "title": work.get("title"),
                "abstract": work.get("abstract"),
                "publication_date": work.get("publication_date"),
                "publication_year": work.get("publication_year"),
                "journal": work.get("journal"),
                "cited_by_count": work.get("cited_by_count"),
                "topics": work.get("topics"),
                "topic": work.get("topic"),
                "subfield": work.get("subfield"),
                "field": work.get("field"),
                "domain": work.get("domain"),
                "author_name": author_name,
                "raw_author_name": clean_text(authorship.get("raw_author_name")),
                "openalex_author_id": author.get("id"),
                "orcid": parsed_profile.get("orcid") or author.get("orcid") or authorship.get("raw_orcid"),
                "author_position": position or None,
                "is_first_author": position.lower() == "first",
                "is_last_author": position.lower() == "last",
                "is_corresponding": bool(authorship.get("is_corresponding")),
                "raw_affiliation_strings": authorship.get("raw_affiliation_strings") or [],
                "msu_coauthors": sorted({name for name in msu_names if name != author_name}),
                "msu_coauthor_count": max(0, len(set(msu_names)) - 1),
                "istina_profile_url": (resolved or {}).get("profile_url") or parsed_profile.get("profile_url"),
                "irid": parsed_profile.get("irid"),
                "spin": parsed_profile.get("spin"),
                "researcher_id": parsed_profile.get("researcher_id"),
                "lab": parsed_profile.get("lab"),
                "department": parsed_profile.get("department"),
                "institute": parsed_profile.get("institute"),
                "position": parsed_profile.get("position"),
                "resolution_score": (resolved or {}).get("resolution_score"),
                "matched_query": (resolved or {}).get("matched_query"),
            }
            row.update(apply_profile_tags(row, profile))
            row["author_identity"] = _author_identity(row)
            row["unit_key"] = _unit_key(row)
            rows.append(row)

    result = pd.DataFrame(rows)
    for key in ("raw_affiliation_strings", "topics", "msu_coauthors"):
        if key in result.columns:
            result[key] = result[key].apply(lambda value: json_dumps_list(json_list(value)))
    for key in ("matched_affiliation_tags", "matched_topic_tags", "matched_role_tags", "match_reasons"):
        if key in result.columns:
            result[key] = result[key].apply(lambda value: json_dumps_list(json_list(value)))
    return result


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    return clean_text(value).lower() in {"1", "true", "yes", "y"}


def _year_is_recent(value: Any, recent_years: set[int]) -> bool:
    try:
        return int(value) in recent_years
    except (TypeError, ValueError):
        return False


def _first_non_empty(series: pd.Series) -> str | None:
    for value in series.tolist():
        text = clean_text(value)
        if text:
            return text
    return None


def _sample_values(series: pd.Series, limit: int = 5) -> str:
    values: list[str] = []
    for value in series.fillna("").astype(str).tolist():
        text = clean_text(value)
        if text and text not in values:
            values.append(text)
        if len(values) >= limit:
            break
    return " | ".join(values)


def _tag_count(series: pd.Series) -> int:
    tags: set[str] = set()
    for value in series.tolist():
        tags.update(str(item) for item in json_list(value) if clean_text(item))
    return len(tags)


def _score(weights: dict[str, Any], metrics: dict[str, Any]) -> tuple[float, dict[str, float]]:
    components = {f"component_{key}": float(metrics.get(key, 0.0)) * float(weight) for key, weight in weights.items()}
    return float(sum(components.values())), components


def _build_author_tables(relevant: pd.DataFrame, profile: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    scoring = profile.get("scoring") or {}
    recent_years = set(int(item) for item in scoring.get("recent_years", [2025, 2026]))
    supervisor_weights = scoring.get("supervisor_weights") or {}
    young_weights = scoring.get("young_contact_weights") or {}
    rows: list[dict[str, Any]] = []

    for author_identity, group in relevant.groupby("author_identity", dropna=False):
        paper_count = int(group["openalex_id"].nunique())
        recent_count = int(sum(_year_is_recent(year, recent_years) for year in group["publication_year"].tolist()))
        first_count = int(group["is_first_author"].apply(_as_bool).sum())
        last_count = int(group["is_last_author"].apply(_as_bool).sum())
        corr_count = int(group["is_corresponding"].apply(_as_bool).sum())
        role_tags = set()
        for value in group["matched_role_tags"].tolist():
            role_tags.update(str(item) for item in json_list(value))
        profile_match_score = float(pd.to_numeric(group["profile_match_score"], errors="coerce").fillna(0).sum())
        unique_msu_coauthors = set()
        for value in group["msu_coauthors"].tolist():
            unique_msu_coauthors.update(str(item) for item in json_list(value) if clean_text(item))
        has_istina_profile = int(group["istina_profile_url"].fillna("").astype(str).str.strip().ne("").any())
        has_unit = int(group["unit_key"].fillna("").astype(str).str.strip().ne("Unknown unit").any())

        metrics = {
            "relevant_pub_count": paper_count,
            "recent_pub_count": recent_count,
            "first_author_count": first_count,
            "last_author_count": last_count,
            "corresponding_author_count": corr_count,
            "senior_role_tag_count": int("senior" in role_tags or "pi" in role_tags),
            "young_role_tag_count": int("young" in role_tags or "trainee" in role_tags),
            "matched_tag_count": _tag_count(group["matched_affiliation_tags"]) + _tag_count(group["matched_topic_tags"]),
            "profile_match_score": profile_match_score,
            "unique_msu_coauthors": len(unique_msu_coauthors),
            "has_istina_profile": has_istina_profile,
            "has_unit": has_unit,
        }
        supervisor_score, supervisor_components_raw = _score(supervisor_weights, metrics)
        young_score, young_components_raw = _score(young_weights, metrics)
        supervisor_components = {
            f"supervisor_{key}": value for key, value in supervisor_components_raw.items()
        }
        young_components = {f"young_{key}": value for key, value in young_components_raw.items()}
        base = {
            "author_identity": author_identity,
            "author_name": _first_non_empty(group["author_name"]),
            "istina_profile_url": _first_non_empty(group["istina_profile_url"]),
            "openalex_author_id": _first_non_empty(group["openalex_author_id"]),
            "orcid": _first_non_empty(group["orcid"]),
            "lab": _first_non_empty(group["lab"]),
            "department": _first_non_empty(group["department"]),
            "institute": _first_non_empty(group["institute"]),
            "position": _first_non_empty(group["position"]),
            "unit_key": _first_non_empty(group["unit_key"]),
            "sample_papers": _sample_values(group["title"]),
            "sample_dois": _sample_values(group["doi"]),
            "matched_affiliation_tags": _sample_values(group["matched_affiliation_tags"]),
            "matched_topic_tags": _sample_values(group["matched_topic_tags"]),
            "matched_role_tags": _sample_values(group["matched_role_tags"]),
            **metrics,
        }
        rows.append({**base, "supervisor_score": supervisor_score, **supervisor_components, "young_contact_score": young_score, **young_components})

    authors = pd.DataFrame(rows)
    if authors.empty:
        return authors, authors.copy()
    supervisors = authors.sort_values(["supervisor_score", "relevant_pub_count"], ascending=[False, False]).reset_index(drop=True)
    young_contacts = authors.sort_values(["young_contact_score", "recent_pub_count"], ascending=[False, False]).reset_index(drop=True)
    return supervisors, young_contacts


def _build_labs(relevant: pd.DataFrame, profile: dict[str, Any]) -> pd.DataFrame:
    scoring = profile.get("scoring") or {}
    recent_years = set(int(item) for item in scoring.get("recent_years", [2025, 2026]))
    rows = []
    for unit_key, group in relevant.groupby("unit_key", dropna=False):
        rows.append(
            {
                "unit_key": unit_key,
                "lab": _first_non_empty(group["lab"]),
                "department": _first_non_empty(group["department"]),
                "institute": _first_non_empty(group["institute"]),
                "relevant_pub_count": int(group["openalex_id"].nunique()),
                "recent_pub_count": int(sum(_year_is_recent(year, recent_years) for year in group["publication_year"].tolist())),
                "unique_authors": int(group["author_identity"].nunique()),
                "sample_papers": _sample_values(group["title"]),
                "matched_affiliation_tags": _sample_values(group["matched_affiliation_tags"]),
                "matched_topic_tags": _sample_values(group["matched_topic_tags"]),
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result = result.sort_values(["relevant_pub_count", "recent_pub_count"], ascending=[False, False]).reset_index(drop=True)
    return result


def _build_edges(relevant: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, row in relevant.iterrows():
        for coauthor in json_list(row.get("msu_coauthors")):
            rows.append(
                {
                    "author_identity": row.get("author_identity"),
                    "author_name": row.get("author_name"),
                    "coauthor_name": coauthor,
                    "openalex_id": row.get("openalex_id"),
                    "title": row.get("title"),
                    "doi": row.get("doi"),
                    "publication_year": row.get("publication_year"),
                    "matched_affiliation_tags": row.get("matched_affiliation_tags"),
                    "matched_topic_tags": row.get("matched_topic_tags"),
                }
            )
    return pd.DataFrame(rows)


def _config_sheet(profile: dict[str, Any]) -> pd.DataFrame:
    rows = []

    def walk(prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                walk(f"{prefix}.{key}" if prefix else str(key), child)
        elif isinstance(value, list):
            rows.append({"key": prefix, "value": json.dumps(value, ensure_ascii=False)})
        else:
            rows.append({"key": prefix, "value": value})

    walk("", profile)
    return pd.DataFrame(rows)


def _required_relevant(row: pd.Series) -> bool:
    return bool(json_list(row.get("matched_affiliation_tags")) or json_list(row.get("matched_topic_tags")))


def export_shortlist(
    *,
    works_path: str | Path,
    output_path: str | Path,
    profile_config: str | Path,
    profile_name: str | None = None,
    resolved_path: str | Path | None = None,
) -> dict[str, Any]:
    """Export a configurable supervisor/contact shortlist workbook."""
    profile = load_research_profile(profile_config, profile_name)
    papers = _rows_from_works(works_path, resolved_path, profile)
    relevant = papers.loc[papers.apply(_required_relevant, axis=1)].copy() if not papers.empty else papers.copy()

    supervisors, young_contacts = _build_author_tables(relevant, profile) if not relevant.empty else (pd.DataFrame(), pd.DataFrame())
    labs = _build_labs(relevant, profile) if not relevant.empty else pd.DataFrame()
    edges = _build_edges(relevant) if not relevant.empty else pd.DataFrame()
    unresolved = pd.DataFrame()
    if not relevant.empty:
        unresolved = (
            relevant.loc[relevant["istina_profile_url"].fillna("").astype(str).str.strip().eq("")]
            .groupby(["author_identity", "author_name"], dropna=False, as_index=False)
            .agg(
                relevant_pub_count=("openalex_id", "nunique"),
                sample_papers=("title", lambda s: _sample_values(s)),
                matched_affiliation_tags=("matched_affiliation_tags", lambda s: _sample_values(s)),
                matched_topic_tags=("matched_topic_tags", lambda s: _sample_values(s)),
            )
            .sort_values(["relevant_pub_count", "author_name"], ascending=[False, True])
        )

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        supervisors.to_excel(writer, index=False, sheet_name="supervisors")
        young_contacts.to_excel(writer, index=False, sheet_name="young_contacts")
        labs.to_excel(writer, index=False, sheet_name="labs_or_units")
        relevant.to_excel(writer, index=False, sheet_name="papers")
        edges.to_excel(writer, index=False, sheet_name="coauthor_edges")
        unresolved.to_excel(writer, index=False, sheet_name="unresolved_istina")
        _config_sheet(profile).to_excel(writer, index=False, sheet_name="config_used")

    return {
        "works_path": str(works_path),
        "resolved_path": str(resolved_path) if resolved_path else None,
        "profile_config": str(profile_config),
        "profile_name": profile["name"],
        "output_path": str(out),
        "papers_total": int(len(papers)),
        "papers_relevant": int(len(relevant)),
        "supervisors": int(len(supervisors)),
        "young_contacts": int(len(young_contacts)),
    }
