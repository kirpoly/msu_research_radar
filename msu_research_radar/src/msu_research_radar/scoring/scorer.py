"""Transparent scoring for labs and authors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_WEIGHTS_PATH = Path("config/scoring_weights.yaml")

TOPIC_BUCKETS = {
    "topic_aging_count": [
        "aging",
        "ageing",
        "senescence",
        "longevity",
        "geriatric",
        "age-associated",
    ],
    "topic_bioengineering_regeneration_count": [
        "bioengineering",
        "regeneration",
        "regenerative",
        "tissue engineering",
        "stem cell",
        "mesenchymal",
    ],
    "topic_systems_omics_count": [
        "systems biology",
        "omics",
        "transcriptomics",
        "proteomics",
        "metabolomics",
        "multiomics",
    ],
    "topic_genetics_epigenetics_count": [
        "genetic",
        "genetics",
        "genomics",
        "epigenetic",
        "epigenomics",
        "dna methylation",
        "chromatin",
    ],
}

SENIOR_POSITION_TERMS = [
    "профессор",
    "доцент",
    "заведующий",
    "руководитель",
    "chief",
    "head",
    "principal investigator",
    "leading researcher",
    "ведущий научный сотрудник",
    "главный научный сотрудник",
]

YOUNG_POSITION_TERMS = [
    "аспирант",
    "студент",
    "intern",
    "junior",
    "младший научный сотрудник",
    "соискатель",
    "ординатор",
]


def _clean_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split())


def _lower_text(value: Any) -> str:
    return _clean_text(value).lower()


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return parsed if isinstance(parsed, list) else [parsed]
    return []


def _as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, (int, float)):
        return bool(value)
    text = _lower_text(value)
    return text in {"1", "true", "yes", "y"}


def _first_non_empty(group: pd.Series) -> str | None:
    for value in group.tolist():
        text = _clean_text(value)
        if text:
            return text
    return None


def _load_scoring_config(config_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(config_path) if config_path is not None else DEFAULT_WEIGHTS_PATH
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    recent_years = [int(item) for item in payload.get("recent_years", [2025, 2026])]
    sample_papers_limit = int(payload.get("sample_papers_limit", 5))
    lab_weights = payload.get("lab_weights") or {}
    author_weights = payload.get("author_weights") or {}
    return {
        "path": str(path),
        "recent_years": recent_years,
        "sample_papers_limit": sample_papers_limit,
        "lab_weights": {str(k): float(v) for k, v in lab_weights.items()},
        "author_weights": {str(k): float(v) for k, v in author_weights.items()},
    }


def _topic_corpus(row: pd.Series) -> str:
    parts: list[str] = []
    for key in ("topic", "subfield", "field", "domain", "title", "journal", "matched_title"):
        parts.append(_clean_text(row.get(key)))
    for value in _json_list(row.get("topics")):
        parts.append(_clean_text(value))
    return _lower_text(" ".join(parts))


def _add_topic_flags(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    corpus = work.apply(_topic_corpus, axis=1)
    for bucket_name, terms in TOPIC_BUCKETS.items():
        work[bucket_name] = corpus.apply(lambda text, t=terms: any(term in text for term in t))
    return work


def _is_recent_year(value: Any, recent_years: set[int]) -> bool:
    if value is None:
        return False
    try:
        return int(value) in recent_years
    except (TypeError, ValueError):
        return False


def _lab_key(row: pd.Series) -> str:
    for key in ("lab", "department", "institute"):
        value = _clean_text(row.get(key))
        if value:
            return value
    return "Unknown unit"


def _is_senior_position(value: Any) -> bool:
    text = _lower_text(value)
    return bool(text) and any(term in text for term in SENIOR_POSITION_TERMS)


def _is_young_position(value: Any) -> bool:
    text = _lower_text(value)
    return bool(text) and any(term in text for term in YOUNG_POSITION_TERMS)


def _sample_papers(papers_df: pd.DataFrame, limit: int) -> tuple[str, str]:
    if papers_df.empty:
        return "", ""
    ordered = papers_df.copy()
    ordered["__publication_date"] = pd.to_datetime(ordered["publication_date"], errors="coerce")
    ordered["__cited"] = pd.to_numeric(ordered["cited_by_count"], errors="coerce").fillna(0)
    ordered = ordered.sort_values(["__publication_date", "__cited"], ascending=[False, False])
    ordered = ordered.drop_duplicates(subset=["openalex_id"], keep="first").head(limit)
    sample_titles = ordered["title"].fillna("").astype(str).tolist()
    sample_dois = ordered["doi"].fillna("").astype(str).tolist()
    sample_titles = [item for item in sample_titles if item.strip()]
    sample_dois = [item for item in sample_dois if item.strip()]
    return " | ".join(sample_titles), " | ".join(sample_dois)


def _paper_level(group: pd.DataFrame) -> pd.DataFrame:
    agg = group.groupby("openalex_id", dropna=False, as_index=False).agg(
        publication_year=("publication_year", "max"),
        publication_date=("publication_date", "max"),
        cited_by_count=("cited_by_count", "max"),
        title=("title", "first"),
        doi=("doi", "first"),
        topic_aging_count=("topic_aging_count", "max"),
        topic_bioengineering_regeneration_count=("topic_bioengineering_regeneration_count", "max"),
        topic_systems_omics_count=("topic_systems_omics_count", "max"),
        topic_genetics_epigenetics_count=("topic_genetics_epigenetics_count", "max"),
    )
    return agg


def score_labs(input_master: str | Path, config_path: str | Path | None = None) -> pd.DataFrame:
    config = _load_scoring_config(config_path)
    weights = config["lab_weights"]
    recent_years = set(config["recent_years"])
    sample_limit = config["sample_papers_limit"]

    df = pd.read_parquet(input_master)
    if df.empty:
        return pd.DataFrame()

    work = _add_topic_flags(df)
    work["lab_group"] = work.apply(_lab_key, axis=1)

    output_rows: list[dict[str, Any]] = []
    for lab_group, group in work.groupby("lab_group", dropna=False):
        papers = _paper_level(group)
        pub_count = int(papers["openalex_id"].nunique())
        recent_pub_count = int(sum(_is_recent_year(year, recent_years) for year in papers["publication_year"].tolist()))
        topic_aging_count = int(papers["topic_aging_count"].sum())
        topic_bio_count = int(papers["topic_bioengineering_regeneration_count"].sum())
        topic_systems_count = int(papers["topic_systems_omics_count"].sum())
        topic_genetics_count = int(papers["topic_genetics_epigenetics_count"].sum())
        unique_active_authors = int(group["author_name"].fillna("").astype(str).replace("", pd.NA).dropna().nunique())
        senior_author_signals = int(
            group.loc[group["position"].apply(_is_senior_position), "author_name"]
            .fillna("")
            .astype(str)
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )
        young_staff_signals_explicit = int(
            group.loc[group["position"].apply(_is_young_position), "author_name"]
            .fillna("")
            .astype(str)
            .replace("", pd.NA)
            .dropna()
            .nunique()
        )

        metrics = {
            "pub_count": pub_count,
            "recent_pub_count_2025_2026": recent_pub_count,
            "topic_aging_count": topic_aging_count,
            "topic_bioengineering_regeneration_count": topic_bio_count,
            "topic_systems_omics_count": topic_systems_count,
            "topic_genetics_epigenetics_count": topic_genetics_count,
            "unique_active_authors": unique_active_authors,
            "senior_author_signals": senior_author_signals,
            "young_staff_signals_explicit": young_staff_signals_explicit,
        }
        component_scores = {f"component_{key}": float(metrics[key]) * float(weights.get(key, 0.0)) for key in metrics}
        total_score = float(sum(component_scores.values()))
        sample_titles, sample_dois = _sample_papers(papers, limit=sample_limit)

        output_rows.append(
            {
                "lab_group": lab_group,
                "lab": _first_non_empty(group["lab"]) if "lab" in group.columns else None,
                "department": _first_non_empty(group["department"]) if "department" in group.columns else None,
                "institute": _first_non_empty(group["institute"]) if "institute" in group.columns else None,
                "total_score": total_score,
                "supporting_publication_count": pub_count,
                "recent_publication_count": recent_pub_count,
                "sample_papers": sample_titles,
                "sample_dois": sample_dois,
                "topic_counts": json.dumps(
                    {
                        "aging_adjacent": topic_aging_count,
                        "bioengineering_regeneration": topic_bio_count,
                        "systems_omics": topic_systems_count,
                        "genetics_epigenetics": topic_genetics_count,
                    },
                    ensure_ascii=False,
                ),
                **metrics,
                **component_scores,
            }
        )

    result = pd.DataFrame(output_rows)
    if not result.empty:
        result = result.sort_values(["total_score", "supporting_publication_count"], ascending=[False, False]).reset_index(drop=True)
    return result


def score_authors(input_master: str | Path, config_path: str | Path | None = None) -> pd.DataFrame:
    config = _load_scoring_config(config_path)
    weights = config["author_weights"]
    recent_years = set(config["recent_years"])
    sample_limit = config["sample_papers_limit"]

    df = pd.read_parquet(input_master)
    if df.empty:
        return pd.DataFrame()

    work = _add_topic_flags(df)
    output_rows: list[dict[str, Any]] = []
    for author_name, group in work.groupby("author_name", dropna=False):
        papers = _paper_level(group)
        relevant_pub_count = int(papers["openalex_id"].nunique())
        recent_pub_count = int(sum(_is_recent_year(year, recent_years) for year in papers["publication_year"].tolist()))
        topic_aging_count = int(papers["topic_aging_count"].sum())
        topic_bio_count = int(papers["topic_bioengineering_regeneration_count"].sum())
        topic_systems_count = int(papers["topic_systems_omics_count"].sum())
        topic_genetics_count = int(papers["topic_genetics_epigenetics_count"].sum())
        topic_match_counts = topic_aging_count + topic_bio_count + topic_systems_count + topic_genetics_count
        first_author_count = int(group["is_first_author"].apply(_as_bool).sum()) if "is_first_author" in group.columns else 0
        last_author_count = int(group["is_last_author"].apply(_as_bool).sum()) if "is_last_author" in group.columns else 0
        corresponding_author_count = (
            int(group["is_corresponding"].apply(_as_bool).sum()) if "is_corresponding" in group.columns else 0
        )
        has_istina_profile = int(group["istina_profile_url"].fillna("").astype(str).str.strip().ne("").any())
        has_lab = int(group["lab"].fillna("").astype(str).str.strip().ne("").any())

        metrics = {
            "relevant_pub_count": relevant_pub_count,
            "first_author_count": first_author_count,
            "last_author_count": last_author_count,
            "corresponding_author_count_explicit": corresponding_author_count,
            "has_istina_profile": has_istina_profile,
            "has_lab": has_lab,
            "topic_match_counts": topic_match_counts,
        }
        component_scores = {f"component_{key}": float(metrics[key]) * float(weights.get(key, 0.0)) for key in metrics}
        total_score = float(sum(component_scores.values()))
        sample_titles, sample_dois = _sample_papers(papers, limit=sample_limit)

        output_rows.append(
            {
                "author_name": _clean_text(author_name),
                "istina_profile_url": _first_non_empty(group["istina_profile_url"]) if "istina_profile_url" in group.columns else None,
                "lab": _first_non_empty(group["lab"]) if "lab" in group.columns else None,
                "department": _first_non_empty(group["department"]) if "department" in group.columns else None,
                "institute": _first_non_empty(group["institute"]) if "institute" in group.columns else None,
                "position": _first_non_empty(group["position"]) if "position" in group.columns else None,
                "total_score": total_score,
                "supporting_publication_count": relevant_pub_count,
                "recent_publication_count": recent_pub_count,
                "sample_papers": sample_titles,
                "sample_dois": sample_dois,
                "topic_counts": json.dumps(
                    {
                        "aging_adjacent": topic_aging_count,
                        "bioengineering_regeneration": topic_bio_count,
                        "systems_omics": topic_systems_count,
                        "genetics_epigenetics": topic_genetics_count,
                    },
                    ensure_ascii=False,
                ),
                "topic_aging_count": topic_aging_count,
                "topic_bioengineering_regeneration_count": topic_bio_count,
                "topic_systems_omics_count": topic_systems_count,
                "topic_genetics_epigenetics_count": topic_genetics_count,
                **metrics,
                **component_scores,
            }
        )

    result = pd.DataFrame(output_rows)
    if not result.empty:
        result = result.sort_values(["total_score", "supporting_publication_count"], ascending=[False, False]).reset_index(drop=True)
    return result


def _weights_sheet(section_name: str, weights: dict[str, float], recent_years: list[int], config_path: str) -> pd.DataFrame:
    rows = [
        {"key": "section", "value": section_name},
        {"key": "weights_config_path", "value": config_path},
        {"key": "recent_years", "value": ", ".join(str(item) for item in recent_years)},
    ]
    for key, value in weights.items():
        rows.append({"key": key, "value": value})
    return pd.DataFrame(rows)


def export_lab_scores(
    input_master: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_scoring_config(config_path)
    scores = score_labs(input_master=input_master, config_path=config_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_df = _weights_sheet("lab", config["lab_weights"], config["recent_years"], config["path"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        scores.to_excel(writer, index=False, sheet_name="lab_scores")
        weights_df.to_excel(writer, index=False, sheet_name="weights")
    return {"output_path": str(out), "rows": int(len(scores))}


def export_author_scores(
    input_master: str | Path,
    output_path: str | Path,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    config = _load_scoring_config(config_path)
    scores = score_authors(input_master=input_master, config_path=config_path)
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    weights_df = _weights_sheet("author", config["author_weights"], config["recent_years"], config["path"])
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        scores.to_excel(writer, index=False, sheet_name="author_scores")
        weights_df.to_excel(writer, index=False, sheet_name="weights")
    return {"output_path": str(out), "rows": int(len(scores))}
