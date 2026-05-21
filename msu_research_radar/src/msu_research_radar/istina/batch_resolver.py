"""Batch resolver: match MSU authors to Istina profiles with strict acceptance."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from msu_research_radar.istina.profile_parser import parse_istina_profile_html
from msu_research_radar.istina.search_client import search_istina_employees_normalized

PROFILE_CACHE_DIR = Path("data/raw/istina/profiles")


def _clean(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _normalize(value: str) -> str:
    lowered = value.lower()
    return " ".join(re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE).split())


def _loads_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value if str(item).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item).strip()]
    return []


def _fetch_with_cache(url: str, cache_path: Path) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return cache_path.read_text(encoding="utf-8")
    response = requests.get(url, timeout=45)
    response.raise_for_status()
    cache_path.write_text(response.text, encoding="utf-8")
    return response.text


def _extract_publications(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    publications: list[dict[str, Any]] = []
    for activity in soup.select("ul.activity"):
        title_anchor = activity.select_one("li a[href*='/publications/article/']")
        if title_anchor is None:
            continue
        title = _clean(title_anchor.get_text(" ", strip=True))
        if not title:
            continue
        authors: list[str] = []
        seen: set[str] = set()
        for author_anchor in activity.select("li a[href*='/workers/']"):
            author_name = _clean(author_anchor.get_text(" ", strip=True))
            key = author_name.lower()
            if author_name and key not in seen:
                seen.add(key)
                authors.append(author_name)
        publications.append({"title": title, "authors": authors})
    return publications


def _best_title_match(sample_titles: list[str], candidate_titles: list[str]) -> tuple[float, str | None]:
    best_score = 0.0
    best_title = None
    normalized_samples = [_normalize(title) for title in sample_titles if _normalize(title)]
    normalized_candidates = [(title, _normalize(title)) for title in candidate_titles if _normalize(title)]
    for sample in normalized_samples:
        for title, normalized in normalized_candidates:
            score = float(fuzz.token_set_ratio(sample, normalized))
            if score > best_score:
                best_score = score
                best_title = title
    return best_score, best_title


def _coauthor_overlap(sample_coauthors: list[str], publication_authors: list[str]) -> list[str]:
    overlap: list[str] = []
    normalized_pub = [(author, _normalize(author)) for author in publication_authors if _normalize(author)]
    for coauthor in sample_coauthors:
        target = _normalize(coauthor)
        if not target:
            continue
        for author, norm_author in normalized_pub:
            if fuzz.token_sort_ratio(target, norm_author) >= 85:
                overlap.append(author)
                break
    return overlap


def _affiliation_overlap(known_affiliations: list[str], affiliation_hint: str, profile: dict[str, Any]) -> bool:
    corpus_parts = [affiliation_hint]
    corpus_parts.extend(profile.get("current_affiliations") or [])
    for field in ["lab", "department", "institute", "position"]:
        corpus_parts.append(profile.get(field) or "")
    corpus = _normalize(" ".join(corpus_parts))
    if not corpus:
        return False
    for affiliation in known_affiliations:
        term = _normalize(affiliation)
        if term and (term in corpus or fuzz.partial_ratio(term, corpus) >= 90):
            return True
    return False


def _score_candidate(
    candidate: dict[str, Any],
    sample_titles: list[str],
    sample_coauthors: list[str],
    known_affiliations: list[str],
) -> dict[str, Any]:
    worker_url = candidate.get("worker_url")
    if not worker_url:
        return {
            "candidate": candidate,
            "score": 0.0,
            "title_score": 0.0,
            "matched_title": None,
            "matched_coauthors": [],
            "affiliation_match": False,
            "parsed_profile": {},
        }

    worker_match = re.search(r"/workers/(\d+)/?", worker_url)
    worker_id = worker_match.group(1) if worker_match else "unknown"

    publications_url = worker_url.rstrip("/") + "/publications/"
    publications_cache = PROFILE_CACHE_DIR / f"worker_{worker_id}_publications.html"
    publications_html = _fetch_with_cache(publications_url, publications_cache)
    publications = _extract_publications(publications_html)
    publication_titles = [publication["title"] for publication in publications]

    title_score, matched_title = _best_title_match(sample_titles, publication_titles)
    matched_publication = next((pub for pub in publications if pub["title"] == matched_title), None)
    publication_authors = matched_publication["authors"] if matched_publication else []
    matched_coauthors = _coauthor_overlap(sample_coauthors, publication_authors)

    profile_url = candidate.get("profile_url") or worker_url
    profile_cache = PROFILE_CACHE_DIR / f"worker_{worker_id}_profile.html"
    profile_html = _fetch_with_cache(profile_url, profile_cache)
    parsed_profile = parse_istina_profile_html(profile_html, url=profile_url)

    affiliation_match = _affiliation_overlap(known_affiliations, candidate.get("affiliation_hint") or "", parsed_profile)
    coauthor_bonus = min(20.0, 5.0 * len(matched_coauthors))
    affiliation_bonus = 8.0 if affiliation_match else 0.0
    score = min(100.0, title_score + coauthor_bonus + affiliation_bonus)

    return {
        "candidate": candidate,
        "score": score,
        "title_score": title_score,
        "matched_title": matched_title,
        "matched_coauthors": matched_coauthors,
        "affiliation_match": affiliation_match,
        "parsed_profile": parsed_profile,
        "profile_url": profile_url,
    }


def resolve_batch_istina_authors(
    input_csv: str | Path,
    output_dir: str | Path,
    *,
    acceptance_score: float = 90.0,
    min_gap: float = 10.0,
) -> dict[str, Any]:
    """Resolve a batch of MSU authors to Istina profiles."""
    in_path = Path(input_csv)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    resolved_path = out_dir / "resolved_profiles.jsonl"
    not_resolved_path = out_dir / "not_resolved.csv"
    summary_path = out_dir / "resolution_summary.json"

    accepted_records: list[dict[str, Any]] = []
    rejected_records: list[dict[str, Any]] = []

    for _, row in df.iterrows():
        author_name = _clean(row.get("author_name"))
        sample_titles = _loads_json_list(row.get("sample_titles"))
        sample_coauthors = _loads_json_list(row.get("coauthors_msu"))
        known_affiliations = _loads_json_list(row.get("known_msu_affiliations"))

        search_payload = search_istina_employees_normalized(
            input_name=author_name,
            delay_seconds=0.0,
            refresh=False,
        )
        queries = search_payload["effective_queries"]
        if not queries:
            rejected_records.append(
                {
                    "author_name": author_name,
                    "input_name": author_name,
                    "effective_queries": "[]",
                    "reason": "no_search_queries",
                }
            )
            continue

        all_candidates = list(search_payload["candidates"])

        if not all_candidates:
            rejected_records.append(
                {
                    "author_name": author_name,
                    "input_name": author_name,
                    "effective_queries": json.dumps(queries, ensure_ascii=False),
                    "reason": "not_found",
                }
            )
            continue

        scored = [
            _score_candidate(
                candidate=candidate,
                sample_titles=sample_titles,
                sample_coauthors=sample_coauthors,
                known_affiliations=known_affiliations,
            )
            for candidate in all_candidates
        ]
        ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
        top = ranked[0]
        second_score = ranked[1]["score"] if len(ranked) > 1 else 0.0
        gap = top["score"] - second_score

        if top["score"] >= acceptance_score and gap >= min_gap:
            candidate = top["candidate"]
            accepted_records.append(
                {
                    "author_name": author_name,
                    "input_name": author_name,
                    "effective_queries": queries,
                    "matched_query": candidate.get("matched_query"),
                    "resolution_score": top["score"],
                    "score_gap": gap,
                    "matched_title": top["matched_title"],
                    "matched_coauthors": top["matched_coauthors"],
                    "candidate_name": candidate.get("name"),
                    "worker_url": candidate.get("worker_url"),
                    "profile_url": top["profile_url"],
                    "affiliation_hint": candidate.get("affiliation_hint"),
                    "parsed_profile": top["parsed_profile"],
                }
            )
        else:
            rejected_records.append(
                {
                    "author_name": author_name,
                    "input_name": author_name,
                    "effective_queries": json.dumps(queries, ensure_ascii=False),
                    "reason": "low_score_or_ambiguous",
                    "top_score": top["score"],
                    "score_gap": gap,
                    "top_candidate_name": (top["candidate"] or {}).get("name"),
                    "top_worker_url": (top["candidate"] or {}).get("worker_url"),
                }
            )

    with resolved_path.open("w", encoding="utf-8") as handle:
        for record in accepted_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    pd.DataFrame(rejected_records).to_csv(not_resolved_path, index=False)

    summary = {
        "input_authors": int(len(df)),
        "accepted": int(len(accepted_records)),
        "rejected": int(len(rejected_records)),
        "acceptance_score": acceptance_score,
        "min_gap": min_gap,
        "resolved_profiles_path": str(resolved_path),
        "not_resolved_path": str(not_resolved_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(summary_path)
    return summary
