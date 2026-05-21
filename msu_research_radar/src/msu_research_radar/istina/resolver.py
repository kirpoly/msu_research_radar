"""Resolve Istina employee by search query and publication context."""

from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
from rapidfuzz import fuzz

from msu_research_radar.istina.profile_parser import parse_istina_profile_html
from msu_research_radar.istina.search_client import (
    search_istina_employees_normalized,
)

PROFILE_CACHE_DIR = Path("data/raw/istina/profiles")
INTERIM_PROFILE_DIR = Path("data/interim/istina_profiles")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _normalize_title(value: str) -> str:
    lowered = value.lower()
    letters = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return " ".join(letters.split())


def _normalize_name(value: str) -> str:
    lowered = value.lower()
    letters = re.sub(r"[^\w\s]", " ", lowered, flags=re.UNICODE)
    return " ".join(letters.split())


def _worker_id_from_url(url: str | None) -> str | None:
    if not url:
        return None
    match = re.search(r"/workers/(\d+)/?", url)
    return match.group(1) if match else None


def _safe_key(value: str) -> str:
    digest = hashlib.sha1(value.encode("utf-8")).hexdigest()[:10]
    clean = re.sub(r"[^A-Za-z0-9А-Яа-яЁё]+", "_", value).strip("_")
    stem = clean[:40] if clean else "value"
    return f"{stem}_{digest}"


def _fetch_with_cache(
    *,
    url: str,
    cache_path: Path,
    delay_seconds: float,
    refresh: bool,
) -> str:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists() and not refresh:
        return cache_path.read_text(encoding="utf-8")
    if delay_seconds > 0:
        time.sleep(delay_seconds)
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

        title = _clean_text(title_anchor.get_text(" ", strip=True))
        if not title:
            continue

        authors: list[str] = []
        seen: set[str] = set()
        for author_anchor in activity.select("li a[href*='/workers/']"):
            author_name = _clean_text(author_anchor.get_text(" ", strip=True))
            if not author_name:
                continue
            key = author_name.lower()
            if key in seen:
                continue
            seen.add(key)
            authors.append(author_name)

        publications.append({"title": title, "authors": authors})
    return publications


def _coauthor_overlap(input_names: list[str], publication_authors: list[str]) -> list[str]:
    if not input_names or not publication_authors:
        return []
    normalized_authors = [(author, _normalize_name(author)) for author in publication_authors]
    overlap: list[str] = []

    for name in input_names:
        target = _normalize_name(name)
        if not target:
            continue
        for author, norm_author in normalized_authors:
            similarity = fuzz.token_sort_ratio(target, norm_author)
            if similarity >= 85:
                overlap.append(author)
                break
    return overlap


def _score_candidate(
    candidate: dict[str, Any],
    article_title: str,
    input_coauthors: list[str],
    delay_seconds: float,
    refresh_profile: bool,
) -> dict[str, Any]:
    worker_url = candidate.get("worker_url")
    worker_id = _worker_id_from_url(worker_url)
    if worker_id is None or worker_url is None:
        return {
            "candidate": candidate,
            "score": 0,
            "title_exact_norm": False,
            "title_fuzzy": 0.0,
            "coauthor_overlap": [],
            "matched_publication_title": None,
            "match_reason": "missing worker URL",
        }

    publications_url = worker_url.rstrip("/") + "/publications/"
    pub_cache = PROFILE_CACHE_DIR / f"worker_{worker_id}_publications.html"
    publications_html = _fetch_with_cache(
        url=publications_url,
        cache_path=pub_cache,
        delay_seconds=delay_seconds,
        refresh=refresh_profile,
    )
    publications = _extract_publications(publications_html)

    target = _normalize_title(article_title)
    best_title = None
    best_fuzzy = 0.0
    best_exact = False
    best_authors: list[str] = []

    for publication in publications:
        pub_title = publication["title"]
        pub_norm = _normalize_title(pub_title)
        exact = pub_norm == target and bool(target)
        fuzzy_score = float(fuzz.token_set_ratio(target, pub_norm)) if target else 0.0

        better = exact and not best_exact
        better = better or (exact == best_exact and fuzzy_score > best_fuzzy)
        if better:
            best_exact = exact
            best_fuzzy = fuzzy_score
            best_title = pub_title
            best_authors = publication["authors"]

    overlap = _coauthor_overlap(input_coauthors, best_authors)
    score = 100.0 if best_exact else best_fuzzy
    if not best_exact and overlap:
        score = min(100.0, score + min(12.0, 4.0 * len(overlap)))

    reason_bits = []
    reason_bits.append("exact title match" if best_exact else f"fuzzy={best_fuzzy:.1f}")
    if overlap:
        reason_bits.append(f"coauthor overlap={len(overlap)}")

    return {
        "candidate": candidate,
        "score": score,
        "title_exact_norm": best_exact,
        "title_fuzzy": best_fuzzy,
        "coauthor_overlap": overlap,
        "matched_publication_title": best_title,
        "match_reason": "; ".join(reason_bits),
    }


def _select_candidate(scored: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    exact = [item for item in scored if item["title_exact_norm"]]
    if len(exact) == 1:
        return exact[0], "single exact title match"

    ranked = sorted(scored, key=lambda item: item["score"], reverse=True)
    if not ranked:
        raise RuntimeError("No candidates available for selection.")

    if len(ranked) == 1:
        return ranked[0], "single candidate"

    top = ranked[0]
    second = ranked[1]
    if top["score"] >= 90 and (top["score"] - second["score"]) >= 15:
        return top, "score threshold auto-select"

    print("Multiple candidates found. Choose a candidate:")
    for index, item in enumerate(ranked, start=1):
        cand = item["candidate"]
        name = cand.get("name") or "<unknown>"
        aff = cand.get("affiliation_hint") or ""
        print(f"[{index}] {name} — {aff}")
        print(f"    score={item['score']:.1f}; {item['match_reason']}")

    while True:
        raw = input("Choose candidate number: ").strip()
        if raw.isdigit():
            selected_index = int(raw)
            if 1 <= selected_index <= len(ranked):
                return ranked[selected_index - 1], "interactive selection"
        print("Invalid choice. Please enter a valid number.")


def resolve_istina_person(
    *,
    query: str,
    article_title: str,
    coauthors: list[str] | None = None,
    delay_seconds: float = 0.0,
    refresh_search: bool = False,
    refresh_profile: bool = False,
) -> dict[str, Any]:
    """Resolve an Istina person using search + publication matching."""
    input_coauthors = coauthors or []
    search_payload = search_istina_employees_normalized(
        input_name=query,
        delay_seconds=delay_seconds,
        refresh=refresh_search,
    )
    effective_queries = search_payload["effective_queries"]
    candidates = search_payload["candidates"]
    if not candidates:
        raise RuntimeError(
            f"No Istina candidates found for input_name={query!r} using effective_queries={effective_queries!r}"
        )

    scored = [
        _score_candidate(
            candidate=candidate,
            article_title=article_title,
            input_coauthors=input_coauthors,
            delay_seconds=delay_seconds,
            refresh_profile=refresh_profile,
        )
        for candidate in candidates
    ]

    chosen_scored, selection_mode = _select_candidate(scored)
    chosen = chosen_scored["candidate"]

    profile_url = chosen.get("profile_url")
    worker_url = chosen.get("worker_url")
    fetch_url = profile_url or worker_url
    if not fetch_url:
        raise RuntimeError("Chosen candidate has no profile/worker URL.")

    worker_id = _worker_id_from_url(worker_url) or "unknown"
    profile_cache = PROFILE_CACHE_DIR / f"worker_{worker_id}_profile.html"
    profile_html = _fetch_with_cache(
        url=fetch_url,
        cache_path=profile_cache,
        delay_seconds=delay_seconds,
        refresh=refresh_profile,
    )
    parsed_profile = parse_istina_profile_html(profile_html, url=profile_url or fetch_url)

    result = {
        "input_name": query,
        "query": query,
        "effective_queries": effective_queries,
        "article_title": article_title,
        "input_coauthors": input_coauthors,
        "selection_mode": selection_mode,
        "chosen_candidate": chosen,
        "chosen_match": {
            "score": chosen_scored["score"],
            "title_exact_norm": chosen_scored["title_exact_norm"],
            "title_fuzzy": chosen_scored["title_fuzzy"],
            "coauthor_overlap": chosen_scored["coauthor_overlap"],
            "matched_publication_title": chosen_scored["matched_publication_title"],
            "match_reason": chosen_scored["match_reason"],
        },
        "candidates": [
            {
                "candidate": item["candidate"],
                "score": item["score"],
                "title_exact_norm": item["title_exact_norm"],
                "title_fuzzy": item["title_fuzzy"],
                "coauthor_overlap": item["coauthor_overlap"],
                "matched_publication_title": item["matched_publication_title"],
                "match_reason": item["match_reason"],
            }
            for item in sorted(scored, key=lambda value: value["score"], reverse=True)
        ],
        "parsed_profile": parsed_profile,
        "profile_cache_path": str(profile_cache),
    }

    INTERIM_PROFILE_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output_name = f"{_safe_key(query)}_{worker_id}_{ts}.json"
    output_path = INTERIM_PROFILE_DIR / output_name
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    result["output_json_path"] = str(output_path)

    return result
