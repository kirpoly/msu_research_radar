"""Istina employee search client for interactive use."""

from __future__ import annotations

import hashlib
import re
import time
from pathlib import Path
from typing import Any
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

from msu_research_radar.istina.name_queries import generate_istina_name_queries

ISTINA_BASE_URL = "https://istina.msu.ru"
WORKER_SEARCH_URL = f"{ISTINA_BASE_URL}/workers/worker_search/"
SEARCH_CACHE_DIR = Path("data/raw/istina/search_cache")


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(value.split())


def _query_key(query: str) -> str:
    digest = hashlib.sha1(query.encode("utf-8")).hexdigest()[:12]
    normalized = re.sub(r"[^A-Za-z0-9А-Яа-яЁё]+", "_", query).strip("_")
    safe = normalized[:48] if normalized else "query"
    return f"{safe}_{digest}"


def _cache_path_for_query(query: str) -> Path:
    return SEARCH_CACHE_DIR / f"{_query_key(query)}.html"


def _print_search_diagnostics(
    *,
    query: str,
    page_url: str,
    status_code: int,
    cache_path: Path,
    card_count: int,
    worker_link_count: int,
) -> None:
    print(
        "[istina-search] diagnostics: "
        f"query={query!r}, url={page_url}, status={status_code}, "
        f"cache={cache_path}, cards={card_count}, worker_links={worker_link_count}"
    )


def _parse_worker_cards(html: str) -> list[dict[str, Any]]:
    soup = BeautifulSoup(html, "html.parser")
    cards = soup.select("div.worker_short")
    candidates: list[dict[str, Any]] = []

    for card in cards:
        worker_anchor = card.select_one("h3 a[href*='/workers/']")
        if worker_anchor is None:
            continue

        name = _clean_text(worker_anchor.get_text(" ", strip=True))
        worker_url = urljoin(ISTINA_BASE_URL, worker_anchor.get("href", ""))

        profile_anchor = card.select_one("a[href*='/profile/']")
        profile_href = profile_anchor.get("href", "") if profile_anchor else ""
        profile_url = urljoin(ISTINA_BASE_URL, profile_href) if profile_href else None

        info_blocks = card.select("div.span-22.last > div")
        affiliation_hint = _clean_text(info_blocks[0].get_text(" ", strip=True)) if info_blocks else ""
        snippet = _clean_text(card.get_text(" ", strip=True))

        candidates.append(
            {
                "name": name or None,
                "profile_url": profile_url,
                "worker_url": worker_url or None,
                "snippet": snippet or None,
                "affiliation_hint": affiliation_hint or None,
            }
        )

    return candidates


def search_istina_employees(
    query: str,
    delay_seconds: float = 0.0,
    refresh: bool = False,
) -> list[dict[str, Any]]:
    """Search Istina employees by free-text query."""
    SEARCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = _cache_path_for_query(query)

    page_url = f"{WORKER_SEARCH_URL}?s={query}"
    status_code = 200

    if cache_path.exists() and not refresh:
        html = cache_path.read_text(encoding="utf-8")
    else:
        if delay_seconds > 0:
            time.sleep(delay_seconds)
        response = requests.get(WORKER_SEARCH_URL, params={"s": query}, timeout=30)
        response.raise_for_status()
        html = response.text
        page_url = response.url
        status_code = response.status_code
        cache_path.write_text(html, encoding="utf-8")

    candidates = _parse_worker_cards(html)
    if not candidates:
        soup = BeautifulSoup(html, "html.parser")
        worker_links = soup.select("a[href*='/workers/']")
        _print_search_diagnostics(
            query=query,
            page_url=page_url,
            status_code=status_code,
            cache_path=cache_path,
            card_count=0,
            worker_link_count=len(worker_links),
        )

    return candidates


def build_effective_istina_queries(input_name: str) -> list[str]:
    """Normalize an input name to Istina-compatible surname+initial queries."""
    variants = generate_istina_name_queries(input_name)
    if not variants:
        return []

    effective: list[str] = []
    seen: set[str] = set()
    for query in variants:
        tokens = re.findall(r"[A-Za-zА-Яа-яЁё]+", query)
        has_initial = any(len(token) == 1 for token in tokens)
        is_single_token = len(tokens) == 1
        if not has_initial and not is_single_token:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        effective.append(query)
    return effective


def search_istina_employees_normalized(
    input_name: str,
    *,
    delay_seconds: float = 0.0,
    refresh: bool = False,
) -> dict[str, Any]:
    """Search Istina with normalized surname+initial queries only."""
    effective_queries = build_effective_istina_queries(input_name)
    all_candidates: list[dict[str, Any]] = []
    seen_keys: set[str] = set()

    for query in effective_queries:
        for candidate in search_istina_employees(
            query=query,
            delay_seconds=delay_seconds,
            refresh=refresh,
        ):
            key = candidate.get("worker_url") or candidate.get("profile_url") or candidate.get("name") or ""
            if not key or key in seen_keys:
                continue
            seen_keys.add(key)
            enriched = dict(candidate)
            enriched["matched_query"] = query
            all_candidates.append(enriched)

    return {
        "input_name": input_name,
        "effective_queries": effective_queries,
        "candidates": all_candidates,
    }
