"""OpenAlex works collection for MSU-affiliated publications."""

from __future__ import annotations

import json
import random
import time
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import requests

OPENALEX_WORKS_URL = "https://api.openalex.org/works"
MSU_ROR = "https://ror.org/010pmpe69"
RETRYABLE_STATUS_CODES = {429, 500, 502, 503, 504}

RAW_OPENALEX_DIR = Path("data/raw/openalex")
INTERIM_DIR = Path("data/interim")


def _build_http_session() -> requests.Session:
    session = requests.Session()
    session.headers.update({"User-Agent": "msu_research_radar/0.1 (openalex collector)"})
    return session


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
    primary_topic = work.get("primary_topic") or {}
    subfield = primary_topic.get("subfield") or {}
    field = primary_topic.get("field") or {}
    domain = primary_topic.get("domain") or {}
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
        "topic": primary_topic.get("display_name"),
        "subfield": subfield.get("display_name"),
        "field": field.get("display_name"),
        "domain": domain.get("display_name"),
        "authorships": authorships,
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
                "topic": record.get("topic"),
                "subfield": record.get("subfield"),
                "field": record.get("field"),
                "domain": record.get("domain"),
                "authorships": json.dumps(record.get("authorships") or [], ensure_ascii=False),
            }
        )
    return pd.DataFrame(serialized)


def _parse_iso_date(value: str) -> date:
    return date.fromisoformat(value)


def _date_chunks(from_date: str, to_date: str, chunk_by: str) -> list[tuple[str, str]]:
    start = _parse_iso_date(from_date)
    end = _parse_iso_date(to_date)
    if start > end:
        raise ValueError("from_date must be <= to_date")
    if chunk_by == "none":
        return [(from_date, to_date)]

    chunks: list[tuple[str, str]] = []
    if chunk_by == "year":
        year = start.year
        while year <= end.year:
            chunk_start = date(year, 1, 1) if year > start.year else start
            chunk_end = date(year, 12, 31) if year < end.year else end
            chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
            year += 1
        return chunks

    if chunk_by == "month":
        current = date(start.year, start.month, 1)
        while current <= end:
            chunk_start = current if current > start else start
            next_month = date(current.year + (1 if current.month == 12 else 0), 1 if current.month == 12 else current.month + 1, 1)
            month_last = date.fromordinal(next_month.toordinal() - 1)
            chunk_end = month_last if month_last < end else end
            chunks.append((chunk_start.isoformat(), chunk_end.isoformat()))
            current = next_month
        return chunks

    raise ValueError("chunk_by must be one of: year, month, none")


def _sleep_with_jitter(base_seconds: float) -> None:
    if base_seconds <= 0:
        return
    delay = base_seconds + random.uniform(0.0, min(0.35, base_seconds))
    time.sleep(delay)


def _retry_after_seconds(response: requests.Response) -> float | None:
    header_value = response.headers.get("Retry-After")
    if not header_value:
        return None
    try:
        parsed = float(header_value)
    except ValueError:
        return None
    if parsed < 0:
        return None
    return parsed


def _request_with_retries(
    *,
    session: requests.Session,
    params: dict[str, Any],
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    retry_budget: dict[str, int],
) -> requests.Response:
    attempt = 0
    while True:
        attempt += 1
        try:
            response = session.get(
                OPENALEX_WORKS_URL,
                params=params,
                timeout=(connect_timeout, read_timeout),
            )
        except requests.RequestException:
            retry_budget["used"] += 1
            exhausted = attempt > (max_retries + 1) or retry_budget["used"] > retry_budget["max"]
            if exhausted:
                raise
            backoff = 0.7 * (2 ** (attempt - 1))
            _sleep_with_jitter(backoff)
            continue

        if response.status_code in RETRYABLE_STATUS_CODES:
            retry_budget["used"] += 1
            exhausted = attempt > (max_retries + 1) or retry_budget["used"] > retry_budget["max"]
            if exhausted:
                response.raise_for_status()
            retry_after = _retry_after_seconds(response)
            if retry_after is not None:
                _sleep_with_jitter(retry_after)
            else:
                backoff = 0.7 * (2 ** (attempt - 1))
                _sleep_with_jitter(backoff)
            continue

        response.raise_for_status()
        return response


def _collect_single_chunk(
    *,
    session: requests.Session,
    chunk_from: str,
    chunk_to: str,
    per_page: int,
    remaining_limit: int | None,
    run_raw_dir: Path,
    chunk_index: int,
    connect_timeout: float,
    read_timeout: float,
    max_retries: int,
    request_delay_seconds: float,
) -> dict[str, Any]:
    started_at = time.perf_counter()
    filter_value = (
        f"institutions.ror:{MSU_ROR},"
        f"from_publication_date:{chunk_from},"
        f"to_publication_date:{chunk_to}"
    )
    params = {
        "filter": filter_value,
        "per-page": min(per_page, 200),
        "cursor": "*",
        "sort": "publication_date:desc",
        "select": (
            "id,doi,display_name,abstract_inverted_index,publication_date,publication_year,"
            "cited_by_count,authorships,primary_location,topics,primary_topic"
        ),
    }

    results: list[dict[str, Any]] = []
    cursors: list[str] = []
    page_count = 0
    retry_budget = {"used": 0, "max": max(4, max_retries * 20)}

    while True:
        if remaining_limit is not None and len(results) >= remaining_limit:
            break
        response = _request_with_retries(
            session=session,
            params=params,
            connect_timeout=connect_timeout,
            read_timeout=read_timeout,
            max_retries=max_retries,
            retry_budget=retry_budget,
        )
        payload = response.json()
        page_count += 1
        page_path = run_raw_dir / f"chunk_{chunk_index:03d}_page_{page_count:04d}.json"
        page_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        page_results = payload.get("results") or []
        if not page_results:
            break
        results.extend(page_results)

        next_cursor = (payload.get("meta") or {}).get("next_cursor")
        if not next_cursor:
            break
        params["cursor"] = next_cursor
        cursors.append(next_cursor)
        _sleep_with_jitter(request_delay_seconds)

    duration_seconds = round(time.perf_counter() - started_at, 3)
    return {
        "chunk_index": chunk_index,
        "from_date": chunk_from,
        "to_date": chunk_to,
        "status": "ok",
        "pages_collected": page_count,
        "records_collected": len(results),
        "retry_events_used": retry_budget["used"],
        "next_cursors_seen": cursors,
        "duration_seconds": duration_seconds,
        "results": results if remaining_limit is None else results[:remaining_limit],
    }


def collect_openalex_works(
    from_date: str,
    to_date: str,
    limit: int | None,
    per_page: int = 200,
    *,
    raw_base_dir: Path = RAW_OPENALEX_DIR,
    interim_dir: Path = INTERIM_DIR,
    chunk_by: str = "year",
    connect_timeout: float = 10.0,
    read_timeout: float = 60.0,
    max_retries: int = 5,
    request_delay_seconds: float = 0.3,
) -> dict[str, Any]:
    """Collect OpenAlex works for MSU ROR in a date range."""
    if limit is not None and limit <= 0:
        raise ValueError("limit must be > 0")
    if connect_timeout <= 0 or read_timeout <= 0:
        raise ValueError("connect_timeout and read_timeout must be > 0")
    if max_retries < 0:
        raise ValueError("max_retries must be >= 0")

    run_id = datetime.now(timezone.utc).strftime("openalex_%Y%m%dT%H%M%SZ")
    run_raw_dir = raw_base_dir / run_id
    run_raw_dir.mkdir(parents=True, exist_ok=True)
    interim_dir.mkdir(parents=True, exist_ok=True)

    all_works: list[dict[str, Any]] = []
    session = _build_http_session()
    chunks = _date_chunks(from_date, to_date, chunk_by)
    chunk_summaries: list[dict[str, Any]] = []
    total_pages = 0
    failed_chunk = None

    for idx, (chunk_from, chunk_to) in enumerate(chunks, start=1):
        remaining_limit = None if limit is None else max(0, limit - len(all_works))
        if remaining_limit == 0:
            chunk_summaries.append(
                {
                    "chunk_index": idx,
                    "from_date": chunk_from,
                    "to_date": chunk_to,
                    "status": "skipped_due_to_limit",
                    "pages_collected": 0,
                    "records_collected": 0,
                    "retry_events_used": 0,
                    "next_cursors_seen": [],
                    "duration_seconds": 0.0,
                }
            )
            continue

        try:
            chunk = _collect_single_chunk(
                session=session,
                chunk_from=chunk_from,
                chunk_to=chunk_to,
                per_page=per_page,
                remaining_limit=remaining_limit,
                run_raw_dir=run_raw_dir,
                chunk_index=idx,
                connect_timeout=connect_timeout,
                read_timeout=read_timeout,
                max_retries=max_retries,
                request_delay_seconds=request_delay_seconds,
            )
        except Exception as exc:
            failed_chunk = {
                "chunk_index": idx,
                "from_date": chunk_from,
                "to_date": chunk_to,
                "status": "failed",
                "pages_collected": 0,
                "records_collected": 0,
                "retry_events_used": 0,
                "next_cursors_seen": [],
                "duration_seconds": 0.0,
                "error": str(exc),
            }
            chunk_summaries.append(failed_chunk)
            break

        chunk_results = chunk.pop("results")
        total_pages += chunk["pages_collected"]
        all_works.extend(chunk_results)
        chunk_summaries.append(chunk)
        if limit is not None and len(all_works) >= limit:
            all_works = all_works[:limit]
            break

    deduped_works: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for work in all_works:
        work_id = work.get("id")
        key = str(work_id) if work_id is not None else ""
        if key and key in seen_ids:
            continue
        if key:
            seen_ids.add(key)
        deduped_works.append(work)

    collected_raw = deduped_works if limit is None else deduped_works[:limit]
    normalized_records = [_normalize_work(work) for work in collected_raw]
    table = _serialize_for_table(normalized_records)

    parquet_path = interim_dir / "openalex_works.parquet"
    csv_path = interim_dir / "openalex_works.csv"
    temp_parquet = interim_dir / "openalex_works.parquet.tmp"
    temp_csv = interim_dir / "openalex_works.csv.tmp"

    manifest = {
        "run_id": run_id,
        "collected_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "OpenAlex",
        "endpoint": OPENALEX_WORKS_URL,
        "filter_base": f"institutions.ror:{MSU_ROR}",
        "from_date": from_date,
        "to_date": to_date,
        "limit": limit,
        "collection_mode": "no_limit" if limit is None else "limited",
        "chunk_by": chunk_by,
        "per_page": min(per_page, 200),
        "sort": "publication_date:desc",
        "connect_timeout": connect_timeout,
        "read_timeout": read_timeout,
        "max_retries": max_retries,
        "request_delay_seconds": request_delay_seconds,
        "pages_collected": total_pages,
        "records_collected": len(collected_raw),
        "chunks_total": len(chunks),
        "chunks_ok": sum(1 for item in chunk_summaries if item.get("status") == "ok"),
        "chunks_failed": sum(1 for item in chunk_summaries if item.get("status") == "failed"),
        "chunk_summaries": chunk_summaries,
        "output_csv": str(csv_path),
        "output_parquet": str(parquet_path),
    }
    manifest_path = run_raw_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    if failed_chunk is not None:
        raise RuntimeError(
            f"OpenAlex chunk failed: index={failed_chunk['chunk_index']}, "
            f"range={failed_chunk['from_date']}..{failed_chunk['to_date']}, "
            f"error={failed_chunk.get('error')}"
        )

    table.to_parquet(temp_parquet, index=False, engine="pyarrow")
    table.to_csv(temp_csv, index=False)
    temp_parquet.replace(parquet_path)
    temp_csv.replace(csv_path)

    return {
        "run_id": run_id,
        "records_collected": len(collected_raw),
        "requested_limit": limit,
        "pages_collected": total_pages,
        "chunks_total": len(chunks),
        "chunks_ok": sum(1 for item in chunk_summaries if item.get("status") == "ok"),
        "chunks_failed": 0,
        "chunk_summaries": chunk_summaries,
        "raw_run_dir": str(run_raw_dir),
        "manifest_path": str(manifest_path),
        "csv_path": str(csv_path),
        "parquet_path": str(parquet_path),
        "records": normalized_records,
    }
