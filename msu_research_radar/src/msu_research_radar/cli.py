"""Command-line interface for msu_research_radar."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from msu_research_radar.istina.resolver import resolve_istina_person
from msu_research_radar.istina.search_client import search_istina_employees
from msu_research_radar.publications.openalex_client import collect_openalex_works


def build_parser() -> argparse.ArgumentParser:
    """Create and return the root argument parser."""
    parser = argparse.ArgumentParser(
        prog="msu_research_radar",
        description="CLI for MSU Research Radar.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show package version and exit.",
    )

    subparsers = parser.add_subparsers(dest="top_command")

    collect_parser = subparsers.add_parser("collect", help="Collect data from external sources")
    collect_sub = collect_parser.add_subparsers(dest="collect_command")
    openalex_collect = collect_sub.add_parser("openalex", help="Collect OpenAlex works for MSU ROR")
    openalex_collect.add_argument("--from-date", required=True, help="Start publication date, e.g. 2022-01-01")
    openalex_collect.add_argument("--to-date", required=True, help="End publication date, e.g. 2026-05-20")
    limit_group = openalex_collect.add_mutually_exclusive_group(required=True)
    limit_group.add_argument("--limit", type=int, help="Max number of works to collect")
    limit_group.add_argument(
        "--no-limit",
        action="store_true",
        help="Collect all works in the date range until OpenAlex cursor is exhausted.",
    )
    openalex_collect.add_argument(
        "--per-page",
        type=int,
        default=200,
        help="OpenAlex page size (max 200).",
    )

    istina_parser = subparsers.add_parser("istina", help="Istina utilities")
    istina_sub = istina_parser.add_subparsers(dest="istina_command")

    search_person = istina_sub.add_parser("search-person", help="Search Istina employees")
    search_person.add_argument("query", help="Employee search query")
    search_person.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Delay before network request (seconds).",
    )
    search_person.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh cached search response.",
    )

    resolve_person = istina_sub.add_parser(
        "resolve-person",
        help="Resolve a person by article title and optional coauthors",
    )
    resolve_person.add_argument("query", help="Employee search query")
    resolve_person.add_argument(
        "--article-title",
        required=True,
        help="Article title used to rank matching candidates.",
    )
    resolve_person.add_argument(
        "--coauthor",
        action="append",
        default=[],
        help="Coauthor name (repeatable).",
    )
    resolve_person.add_argument(
        "--delay-seconds",
        type=float,
        default=0.0,
        help="Delay before network requests (seconds).",
    )
    resolve_person.add_argument(
        "--refresh-search",
        action="store_true",
        help="Refresh cached search response.",
    )
    resolve_person.add_argument(
        "--refresh-profile",
        action="store_true",
        help="Refresh cached profile/publications responses.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from msu_research_radar import __version__

        print(__version__)
        return 0

    if args.top_command == "istina" and args.istina_command == "search-person":
        results = search_istina_employees(
            query=args.query,
            delay_seconds=args.delay_seconds,
            refresh=args.refresh,
        )
        print(json.dumps(results, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "istina" and args.istina_command == "resolve-person":
        resolved = resolve_istina_person(
            query=args.query,
            article_title=args.article_title,
            coauthors=args.coauthor,
            delay_seconds=args.delay_seconds,
            refresh_search=args.refresh_search,
            refresh_profile=args.refresh_profile,
        )
        print(json.dumps(resolved, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "collect" and args.collect_command == "openalex":
        requested_limit = None if args.no_limit else args.limit
        result = collect_openalex_works(
            from_date=args.from_date,
            to_date=args.to_date,
            limit=requested_limit,
            per_page=args.per_page,
        )
        summary = {
            "records_collected": result["records_collected"],
            "requested_limit": result["requested_limit"],
            "pages_collected": result["pages_collected"],
            "raw_run_dir": result["raw_run_dir"],
            "manifest_path": result["manifest_path"],
            "csv_path": result["csv_path"],
            "parquet_path": result["parquet_path"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
