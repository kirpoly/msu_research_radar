"""Command-line interface for msu_research_radar."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from msu_research_radar.istina.resolver import resolve_istina_person
from msu_research_radar.istina.search_client import search_istina_employees


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

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

