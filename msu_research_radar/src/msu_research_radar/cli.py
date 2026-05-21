"""Command-line interface for msu_research_radar."""

from __future__ import annotations

import argparse
import json
from typing import Sequence

from msu_research_radar.authors.msu_author_extractor import extract_msu_authors
from msu_research_radar.export.master_builder import build_master_author_graph
from msu_research_radar.istina.batch_resolver import resolve_batch_istina_authors
from msu_research_radar.istina.resolver import resolve_istina_person
from msu_research_radar.istina.search_client import search_istina_employees_normalized
from msu_research_radar.publications.openalex_client import collect_openalex_works
from msu_research_radar.radar import export_shortlist, inspect_affiliations, inspect_topics
from msu_research_radar.scoring import export_author_scores, export_lab_scores


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
    openalex_collect.add_argument(
        "--chunk-by",
        choices=["year", "month", "none"],
        default="year",
        help="Date range chunking strategy.",
    )
    openalex_collect.add_argument(
        "--connect-timeout",
        type=float,
        default=10.0,
        help="HTTP connect timeout in seconds.",
    )
    openalex_collect.add_argument(
        "--read-timeout",
        type=float,
        default=60.0,
        help="HTTP read timeout in seconds.",
    )
    openalex_collect.add_argument(
        "--max-retries",
        type=int,
        default=5,
        help="Max retries for retryable HTTP/network errors.",
    )
    openalex_collect.add_argument(
        "--request-delay-seconds",
        type=float,
        default=0.3,
        help="Base delay between OpenAlex page requests.",
    )

    authors_parser = subparsers.add_parser("authors", help="Author-level utilities")
    authors_sub = authors_parser.add_subparsers(dest="authors_command")
    extract_msu = authors_sub.add_parser("extract-msu", help="Extract MSU-affiliated authors from OpenAlex works")
    extract_msu.add_argument("--input", required=True, help="Input OpenAlex parquet file")
    extract_msu.add_argument("--output", required=True, help="Output CSV path for aggregated MSU authors")

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
    resolve_batch = istina_sub.add_parser("resolve-batch", help="Resolve a batch of authors against Istina")
    resolve_batch.add_argument("--input", required=True, help="Input CSV of MSU authors")
    resolve_batch.add_argument("--output-dir", required=True, help="Output directory for batch resolution")
    resolve_batch.add_argument(
        "--acceptance-score",
        type=float,
        default=90.0,
        help="Minimum score to accept a match.",
    )
    resolve_batch.add_argument(
        "--min-gap",
        type=float,
        default=10.0,
        help="Minimum score gap between top and second candidate.",
    )

    export_parser = subparsers.add_parser("export", help="Export integrated datasets")
    export_sub = export_parser.add_subparsers(dest="export_command")
    build_master = export_sub.add_parser("build-master", help="Build master MSU author-paper graph dataset")
    build_master.add_argument("--works", required=True, help="Input OpenAlex works parquet")
    build_master.add_argument("--authors", required=True, help="Input MSU authors CSV")
    build_master.add_argument("--resolved", required=True, help="Input resolved Istina profiles JSONL")
    build_master.add_argument("--output", required=True, help="Output master parquet path")
    build_master.add_argument(
        "--profile-config",
        help="Optional research profile YAML for generic tag columns.",
    )
    build_master.add_argument(
        "--profile-name",
        help="Research profile name inside --profile-config.",
    )

    score_parser = subparsers.add_parser("score", help="Score labs and authors for human review")
    score_sub = score_parser.add_subparsers(dest="score_command")
    score_labs_cmd = score_sub.add_parser("labs", help="Score labs from master dataset")
    score_labs_cmd.add_argument("--input", required=True, help="Input master parquet path")
    score_labs_cmd.add_argument("--output", required=True, help="Output xlsx path")
    score_labs_cmd.add_argument(
        "--weights-config",
        default="config/scoring_weights.yaml",
        help="Path to scoring weights YAML config.",
    )
    score_authors_cmd = score_sub.add_parser("authors", help="Score authors from master dataset")
    score_authors_cmd.add_argument("--input", required=True, help="Input master parquet path")
    score_authors_cmd.add_argument("--output", required=True, help="Output xlsx path")
    score_authors_cmd.add_argument(
        "--weights-config",
        default="config/scoring_weights.yaml",
        help="Path to scoring weights YAML config.",
    )
    score_all_cmd = score_sub.add_parser("all", help="Score labs and authors in one run")
    score_all_cmd.add_argument("--input", required=True, help="Input master parquet path")
    score_all_cmd.add_argument(
        "--labs-output",
        default="exports/lab_scores.xlsx",
        help="Output xlsx path for lab scores.",
    )
    score_all_cmd.add_argument(
        "--authors-output",
        default="exports/author_scores.xlsx",
        help="Output xlsx path for author scores.",
    )
    score_all_cmd.add_argument(
        "--weights-config",
        default="config/scoring_weights.yaml",
        help="Path to scoring weights YAML config.",
    )

    radar_parser = subparsers.add_parser("radar", help="Configurable shortlist and inspection tools")
    radar_sub = radar_parser.add_subparsers(dest="radar_command")

    inspect_aff_cmd = radar_sub.add_parser(
        "inspect-affiliations",
        help="Export frequent affiliation-like strings for profile tuning.",
    )
    inspect_aff_cmd.add_argument("--input", required=True, help="Input works/master/authors parquet or CSV")
    inspect_aff_cmd.add_argument("--output", required=True, help="Output CSV/XLSX path")
    inspect_aff_cmd.add_argument("--limit", type=int, default=500, help="Maximum rows to export")

    inspect_topics_cmd = radar_sub.add_parser(
        "inspect-topics",
        help="Export frequent topic-like strings for profile tuning.",
    )
    inspect_topics_cmd.add_argument("--input", required=True, help="Input works/master parquet or CSV")
    inspect_topics_cmd.add_argument("--output", required=True, help="Output CSV/XLSX path")
    inspect_topics_cmd.add_argument("--limit", type=int, default=500, help="Maximum rows to export")

    shortlist_cmd = radar_sub.add_parser(
        "shortlist",
        help="Export configurable supervisor/contact shortlist workbook.",
    )
    shortlist_cmd.add_argument("--works", required=True, help="Input OpenAlex works parquet")
    shortlist_cmd.add_argument("--output", required=True, help="Output shortlist XLSX path")
    shortlist_cmd.add_argument(
        "--profile-config",
        default="config/research_profiles.yaml",
        help="Research profiles YAML path.",
    )
    shortlist_cmd.add_argument(
        "--profile-name",
        default=None,
        help="Research profile name. Defaults to the first profile in YAML.",
    )
    shortlist_cmd.add_argument(
        "--resolved",
        default=None,
        help="Optional resolved Istina profiles JSONL.",
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
        results = search_istina_employees_normalized(
            input_name=args.query,
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

    if args.top_command == "istina" and args.istina_command == "resolve-batch":
        summary = resolve_batch_istina_authors(
            input_csv=args.input,
            output_dir=args.output_dir,
            acceptance_score=args.acceptance_score,
            min_gap=args.min_gap,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "collect" and args.collect_command == "openalex":
        requested_limit = None if args.no_limit else args.limit
        result = collect_openalex_works(
            from_date=args.from_date,
            to_date=args.to_date,
            limit=requested_limit,
            per_page=args.per_page,
            chunk_by=args.chunk_by,
            connect_timeout=args.connect_timeout,
            read_timeout=args.read_timeout,
            max_retries=args.max_retries,
            request_delay_seconds=args.request_delay_seconds,
        )
        summary = {
            "records_collected": result["records_collected"],
            "requested_limit": result["requested_limit"],
            "pages_collected": result["pages_collected"],
            "chunks_total": result["chunks_total"],
            "chunks_ok": result["chunks_ok"],
            "chunks_failed": result["chunks_failed"],
            "raw_run_dir": result["raw_run_dir"],
            "manifest_path": result["manifest_path"],
            "csv_path": result["csv_path"],
            "parquet_path": result["parquet_path"],
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "authors" and args.authors_command == "extract-msu":
        summary = extract_msu_authors(input_path=args.input, output_path=args.output)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "export" and args.export_command == "build-master":
        summary = build_master_author_graph(
            works_path=args.works,
            authors_path=args.authors,
            resolved_path=args.resolved,
            output_path=args.output,
            profile_config=args.profile_config,
            profile_name=args.profile_name,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "score" and args.score_command == "labs":
        summary = export_lab_scores(
            input_master=args.input,
            output_path=args.output,
            config_path=args.weights_config,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "score" and args.score_command == "authors":
        summary = export_author_scores(
            input_master=args.input,
            output_path=args.output,
            config_path=args.weights_config,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "score" and args.score_command == "all":
        lab_summary = export_lab_scores(
            input_master=args.input,
            output_path=args.labs_output,
            config_path=args.weights_config,
        )
        author_summary = export_author_scores(
            input_master=args.input,
            output_path=args.authors_output,
            config_path=args.weights_config,
        )
        summary = {"labs": lab_summary, "authors": author_summary}
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "radar" and args.radar_command == "inspect-affiliations":
        summary = inspect_affiliations(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "radar" and args.radar_command == "inspect-topics":
        summary = inspect_topics(
            input_path=args.input,
            output_path=args.output,
            limit=args.limit,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    if args.top_command == "radar" and args.radar_command == "shortlist":
        summary = export_shortlist(
            works_path=args.works,
            output_path=args.output,
            profile_config=args.profile_config,
            profile_name=args.profile_name,
            resolved_path=args.resolved,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
