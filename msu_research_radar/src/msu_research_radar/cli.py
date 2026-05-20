"""Интерфейс командной строки для msu_research_radar."""

from __future__ import annotations

import argparse
from typing import Sequence


def build_parser() -> argparse.ArgumentParser:
    """Создает и возвращает корневой парсер аргументов."""
    parser = argparse.ArgumentParser(
        prog="msu_research_radar",
        description="CLI каркаса проекта MSU Research Radar.",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Показать версию пакета и завершить работу.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Запускает CLI и возвращает код завершения."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        from msu_research_radar import __version__

        print(__version__)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
