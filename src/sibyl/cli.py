"""Minimal command-line boundary for Sibyl."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

from sibyl import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sibyl",
        description="Faithful recovery of handwritten material into structured artifacts.",
    )
    parser.add_argument("--version", action="store_true", help="show the executable version")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.version:
        print(__version__)
    return 0
