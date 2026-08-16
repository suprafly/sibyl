"""Minimal command-line boundary for Sibyl."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sibyl import __version__
from sibyl.experiments.trocr import format_result, run_experiment
from sibyl.recovery import format_recovery, recover_page


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sibyl",
        description="Faithful recovery of handwritten material into structured artifacts.",
    )
    parser.add_argument("--version", action="store_true", help="show the executable version")
    commands = parser.add_subparsers(dest="command")
    experiment_parser = commands.add_parser("experiment", help="run an empirical experiment")
    experiment_commands = experiment_parser.add_subparsers(dest="experiment_name", required=True)
    trocr = experiment_commands.add_parser(
        "trocr", help="recognize one handwritten line with TrOCR Large"
    )
    trocr.add_argument("image", type=Path, help="image containing one handwritten line or crop")
    trocr.add_argument("--json", action="store_true", help="emit the result as JSON")
    recover_parser = commands.add_parser("recover", help="recover one handwritten page")
    recover_parser.add_argument("image", type=Path, help="page image")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    if arguments.version:
        print(__version__)
        return 0
    if arguments.command == "experiment" and arguments.experiment_name == "trocr":
        try:
            result = run_experiment(arguments.image)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_result(result, arguments.json))
    if arguments.command == "recover":
        try:
            print(format_recovery(recover_page(arguments.image)))
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
    return 0
