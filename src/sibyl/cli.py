"""Minimal command-line boundary for Sibyl."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sibyl import __version__
from sibyl.experiments.trocr import format_result, run_experiment
from sibyl.transform import (
    format_text_transform,
    format_transform,
    transform_page,
    write_markdown_transform,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sibyl",
        description="Faithful transform of handwritten material into structured artifacts.",
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
    run_parser = commands.add_parser("run", help="transform one handwritten page")
    run_parser.add_argument("image", type=Path, help="page image")
    output = run_parser.add_mutually_exclusive_group()
    output.add_argument("--markdown", action="store_true", help="write a Markdown projection")
    output.add_argument("--json", action="store_true", help="emit the structured transform JSON")
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
    if arguments.command == "run":
        try:
            page = transform_page(arguments.image)
            if arguments.json:
                print(format_transform(page))
            elif arguments.markdown:
                output_path = write_markdown_transform(page)
                print(f"wrote transform projections: {output_path.parent}")
            else:
                print(format_text_transform(page))
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
    return 0
