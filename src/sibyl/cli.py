"""Minimal command-line boundary for Sibyl."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from pathlib import Path

from sibyl import __version__
from sibyl.experiments.convergence import (
    DEFAULT_JSON as CONVERGENCE_DEFAULT_JSON,
)
from sibyl.experiments.convergence import (
    DEFAULT_MARKDOWN as CONVERGENCE_DEFAULT_MARKDOWN,
)
from sibyl.experiments.convergence import (
    format_convergence_result,
    run_convergence,
)
from sibyl.experiments.transcription_reread import (
    DEFAULT_OUTPUT as REREAD_DEFAULT_OUTPUT,
)
from sibyl.experiments.transcription_reread import (
    DEFAULT_RUNS,
    format_reread_result,
    run_reread_experiment,
)
from sibyl.experiments.transcription_variance import (
    DEFAULT_OUTPUT,
    format_variance_result,
    run_variance_experiment,
)
from sibyl.experiments.trocr import format_result, run_experiment
from sibyl.experiments.trocr_compare import (
    DEFAULT_OUTPUT as TROCR_COMPARE_DEFAULT_OUTPUT,
)
from sibyl.experiments.trocr_compare import (
    format_compare_result,
    run_compare_experiment,
)
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
    variance = experiment_commands.add_parser(
        "transcription-variance", help="repeat page transcription for variance measurement"
    )
    variance.add_argument("image", type=Path, help="page image")
    variance.add_argument("--runs", type=int, help="number of transcription runs")
    variance.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "experimental JSON output path "
            "(default: .sibyl/experiments/transcription-variance.json)"
        ),
    )
    reread = experiment_commands.add_parser(
        "transcription-reread", help="measure repeated Qwen reads of localized text regions"
    )
    reread.add_argument("image", type=Path, help="page image")
    reread.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"independent regional reads per crop (default: {DEFAULT_RUNS})",
    )
    compare = experiment_commands.add_parser(
        "trocr-compare", help="compare Qwen and TrOCR on identical source-resolution crops"
    )
    compare.add_argument("image", type=Path, help="page image")
    compare.add_argument("--runs", type=int, default=DEFAULT_RUNS, help="reads per recognizer")
    compare.add_argument(
        "--regions", help="comma-separated region IDs (default: all accepted coarse regions)"
    )
    compare.add_argument("--output", type=Path, default=None, help="experimental JSON output path")
    converge = experiment_commands.add_parser(
        "converge", help="synthesize preserved Qwen/TrOCR evidence into a Markdown candidate"
    )
    converge.add_argument("input", type=Path, help="trocr-compare JSON artifact")
    converge.add_argument("--review", type=Path, help="optional explicit human review YAML")
    converge.add_argument(
        "--output", type=Path, default=CONVERGENCE_DEFAULT_MARKDOWN, help="candidate Markdown path"
    )
    converge.add_argument(
        "--json-output", type=Path, default=CONVERGENCE_DEFAULT_JSON, help="provenance JSON path"
    )
    reread.add_argument(
        "--output",
        type=Path,
        default=None,
        help=(
            "experimental JSON output path (default: .sibyl/experiments/transcription-reread.json)"
        ),
    )
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
    if arguments.command == "experiment" and arguments.experiment_name == "transcription-variance":
        try:
            variance_result = run_variance_experiment(
                arguments.image,
                runs=arguments.runs,
                output_path=arguments.output or DEFAULT_OUTPUT,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_variance_result(variance_result))
    if arguments.command == "experiment" and arguments.experiment_name == "transcription-reread":
        try:
            reread_result = run_reread_experiment(
                arguments.image,
                runs=arguments.runs,
                output_path=arguments.output or REREAD_DEFAULT_OUTPUT,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_reread_result(reread_result))
    if arguments.command == "experiment" and arguments.experiment_name == "trocr-compare":
        try:
            compare_result = run_compare_experiment(
                arguments.image,
                runs=arguments.runs,
                regions=arguments.regions,
                output_path=arguments.output or TROCR_COMPARE_DEFAULT_OUTPUT,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_compare_result(compare_result))
    if arguments.command == "experiment" and arguments.experiment_name == "converge":
        try:
            convergence_result = run_convergence(
                arguments.input,
                review_path=arguments.review,
                markdown_path=arguments.output,
                json_path=arguments.json_output,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_convergence_result(convergence_result))
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
