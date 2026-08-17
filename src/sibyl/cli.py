"""Minimal command-line boundary for Sibyl."""

from __future__ import annotations

import argparse
import json
import zipfile
from collections.abc import Sequence
from pathlib import Path

from sibyl import __version__
from sibyl.corpus import format_corpus_result, prepare_boox_corpus
from sibyl.experiments.boox_recognition import (
    BOOX_RECOGNITION_NUM_CTX,
    format_boox_recognition,
    run_boox_recognition,
)
from sibyl.experiments.boox_recognition import (
    DEFAULT_IMAGE as BOOX_RECOGNITION_DEFAULT_IMAGE,
)
from sibyl.experiments.boox_recognition import (
    DEFAULT_NOTE as BOOX_RECOGNITION_DEFAULT_NOTE,
)
from sibyl.experiments.boox_recognition import (
    DEFAULT_OUTPUT as BOOX_RECOGNITION_DEFAULT_OUTPUT,
)
from sibyl.experiments.boox_recognition import (
    DEFAULT_RUNS as BOOX_RECOGNITION_DEFAULT_RUNS,
)
from sibyl.experiments.boox_stroke_segmentation import (
    DEFAULT_IMAGE as BOOX_SEGMENTATION_DEFAULT_IMAGE,
)
from sibyl.experiments.boox_stroke_segmentation import (
    DEFAULT_NOTE as BOOX_SEGMENTATION_DEFAULT_NOTE,
)
from sibyl.experiments.boox_stroke_segmentation import (
    DEFAULT_OUTPUT as BOOX_SEGMENTATION_DEFAULT_OUTPUT,
)
from sibyl.experiments.boox_stroke_segmentation import (
    format_stroke_segmentation,
    run_stroke_segmentation,
)
from sibyl.experiments.boox_strokes import (
    DEFAULT_NOTE as BOOX_STROKES_DEFAULT_NOTE,
)
from sibyl.experiments.boox_strokes import (
    DEFAULT_PAGE as BOOX_STROKES_DEFAULT_PAGE,
)
from sibyl.experiments.boox_strokes import (
    format_boox_strokes,
    format_boox_strokes_corpus,
    inspect_boox_strokes,
    inspect_boox_strokes_all,
)
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
from sibyl.experiments.handwriting_exemplars import (
    DEFAULT_OUTPUT as HANDWRITING_EXEMPLARS_DEFAULT_OUTPUT,
)
from sibyl.experiments.handwriting_exemplars import (
    DEFAULT_RUNS as HANDWRITING_EXEMPLARS_DEFAULT_RUNS,
)
from sibyl.experiments.handwriting_exemplars import (
    format_handwriting_exemplars,
    run_handwriting_exemplars,
)
from sibyl.experiments.handwriting_preprocess import (
    DEFAULT_OUTPUT as HANDWRITING_PREPROCESS_DEFAULT_OUTPUT,
)
from sibyl.experiments.handwriting_preprocess import (
    format_handwriting_preprocess_result,
    run_handwriting_preprocess,
)
from sibyl.experiments.qwen_recognition_knobs import (
    DEFAULT_OUTPUT as QWEN_KNOBS_DEFAULT_OUTPUT,
)
from sibyl.experiments.qwen_recognition_knobs import (
    DEFAULT_RUNS as QWEN_KNOBS_DEFAULT_RUNS,
)
from sibyl.experiments.qwen_recognition_knobs import (
    DEFAULT_SEEDS,
    DEFAULT_TEMPERATURES,
    DEFAULT_TOP_P,
    format_qwen_recognition_knobs,
    run_qwen_recognition_knobs,
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
    corpus = commands.add_parser("corpus", help="prepare local corpus source artifacts")
    corpus_commands = corpus.add_subparsers(dest="corpus_name", required=True)
    boox = corpus_commands.add_parser(
        "boox-reduce", help="create a source-preserving PDF from a BOOX note page set"
    )
    boox.add_argument("note", type=Path, help="BOOX .note file")
    boox.add_argument("pdf", type=Path, help="full source PDF")
    boox.add_argument(
        "--output",
        type=Path,
        default=Path("samples/Grafting-101-corpus.pdf"),
        help="reduced PDF output path",
    )
    boox.add_argument(
        "--manifest",
        type=Path,
        default=Path("samples/Grafting-101-corpus.json"),
        help="provenance manifest path",
    )
    experiment_parser = commands.add_parser("experiment", help="run an empirical experiment")
    experiment_commands = experiment_parser.add_subparsers(dest="experiment_name", required=True)
    boox_strokes = experiment_commands.add_parser(
        "boox-strokes", help="inspect native BOOX handwriting stroke resources"
    )
    boox_strokes.add_argument(
        "note", type=Path, nargs="?", default=BOOX_STROKES_DEFAULT_NOTE, help="BOOX .note file"
    )
    boox_page_group = boox_strokes.add_mutually_exclusive_group()
    boox_page_group.add_argument(
        "--page", type=int, default=BOOX_STROKES_DEFAULT_PAGE, help="one-based page number"
    )
    boox_page_group.add_argument(
        "--all-pages", action="store_true", help="inspect every page in the BOOX note"
    )
    boox_strokes.add_argument(
        "--output",
        type=Path,
        default=Path(".sibyl/experiments/boox-strokes"),
        help="artifact directory",
    )
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
    preprocess = experiment_commands.add_parser(
        "handwriting-preprocess", help="compare deterministic visual preprocessing variants"
    )
    preprocess.add_argument("image", type=Path, help="page image containing preserved crops")
    preprocess.add_argument("--regions", help="comma-separated existing region IDs")
    preprocess.add_argument("--lines", help="comma-separated existing line IDs (takes precedence)")
    preprocess.add_argument("--crop", type=Path, help="explicit existing source crop path")
    preprocess.add_argument(
        "--runs", type=int, default=DEFAULT_RUNS, help="reads per recognizer and variant"
    )
    preprocess.add_argument(
        "--review", type=Path, help="optional review file containing ground_truth"
    )
    preprocess.add_argument(
        "--output",
        type=Path,
        default=HANDWRITING_PREPROCESS_DEFAULT_OUTPUT,
        help="experimental JSON output path",
    )
    exemplars = experiment_commands.add_parser(
        "handwriting-exemplars",
        help="test Qwen handwriting recognition with same-writer visual examples",
    )
    exemplars.add_argument("image", type=Path, help="page image")
    exemplars.add_argument("--regions", help="comma-separated existing region IDs")
    exemplars.add_argument("--lines", help="comma-separated existing line IDs (takes precedence)")
    exemplars.add_argument("--target-crop", type=Path, help="explicit existing target crop path")
    exemplars.add_argument("--references", help="comma-separated reference IDs")
    exemplars.add_argument(
        "--reference-manifest", type=Path, help="JSON/YAML confirmed reference manifest"
    )
    exemplars.add_argument("--reference-set", help="explicit comma-separated reference-set IDs")
    exemplars.add_argument(
        "--runs",
        type=int,
        default=HANDWRITING_EXEMPLARS_DEFAULT_RUNS,
        help="runs per reference set",
    )
    exemplars.add_argument("--review", type=Path, help="optional target review JSON")
    exemplars.add_argument(
        "--output", type=Path, default=HANDWRITING_EXEMPLARS_DEFAULT_OUTPUT, help="artifact path"
    )
    boox_recognition = experiment_commands.add_parser(
        "boox-recognition",
        help="measure whether native BOOX strokes help Qwen read page-4 handwriting",
    )
    boox_recognition.add_argument(
        "image", type=Path, nargs="?", default=BOOX_RECOGNITION_DEFAULT_IMAGE, help="page-4 image"
    )
    boox_recognition.add_argument(
        "--note", type=Path, default=BOOX_RECOGNITION_DEFAULT_NOTE, help="BOOX .note source"
    )
    boox_recognition.add_argument("--regions", help="comma-separated existing region IDs")
    boox_recognition.add_argument(
        "--lines", help="comma-separated existing line IDs (takes precedence)"
    )
    boox_recognition.add_argument(
        "--runs", type=int, default=BOOX_RECOGNITION_DEFAULT_RUNS, help="reads per condition"
    )
    boox_recognition.add_argument(
        "--num-predict",
        type=int,
        default=None,
        help="maximum output tokens per read (default: 2048 for recovery, 1024 for experiments)",
    )
    boox_recognition.add_argument(
        "--num-ctx",
        type=int,
        default=BOOX_RECOGNITION_NUM_CTX,
        help="Ollama context size per read",
    )
    boox_recognition.add_argument(
        "--native-stroke-width",
        type=int,
        default=None,
        help="rendered native-reference stroke width (default: 6 for recovery, 2 for experiments)",
    )
    boox_recognition.add_argument(
        "--reference-height",
        type=int,
        help="resize native line references to this presentation height",
    )
    boox_recognition.add_argument(
        "--reference-lines",
        help="comma-separated non-target line IDs to use as references",
    )
    boox_recognition.add_argument(
        "--markdown",
        action="store_true",
        help="recover the page with BOOX evidence and write recovery.md",
    )
    boox_recognition.add_argument(
        "--baseline-attempts",
        type=int,
        default=2,
        help="recovery baseline attempts including the initial read",
    )
    boox_recognition.add_argument(
        "--targeted-rereads",
        type=int,
        default=1,
        help="targeted baseline rereads for unresolved recovery regions",
    )
    boox_recognition.add_argument(
        "--no-native-escalation",
        action="store_true",
        help="disable native-reference escalation during recovery",
    )
    boox_recognition.add_argument(
        "--native-reference-sizes",
        default=None,
        help="recovery native reference sizes, comma-separated integers or all",
    )
    boox_recognition.add_argument(
        "--conditions",
        help="comma-separated conditions for targeted experiments; recovery is adaptive",
    )
    boox_recognition.add_argument(
        "--review", type=Path, help="confirmed evaluation-only review JSON/YAML"
    )
    boox_recognition.add_argument(
        "--output",
        type=Path,
        default=BOOX_RECOGNITION_DEFAULT_OUTPUT,
        help="experimental JSON artifact",
    )
    stroke_segmentation = experiment_commands.add_parser(
        "boox-stroke-segmentation",
        help="compare raster segmentation strategies using BOOX stroke geometry",
    )
    stroke_segmentation.add_argument(
        "image", type=Path, nargs="?", default=BOOX_SEGMENTATION_DEFAULT_IMAGE
    )
    stroke_segmentation.add_argument(
        "--note", type=Path, default=BOOX_SEGMENTATION_DEFAULT_NOTE
    )
    stroke_segmentation.add_argument("--runs", type=int, default=1)
    stroke_segmentation.add_argument("--num-predict", type=int, default=2048)
    stroke_segmentation.add_argument("--num-ctx", type=int, default=8192)
    stroke_segmentation.add_argument("--review", type=Path)
    stroke_segmentation.add_argument(
        "--output", type=Path, default=BOOX_SEGMENTATION_DEFAULT_OUTPUT
    )
    stroke_segmentation.add_argument("--reread-artifact", type=Path)
    stroke_segmentation.add_argument("--compare-artifact", type=Path)
    stroke_segmentation.add_argument("--max-vertical-gap", type=float, default=70.0)
    stroke_segmentation.add_argument("--max-word-gap", type=float, default=55.0)
    knobs = experiment_commands.add_parser(
        "qwen-recognition-knobs",
        help="measure Qwen handwriting prompt, context, and decoding controls",
    )
    knobs.add_argument("image", type=Path, help="page image")
    knobs.add_argument("--regions", help="comma-separated existing region IDs")
    knobs.add_argument("--lines", help="comma-separated existing line IDs (takes precedence)")
    knobs.add_argument("--crop", type=Path, help="explicit existing source crop path")
    knobs.add_argument(
        "--runs", type=int, default=QWEN_KNOBS_DEFAULT_RUNS, help="runs per configuration"
    )
    knobs.add_argument(
        "--contexts", help="comma-separated context variants, such as tight,padding-10"
    )
    knobs.add_argument(
        "--prompts", default="regional,isolated,exact-word", help="comma-separated prompt variants"
    )
    knobs.add_argument(
        "--temperatures",
        default="0.0",
        help="comma-separated temperatures; baseline is always included",
    )
    knobs.add_argument(
        "--top-p", default="1.0", help="comma-separated top_p values; baseline is always included"
    )
    knobs.add_argument(
        "--seeds", default="", help="comma-separated deterministic seeds; empty means baseline seed"
    )
    knobs.add_argument(
        "--decode-sweep",
        action="store_true",
        help="expand to the documented temperature/top_p/seed sweep",
    )
    knobs.add_argument("--num-predict", type=int, default=256, help="Ollama num_predict control")
    knobs.add_argument(
        "--review", type=Path, help="optional JSON/YAML human review with confirmed ground_truth"
    )
    knobs.add_argument(
        "--output", type=Path, default=QWEN_KNOBS_DEFAULT_OUTPUT, help="artifact path"
    )
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
    if arguments.command == "corpus" and arguments.corpus_name == "boox-reduce":
        try:
            corpus_result = prepare_boox_corpus(
                arguments.note,
                arguments.pdf,
                output_pdf=arguments.output,
                manifest_path=arguments.manifest,
            )
        except (FileNotFoundError, RuntimeError, ValueError, OSError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_corpus_result(corpus_result))
        return 0
    if arguments.command == "experiment" and arguments.experiment_name == "trocr":
        try:
            result = run_experiment(arguments.image)
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_result(result, arguments.json))
    if arguments.command == "experiment" and arguments.experiment_name == "boox-strokes":
        try:
            if arguments.all_pages:
                boox_result = inspect_boox_strokes_all(arguments.note, output=arguments.output)
                print(format_boox_strokes_corpus(boox_result))
                return 0
            boox_result = inspect_boox_strokes(
                arguments.note, page=arguments.page, output=arguments.output
            )
        except (FileNotFoundError, RuntimeError, ValueError, OSError, zipfile.BadZipFile) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_boox_strokes(boox_result))
    if (
        arguments.command == "experiment"
        and arguments.experiment_name == "boox-stroke-segmentation"
    ):
        try:
            segmentation_result = run_stroke_segmentation(
                arguments.image,
                note_path=arguments.note,
                runs=arguments.runs,
                num_predict=arguments.num_predict,
                num_ctx=arguments.num_ctx,
                review_path=arguments.review,
                output_path=arguments.output,
                reread_path=arguments.reread_artifact
                or Path(".sibyl/experiments/transcription-reread.json"),
                compare_path=arguments.compare_artifact
                or Path(".sibyl/experiments/trocr-compare.json"),
                max_vertical_gap=arguments.max_vertical_gap,
                max_word_gap=arguments.max_word_gap,
            )
        except (FileNotFoundError, RuntimeError, ValueError, OSError, zipfile.BadZipFile) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_stroke_segmentation(segmentation_result))
        return 0
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
    if arguments.command == "experiment" and arguments.experiment_name == "handwriting-preprocess":
        try:
            preprocess_result = run_handwriting_preprocess(
                arguments.image,
                runs=arguments.runs,
                regions=arguments.regions,
                lines=arguments.lines,
                crop_path=arguments.crop,
                review_path=arguments.review,
                output_path=arguments.output,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_handwriting_preprocess_result(preprocess_result))
    if arguments.command == "experiment" and arguments.experiment_name == "qwen-recognition-knobs":
        try:
            temperatures: tuple[float, ...]
            top_ps: tuple[float, ...]
            seeds: tuple[int | None, ...]
            if arguments.decode_sweep:
                temperatures, top_ps, seeds = DEFAULT_TEMPERATURES, DEFAULT_TOP_P, DEFAULT_SEEDS
            else:
                temperatures = tuple(
                    float(value) for value in arguments.temperatures.split(",") if value
                )
                top_ps = tuple(float(value) for value in arguments.top_p.split(",") if value)
                seeds = tuple(int(value) for value in arguments.seeds.split(",") if value) or (
                    None,
                )
            contexts = (
                tuple(value.strip() for value in arguments.contexts.split(",") if value.strip())
                if arguments.contexts
                else None
            )
            prompts = tuple(
                value.strip() for value in arguments.prompts.split(",") if value.strip()
            )
            knobs_result = run_qwen_recognition_knobs(
                arguments.image,
                runs=arguments.runs,
                regions=arguments.regions,
                lines=arguments.lines,
                crop_path=arguments.crop,
                contexts=contexts,
                prompt_variants=prompts,
                temperatures=temperatures,
                top_ps=top_ps,
                seeds=seeds,
                num_predict=arguments.num_predict,
                review_path=arguments.review,
                output_path=arguments.output,
            )
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_qwen_recognition_knobs(knobs_result))
    if arguments.command == "experiment" and arguments.experiment_name == "handwriting-exemplars":
        try:
            exemplar_result = run_handwriting_exemplars(
                arguments.image,
                regions=arguments.regions,
                lines=arguments.lines,
                target_crop=arguments.target_crop,
                references=arguments.references,
                reference_manifest=arguments.reference_manifest,
                reference_set=arguments.reference_set,
                runs=arguments.runs,
                review_path=arguments.review,
                output_path=arguments.output,
            )
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_handwriting_exemplars(exemplar_result))
    if arguments.command == "experiment" and arguments.experiment_name == "boox-recognition":
        try:
            recognition_result = run_boox_recognition(
                arguments.image,
                note_path=arguments.note,
                regions=arguments.regions,
                lines=arguments.lines,
                runs=arguments.runs,
                num_predict=arguments.num_predict,
                num_ctx=arguments.num_ctx,
                native_stroke_width=arguments.native_stroke_width,
                reference_height=arguments.reference_height,
                reference_lines=arguments.reference_lines,
                conditions=arguments.conditions,
                baseline_attempts=arguments.baseline_attempts,
                targeted_rereads=arguments.targeted_rereads,
                native_escalation=not arguments.no_native_escalation,
                native_reference_sizes=arguments.native_reference_sizes,
                markdown=arguments.markdown,
                review_path=arguments.review,
                output_path=arguments.output,
            )
        except (
            FileNotFoundError,
            RuntimeError,
            ValueError,
            OSError,
            json.JSONDecodeError,
        ) as error:
            print(f"sibyl: error: {error}", file=__import__("sys").stderr)
            return 2
        print(format_boox_recognition(recognition_result))
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
