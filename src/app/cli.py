"""
CLI\u2011\u0438\u043d\u0442\u0435\u0440\u0444\u0435\u0439\u0441.

\u041f\u0440\u0438\u043c\u0435\u0440\u044b
-------
$ python -m src.app.cli --pdf docs/example.pdf
$ python -m src.app.cli --pdf docs/example.pdf --topk 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from model.predict import classify

__all__ = ["main"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a PDF document into topical categories."
    )
    parser.add_argument(
        "--pdf",
        required=True,
        type=Path,
        help="Path to a PDF file.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=1,
        help="Number of top categories to show.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Increase log verbosity.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.verbose:
        logger.remove()
        logger.add(sys.stderr, level="DEBUG")

    if not args.pdf.is_file():
        logger.error("File not found: %s", args.pdf)
        sys.exit(1)

    try:
        predictions = classify(args.pdf, top_k=args.topk)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to classify: %s", exc)
        sys.exit(2)

    for label, prob in predictions:
        print(f"{label:<15} {prob:.2%}")


if __name__ == "__main__":
    main()
