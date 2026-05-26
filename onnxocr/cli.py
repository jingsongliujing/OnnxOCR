from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from onnxocr.cli_runtime import (
    OnnxOCREngine,
    CLIInput,
    create_candidate_registry,
    create_default_registry,
)

SKILL_ALIASES = {"skill", "skills", "skill_cli", "onnxocr.skill_cli"}


def normalize_args(argv: Sequence[str] | None = None) -> list[str]:
    """Normalize CLI arguments, removing legacy skill aliases."""
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in SKILL_ALIASES:
        args = args[1:]
    return args


def _print_text(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(text)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run vertical OCR CLI on top of OnnxOCR.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available OCR scenarios.")
    list_parser.add_argument("--candidates", action="store_true", help="Include experimental scenarios.")

    schema_parser = subparsers.add_parser("schema", help="Print one scenario schema.")
    schema_parser.add_argument("scenario_id")
    schema_parser.add_argument("--candidates", action="store_true", help="Include experimental scenarios.")

    run_parser = subparsers.add_parser("run", help="Run a scenario against an image.")
    run_parser.add_argument("scenario_id")
    run_parser.add_argument("image_path")
    run_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    run_parser.add_argument("--candidates", action="store_true", help="Include experimental scenarios.")

    args = parser.parse_args(argv)
    engine = OnnxOCREngine()
    registry = create_candidate_registry() if getattr(args, "candidates", False) else create_default_registry()

    if args.command == "list":
        for scenario_id in registry.list_ids():
            scenario = registry.get(scenario_id, engine)
            print(f"{scenario.cli_id}\t{scenario.name}")
        return 0

    if args.command == "schema":
        scenario = registry.get(args.scenario_id, engine)
        _print_text(json.dumps(scenario.schema(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        image_path = Path(args.image_path)
        scenario = registry.get(args.scenario_id, engine)
        result = scenario.run(CLIInput(image_path=str(image_path)))
        indent = 2 if args.pretty else None
        _print_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=indent))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
