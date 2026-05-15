from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from onnxocr.skills import (
    OnnxOCREngine,
    SkillInput,
    create_candidate_registry,
    create_default_registry,
)


def _print_text(text: str) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    print(text)


def main(argv: Any = None) -> int:
    parser = argparse.ArgumentParser(description="Run vertical OCR skills on top of OnnxOCR.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list", help="List available OCR skills.")
    list_parser.add_argument("--candidates", action="store_true", help="Include experimental skills.")

    schema_parser = subparsers.add_parser("schema", help="Print one skill schema.")
    schema_parser.add_argument("skill_id")
    schema_parser.add_argument("--candidates", action="store_true", help="Include experimental skills.")

    run_parser = subparsers.add_parser("run", help="Run a skill against an image.")
    run_parser.add_argument("skill_id")
    run_parser.add_argument("image_path")
    run_parser.add_argument("--pretty", action="store_true", help="Pretty-print JSON.")
    run_parser.add_argument("--candidates", action="store_true", help="Include experimental skills.")

    args = parser.parse_args(argv)
    engine = OnnxOCREngine()
    registry = create_candidate_registry() if getattr(args, "candidates", False) else create_default_registry()

    if args.command == "list":
        for skill_id in registry.list_ids():
            skill = registry.get(skill_id, engine)
            print(f"{skill.skill_id}\t{skill.name}")
        return 0

    if args.command == "schema":
        skill = registry.get(args.skill_id, engine)
        _print_text(json.dumps(skill.schema(), ensure_ascii=False, indent=2))
        return 0

    if args.command == "run":
        image_path = Path(args.image_path)
        skill = registry.get(args.skill_id, engine)
        result = skill.run(SkillInput(image_path=str(image_path)))
        indent = 2 if args.pretty else None
        _print_text(json.dumps(result.to_dict(), ensure_ascii=False, indent=indent))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
