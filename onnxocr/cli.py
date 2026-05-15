from __future__ import annotations

import sys
from typing import Sequence

from onnxocr.skill_cli import main as skill_main


SKILL_ALIASES = {"skill", "skills", "skill_cli", "onnxocr.skill_cli"}


def normalize_args(argv: Sequence[str] | None = None) -> list[str]:
    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] in SKILL_ALIASES:
        args = args[1:]
    return args


def main(argv: Sequence[str] | None = None) -> int:
    return skill_main(normalize_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
