from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path


def default_target() -> Path:
    codex_home = os.environ.get("CODEX_HOME")
    if codex_home:
        return Path(codex_home) / "skills" / "onnxocr-industry-ocr"
    return Path.home() / ".codex" / "skills" / "onnxocr-industry-ocr"


def install_skill(target: Path, force: bool = False) -> Path:
    root = Path(__file__).resolve().parents[1]
    source = root / "skills" / "industry-ocr"
    if not source.exists():
        raise FileNotFoundError(f"Skill source not found: {source}")

    target = target.expanduser().resolve()
    if target.exists():
        if not force:
            raise FileExistsError(f"Target already exists: {target}. Use --force to overwrite.")
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target)
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Install OnnxOCR industry OCR Skill for local Agent tools.")
    parser.add_argument("--target", type=Path, default=default_target(), help="Target skill directory.")
    parser.add_argument("--force", action="store_true", help="Overwrite existing target.")
    args = parser.parse_args()

    target = install_skill(args.target, force=args.force)
    print(f"Installed OnnxOCR skill to: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
