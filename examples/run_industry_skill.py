from __future__ import annotations

import argparse
import json

from onnxocr.skills import OnnxOCREngine, SkillInput, create_default_registry


def main():
    parser = argparse.ArgumentParser(description="Run an OnnxOCR industry skill.")
    parser.add_argument("skill_id", help="Example: transport.train_ticket")
    parser.add_argument("image_path")
    args = parser.parse_args()

    engine = OnnxOCREngine()
    registry = create_default_registry()
    skill = registry.get(args.skill_id, engine)
    result = skill.run(SkillInput(image_path=args.image_path))
    print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
