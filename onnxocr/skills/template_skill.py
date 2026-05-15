from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from .base import BaseOCRSkill, SkillInput, SkillOutput
from .extractors import FieldSpec, extract_fields, fields_to_dict, merge_text


@dataclass
class TemplateSpec:
    skill_id: str
    name: str
    description: str
    domains: List[str]
    fields: List[FieldSpec] = field(default_factory=list)
    version: str = "0.1.0"


class TemplateOCRSkill(BaseOCRSkill):
    """Configurable OCR skill for a fixed industry document template."""

    template: TemplateSpec

    def __init__(self, ocr_engine, template: TemplateSpec):
        super().__init__(ocr_engine)
        self.template = template
        self.skill_id = template.skill_id
        self.name = template.name
        self.description = template.description
        self.domains = template.domains
        self.version = template.version

    def run(self, skill_input: SkillInput) -> SkillOutput:
        lines = self.ocr_engine.recognize(skill_input)
        field_results = extract_fields(lines, self.template.fields)
        confidences = [field.confidence for field in field_results if field.value not in (None, "")]
        confidence = sum(confidences) / len(confidences) if confidences else 0.0
        return SkillOutput(
            skill_id=self.skill_id,
            skill_name=self.name,
            fields=fields_to_dict(field_results),
            field_results=field_results,
            raw_text=merge_text(lines),
            confidence=confidence,
            metadata={"domains": self.domains, "version": self.version},
        )

    def schema(self) -> Dict:
        schema = super().schema()
        schema["fields"] = [
            {
                "name": field.name,
                "labels": field.labels,
                "patterns": field.patterns,
                "required": field.required,
                "normalizer": field.normalizer,
            }
            for field in self.template.fields
        ]
        return schema
