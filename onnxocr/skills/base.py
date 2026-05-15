from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol

import numpy as np


@dataclass
class OcrLine:
    """One OCR text line with its quadrilateral box and confidence."""

    text: str
    score: float
    box: List[List[float]] = field(default_factory=list)


@dataclass
class FieldResult:
    """A normalized field extracted by a skill."""

    name: str
    value: Any
    confidence: float = 0.0
    source_text: str = ""
    box: List[List[float]] = field(default_factory=list)


@dataclass
class SkillInput:
    """Input passed to an OCR skill."""

    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    ocr_lines: Optional[List[OcrLine]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SkillOutput:
    """Standard result returned by every OCR skill."""

    skill_id: str
    skill_name: str
    fields: Dict[str, Any]
    field_results: List[FieldResult]
    raw_text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "fields": self.fields,
            "field_results": [field.__dict__ for field in self.field_results],
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class OcrEngine(Protocol):
    """Protocol implemented by OCR backends used by skills."""

    def recognize(self, skill_input: SkillInput) -> List[OcrLine]:
        ...

    def recognize_table(self, skill_input: SkillInput) -> Dict[str, Any]:
        ...

    def recognize_plate(self, skill_input: SkillInput) -> List[Dict[str, Any]]:
        ...


class BaseOCRSkill:
    """Base class for vertical OCR skills."""

    skill_id = ""
    name = ""
    description = ""
    domains: List[str] = []
    version = "0.1.0"

    def __init__(self, ocr_engine: OcrEngine):
        self.ocr_engine = ocr_engine

    def run(self, skill_input: SkillInput) -> SkillOutput:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
            "version": self.version,
        }
