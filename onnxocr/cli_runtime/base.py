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
    """A normalized field extracted by a CLI scenario."""

    name: str
    value: Any
    confidence: float = 0.0
    source_text: str = ""
    box: List[List[float]] = field(default_factory=list)


@dataclass
class CLIInput:
    """Input passed to an OCR CLI scenario."""

    image: Optional[np.ndarray] = None
    image_path: Optional[str] = None
    ocr_lines: Optional[List[OcrLine]] = None
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CLIOutput:
    """Standard result returned by every OCR CLI scenario."""

    cli_id: str
    cli_name: str
    fields: Dict[str, Any]
    field_results: List[FieldResult]
    raw_text: str
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cli_id": self.cli_id,
            "cli_name": self.cli_name,
            "fields": self.fields,
            "field_results": [field.__dict__ for field in self.field_results],
            "raw_text": self.raw_text,
            "confidence": self.confidence,
            "metadata": self.metadata,
        }


class OcrEngine(Protocol):
    """Protocol implemented by OCR backends used by CLI scenarios."""

    def recognize(self, cli_input: CLIInput) -> List[OcrLine]:
        ...

    def recognize_table(self, cli_input: CLIInput) -> Dict[str, Any]:
        ...

    def recognize_plate(self, cli_input: CLIInput) -> List[Dict[str, Any]]:
        ...


class BaseOCREngine:
    """Base class for vertical OCR CLI scenarios."""

    cli_id = ""
    name = ""
    description = ""
    domains: List[str] = []
    version = "0.1.0"

    def __init__(self, ocr_engine: OcrEngine):
        self.ocr_engine = ocr_engine

    def run(self, cli_input: CLIInput) -> CLIOutput:
        raise NotImplementedError

    def schema(self) -> Dict[str, Any]:
        return {
            "cli_id": self.cli_id,
            "name": self.name,
            "description": self.description,
            "domains": self.domains,
            "version": self.version,
        }


# Backward compatibility aliases
SkillInput = CLIInput
SkillOutput = CLIOutput
BaseOCRSkill = BaseOCREngine
