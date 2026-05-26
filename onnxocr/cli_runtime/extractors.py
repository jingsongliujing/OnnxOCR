from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional

from .base import FieldResult, OcrLine


@dataclass
class FieldSpec:
    name: str
    labels: List[str] = field(default_factory=list)
    patterns: List[str] = field(default_factory=list)
    required: bool = False
    normalizer: Optional[str] = None


def normalize_onnxocr_result(result: Any) -> List[OcrLine]:
    """Convert ONNXPaddleOcr general OCR output into OcrLine objects."""

    if not result:
        return []

    lines = result[0] if isinstance(result, list) and result and isinstance(result[0], list) else result
    normalized = []
    for line in lines:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        box = line[0]
        rec = line[1]
        if isinstance(rec, (list, tuple)) and len(rec) >= 2:
            text, score = rec[0], rec[1]
        else:
            text, score = rec, 0.0
        normalized.append(OcrLine(text=str(text), score=float(score), box=box))
    return normalized


def merge_text(lines: Iterable[OcrLine]) -> str:
    return "\n".join(line.text for line in lines if line.text)


def extract_fields(lines: List[OcrLine], field_specs: List[FieldSpec]) -> List[FieldResult]:
    raw_text = merge_text(lines)
    results = []
    for spec in field_specs:
        match = _extract_by_pattern(raw_text, spec)
        if match is None:
            match = _extract_by_label(lines, spec)
        if match is None:
            if spec.required:
                results.append(FieldResult(name=spec.name, value=None, confidence=0.0))
            continue
        value = _normalize_value(match.value, spec.normalizer)
        results.append(
            FieldResult(
                name=spec.name,
                value=value,
                confidence=match.confidence,
                source_text=match.source_text,
                box=match.box,
            )
        )
    return results


def fields_to_dict(field_results: List[FieldResult]) -> Dict[str, Any]:
    return {result.name: result.value for result in field_results}


@dataclass
class _Match:
    value: str
    confidence: float
    source_text: str
    box: List[List[float]] = field(default_factory=list)


def _extract_by_pattern(raw_text: str, spec: FieldSpec) -> Optional[_Match]:
    for pattern in spec.patterns:
        match = re.search(pattern, raw_text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            value = match.group(1) if match.groups() else match.group(0)
            return _Match(
                value=value.strip(" :：\t\r\n"),
                confidence=0.85,
                source_text=match.group(0),
            )
    return None


def _extract_by_label(lines: List[OcrLine], spec: FieldSpec) -> Optional[_Match]:
    for index, line in enumerate(lines):
        text = line.text.strip()
        for label in sorted(spec.labels, key=len, reverse=True):
            if label not in text:
                continue
            value = _value_after_label(text, label)
            source = text
            confidence = line.score * 0.8 if line.score else 0.55
            box = line.box
            if not value and index + 1 < len(lines):
                next_line = lines[index + 1]
                value = next_line.text.strip()
                source = f"{text} {value}"
                confidence = min(line.score, next_line.score) if line.score and next_line.score else 0.5
                box = next_line.box
            if value:
                return _Match(value=value, confidence=float(confidence), source_text=source, box=box)
    return None


def _value_after_label(text: str, label: str) -> str:
    value = text.split(label, 1)[1]
    return value.strip(" :：-\t")


def _normalize_value(value: str, normalizer: Optional[str]) -> str:
    value = value.strip()
    if normalizer == "date":
        value = value.replace("年", "-").replace("月", "-").replace("日", "")
        value = value.replace("/", "-").replace(".", "-").replace("?", "-")
        value = value.strip("-")
    if normalizer == "amount":
        value = value.replace(",", "").replace("￥", "").replace("¥", "")
    if normalizer == "digits":
        value = re.sub(r"\D", "", value)
    return value
