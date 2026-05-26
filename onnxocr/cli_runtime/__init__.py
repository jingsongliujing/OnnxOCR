from .base import BaseOCREngine, FieldResult, OcrLine, CLIInput, CLIOutput
from .engine import OnnxOCREngine
from .registry import CLIRegistry, create_candidate_registry, create_default_registry
from .template_cli import TemplateOCREngine, TemplateSpec

__all__ = [
    "BaseOCREngine",
    "FieldResult",
    "OcrLine",
    "OnnxOCREngine",
    "CLIInput",
    "CLIOutput",
    "CLIRegistry",
    "TemplateOCREngine",
    "TemplateSpec",
    "create_candidate_registry",
    "create_default_registry",
]

# Backward compatibility aliases
BaseOCRSkill = BaseOCREngine
SkillInput = CLIInput
SkillOutput = CLIOutput
SkillRegistry = CLIRegistry
TemplateOCRSkill = TemplateOCREngine
