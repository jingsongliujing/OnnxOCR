from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List

from .base import BaseOCRSkill, OcrEngine
from .builtin.industry import (
    create_agriculture_plant_protection_skill,
    create_agriculture_quality_skill,
    create_agriculture_traceability_skill,
    create_contract_key_info_skill,
    create_business_license_skill,
    create_document_image_to_markdown_skill,
    create_document_pdf_to_markdown_skill,
    create_education_exam_paper_skill,
    create_finance_invoice_skill,
    create_finance_bank_card_skill,
    create_government_red_head_skill,
    create_identity_id_card_skill,
    create_logistics_inbound_skill,
    create_logistics_express_waybill_skill,
    create_medical_lab_report_skill,
    create_oa_reimbursement_skill,
    create_table_structuring_skill,
    create_transport_taxi_invoice_skill,
    create_transport_train_ticket_skill,
    create_vehicle_driving_license_skill,
    create_vehicle_driver_license_skill,
    create_vehicle_plate_skill,
)

SkillFactory = Callable[[OcrEngine], BaseOCRSkill]


class SkillRegistry:
    """Registry for built-in and user-defined OCR skills."""

    def __init__(self):
        self._factories: Dict[str, SkillFactory] = OrderedDict()

    def register(self, skill_id: str, factory: SkillFactory) -> None:
        if skill_id in self._factories:
            raise ValueError(f"Skill already registered: {skill_id}")
        self._factories[skill_id] = factory

    def get(self, skill_id: str, ocr_engine: OcrEngine) -> BaseOCRSkill:
        try:
            return self._factories[skill_id](ocr_engine)
        except KeyError as exc:
            supported = ", ".join(self._factories)
            raise KeyError(f"Unknown skill_id: {skill_id}. Supported: {supported}") from exc

    def list_ids(self) -> List[str]:
        return list(self._factories.keys())

    def schemas(self, ocr_engine: OcrEngine) -> List[Dict]:
        return [self.get(skill_id, ocr_engine).schema() for skill_id in self.list_ids()]

    def update(self, items: Iterable[tuple[str, SkillFactory]]) -> None:
        for skill_id, factory in items:
            self.register(skill_id, factory)


def create_default_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.update(
        [
            ("transport.train_ticket", create_transport_train_ticket_skill),
            ("education.exam_paper", create_education_exam_paper_skill),
            ("vehicle.plate", create_vehicle_plate_skill),
            ("table.structure", create_table_structuring_skill),
            ("document.image_to_markdown", create_document_image_to_markdown_skill),
        ]
    )
    return registry


def create_candidate_registry() -> SkillRegistry:
    """Experimental templates that need real sample validation before default use."""

    registry = create_default_registry()
    registry.update(
        [
            ("agriculture.quality_inspection", create_agriculture_quality_skill),
            ("agriculture.traceability_label", create_agriculture_traceability_skill),
            ("agriculture.plant_protection_record", create_agriculture_plant_protection_skill),
            ("oa.reimbursement", create_oa_reimbursement_skill),
            ("finance.invoice", create_finance_invoice_skill),
            ("finance.bank_card", create_finance_bank_card_skill),
            ("identity.id_card", create_identity_id_card_skill),
            ("business.license", create_business_license_skill),
            ("legal.contract_key_info", create_contract_key_info_skill),
            ("government.red_head_document", create_government_red_head_skill),
            ("logistics.inbound_order", create_logistics_inbound_skill),
            ("logistics.express_waybill", create_logistics_express_waybill_skill),
            ("medical.lab_report", create_medical_lab_report_skill),
            ("transport.taxi_invoice", create_transport_taxi_invoice_skill),
            ("vehicle.driving_license", create_vehicle_driving_license_skill),
            ("vehicle.driver_license", create_vehicle_driver_license_skill),
            ("document.pdf_to_markdown", create_document_pdf_to_markdown_skill),
        ]
    )
    return registry
