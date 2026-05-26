from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, Iterable, List

from .base import BaseOCREngine, OcrEngine
from .builtin.industry import (
    create_agriculture_plant_protection_cli,
    create_agriculture_quality_cli,
    create_agriculture_traceability_cli,
    create_contract_key_info_cli,
    create_business_license_cli,
    create_document_image_to_markdown_cli,
    create_document_pdf_to_markdown_cli,
    create_education_exam_paper_cli,
    create_finance_invoice_cli,
    create_finance_bank_card_cli,
    create_government_red_head_cli,
    create_identity_id_card_cli,
    create_logistics_inbound_cli,
    create_logistics_express_waybill_cli,
    create_medical_lab_report_cli,
    create_oa_reimbursement_cli,
    create_table_structuring_cli,
    create_transport_taxi_invoice_cli,
    create_transport_train_ticket_cli,
    create_vehicle_driving_license_cli,
    create_vehicle_driver_license_cli,
    create_vehicle_plate_cli,
)

CLIFactory = Callable[[OcrEngine], BaseOCREngine]


class CLIRegistry:
    """Registry for built-in and user-defined OCR CLI scenarios."""

    def __init__(self):
        self._factories: Dict[str, CLIFactory] = OrderedDict()

    def register(self, cli_id: str, factory: CLIFactory) -> None:
        if cli_id in self._factories:
            raise ValueError(f"CLI scenario already registered: {cli_id}")
        self._factories[cli_id] = factory

    def get(self, cli_id: str, ocr_engine: OcrEngine) -> BaseOCREngine:
        try:
            return self._factories[cli_id](ocr_engine)
        except KeyError as exc:
            supported = ", ".join(self._factories)
            raise KeyError(f"Unknown cli_id: {cli_id}. Supported: {supported}") from exc

    def list_ids(self) -> List[str]:
        return list(self._factories.keys())

    def schemas(self, ocr_engine: OcrEngine) -> List[Dict]:
        return [self.get(cli_id, ocr_engine).schema() for cli_id in self.list_ids()]

    def update(self, items: Iterable[tuple[str, CLIFactory]]) -> None:
        for cli_id, factory in items:
            self.register(cli_id, factory)


def create_default_registry() -> CLIRegistry:
    registry = CLIRegistry()
    registry.update(
        [
            ("education.exam_paper", create_education_exam_paper_cli),
            ("identity.id_card", create_identity_id_card_cli),
            ("finance.bank_card", create_finance_bank_card_cli),
            ("finance.invoice", create_finance_invoice_cli),
            ("vehicle.plate", create_vehicle_plate_cli),
            ("table.structure", create_table_structuring_cli),
            ("document.image_to_markdown", create_document_image_to_markdown_cli),
        ]
    )
    return registry


def create_candidate_registry() -> CLIRegistry:
    """Experimental templates that need real sample validation before default use."""

    registry = create_default_registry()
    candidates = [
        ("agriculture.quality_inspection", create_agriculture_quality_cli),
        ("agriculture.traceability_label", create_agriculture_traceability_cli),
        ("agriculture.plant_protection_record", create_agriculture_plant_protection_cli),
        ("oa.reimbursement", create_oa_reimbursement_cli),
        ("business.license", create_business_license_cli),
        ("legal.contract_key_info", create_contract_key_info_cli),
        ("government.red_head_document", create_government_red_head_cli),
        ("logistics.inbound_order", create_logistics_inbound_cli),
        ("logistics.express_waybill", create_logistics_express_waybill_cli),
        ("medical.lab_report", create_medical_lab_report_cli),
        ("transport.train_ticket", create_transport_train_ticket_cli),
        ("transport.taxi_invoice", create_transport_taxi_invoice_cli),
        ("vehicle.driving_license", create_vehicle_driving_license_cli),
        ("vehicle.driver_license", create_vehicle_driver_license_cli),
        ("document.pdf_to_markdown", create_document_pdf_to_markdown_cli),
    ]
    for cli_id, factory in candidates:
        if cli_id not in registry.list_ids():
            registry.register(cli_id, factory)
    return registry


# Backward compatibility aliases
SkillRegistry = CLIRegistry
