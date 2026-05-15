import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from onnxocr.skills import OcrLine, SkillInput, create_candidate_registry, create_default_registry
from onnxocr.cli import normalize_args


class FakeEngine:
    def __init__(self, lines=None, table=None, plates=None):
        self.lines = lines or []
        self.table = table or {}
        self.plates = plates or []

    def recognize(self, skill_input):
        return skill_input.ocr_lines if skill_input.ocr_lines is not None else self.lines

    def recognize_table(self, skill_input):
        return self.table

    def recognize_plate(self, skill_input):
        return self.plates


def test_cli_normalizes_skill_aliases():
    assert normalize_args(["list"]) == ["list"]
    assert normalize_args(["skill", "list"]) == ["list"]
    assert normalize_args(["onnxocr.skill_cli", "list", "--candidates"]) == ["list", "--candidates"]


def test_default_registry_lists_vertical_skills():
    registry = create_default_registry()

    assert registry.list_ids() == [
        "transport.train_ticket",
        "education.exam_paper",
        "vehicle.plate",
        "table.structure",
        "document.image_to_markdown",
    ]


def test_candidate_registry_keeps_unvalidated_templates_out_of_default():
    registry = create_candidate_registry()

    assert "transport.taxi_invoice" in registry.list_ids()
    assert "agriculture.quality_inspection" in registry.list_ids()
    assert "identity.id_card" in registry.list_ids()
    assert "finance.bank_card" in registry.list_ids()
    assert "business.license" in registry.list_ids()
    assert "logistics.express_waybill" in registry.list_ids()
    assert "vehicle.driver_license" in registry.list_ids()
    assert "transport.taxi_invoice" not in create_default_registry().list_ids()


def test_agriculture_quality_skill_extracts_chinese_template_fields():
    lines = [
        OcrLine("产品名称：苹果", 0.98),
        OcrLine("批号：A-20260515", 0.96),
        OcrLine("检测日期：2026年05月15日", 0.95),
        OcrLine("检测结果：合格", 0.97),
    ]
    registry = create_candidate_registry()
    skill = registry.get("agriculture.quality_inspection", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["product_name"] == "苹果"
    assert result.fields["batch_no"] == "A-20260515"
    assert result.fields["inspection_date"] == "2026-05-15"
    assert result.fields["result"] == "合格"
    assert result.confidence > 0.5


def test_agriculture_quality_skill_supports_english_aliases():
    lines = [
        OcrLine("Product Name: Apple", 0.99),
        OcrLine("Batch No: A-20260515", 0.98),
        OcrLine("Inspection Date: 2026-05-15", 0.97),
        OcrLine("Result: PASS", 0.96),
        OcrLine("Supplier: Green Farm", 0.95),
    ]
    registry = create_candidate_registry()
    skill = registry.get("agriculture.quality_inspection", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["product_name"] == "Apple"
    assert result.fields["batch_no"] == "A-20260515"
    assert result.fields["inspection_date"] == "2026-05-15"
    assert result.fields["result"] == "PASS"
    assert result.fields["supplier"] == "Green Farm"


def test_invoice_skill_extracts_key_fields():
    lines = [
        OcrLine("发票号码：12345678", 0.99),
        OcrLine("开票日期：2026年05月15日", 0.98),
        OcrLine("购买方：北京某某科技有限公司", 0.97),
        OcrLine("价税合计：￥1280.50", 0.96),
    ]
    registry = create_candidate_registry()
    skill = registry.get("finance.invoice", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["invoice_no"] == "12345678"
    assert result.fields["invoice_date"] == "2026-05-15"
    assert result.fields["buyer"] == "北京某某科技有限公司"
    assert result.fields["amount"] == "1280.50"


def test_contract_skill_extracts_parties():
    lines = [
        OcrLine("合同编号：HT-2026-001", 0.99),
        OcrLine("甲方：上海甲方公司", 0.98),
        OcrLine("乙方：杭州乙方公司", 0.98),
        OcrLine("合同金额：50000.00", 0.97),
    ]
    registry = create_candidate_registry()
    skill = registry.get("legal.contract_key_info", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["contract_no"] == "HT-2026-001"
    assert result.fields["party_a"] == "上海甲方公司"
    assert result.fields["party_b"] == "杭州乙方公司"
    assert result.fields["amount"] == "50000.00"


def test_education_exam_paper_skill_extracts_realistic_chinese_fields():
    lines = [
        OcrLine("2024年三年级上册语文期末检测卷", 0.98),
        OcrLine("时间：90分钟满分：100分", 0.97),
    ]
    registry = create_default_registry()
    skill = registry.get("education.exam_paper", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["title"] == "2024年三年级上册语文期末检测卷"
    assert result.fields["grade"] == "三年级"
    assert result.fields["subject"] == "语文"
    assert result.fields["time_limit"] == "90分钟"
    assert result.fields["total_score"] == "100分"


def test_train_ticket_skill_extracts_realistic_chinese_fields():
    lines = [
        OcrLine("001H034094", 0.98),
        OcrLine("鹤壁东站", 0.98),
        OcrLine("郑州东站", 0.98),
        OcrLine("G1289", 0.99),
        OcrLine("2018年01月05日17:41开", 0.99),
        OcrLine("07车08D号", 0.97),
        OcrLine("￥59.5元", 0.98),
        OcrLine("二等座", 0.98),
    ]
    registry = create_default_registry()
    skill = registry.get("transport.train_ticket", FakeEngine(lines=lines))

    result = skill.run(SkillInput())

    assert result.fields["ticket_no"] == "001H034094"
    assert result.fields["from_station"] == "鹤壁东站"
    assert result.fields["to_station"] == "郑州东站"
    assert result.fields["train_no"] == "G1289"
    assert result.fields["price"] == "59.5元"


def test_table_skill_wraps_structured_table_output():
    table = {"html": "<table><tr><td>A</td></tr></table>", "cell_bboxes": [[0, 0, 10, 10]]}
    registry = create_candidate_registry()
    skill = registry.get("table.structure", FakeEngine(table=table))

    result = skill.run(SkillInput())

    assert result.fields["html"].startswith("<table")
    assert result.fields["cell_bboxes"] == [[0, 0, 10, 10]]


def test_id_card_candidate_extracts_fake_sample_fields():
    lines = [
        OcrLine("姓名：张三", 0.98),
        OcrLine("性别：男 民族：汉", 0.98),
        OcrLine("出生：1990年01月02日", 0.98),
        OcrLine("住址：北京市海淀区示例路1号", 0.98),
        OcrLine("公民身份号码：110101199001020011", 0.98),
    ]
    skill = create_candidate_registry().get("identity.id_card", FakeEngine(lines=lines))
    result = skill.run(SkillInput())

    assert result.fields["name"] == "张三"
    assert result.fields["gender"] == "男"
    assert result.fields["id_no"] == "110101199001020011"


def test_bank_card_candidate_extracts_fake_sample_fields():
    lines = [
        OcrLine("中国工商银行", 0.98),
        OcrLine("6222 0200 1234 5678 901", 0.98),
        OcrLine("VALID THRU 12/30", 0.98),
        OcrLine("借记卡", 0.98),
    ]
    skill = create_candidate_registry().get("finance.bank_card", FakeEngine(lines=lines))
    result = skill.run(SkillInput())

    assert result.fields["card_no"] == "6222020012345678901"
    assert result.fields["bank_name"] == "中国工商银行"
    assert result.fields["valid_thru"] == "12/30"


def test_bank_card_candidate_handles_multiline_card_number():
    lines = [
        OcrLine("6222", 0.98),
        OcrLine("0200", 0.98),
        OcrLine("1234 5678", 0.98),
        OcrLine("901", 0.98),
        OcrLine("VALID", 0.98),
        OcrLine("THRU", 0.98),
        OcrLine("12/30", 0.98),
    ]
    skill = create_candidate_registry().get("finance.bank_card", FakeEngine(lines=lines))
    result = skill.run(SkillInput())

    assert result.fields["card_no"] == "6222020012345678901"
    assert result.fields["valid_thru"] == "12/30"


def test_business_license_candidate_extracts_fake_sample_fields():
    lines = [
        OcrLine("统一社会信用代码：91110108MA0000000X", 0.98),
        OcrLine("名称：北京示例科技有限公司", 0.98),
        OcrLine("类型：有限责任公司", 0.98),
        OcrLine("法定代表人：李四", 0.98),
        OcrLine("成立日期：2020年05月01日", 0.98),
    ]
    skill = create_candidate_registry().get("business.license", FakeEngine(lines=lines))
    result = skill.run(SkillInput())

    assert result.fields["credit_code"] == "91110108MA0000000X"
    assert result.fields["company_name"] == "北京示例科技有限公司"
    assert result.fields["establish_date"] == "2020-05-01"


def test_express_waybill_candidate_extracts_fake_sample_fields():
    lines = [
        OcrLine("顺丰速运", 0.98),
        OcrLine("运单号：SF1234567890", 0.98),
        OcrLine("寄件人：王五", 0.98),
        OcrLine("收件人：赵六", 0.98),
        OcrLine("收件电话：138****8888", 0.98),
    ]
    skill = create_candidate_registry().get("logistics.express_waybill", FakeEngine(lines=lines))
    result = skill.run(SkillInput())

    assert result.fields["waybill_no"] == "SF1234567890"
    assert result.fields["sender"] == "王五"
    assert result.fields["receiver"] == "赵六"
