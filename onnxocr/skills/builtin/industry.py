from __future__ import annotations

from typing import Dict

from onnxocr.skills.extractors import FieldSpec
from onnxocr.skills.template_skill import TemplateOCRSkill, TemplateSpec


def create_agriculture_quality_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="agriculture.quality_inspection",
            name="农产品质检单 OCR",
            description="面向农产品质检单、入库单和溯源标签的字段抽取模板。",
            domains=["农业", "质检", "溯源"],
            fields=[
                FieldSpec("product_name", labels=["品名", "产品名称", "农产品", "Product Name", "Product"], required=True),
                FieldSpec("batch_no", labels=["批次", "批号", "追溯码", "Batch No", "Batch"], patterns=[r"(?:批次|批号|追溯码|Batch No|Batch)[:：\s]*([A-Z0-9-]+)"]),
                FieldSpec("inspection_date", labels=["检验日期", "检测日期", "Inspection Date", "Test Date"], patterns=[r"(?:检验|检测)日期[:：\s]*([0-9年月日./-]+)", r"(?:Inspection Date|Test Date)[:：\s]*([0-9./-]+)"], normalizer="date"),
                FieldSpec("result", labels=["检验结果", "检测结果", "结论", "Result", "Conclusion"], required=True),
                FieldSpec("supplier", labels=["供应商", "基地", "产地", "Supplier", "Origin"]),
            ],
        ),
    )


def create_agriculture_traceability_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="agriculture.traceability_label",
            name="农产品溯源标签 OCR",
            description="面向农产品包装、二维码旁标签、产地溯源标签的字段抽取模板。",
            domains=["农业", "溯源", "标签"],
            fields=[
                FieldSpec("trace_code", labels=["追溯码", "溯源码", "Trace Code", "Trace No"], patterns=[r"(?:追溯码|溯源码|Trace Code|Trace No)[:：\s]*([A-Z0-9-]+)"], required=True),
                FieldSpec("product_name", labels=["产品名称", "品名", "Product Name", "Product"], required=True),
                FieldSpec("origin", labels=["产地", "基地", "Origin", "Place of Origin"]),
                FieldSpec("producer", labels=["生产者", "生产企业", "Producer", "Manufacturer"]),
                FieldSpec("harvest_date", labels=["采摘日期", "生产日期", "Harvest Date", "Production Date"], patterns=[r"(?:采摘|生产)日期[:：\s]*([0-9年月日./-]+)", r"(?:Harvest Date|Production Date)[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_agriculture_plant_protection_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="agriculture.plant_protection_record",
            name="植保作业记录 OCR",
            description="面向植保单据、农药使用记录、田间作业记录的字段抽取模板。",
            domains=["农业", "植保", "作业记录"],
            fields=[
                FieldSpec("crop", labels=["作物", "农作物", "Crop"], required=True),
                FieldSpec("pesticide", labels=["农药名称", "药剂名称", "Pesticide"], required=True),
                FieldSpec("dosage", labels=["用量", "剂量", "Dosage"], patterns=[r"(?:用量|剂量|Dosage)[:：\s]*([0-9.]+\s*\S*)"]),
                FieldSpec("operator", labels=["作业人", "操作人", "Operator"]),
                FieldSpec("operation_date", labels=["作业日期", "用药日期", "Operation Date"], patterns=[r"(?:作业|用药)日期[:：\s]*([0-9年月日./-]+)", r"Operation Date[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_oa_reimbursement_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="oa.reimbursement",
            name="企业报销单 OCR",
            description="面向企业 OA 报销单、发票粘贴单和行政单据的字段抽取模板。",
            domains=["企业 OA", "财务", "行政"],
            fields=[
                FieldSpec("applicant", labels=["报销人", "申请人", "经办人", "Applicant", "Employee"], required=True),
                FieldSpec("department", labels=["部门", "所属部门", "Department"]),
                FieldSpec("amount", labels=["金额", "报销金额", "合计", "Amount", "Total"], patterns=[r"(?:报销金额|合计|金额|Amount|Total)[:：\s¥￥$]*([0-9,.]+)"], normalizer="amount", required=True),
                FieldSpec("date", labels=["日期", "申请日期", "报销日期", "Date"], patterns=[r"(?:申请|报销)?日期[:：\s]*([0-9年月日./-]+)", r"Date[:：\s]*([0-9./-]+)"], normalizer="date"),
                FieldSpec("purpose", labels=["用途", "事由", "摘要", "Purpose", "Summary"]),
            ],
        ),
    )


def create_finance_invoice_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="finance.invoice",
            name="发票 OCR",
            description="面向增值税发票、普通发票、电子发票截图的字段抽取模板。",
            domains=["财务", "发票", "税务"],
            fields=[
                FieldSpec("invoice_code", labels=["发票代码", "Invoice Code"], patterns=[r"(?:发票代码|Invoice Code)[:：\s]*([0-9]{8,12})"]),
                FieldSpec("invoice_no", labels=["发票号码", "Invoice No"], patterns=[r"(?:发票号码|Invoice No)[:：\s]*([0-9]{6,12})"], required=True),
                FieldSpec("invoice_date", labels=["开票日期", "Invoice Date"], patterns=[r"开票日期[:：\s]*([0-9年月日./-]+)", r"Invoice Date[:：\s]*([0-9./-]+)"], normalizer="date"),
                FieldSpec("buyer", labels=["购买方", "购方名称", "Buyer"]),
                FieldSpec("seller", labels=["销售方", "销方名称", "Seller"]),
                FieldSpec("amount", labels=["价税合计", "合计金额", "Amount", "Total"], patterns=[r"(?:价税合计|合计金额|Amount|Total)[:：\s¥￥$]*([0-9,.]+)"], normalizer="amount", required=True),
            ],
        ),
    )


def create_contract_key_info_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="legal.contract_key_info",
            name="合同关键信息 OCR",
            description="面向采购合同、服务合同、租赁合同的合同编号、主体、金额和签署日期抽取模板。",
            domains=["法务", "合同", "企业管理"],
            fields=[
                FieldSpec("contract_no", labels=["合同编号", "Contract No"], patterns=[r"(?:合同编号|Contract No)[:：\s]*([A-Z0-9-]+)"], required=True),
                FieldSpec("party_a", labels=["甲方", "委托方", "Party A"], required=True),
                FieldSpec("party_b", labels=["乙方", "受托方", "Party B"], required=True),
                FieldSpec("amount", labels=["合同金额", "总金额", "Amount"], patterns=[r"(?:合同金额|总金额|Amount)[:：\s¥￥$]*([0-9,.]+)"], normalizer="amount"),
                FieldSpec("sign_date", labels=["签订日期", "签署日期", "Sign Date"], patterns=[r"(?:签订|签署)日期[:：\s]*([0-9年月日./-]+)", r"Sign Date[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_government_red_head_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="government.red_head_document",
            name="红头文件 OCR",
            description="面向红头文件、公文通知、行政批复的文号、标题、发文机关和日期抽取模板。",
            domains=["政务", "公文", "行政"],
            fields=[
                FieldSpec("document_no", labels=["文号", "发文字号", "Document No"], patterns=[r"(?:文号|发文字号|Document No)[:：\s]*([A-Za-z0-9〔〕\[\]()-]+)"]),
                FieldSpec("issuer", labels=["发文机关", "发布单位", "Issuer"], required=True),
                FieldSpec("title", labels=["标题", "Title"], required=True),
                FieldSpec("date", labels=["发文日期", "日期", "Date"], patterns=[r"(?:发文)?日期[:：\s]*([0-9年月日./-]+)", r"Date[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_education_exam_paper_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="education.exam_paper",
            name="试卷信息 OCR",
            description="面向中小学试卷、练习卷、检测卷的标题、年级、科目、考试时间和满分抽取模板。",
            domains=["教育", "试卷", "教培"],
            fields=[
                FieldSpec("title", labels=["标题", "试卷名称"], patterns=[r"([0-9]{4}年[^\n]{0,40}(?:检测卷|试卷|练习卷))"], required=True),
                FieldSpec("grade", labels=["年级"], patterns=[r"([一二三四五六七八九十0-9]+年级)"], required=True),
                FieldSpec("subject", labels=["科目"], patterns=[r"(语文|数学|英语|物理|化学|生物|历史|地理|道德与法治|科学)"], required=True),
                FieldSpec("time_limit", labels=["时间", "考试时间"], patterns=[r"时间[:：]?\s*([0-9]+分钟)"]),
                FieldSpec("total_score", labels=["满分"], patterns=[r"满分[:：]?\s*([0-9]+分)"]),
            ],
        ),
    )


def create_logistics_inbound_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="logistics.inbound_order",
            name="物流仓储入库单 OCR",
            description="面向快递面单、入库单、外采原果入库单据的字段抽取模板。",
            domains=["物流", "仓储", "入库"],
            fields=[
                FieldSpec("order_no", labels=["单号", "入库单号", "快递单号", "Order No", "Waybill No"], patterns=[r"(?:入库单号|快递单号|单号|Order No|Waybill No)[:：\s]*([A-Z0-9-]+)"], required=True),
                FieldSpec("sender", labels=["发货方", "寄件人", "供应商", "Sender", "Supplier"]),
                FieldSpec("receiver", labels=["收货方", "收件人", "仓库", "Receiver", "Warehouse"]),
                FieldSpec("product_name", labels=["货品", "品名", "物料名称", "Product", "Item"]),
                FieldSpec("quantity", labels=["数量", "件数", "重量", "Quantity", "Weight"], patterns=[r"(?:数量|件数|重量|Quantity|Weight)[:：\s]*([0-9.]+\s*\S*)"]),
            ],
        ),
    )


def create_transport_taxi_invoice_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="transport.taxi_invoice",
            name="出租车票 OCR",
            description="面向出租汽车通用机打发票的发票代码、发票号码、信息码、车号、日期、上下车时间、里程和金额抽取模板。",
            domains=["交通", "出租车票", "发票"],
            fields=[
                FieldSpec("invoice_code", labels=["发票代码"], patterns=[r"发票代码[:：\s]*([0-9]{8,12})"], required=True),
                FieldSpec("invoice_no", labels=["发票号码"], patterns=[r"发票号码[:：\s]*([0-9]{6,12})"], required=True),
                FieldSpec("info_code", labels=["信息码"], patterns=[r"信息码[:：\s]*([0-9]{6,12})"]),
                FieldSpec("company", labels=["公司"], patterns=[r"([^\n]{2,12}公司)"]),
                FieldSpec("plate_no", labels=["车号"], patterns=[r"车\s*号[:：]?\s*([京津沪渝冀豫云辽黑湘皖鲁新苏浙赣鄂桂甘晋蒙陕吉闽贵粤青藏川宁琼][A-Z0-9]{1,8})"]),
                FieldSpec("ride_date", labels=["日期"], patterns=[r"日期[:：]?\s*([0-9年月日./-]+)"], normalizer="date"),
                FieldSpec("start_time", labels=["上车"], patterns=[r"上\s*车[:：]?\s*([0-9]{1,2}[:：][0-9]{2})"]),
                FieldSpec("end_time", labels=["下车"], patterns=[r"(?:下\s*车|长)[:：]?\s*([0-9]{1,2}[:：][0-9]{2})"]),
                FieldSpec("distance", labels=["里程"], patterns=[r"(?:里\s*程|程)[:：]?\s*([0-9.]+ ?[Kk][Mm])"]),
                FieldSpec("amount", labels=["金额"], patterns=[r"金额[:：]?\s*([0-9.]+元?)"], normalizer="amount"),
            ],
        ),
    )


def create_transport_train_ticket_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="transport.train_ticket",
            name="火车票 OCR",
            description="面向铁路火车票、高铁票的出发站、到达站、车次、发车时间、席别、票价和票号抽取模板。",
            domains=["交通", "火车票", "高铁票"],
            fields=[
                FieldSpec("ticket_no", labels=["票号"], patterns=[r"^([A-Z0-9]{6,})"], required=True),
                FieldSpec("from_station", labels=["出发站"], patterns=[r"^[A-Z0-9]+\n([\u4e00-\u9fa5]{2,}站)"], required=True),
                FieldSpec("to_station", labels=["到达站"], patterns=[r"^[A-Z0-9]+\n[\u4e00-\u9fa5]{2,}站\n([\u4e00-\u9fa5]{2,}站)"], required=True),
                FieldSpec("train_no", labels=["车次"], patterns=[r"\b([GDCZTK][0-9]{1,4})\b"], required=True),
                FieldSpec("departure_time", labels=["发车时间"], patterns=[r"([0-9]{4}年[0-9]{2}月[0-9]{2}日[0-9]{1,2}[:：][0-9]{2})开"], required=True),
                FieldSpec("seat_no", labels=["座位号"], patterns=[r"([0-9]{1,2}车[0-9]{1,2}[A-Z]号)"]),
                FieldSpec("seat_class", labels=["席别"], patterns=[r"(商务座|一等座|二等座|硬座|软座|硬卧|软卧|无座)"]),
                FieldSpec("price", labels=["票价"], patterns=[r"[￥¥]\s*([0-9.]+元?)"], normalizer="amount", required=True),
                FieldSpec("sale_station", labels=["售票站"], patterns=[r"([^\n]{2,}售)$"]),
            ],
        ),
    )


def create_medical_lab_report_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="medical.lab_report",
            name="检验报告 OCR",
            description="面向医院检验报告、体检报告、第三方检测报告的基础字段抽取模板。",
            domains=["医疗", "检验报告", "体检"],
            fields=[
                FieldSpec("patient_name", labels=["姓名", "患者姓名", "Name"], required=True),
                FieldSpec("report_no", labels=["报告单号", "样本编号", "Report No", "Sample No"], patterns=[r"(?:报告单号|样本编号|Report No|Sample No)[:：\s]*([A-Z0-9-]+)"]),
                FieldSpec("department", labels=["科室", "Department"]),
                FieldSpec("test_item", labels=["检验项目", "项目名称", "Test Item"], required=True),
                FieldSpec("report_date", labels=["报告日期", "检验日期", "Report Date"], patterns=[r"(?:报告|检验)日期[:：\s]*([0-9年月日./-]+)", r"Report Date[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_identity_id_card_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="identity.id_card",
            name="中国公民身份证 OCR",
            description="面向中国居民身份证样张/脱敏图的姓名、性别、民族、出生日期、住址、身份证号抽取模板。",
            domains=["证照", "身份证", "实名认证"],
            fields=[
                FieldSpec("name", labels=["姓名"], required=True),
                FieldSpec("gender", labels=["性别"], patterns=[r"性别[:：]?\s*(男|女)"], required=True),
                FieldSpec("nation", labels=["民族"], patterns=[r"民族[:：]?\s*([\u4e00-\u9fa5]{1,4})"]),
                FieldSpec(
                    "birth_date",
                    labels=["出生", "出生日期"],
                    patterns=[
                        r"出生[:：]?\s*([0-9]{4}年[0-9]{1,2}月[0-9]{1,2}日?)",
                        r"([12][0-9]{3}[年?/.-]?[01]?[0-9][月?/.-]?[0-3]?[0-9]日?)",
                    ],
                    normalizer="date",
                ),
                FieldSpec("address", labels=["住址", "地址"]),
                FieldSpec(
                    "id_no",
                    labels=["公民身份号码", "身份证号"],
                    patterns=[
                        r"(?:公民身份号码|身份证号)[:：]?\s*([0-9Xx]{15,18})",
                        r"\b([1-9][0-9]{5}(?:18|19|20)[0-9]{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12][0-9]|3[01])[0-9]{3}[0-9Xx])\b",
                    ],
                    required=True,
                ),
            ],
        ),
    )


def create_finance_bank_card_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="finance.bank_card",
            name="银行卡 OCR",
            description="面向银行卡样张/脱敏图的卡号、银行名称、卡组织和有效期抽取模板。",
            domains=["金融", "银行卡", "支付"],
            fields=[
                FieldSpec(
                    "card_no",
                    labels=["卡号", "银行卡号"],
                    patterns=[r"([0-9]{4}(?:[\s-]?[0-9]{3,4}){3,5})"],
                    normalizer="digits",
                    required=True,
                ),
                FieldSpec("bank_name", labels=["银行", "银行名称"], patterns=[r"([\u4e00-\u9fa5]{2,20}银行)"]),
                FieldSpec("valid_thru", labels=["有效期", "VALID THRU"], patterns=[r"(?:有效期|VALID\s*THRU)[:：]?\s*([0-9]{2}/[0-9]{2})"]),
                FieldSpec("card_type", labels=["卡类型"], patterns=[r"(借记卡|储蓄卡|信用卡|贷记卡|Debit|Credit)"]),
            ],
        ),
    )


def create_business_license_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="business.license",
            name="营业执照 OCR",
            description="面向营业执照样张/脱敏图的统一社会信用代码、名称、类型、法定代表人、住所、成立日期抽取模板。",
            domains=["企业", "营业执照", "工商"],
            fields=[
                FieldSpec("credit_code", labels=["统一社会信用代码"], patterns=[r"统一社会信用代码[:：]?\s*([0-9A-Z]{15,18})"], required=True),
                FieldSpec("company_name", labels=["名称", "企业名称"], required=True),
                FieldSpec("company_type", labels=["类型"]),
                FieldSpec("legal_representative", labels=["法定代表人", "负责人"]),
                FieldSpec("address", labels=["住所", "经营场所"]),
                FieldSpec("establish_date", labels=["成立日期"], patterns=[r"成立日期[:：]?\s*([0-9年月日./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_logistics_express_waybill_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="logistics.express_waybill",
            name="快递面单 OCR",
            description="面向快递面单样张/脱敏图的运单号、寄件人、收件人、地址、电话和快递公司抽取模板。",
            domains=["物流", "快递面单", "地址识别"],
            fields=[
                FieldSpec("waybill_no", labels=["运单号", "快递单号", "单号"], patterns=[r"(?:运单号|快递单号|单号)[:：]?\s*([A-Z0-9-]{8,30})"], required=True),
                FieldSpec("sender", labels=["寄件人", "寄方", "发件人"]),
                FieldSpec("receiver", labels=["收件人", "收方"]),
                FieldSpec("sender_phone", labels=["寄件电话"], patterns=[r"寄件电话[:：]?\s*([0-9* -]{7,20})"]),
                FieldSpec("receiver_phone", labels=["收件电话"], patterns=[r"收件电话[:：]?\s*([0-9* -]{7,20})"]),
                FieldSpec("address", labels=["地址", "收件地址"]),
                FieldSpec("company", labels=["快递公司"], patterns=[r"(顺丰|中通|圆通|申通|韵达|邮政|京东|极兔)[^\n]*"]),
            ],
        ),
    )


def create_vehicle_driving_license_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="vehicle.driving_license",
            name="行驶证 OCR",
            description="面向机动车行驶证主页字段的轻量抽取模板。",
            domains=["车辆", "证照", "行驶证"],
            fields=[
                FieldSpec("plate_no", labels=["号牌号码", "车牌号码", "Plate No"], required=True),
                FieldSpec("owner", labels=["所有人", "Owner"], required=True),
                FieldSpec("vehicle_type", labels=["车辆类型", "Vehicle Type"]),
                FieldSpec("vin", labels=["车辆识别代号", "车架号", "VIN"], patterns=[r"(?:车辆识别代号|车架号|VIN)[:：\s]*([A-Z0-9]{8,20})"]),
                FieldSpec("register_date", labels=["注册日期", "Register Date"], patterns=[r"注册日期[:：\s]*([0-9年月日./-]+)", r"Register Date[:：\s]*([0-9./-]+)"], normalizer="date"),
            ],
        ),
    )


def create_vehicle_driver_license_skill(ocr_engine):
    return TemplateOCRSkill(
        ocr_engine,
        TemplateSpec(
            skill_id="vehicle.driver_license",
            name="驾驶证 OCR",
            description="面向机动车驾驶证样张/脱敏图的姓名、证号、准驾车型、有效期限抽取模板。",
            domains=["车辆", "证照", "驾驶证"],
            fields=[
                FieldSpec("name", labels=["姓名"], required=True),
                FieldSpec("license_no", labels=["证号", "驾驶证号"], patterns=[r"(?:证号|驾驶证号)[:：]?\s*([0-9Xx]{15,18})"], required=True),
                FieldSpec("vehicle_class", labels=["准驾车型"], patterns=[r"准驾车型[:：]?\s*([A-Z0-9]+)"]),
                FieldSpec("valid_from", labels=["有效起始日期"], patterns=[r"有效起始日期[:：]?\s*([0-9年月日./-]+)"], normalizer="date"),
                FieldSpec("valid_for", labels=["有效期限"], patterns=[r"有效期限[:：]?\s*([0-9]+年|长期)"]),
            ],
        ),
    )


def create_vehicle_plate_skill(ocr_engine):
    return _VehiclePlateSkill(ocr_engine)


def create_table_structuring_skill(ocr_engine):
    return _TableStructuringSkill(ocr_engine)


def create_document_image_to_markdown_skill(ocr_engine):
    return _DocumentMarkdownSkill(
        ocr_engine,
        skill_id="document.image_to_markdown",
        name="图片转 Markdown",
        description="将论文截图、扫描图片、表格图片解析为 Markdown。",
        domains=["文档", "Markdown", "图片解析"],
    )


def create_document_pdf_to_markdown_skill(ocr_engine):
    return _DocumentMarkdownSkill(
        ocr_engine,
        skill_id="document.pdf_to_markdown",
        name="PDF 转 Markdown",
        description="将 PDF 文档解析为 Markdown。",
        domains=["文档", "PDF", "Markdown"],
    )


class _VehiclePlateSkill(TemplateOCRSkill):
    def __init__(self, ocr_engine):
        super().__init__(
            ocr_engine,
            TemplateSpec(
                skill_id="vehicle.plate",
                name="车牌识别 OCR",
                description="面向停车场、园区门禁、物流车辆登记等场景的车牌识别入口。",
                domains=["车辆", "车牌", "门禁"],
            ),
        )

    def run(self, skill_input):
        plates = self.ocr_engine.recognize_plate(skill_input)
        fields = {"plates": [item.get("plate") or item.get("plate_no") or item.get("text") for item in plates]}
        return self._make_output(fields, plates)

    def _make_output(self, fields: Dict, plates):
        from onnxocr.skills.base import FieldResult, SkillOutput

        field_results = [
            FieldResult(
                name="plates",
                value=fields["plates"],
                confidence=max([float(item.get("score", 0.0)) for item in plates] or [0.0]),
            )
        ]
        return SkillOutput(
            skill_id=self.skill_id,
            skill_name=self.name,
            fields=fields,
            field_results=field_results,
            raw_text="\n".join(str(value) for value in fields["plates"] if value),
            confidence=field_results[0].confidence,
            metadata={"domains": self.domains, "version": self.version, "raw": plates},
        )


class _TableStructuringSkill(TemplateOCRSkill):
    def __init__(self, ocr_engine):
        super().__init__(
            ocr_engine,
            TemplateSpec(
                skill_id="table.structure",
                name="表格结构化 OCR",
                description="将 Excel 截图、纸质表格或扫描件转成结构化 HTML、单元格框和逻辑行列数据。",
                domains=["表格", "Excel", "多维表格"],
            ),
        )

    def run(self, skill_input):
        from onnxocr.skills.base import FieldResult, SkillOutput

        table = self.ocr_engine.recognize_table(skill_input)
        fields = {
            "html": table.get("html", ""),
            "cell_bboxes": table.get("cell_bboxes", []),
            "logic_points": table.get("logic_points", []),
        }
        return SkillOutput(
            skill_id=self.skill_id,
            skill_name=self.name,
            fields=fields,
            field_results=[FieldResult(name="html", value=fields["html"], confidence=0.8 if fields["html"] else 0.0)],
            raw_text=fields["html"],
            confidence=0.8 if fields["html"] else 0.0,
            metadata={"domains": self.domains, "version": self.version, "raw": table},
        )


class _DocumentMarkdownSkill(TemplateOCRSkill):
    def __init__(self, ocr_engine, skill_id: str, name: str, description: str, domains):
        super().__init__(
            ocr_engine,
            TemplateSpec(
                skill_id=skill_id,
                name=name,
                description=description,
                domains=domains,
            ),
        )

    def run(self, skill_input):
        from pathlib import Path

        from onnxocr.layout_markdown import LayoutMarkdownConverter
        from onnxocr.skills.base import FieldResult, SkillOutput

        if not skill_input.image_path:
            raise ValueError("Document Markdown skill requires image_path.")
        source_path = Path(skill_input.image_path)
        output_dir = Path(skill_input.options.get("output_dir", "result_img"))
        output_md_path = output_dir / f"{source_path.stem}.md"
        converter = LayoutMarkdownConverter(
            layout_model_type="pp_doclayoutv2",
            formula_enable=False,
            table_enable=True,
        )
        result = converter.convert_file(str(source_path), output_md_path=str(output_md_path))
        fields = {
            "markdown_path": result["markdown_path"],
            "assets_dir": result["assets_dir"],
            "markdown_preview": result["markdown"][:500],
        }
        return SkillOutput(
            skill_id=self.skill_id,
            skill_name=self.name,
            fields=fields,
            field_results=[
                FieldResult(
                    name="markdown_path",
                    value=result["markdown_path"],
                    confidence=0.8 if result["markdown"].strip() else 0.0,
                )
            ],
            raw_text=result["markdown"],
            confidence=0.8 if result["markdown"].strip() else 0.0,
            metadata={"domains": self.domains, "version": self.version, "engine": result["engine"]},
        )
