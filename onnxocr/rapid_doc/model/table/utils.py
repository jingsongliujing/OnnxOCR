from bs4 import BeautifulSoup

def count_table_cells_physical(html_code):
    """计算表格的物理单元格数量（合并单元格算一个）"""
    if not html_code:
        return 0

    # 简单计数td和th标签的数量
    html_lower = html_code.lower()
    td_count = html_lower.count('<td')
    th_count = html_lower.count('<th')
    return td_count + th_count

def select_best_table_model(ocr_result, wired_html_code, wireless_html_code):
    wired_html_code = wired_html_code or ""
    wireless_html_code = wireless_html_code or ""

    wired_len = count_table_cells_physical(wired_html_code)
    wireless_len = count_table_cells_physical(wireless_html_code)
    # 计算两种模型检测的单元格数量差异
    gap_of_len = wireless_len - wired_len
    # logger.debug(f"wired table cell bboxes: {wired_len}, wireless table cell bboxes: {wireless_len}")

    # 使用OCR结果计算两种模型填入的文字数量
    wireless_text_count = 0
    wired_text_count = 0
    for text in ocr_result[1]:
        if text in wireless_html_code:
            wireless_text_count += 1
        if text in wired_html_code:
            wired_text_count += 1
    # logger.debug(f"wireless table ocr text count: {wireless_text_count}, wired table ocr text count: {wired_text_count}")

    # 使用HTML解析器计算空单元格数量
    wireless_soup = BeautifulSoup(wireless_html_code, 'html.parser') if wireless_html_code else BeautifulSoup("",
                                                                                                              'html.parser')
    wired_soup = BeautifulSoup(wired_html_code, 'html.parser') if wired_html_code else BeautifulSoup("", 'html.parser')
    # 计算空单元格数量(没有文本内容或只有空白字符)
    wireless_blank_count = sum(1 for cell in wireless_soup.find_all(['td', 'th']) if not cell.text.strip())
    wired_blank_count = sum(1 for cell in wired_soup.find_all(['td', 'th']) if not cell.text.strip())
    # logger.debug(f"wireless table blank cell count: {wireless_blank_count}, wired table blank cell count: {wired_blank_count}")

    # 计算非空单元格数量
    wireless_non_blank_count = wireless_len - wireless_blank_count
    wired_non_blank_count = wired_len - wired_blank_count
    # 无线表非空格数量大于有线表非空格数量时，才考虑切换
    switch_flag = False
    if wireless_non_blank_count > wired_non_blank_count:
        # 假设非空表格是接近正方表，使用非空单元格数量开平方作为表格规模的估计
        wired_table_scale = round(wired_non_blank_count ** 0.5)
        # logger.debug(f"wireless non-blank cell count: {wireless_non_blank_count}, wired non-blank cell count: {wired_non_blank_count}, wired table scale: {wired_table_scale}")
        # 如果无线表非空格的数量比有线表多一列或以上，需要切换到无线表
        wired_scale_plus_2_cols = wired_non_blank_count + (wired_table_scale * 2)
        wired_scale_squared_plus_2_rows = wired_table_scale * (wired_table_scale + 2)
        if (wireless_non_blank_count + 3) >= max(wired_scale_plus_2_cols, wired_scale_squared_plus_2_rows):
            switch_flag = True

    # 判断是否使用无线表格模型的结果
    if (
            switch_flag
            or (0 <= gap_of_len <= 5 and wired_len <= round(wireless_len * 0.75))  # 两者相差不大但有线模型结果较少
            or (gap_of_len == 0 and wired_len <= 4)  # 单元格数量完全相等且总量小于等于4
            or (wired_text_count <= wireless_text_count * 0.6 and wireless_text_count >= 10)  # 有线模型填入的文字明显少于无线模型
    ):
        # logger.debug("fall back to wireless table model")
        html_code = wireless_html_code
    else:
        html_code = wired_html_code

    return html_code