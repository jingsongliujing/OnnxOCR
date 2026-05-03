import json
from io import BytesIO

from loguru import logger
from pypdf import PdfReader, PdfWriter, PageObject
from reportlab.pdfgen import canvas

from onnxocr.rapid_doc.data.data_reader_writer import DataWriter
from .enum_class import BlockType, ContentType, SplitFlag


def cal_canvas_rect(page, bbox):
    """
    Calculate the rectangle coordinates on the canvas based on the original PDF page and bounding box.

    Args:
        page: A PyPDF2 Page object representing a single page in the PDF.
        bbox: [x0, y0, x1, y1] representing the bounding box coordinates.

    Returns:
        rect: [x0, y0, width, height] representing the rectangle coordinates on the canvas.
    """
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
    
    actual_width = page_width    # The width of the final PDF display
    actual_height = page_height  # The height of the final PDF display
    
    rotation_obj = page.get("/Rotate", 0)
    try:
        rotation = int(rotation_obj) % 360  # cast rotation to int to handle IndirectObject
    except (ValueError, TypeError) as e:
        logger.warning(f"Invalid /Rotate value {rotation_obj!r} on page; defaulting to 0. Error: {e}")
        rotation = 0
    
    if rotation in [90, 270]:
        # PDF is rotated 90 degrees or 270 degrees, and the width and height need to be swapped
        actual_width, actual_height = actual_height, actual_width
        
    x0, y0, x1, y1 = bbox
    rect_w = abs(x1 - x0)
    rect_h = abs(y1 - y0)
    
    if rotation == 270:
        rect_w, rect_h = rect_h, rect_w
        x0 = actual_height - y1
        y0 = actual_width - x1
    elif rotation == 180:
        x0 = page_width - x1
        y0 = y0
        # y0 stays the same
    elif rotation == 90:
        rect_w, rect_h = rect_h, rect_w
        x0, y0 = y0, x0 
    else:
        # rotation == 0
        y0 = page_height - y1
    
    rect = [x0, y0, rect_w, rect_h]        
    return rect


def cal_canvas_polygon(page, polygon_points):
    """
    Calculate the polygon coordinates on the canvas based on the original PDF page and polygon points.

    Args:
        page: A PyPDF2 Page object representing a single page in the PDF.
        polygon_points: List of [x, y] points representing the polygon vertices.

    Returns:
        List of [x, y] points in canvas coordinates.
    """
    page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
    
    rotation_obj = page.get("/Rotate", 0)
    try:
        rotation = int(rotation_obj) % 360
    except (ValueError, TypeError):
        rotation = 0
    
    if rotation in [90, 270]:
        page_width, page_height = page_height, page_width
    
    canvas_points = []
    for x, y in polygon_points:
        if rotation == 270:
            cx = page_height - y
            cy = page_width - x
        elif rotation == 180:
            cx = page_width - x
            cy = y
        elif rotation == 90:
            cx = y
            cy = x
        else:  # rotation == 0
            cx = x
            cy = page_height - y
        canvas_points.append([cx, cy])
    
    return canvas_points


def draw_polygon(c, polygon_points, rgb_config, fill_config):
    """
    Draw a polygon on the canvas.
    
    Args:
        c: reportlab canvas object
        polygon_points: List of [x, y] points in canvas coordinates
        rgb_config: RGB color values (0-1 range)
        fill_config: Whether to fill the polygon
    """
    if len(polygon_points) < 3:
        return
    
    path = c.beginPath()
    path.moveTo(polygon_points[0][0], polygon_points[0][1])
    for point in polygon_points[1:]:
        path.lineTo(point[0], point[1])
    path.close()
    
    if fill_config:
        c.setFillColorRGB(rgb_config[0], rgb_config[1], rgb_config[2], 0.3)
        c.drawPath(path, stroke=0, fill=1)
    else:
        c.setStrokeColorRGB(rgb_config[0], rgb_config[1], rgb_config[2])
        c.drawPath(path, stroke=1, fill=0)


def draw_bbox_without_number(i, bbox_list, page, c, rgb_config, fill_config):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]

    for item in page_data:
        # 支持两种格式: bbox 列表或包含 polygon_points 的字典
        if isinstance(item, dict):
            bbox = item.get('bbox')
            polygon_points = item.get('polygon_points')
        else:
            bbox = item
            polygon_points = None
        
        # 优先使用多边形绘制
        if polygon_points is not None and len(polygon_points) >= 3:
            canvas_polygon = cal_canvas_polygon(page, polygon_points)
            draw_polygon(c, canvas_polygon, new_rgb, fill_config)
        elif bbox is not None:
            rect = cal_canvas_rect(page, bbox)
            if fill_config:
                c.setFillColorRGB(new_rgb[0], new_rgb[1], new_rgb[2], 0.3)
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
            else:
                c.setStrokeColorRGB(new_rgb[0], new_rgb[1], new_rgb[2])
                c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
    return c


def draw_bbox_with_number(i, bbox_list, page, c, rgb_config, fill_config, draw_bbox=True):
    new_rgb = [float(color) / 255 for color in rgb_config]
    page_data = bbox_list[i]

    for j, item in enumerate(page_data):
        # 支持两种格式: bbox 列表或包含 polygon_points 的字典
        if isinstance(item, dict):
            bbox = item.get('bbox')
            polygon_points = item.get('polygon_points')

            restore_scale = 200 / 72
            if polygon_points is not None:
                polygon_points = [
                    [float(x / restore_scale), float(y / restore_scale)]
                    for x, y in polygon_points
                ]
        else:
            bbox = item
            polygon_points = None
        
        if bbox is None:
            continue
            
        rect = cal_canvas_rect(page, bbox)
        
        if draw_bbox:
            # 优先使用多边形绘制
            if polygon_points is not None and len(polygon_points) >= 3:
                canvas_polygon = cal_canvas_polygon(page, polygon_points)
                draw_polygon(c, canvas_polygon, new_rgb, fill_config)
            else:
                if fill_config:
                    c.setFillColorRGB(*new_rgb, 0.3)
                    c.rect(rect[0], rect[1], rect[2], rect[3], stroke=0, fill=1)
                else:
                    c.setStrokeColorRGB(*new_rgb)
                    c.rect(rect[0], rect[1], rect[2], rect[3], stroke=1, fill=0)
        
        c.setFillColorRGB(*new_rgb, 1.0)
        c.setFontSize(size=10)
        
        c.saveState()
        rotation_obj = page.get("/Rotate", 0)
        try:
            rotation = int(rotation_obj) % 360
        except (ValueError, TypeError):
            logger.warning(f"Invalid /Rotate value: {rotation_obj!r}, defaulting to 0")
            rotation = 0

        if rotation == 0:
            c.translate(rect[0] + rect[2] + 2, rect[1] + rect[3] - 10)
        elif rotation == 90:
            c.translate(rect[0] + 10, rect[1] + rect[3] + 2)
        elif rotation == 180:
            c.translate(rect[0] - 2, rect[1] + 10)
        elif rotation == 270:
            c.translate(rect[0] + rect[2] - 10, rect[1] - 2)
            
        c.rotate(rotation)
        c.drawString(0, 0, str(j + 1))
        c.restoreState()

    return c


def _layout_item(bbox, polygon_points=None):
    """统一 layout 项格式：bbox 与 polygon_points 同级，供画框使用。"""
    if polygon_points is not None and len(polygon_points) >= 3:
        return {"bbox": bbox, "polygon_points": polygon_points}
    return {"bbox": bbox, "polygon_points": None}


def draw_layout_bbox(pdf_info, pdf_bytes, out_path: str | DataWriter, filename):
    dropped_bbox_list = []
    tables_body_list, tables_caption_list, tables_footnote_list = [], [], []
    imgs_body_list, imgs_caption_list, imgs_footnote_list = [], [], []
    codes_body_list, codes_caption_list = [], []
    titles_list = []
    texts_list = []
    interequations_list = []
    lists_list = []
    list_items_list = []
    indexs_list = []

    for page in pdf_info:
        page_dropped_list = []
        tables_body, tables_caption, tables_footnote = [], [], []
        imgs_body, imgs_caption, imgs_footnote = [], [], []
        codes_body, codes_caption = [], []
        titles = []
        texts = []
        interequations = []
        lists = []
        list_items = []
        indices = []

        for dropped_bbox in page['discarded_blocks']:
            page_dropped_list.append(_layout_item(dropped_bbox["bbox"], dropped_bbox.get("polygon_points")))
        dropped_bbox_list.append(page_dropped_list)

        for block in page["para_blocks"]:
            bbox = block["bbox"]
            poly = block.get("polygon_points")
            if block["type"] == BlockType.TABLE:
                for nested_block in block["blocks"]:
                    nb_bbox = nested_block["bbox"]
                    nb_poly = nested_block.get("polygon_points")
                    item = _layout_item(nb_bbox, nb_poly)
                    if nested_block["type"] == BlockType.TABLE_BODY:
                        tables_body.append(item)
                    elif nested_block["type"] == BlockType.TABLE_CAPTION:
                        tables_caption.append(item)
                    elif nested_block["type"] == BlockType.TABLE_FOOTNOTE:
                        if nested_block.get(SplitFlag.CROSS_PAGE, False):
                            continue
                        tables_footnote.append(item)
            elif block["type"] == BlockType.IMAGE:
                for nested_block in block["blocks"]:
                    nb_bbox = nested_block["bbox"]
                    nb_poly = nested_block.get("polygon_points")
                    item = _layout_item(nb_bbox, nb_poly)
                    if nested_block["type"] == BlockType.IMAGE_BODY:
                        imgs_body.append(item)
                    elif nested_block["type"] == BlockType.IMAGE_CAPTION:
                        imgs_caption.append(item)
                    elif nested_block["type"] == BlockType.IMAGE_FOOTNOTE:
                        imgs_footnote.append(item)
            elif block["type"] == BlockType.CODE:
                for nested_block in block["blocks"]:
                    nb_bbox = nested_block["bbox"]
                    nb_poly = nested_block.get("polygon_points")
                    item = _layout_item(nb_bbox, nb_poly)
                    if nested_block["type"] == BlockType.CODE_BODY:
                        codes_body.append(item)
                    elif nested_block["type"] == BlockType.CODE_CAPTION:
                        codes_caption.append(item)
            elif block["type"] == BlockType.TITLE:
                titles.append(_layout_item(bbox, poly))
            elif block["type"] in [BlockType.TEXT, BlockType.REF_TEXT]:
                texts.append(_layout_item(bbox, poly))
            elif block["type"] == BlockType.INTERLINE_EQUATION:
                interequations.append(_layout_item(bbox, poly))
            elif block["type"] == BlockType.LIST:
                lists.append(_layout_item(bbox, poly))
                if "blocks" in block:
                    for sub_block in block["blocks"]:
                        sp_bbox = sub_block.get("bbox")
                        sp_poly = sub_block.get("polygon_points")
                        list_items.append(_layout_item(sp_bbox, sp_poly))
            elif block["type"] == BlockType.INDEX:
                indices.append(_layout_item(bbox, poly))

        tables_body_list.append(tables_body)
        tables_caption_list.append(tables_caption)
        tables_footnote_list.append(tables_footnote)
        imgs_body_list.append(imgs_body)
        imgs_caption_list.append(imgs_caption)
        imgs_footnote_list.append(imgs_footnote)
        titles_list.append(titles)
        texts_list.append(texts)
        interequations_list.append(interequations)
        lists_list.append(lists)
        list_items_list.append(list_items)
        indexs_list.append(indices)
        codes_body_list.append(codes_body)
        codes_caption_list.append(codes_caption)

    layout_bbox_list = []
    inner_layout_bbox_list = []
    table_type_order = {"table_caption": 1, "table_body": 2, "table_footnote": 3}
    for page in pdf_info:
        page_block_list = []
        page_inner_list = []
        for block in page["para_blocks"]:
            bbox = block["bbox"]
            poly = block.get("polygon_points")
            if block["type"] in [
                BlockType.TEXT,
                BlockType.REF_TEXT,
                BlockType.TITLE,
                BlockType.INTERLINE_EQUATION,
                BlockType.LIST,
                BlockType.INDEX,
            ]:
                page_block_list.append(_layout_item(bbox, poly))
            elif block["type"] in [BlockType.IMAGE]:
                for sub_block in block["blocks"]:
                    sb_bbox = sub_block["bbox"]
                    sb_poly = sub_block.get("polygon_points")
                    page_block_list.append(_layout_item(sb_bbox, sb_poly))
            elif block["type"] in [BlockType.TABLE]:
                sorted_blocks = sorted(block["blocks"], key=lambda x: table_type_order.get(x["type"], 0))
                for sub_block in sorted_blocks:
                    if sub_block.get(SplitFlag.CROSS_PAGE, False):
                        continue
                    sb_bbox = sub_block["bbox"]
                    sb_poly = sub_block.get("polygon_points")
                    page_block_list.append(_layout_item(sb_bbox, sb_poly))
                    # 表格内的图片和公式
                    for line in sub_block.get("lines", []):
                        for span in line.get("spans", []):
                            if "img_boxes" in span:
                                page_inner_list.extend(span["img_boxes"])
                            if "latex_boxes" in span:
                                page_inner_list.extend(span["latex_boxes"])
            elif block["type"] in [BlockType.CODE]:
                for sub_block in block["blocks"]:
                    sb_bbox = sub_block["bbox"]
                    sb_poly = sub_block.get("polygon_points")
                    page_block_list.append(_layout_item(sb_bbox, sb_poly))

        layout_bbox_list.append(page_block_list)
        inner_layout_bbox_list.append(page_inner_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        c = draw_bbox_without_number(i, codes_body_list, page, c, [102, 0, 204], True)
        c = draw_bbox_without_number(i, codes_caption_list, page, c, [204, 153, 255], True)
        c = draw_bbox_without_number(i, dropped_bbox_list, page, c, [158, 158, 158], True)
        c = draw_bbox_without_number(i, tables_body_list, page, c, [204, 204, 0], True)
        c = draw_bbox_without_number(i, tables_caption_list, page, c, [255, 255, 102], True)
        c = draw_bbox_without_number(i, tables_footnote_list, page, c, [229, 255, 204], True)
        c = draw_bbox_without_number(i, imgs_body_list, page, c, [153, 255, 51], True)
        c = draw_bbox_without_number(i, imgs_caption_list, page, c, [102, 178, 255], True)
        c = draw_bbox_without_number(i, imgs_footnote_list, page, c, [255, 178, 102], True)
        c = draw_bbox_without_number(i, titles_list, page, c, [102, 102, 255], True)
        c = draw_bbox_without_number(i, texts_list, page, c, [153, 0, 76], True)
        c = draw_bbox_without_number(i, interequations_list, page, c, [0, 255, 0], True)
        c = draw_bbox_without_number(i, lists_list, page, c, [40, 169, 92], True)
        c = draw_bbox_without_number(i, list_items_list, page, c, [40, 169, 92], False)
        c = draw_bbox_without_number(i, indexs_list, page, c, [40, 169, 92], True)
        c = draw_bbox_with_number(i, layout_bbox_list, page, c, [255, 0, 0], False, draw_bbox=False)
        c = draw_bbox_without_number(i, inner_layout_bbox_list, page, c,  [0, 255, 0], False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"layout.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # 保存结果
    if isinstance(out_path, DataWriter):
        buffer = BytesIO()
        output_pdf.write(buffer)
        pdf_bytes = buffer.getvalue()
        out_path.write(
            f"{filename}",
            pdf_bytes,
        )
    else:
        with open(f"{out_path}/{filename}", "wb") as f:
            output_pdf.write(f)


def draw_span_bbox(pdf_info, pdf_bytes, out_path: str | DataWriter, filename):
    text_list = []
    inline_equation_list = []
    interline_equation_list = []
    image_list = []
    table_list = []
    dropped_list = []

    def get_span_info(span):
        # span 级：普通 OCR 为行级框，与 layout 的 block 级区分；bbox 与 polygon_points 同级
        item = _layout_item(span["bbox"], span.get("polygon_points"))
        if span['type'] == ContentType.TEXT:
            page_text_list.append(item)
        elif span['type'] == ContentType.INLINE_EQUATION:
            page_inline_equation_list.append(item)
        elif span['type'] == ContentType.INTERLINE_EQUATION:
            page_interline_equation_list.append(item)
        elif span['type'] == ContentType.CHECKBOX:
            page_inline_equation_list.append(item)
        elif span['type'] == ContentType.IMAGE:
            page_image_list.append(item)
        elif span['type'] == ContentType.TABLE:
            page_table_list.append(item)

    for page in pdf_info:
        page_text_list = []
        page_inline_equation_list = []
        page_interline_equation_list = []
        page_image_list = []
        page_table_list = []
        page_dropped_list = []


        # 构造 dropped_list（span 级，与正文 span 一致）
        for block in page['discarded_blocks']:
            if block['type'] == BlockType.DISCARDED:
                for line in block['lines']:
                    for span in line['spans']:
                        page_dropped_list.append(_layout_item(span["bbox"], span.get("polygon_points")))
        dropped_list.append(page_dropped_list)
        # 构造其余useful_list
        # for block in page['para_blocks']:  # span直接用分段合并前的结果就可以
        for block in page['preproc_blocks']:
            if block['type'] in [
                BlockType.TEXT,
                BlockType.TITLE,
                BlockType.INTERLINE_EQUATION,
                BlockType.LIST,
                BlockType.INDEX,
            ]:
                for line in block['lines']:
                    for span in line['spans']:
                        get_span_info(span)
            elif block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
                for sub_block in block['blocks']:
                    for line in sub_block['lines']:
                        for span in line['spans']:
                            get_span_info(span)
        text_list.append(page_text_list)
        inline_equation_list.append(page_inline_equation_list)
        interline_equation_list.append(page_interline_equation_list)
        image_list.append(page_image_list)
        table_list.append(page_table_list)

    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        # 获取当前页面的数据
        draw_bbox_without_number(i, text_list, page, c,[255, 0, 0], False)
        draw_bbox_without_number(i, inline_equation_list, page, c, [0, 255, 0], False)
        draw_bbox_without_number(i, interline_equation_list, page, c, [0, 0, 255], False)
        draw_bbox_without_number(i, image_list, page, c, [255, 204, 0], False)
        draw_bbox_without_number(i, table_list, page, c, [204, 0, 255], False)
        draw_bbox_without_number(i, dropped_list, page, c, [158, 158, 158], False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"span.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # Save the PDF
    if isinstance(out_path, DataWriter):
        buffer = BytesIO()
        output_pdf.write(buffer)
        pdf_bytes = buffer.getvalue()
        out_path.write(
            f"{filename}",
            pdf_bytes,
        )
    else:
        with open(f"{out_path}/{filename}", "wb") as f:
            output_pdf.write(f)


def draw_line_sort_bbox(pdf_info, pdf_bytes, out_path, filename):
    layout_bbox_list = []

    for page in pdf_info:
        page_line_list = []
        for block in page['preproc_blocks']:
            if block['type'] in [BlockType.TEXT]:
                for line in block['lines']:
                    bbox = line['bbox']
                    index = line['index']
                    page_line_list.append({'index': index, 'bbox': bbox})
            elif block['type'] in [BlockType.TITLE, BlockType.INTERLINE_EQUATION]:
                if 'virtual_lines' in block:
                    if len(block['virtual_lines']) > 0 and block['virtual_lines'][0].get('index', None) is not None:
                        for line in block['virtual_lines']:
                            bbox = line['bbox']
                            index = line['index']
                            page_line_list.append({'index': index, 'bbox': bbox})
                else:
                    for line in block['lines']:
                        bbox = line['bbox']
                        index = line['index']
                        page_line_list.append({'index': index, 'bbox': bbox})
            elif block['type'] in [BlockType.IMAGE, BlockType.TABLE]:
                for sub_block in block['blocks']:
                    if sub_block['type'] in [BlockType.IMAGE_BODY, BlockType.TABLE_BODY]:
                        if len(sub_block['virtual_lines']) > 0 and sub_block['virtual_lines'][0].get('index', None) is not None:
                            for line in sub_block['virtual_lines']:
                                bbox = line['bbox']
                                index = line['index']
                                page_line_list.append({'index': index, 'bbox': bbox})
                        else:
                            for line in sub_block['lines']:
                                bbox = line['bbox']
                                index = line['index']
                                page_line_list.append({'index': index, 'bbox': bbox})
                    elif sub_block['type'] in [BlockType.IMAGE_CAPTION, BlockType.TABLE_CAPTION, BlockType.IMAGE_FOOTNOTE, BlockType.TABLE_FOOTNOTE]:
                        for line in sub_block['lines']:
                            bbox = line['bbox']
                            index = line['index']
                            page_line_list.append({'index': index, 'bbox': bbox})
        sorted_bboxes = sorted(page_line_list, key=lambda x: x['index'])
        layout_bbox_list.append(sorted_bbox['bbox'] for sorted_bbox in sorted_bboxes)
    pdf_bytes_io = BytesIO(pdf_bytes)
    pdf_docs = PdfReader(pdf_bytes_io)
    output_pdf = PdfWriter()

    for i, page in enumerate(pdf_docs.pages):
        # 获取原始页面尺寸
        page_width, page_height = float(page.cropbox[2]), float(page.cropbox[3])
        custom_page_size = (page_width, page_height)

        packet = BytesIO()
        # 使用原始PDF的尺寸创建canvas
        c = canvas.Canvas(packet, pagesize=custom_page_size)

        # 获取当前页面的数据
        draw_bbox_with_number(i, layout_bbox_list, page, c, [255, 0, 0], False)

        c.save()
        packet.seek(0)
        overlay_pdf = PdfReader(packet)

        # 添加检查确保overlay_pdf.pages不为空
        if len(overlay_pdf.pages) > 0:
            new_page = PageObject(pdf=None)
            new_page.update(page)
            page = new_page
            page.merge_page(overlay_pdf.pages[0])
        else:
            # 记录日志并继续处理下一个页面
            # logger.warning(f"span.pdf: 第{i + 1}页未能生成有效的overlay PDF")
            pass

        output_pdf.add_page(page)

    # Save the PDF
    with open(f"{out_path}/{filename}", "wb") as f:
        output_pdf.write(f)

if __name__ == "__main__":
    # 读取PDF文件
    pdf_path = "examples/demo1.pdf"
    with open(pdf_path, "rb") as f:
        pdf_bytes = f.read()

    # 从json文件读取pdf_info

    json_path = "examples/demo1_1746005777.0863056_middle.json"
    with open(json_path, "r", encoding="utf-8") as f:
        pdf_ann = json.load(f)
    pdf_info = pdf_ann["pdf_info"]
    # 调用可视化函数,输出到examples目录
    draw_layout_bbox(pdf_info, pdf_bytes, "examples", "output_with_layout.pdf")
