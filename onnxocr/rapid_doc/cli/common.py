# Copyright (c) Opendatalab. All rights reserved.
import io
import os
import json
import copy
from pathlib import Path

from loguru import logger
import pypdfium2 as pdfium

from onnxocr.rapid_doc.utils import PyPDFium2Parser
from onnxocr.rapid_doc.data.data_reader_writer import FileBasedDataWriter
from onnxocr.rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox, draw_line_sort_bbox
from onnxocr.rapid_doc.utils.enum_class import MakeMode
from onnxocr.rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from onnxocr.rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes
from onnxocr.rapid_doc.backend.office.office_middle_json_mkcontent import union_make as office_union_make
from onnxocr.rapid_doc.backend.office.office_analyze import office_analyze
from onnxocr.rapid_doc.utils.config_reader import get_processing_window_size
from onnxocr.rapid_doc.utils.pdf_page_id import get_end_page_id

pdf_suffixes = ["pdf"]
image_suffixes = ["png", "jpeg", "jp2", "webp", "gif", "bmp", "jpg", "tiff"]
docx_suffixes = ["docx"]
pptx_suffixes = ["pptx"]
xlsx_suffixes = ["xlsx", "xlsm"]
office_suffixes = docx_suffixes + pptx_suffixes + xlsx_suffixes
old_office_suffixes = ["doc", "ppt", "xls"]

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def read_fn(path, file_suffix: str | None = None):
    if not isinstance(path, Path):
        path = Path(path)
    with open(str(path), "rb") as input_file:
        file_bytes = input_file.read()
        if file_suffix is None:
            file_suffix = guess_suffix_by_bytes(file_bytes, path)
        if file_suffix in image_suffixes:
            return images_bytes_to_pdf_bytes(file_bytes)
        elif file_suffix in pdf_suffixes + office_suffixes:
            return file_bytes
        else:
            raise Exception(f"Unknown file suffix: {file_suffix}")


def prepare_env(output_dir, pdf_file_name, parse_method):
    local_md_dir = str(os.path.join(output_dir, pdf_file_name, parse_method))
    local_image_dir = os.path.join(str(local_md_dir), "images")
    os.makedirs(local_image_dir, exist_ok=True)
    os.makedirs(local_md_dir, exist_ok=True)
    return local_image_dir, local_md_dir

def convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id=0, end_page_id=None):
    original_image = None
    if isinstance(pdf_bytes, dict):
        original_image = pdf_bytes.get("original_image")
        pdf_bytes = pdf_bytes["pdf_bytes"]
    pdf = None
    output_pdf = None
    try:
        with PyPDFium2Parser.lock:
            pdf = pdfium.PdfDocument(pdf_bytes)
            output_pdf = pdfium.PdfDocument.new()
            end_page_id = get_end_page_id(end_page_id, len(pdf))

            # 选择要导入的页面索引
            page_indices = list(range(start_page_id, end_page_id + 1))

            # 从原PDF导入页面到新PDF
            output_pdf.import_pages(pdf, page_indices)

            # 将新PDF保存到内存缓冲区
            output_buffer = io.BytesIO()
            output_pdf.save(output_buffer)

            # 获取字节数据
            output_bytes = output_buffer.getvalue()

    except Exception as e:
        logger.warning(f"Error in converting PDF bytes: {e}, Using original PDF bytes.")
        output_bytes = pdf_bytes

    with PyPDFium2Parser.lock:
        if pdf is not None:
            pdf.close()  # 关闭原PDF文档以释放资源
        if output_pdf is not None:
            output_pdf.close()  # 关闭新PDF文档以释放资源
    if original_image is not None:
        return {
            "pdf_bytes": output_bytes,
            "original_image": original_image,
        }
    return output_bytes

def convert_pdf_to_bytes_by_pypdfium2(
    pdf_bytes,
    start_page_id=0,
    end_page_id=None,
    pdf_pages_batch=0,
):
    original_image = None
    if isinstance(pdf_bytes, dict):
        original_image = pdf_bytes.get("original_image")
        pdf_bytes = pdf_bytes["pdf_bytes"]
    pdf = None
    output_pdf = None
    file_end = False
    try:
        pdf = pdfium.PdfDocument(pdf_bytes)
        output_pdf = pdfium.PdfDocument.new()
        with PyPDFium2Parser.lock:
            total_pages = len(pdf)

            if total_pages == 0:
                return b"", True

            if start_page_id < 0:
                start_page_id = 0

            if pdf_pages_batch > 0:
                end_page_id = start_page_id + pdf_pages_batch - 1
            else:
                end_page_id = end_page_id if end_page_id is not None and end_page_id >= 0 else total_pages - 1

            if end_page_id > total_pages - 1:
                logger.warning("end_page_id is out of range, use pdf length")
                end_page_id = total_pages - 1
                file_end = True
            elif end_page_id == total_pages - 1:
                file_end = True

            # 逐页导入，失败则跳过
            for page_index in range(start_page_id, end_page_id + 1):
                try:
                    output_pdf.import_pages(pdf, pages=[page_index])
                except Exception as page_error:
                    logger.warning(f"Failed to import page {page_index}: {page_error}, skipping this page.")
                    continue

            output_buffer = io.BytesIO()
            output_pdf.save(output_buffer)
            output_bytes = output_buffer.getvalue()

    except Exception as e:
        logger.warning(f"Error in converting PDF bytes: {e}, using original PDF bytes.")
        output_bytes = b""
        file_end = True

    finally:
        with PyPDFium2Parser.lock:
            if pdf is not None:
                pdf.close()  # 关闭原PDF文档以释放资源
            if output_pdf is not None:
                output_pdf.close()  # 关闭新PDF文档以释放资源
    if original_image is not None:
        return {
            "pdf_bytes": output_bytes,
            "original_image": original_image,
        }, file_end
    return output_bytes, file_end

#=============================================app.py相关调用=============================================
def _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id):
    """准备处理PDF字节数据"""
    result = []
    for pdf_bytes in pdf_bytes_list:
        new_pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
        result.append(new_pdf_bytes)
    return result


def _process_output(
        pdf_info,
        pdf_bytes,
        pdf_file_name,
        local_md_dir,
        local_image_dir,
        md_writer,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_orig_pdf,
        f_dump_md,
        f_dump_content_list,
        f_dump_middle_json,
        f_dump_model_output,
        f_make_md_mode,
        middle_json,
        model_output=None,
        process_mode="pipeline",
):
    if isinstance(pdf_bytes, dict):
        pdf_bytes = pdf_bytes["pdf_bytes"]
    f_draw_line_sort_bbox = False
    from onnxocr.rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
    if process_mode == "pipeline":
        make_func = pipeline_union_make
    elif process_mode in office_suffixes:
        make_func = office_union_make
    else:
        raise Exception(f"Unknown process_mode: {process_mode}")
    """处理输出文件"""
    if f_draw_layout_bbox:
        draw_layout_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_layout.pdf")

    if f_draw_span_bbox:
        draw_span_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_span.pdf")

    if f_dump_orig_pdf:
        if process_mode in ["pipeline", "vlm"]:
            md_writer.write(
                f"{pdf_file_name}_origin.pdf",
                pdf_bytes,
            )
        elif process_mode in office_suffixes:
            md_writer.write(
                f"{pdf_file_name}_origin.{process_mode}",
                pdf_bytes,
            )

    if f_draw_line_sort_bbox:
        draw_line_sort_bbox(pdf_info, pdf_bytes, local_md_dir, f"{pdf_file_name}_line_sort.pdf")

    image_dir = str(os.path.basename(local_image_dir))

    if f_dump_md:
        md_content_str = make_func(pdf_info, f_make_md_mode, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}.md",
            md_content_str,
        )

    if f_dump_content_list:
        content_list = make_func(pdf_info, MakeMode.CONTENT_LIST, image_dir)
        md_writer.write_string(
            f"{pdf_file_name}_content_list.json",
            json.dumps(content_list, ensure_ascii=False, indent=4),
        )
        if process_mode != "pipeline":
            content_list_v2 = make_func(pdf_info, MakeMode.CONTENT_LIST_V2, image_dir)
            md_writer.write_string(
                f"{pdf_file_name}_content_list_v2.json",
                json.dumps(content_list_v2, ensure_ascii=False, indent=4),
            )

    if f_dump_middle_json:
        md_writer.write_string(
            f"{pdf_file_name}_middle.json",
            json.dumps(middle_json, ensure_ascii=False, indent=4),
        )

    if f_dump_model_output:
        md_writer.write_string(
            f"{pdf_file_name}_model.json",
            json.dumps(model_output, ensure_ascii=False, indent=4),
        )

    logger.info(f"local output dir is {local_md_dir}")


def _process_pipeline(
        output_dir,
        pdf_file_names,
        pdf_bytes_list,
        p_lang_list,
        parse_method,
        p_formula_enable,
        p_table_enable,
        layout_config,
        ocr_config,
        formula_config,
        table_config,
        checkbox_config,
        image_config,
        f_draw_layout_bbox,
        f_draw_span_bbox,
        f_dump_md,
        f_dump_middle_json,
        f_dump_model_output,
        f_dump_orig_pdf,
        f_dump_content_list,
        f_make_md_mode,
):
    """处理pipeline后端逻辑"""
    from onnxocr.rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
    from onnxocr.rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze

    local_image_dirs = []
    local_md_dirs = []
    image_writers = []
    md_writers = []
    for pdf_file_name in pdf_file_names:
        local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, parse_method)
        local_image_dirs.append(local_image_dir)
        local_md_dirs.append(local_md_dir)
        image_writers.append(FileBasedDataWriter(local_image_dir))
        md_writers.append(FileBasedDataWriter(local_md_dir))

    pdf_pages_batch = get_processing_window_size(default=64)
    tmp_start_page_id = 0
    batch_idx = 0
    middle_json_list = [None] * len(pdf_bytes_list)
    model_json_list = [[] if f_dump_model_output else None for _ in pdf_bytes_list]
    finished = [False] * len(pdf_bytes_list)

    while not all(finished):
        active_indexes = [idx for idx, is_finished in enumerate(finished) if not is_finished]
        active_pdf_bytes_list = [pdf_bytes_list[idx] for idx in active_indexes]
        active_lang_list = [p_lang_list[idx] for idx in active_indexes]
        infer_results, all_image_lists, all_pdf_docs, lang_list, ocr_enabled_list, file_end_list = (
            pipeline_doc_analyze(
                active_pdf_bytes_list, active_lang_list, parse_method=parse_method,
                formula_enable=p_formula_enable, table_enable=p_table_enable,
                layout_config=layout_config, ocr_config=ocr_config, formula_config=formula_config,
                table_config=table_config, checkbox_config=checkbox_config,
                start_page_id=tmp_start_page_id, end_page_id=None, pdf_pages_batch=pdf_pages_batch
            )
        )

        for active_idx, model_list in enumerate(infer_results):
            original_idx = active_indexes[active_idx]
            if f_dump_model_output:
                model_json_list[original_idx].extend(copy.deepcopy(model_list))

            tmp_middle_json = pipeline_result_to_middle_json(
                model_list, all_image_lists[active_idx], all_pdf_docs[active_idx], image_writers[original_idx],
                lang_list[active_idx], ocr_enabled_list[active_idx], p_formula_enable,
                ocr_config=ocr_config, image_config=image_config, batch_idx=batch_idx, pdf_pages_batch=pdf_pages_batch
            )

            if middle_json_list[original_idx] is None:
                middle_json_list[original_idx] = tmp_middle_json
            else:
                middle_json_list[original_idx]["pdf_info"].extend(tmp_middle_json["pdf_info"])

            if file_end_list[active_idx]:
                _process_output(
                    middle_json_list[original_idx]["pdf_info"], pdf_bytes_list[original_idx], pdf_file_names[original_idx],
                    local_md_dirs[original_idx], local_image_dirs[original_idx], md_writers[original_idx],
                    f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_pdf, f_dump_md, f_dump_content_list,
                    f_dump_middle_json, f_dump_model_output, f_make_md_mode,
                    middle_json_list[original_idx], model_json_list[original_idx], process_mode="pipeline"
                )
                finished[original_idx] = True
            elif not model_list:
                logger.warning(f"No pages parsed for {pdf_file_names[original_idx]}, stop batch processing.")
                finished[original_idx] = True

        tmp_start_page_id += pdf_pages_batch
        batch_idx += 1

def _process_office_doc(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_file=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
):
    need_remove_index = []
    for i, file_bytes in enumerate(pdf_bytes_list):
        pdf_file_name = pdf_file_names[i]
        file_suffix = guess_suffix_by_bytes(file_bytes)
        if file_suffix in office_suffixes:

            need_remove_index.append(i)

            local_image_dir, local_md_dir = prepare_env(output_dir, pdf_file_name, f"office")
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
            middle_json, infer_result = office_analyze(
                file_bytes,
                image_writer=image_writer,
            )

            f_draw_layout_bbox = False
            f_draw_span_bbox = False
            pdf_info = middle_json["pdf_info"]

            _process_output(
                pdf_info, file_bytes, pdf_file_name, local_md_dir, local_image_dir,
                md_writer, f_draw_layout_bbox, f_draw_span_bbox, f_dump_orig_file,
                f_dump_md, f_dump_content_list, f_dump_middle_json, f_dump_model_output,
                f_make_md_mode, middle_json, infer_result, process_mode=file_suffix
            )

    return need_remove_index


def do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,
        image_config=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        **kwargs,
):
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return

    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode
        )


async def aio_do_parse(
        output_dir,
        pdf_file_names: list[str],
        pdf_bytes_list: list[bytes],
        p_lang_list: list[str],
        backend="pipeline",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        layout_config=None,
        ocr_config=None,
        formula_config=None,
        table_config=None,
        checkbox_config=None,
        image_config=None,
        f_draw_layout_bbox=True,
        f_draw_span_bbox=True,
        f_dump_md=True,
        f_dump_middle_json=True,
        f_dump_model_output=True,
        f_dump_orig_pdf=True,
        f_dump_content_list=True,
        f_make_md_mode=MakeMode.MM_MD,
        start_page_id=0,
        end_page_id=None,
        **kwargs,
):
    need_remove_index = _process_office_doc(
        output_dir,
        pdf_file_names=pdf_file_names,
        pdf_bytes_list=pdf_bytes_list,
        f_dump_md=f_dump_md,
        f_dump_middle_json=f_dump_middle_json,
        f_dump_model_output=f_dump_model_output,
        f_dump_orig_file=f_dump_orig_pdf,
        f_dump_content_list=f_dump_content_list,
        f_make_md_mode=f_make_md_mode,
    )
    for index in sorted(need_remove_index, reverse=True):
        del pdf_bytes_list[index]
        del pdf_file_names[index]
        del p_lang_list[index]
    if not pdf_bytes_list:
        logger.warning("No valid PDF or image files to process.")
        return
    # 预处理PDF字节数据
    pdf_bytes_list = _prepare_pdf_bytes(pdf_bytes_list, start_page_id, end_page_id)

    if backend == "pipeline":
        # pipeline模式暂不支持异步，使用同步处理方式
        _process_pipeline(
            output_dir, pdf_file_names, pdf_bytes_list, p_lang_list,
            parse_method, formula_enable, table_enable,
            layout_config, ocr_config, formula_config, table_config, checkbox_config, image_config,
            f_draw_layout_bbox, f_draw_span_bbox, f_dump_md, f_dump_middle_json,
            f_dump_model_output, f_dump_orig_pdf, f_dump_content_list, f_make_md_mode,
        )



if __name__ == "__main__":
    # pdf_path = "../../demo/pdfs/demo3.pdf"
    pdf_path = "C:/Users/zhaoxiaomeng/Downloads/4546d0e2-ba60-40a5-a17e-b68555cec741.pdf"

    try:
       do_parse("./output", [Path(pdf_path).stem], [read_fn(Path(pdf_path))],["ch"],
                end_page_id=10,
                backend = 'pipeline'
                )
    except Exception as e:
        logger.exception(e)

