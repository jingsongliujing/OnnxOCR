import base64
import json
import os
import tempfile
import requests
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote, urlparse

from onnxocr.rapid_doc.backend.office.office_analyze import office_analyze
from onnxocr.rapid_doc.backend.office.office_middle_json_mkcontent import union_make as office_union_make
from onnxocr.rapid_doc.backend.pipeline.model_json_to_middle_json import result_to_middle_json as pipeline_result_to_middle_json
from onnxocr.rapid_doc.backend.pipeline.pipeline_analyze import ModelSingleton
from onnxocr.rapid_doc.backend.pipeline.pipeline_analyze import doc_analyze as pipeline_doc_analyze
from onnxocr.rapid_doc.backend.pipeline.pipeline_middle_json_mkcontent import union_make as pipeline_union_make
from onnxocr.rapid_doc.cli.common import image_suffixes, office_suffixes, old_office_suffixes, prepare_env
from onnxocr.rapid_doc.data.data_reader_writer import FileBasedDataWriter
from onnxocr.rapid_doc.data.data_reader_writer.base import DataWriter
from onnxocr.rapid_doc.utils.draw_bbox import draw_layout_bbox, draw_span_bbox
from onnxocr.rapid_doc.utils.enum_class import MakeMode
from onnxocr.rapid_doc.utils.config_reader import get_processing_window_size
from onnxocr.rapid_doc.utils.guess_suffix_or_lang import guess_suffix_by_bytes
from onnxocr.rapid_doc.utils.office_converter import convert_legacy_office_to_modern
from onnxocr.rapid_doc.utils.pdf_image_tools import images_bytes_to_pdf_bytes


@dataclass
class RapidDocOutput:
    # 统一的解析输出对象，单文档调用时直接返回它；
    # 同时实现了可迭代协议，因此也支持：
    # markdown, images = engine(pdf_bytes)
    markdown: str = ""
    images: dict[str, bytes] = field(default_factory=dict)
    middle_json: dict[str, Any] | None = None
    content_list_json: list[Any] | None = None

    def __iter__(self):
        yield self.markdown
        yield self.images


class MemoryDataWriter(DataWriter):
    # 将写入的数据保存在内存中，便于直接返回给 Python 调用方。
    def __init__(self, parent_dir: str = "images") -> None:
        self._parent_dir = parent_dir
        self.data: dict[str, bytes] = {}

    def write(self, path: str, data: bytes) -> None:
        self.data[path.replace("\\", "/")] = data


class FanoutDataWriter(DataWriter):
    # 同时写入多个 writer。
    def __init__(self, *writers: DataWriter | None) -> None:
        self._writers = [writer for writer in writers if writer is not None]
        self._parent_dir = ""
        for writer in self._writers:
            parent_dir = getattr(writer, "_parent_dir", "")
            if parent_dir:
                self._parent_dir = parent_dir
                break

    def write(self, path: str, data: bytes) -> None:
        for writer in self._writers:
            writer.write(path, data)


class RapidDoc:
    def __init__(
        self,
        layout_config: dict[str, Any] | None = None,
        ocr_config: dict[str, Any] | None = None,
        formula_config: dict[str, Any] | None = None,
        table_config: dict[str, Any] | None = None,
        checkbox_config: dict[str, Any] | None = None,
        image_config: dict[str, Any] | None = None,
        parse_method: str = "auto",
        formula_enable: bool = True,
        table_enable: bool = True,
        lang: str = "ch",
        make_md_mode: str = MakeMode.MM_MD,
        output_dir: str | Path | None = None, # 如果有output_dir会覆盖image_writer和md_writer
        image_writer: DataWriter | None = None,
        md_writer: DataWriter | None = None,
        image_dir_name: str = "images",
        image_output_mode: str = "url", # url / data_uri
        preload_model: bool = False,
        pdf_pages_batch: int = 64,
    ) -> None:
        self.layout_config = layout_config or {}
        self.ocr_config = ocr_config or {}
        self.formula_config = formula_config or {}
        self.table_config = table_config or {}
        self.checkbox_config = checkbox_config or {}
        self.image_config = image_config or {}

        self.parse_method = parse_method
        self.formula_enable = formula_enable
        self.table_enable = table_enable
        self.lang = lang
        self.make_md_mode = make_md_mode
        self.default_output_dir = output_dir
        self.default_image_writer = image_writer
        self.default_md_writer = md_writer
        self.image_dir_name = image_dir_name or "images"
        self.image_output_mode = image_output_mode
        self.pdf_pages_batch = pdf_pages_batch

        self._validate_image_output_mode(self.image_output_mode)

        # 在实例化阶段主动预热模型，后续 __call__ 时能复用缓存。
        if preload_model:
            self.warmup(lang=lang)

    def warmup(
        self,
        lang: str | None = None,
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
    ) -> None:
        warmup_lang = lang or self.lang
        warmup_formula_enable = (
            self.formula_enable if formula_enable is None else formula_enable
        )
        warmup_table_enable = (
            self.table_enable if table_enable is None else table_enable
        )

        ModelSingleton().get_model(
            lang=warmup_lang,
            formula_enable=warmup_formula_enable,
            table_enable=warmup_table_enable,
            layout_config=self.layout_config,
            ocr_config=self.ocr_config,
            formula_config=self.formula_config,
            table_config=self.table_config,
        )

    def __call__(
        self,
        inputs: str | bytes | Path | list[str | bytes | Path],
        output_dir: str | Path | None = None,
        image_writer: DataWriter | None = None,
        md_writer: DataWriter | None = None,
        image_output_mode: str | None = None,
        image_dir_name: str | None = None,
        parse_method: str | None = None,
        formula_enable: bool | None = None,
        table_enable: bool | None = None,
        lang: str | list[str] | None = None,
        start_page_id: int = 0,
        end_page_id: int | None = None,
        f_dump_middle_json: bool = True,
        f_dump_content_list: bool = True,
        f_draw_layout_bbox: bool = False,
        f_draw_span_bbox: bool = False,
    ) -> RapidDocOutput | list[RapidDocOutput]:
        is_batch = self._is_batch_input(inputs)
        normalized_inputs = list(inputs) if is_batch else [inputs]

        final_image_output_mode = image_output_mode or self.image_output_mode
        final_image_dir_name = image_dir_name or self.image_dir_name
        final_parse_method = parse_method or self.parse_method
        final_formula_enable = (
            self.formula_enable if formula_enable is None else formula_enable
        )
        final_table_enable = (
            self.table_enable if table_enable is None else table_enable
        )
        final_output_dir = output_dir or self.default_output_dir
        final_image_writer = image_writer or self.default_image_writer
        final_md_writer = md_writer or self.default_md_writer

        self._validate_image_output_mode(final_image_output_mode)

        normalized_docs = self._normalize_inputs(normalized_inputs)
        lang_list = self._normalize_lang_list(lang, len(normalized_docs))

        outputs: list[RapidDocOutput] = []
        pipeline_indexes: list[int] = []
        office_indexes: list[int] = []
        pipeline_pdf_bytes_list: list[bytes | dict[str, Any]] = []
        pipeline_name_list: list[str] = []
        pipeline_lang_list: list[str] = []

        for index, doc in enumerate(normalized_docs):
            if doc["suffix"] in office_suffixes:
                office_indexes.append(index)
                outputs.append(self._empty_output())
                continue

            if doc["suffix"] not in ["pdf", *image_suffixes]:
                raise ValueError(f"Unsupported input suffix: {doc['suffix']}")

            pipeline_indexes.append(index)
            outputs.append(self._empty_output())
            pipeline_pdf_bytes_list.append(doc["pdf_bytes"])
            pipeline_name_list.append(doc["name"])
            pipeline_lang_list.append(lang_list[index])

        for index in office_indexes:
            doc = normalized_docs[index]
            outputs[index] = self._parse_office(
                name=doc["name"],
                file_bytes=doc["raw_bytes"],
                output_dir=final_output_dir,
                image_writer=final_image_writer,
                md_writer=final_md_writer,
                image_dir_name=final_image_dir_name,
                image_output_mode=final_image_output_mode,
                f_dump_middle_json=f_dump_middle_json,
                f_dump_content_list=f_dump_content_list,
            )

        if pipeline_pdf_bytes_list:
            pipeline_outputs = self._parse_pipeline_batch(
                names=pipeline_name_list,
                pdf_bytes_list=pipeline_pdf_bytes_list,
                lang_list=pipeline_lang_list,
                parse_method=final_parse_method,
                formula_enable=final_formula_enable,
                table_enable=final_table_enable,
                output_dir=final_output_dir,
                image_writer=final_image_writer,
                md_writer=final_md_writer,
                image_dir_name=final_image_dir_name,
                image_output_mode=final_image_output_mode,
                start_page_id=start_page_id,
                end_page_id=end_page_id,
                f_dump_middle_json=f_dump_middle_json,
                f_dump_content_list=f_dump_content_list,
                f_draw_layout_bbox=f_draw_layout_bbox,
                f_draw_span_bbox=f_draw_span_bbox,
                pdf_pages_batch=self.pdf_pages_batch,
            )
            for output_index, result in zip(pipeline_indexes, pipeline_outputs):
                outputs[output_index] = result

        if is_batch:
            return outputs
        return outputs[0]

    def _parse_pipeline_batch(
        self,
        names: list[str],
        pdf_bytes_list: list[bytes | dict[str, Any]],
        lang_list: list[str],
        parse_method: str,
        formula_enable: bool,
        table_enable: bool,
        output_dir: str | Path | None,
        image_writer: DataWriter | None,
        md_writer: DataWriter | None,
        image_dir_name: str,
        image_output_mode: str,
        start_page_id: int,
        end_page_id: int | None,
        f_dump_middle_json: bool,
        f_dump_content_list: bool,
        f_draw_layout_bbox: bool,
        f_draw_span_bbox: bool,
        pdf_pages_batch: int,
    ) -> list[RapidDocOutput]:
        from onnxocr.rapid_doc.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2

        sliced_pdf_bytes_list = [
            convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_bytes, start_page_id, end_page_id)
            for pdf_bytes in pdf_bytes_list
        ]

        memory_image_writers: list[MemoryDataWriter] = []
        combined_image_writers: list[DataWriter] = []
        md_writers: list[DataWriter | None] = []
        for name in names:
            current_image_writer = image_writer
            current_md_writer = md_writer
            if output_dir:
                local_image_dir, local_md_dir = prepare_env(output_dir, name, parse_method)
                current_image_writer = FileBasedDataWriter(local_image_dir)
                current_md_writer = FileBasedDataWriter(local_md_dir)
            memory_image_writer, combined_image_writer = self._build_image_writer(
                image_dir_name=image_dir_name,
                extra_image_writer=current_image_writer,
            )
            memory_image_writers.append(memory_image_writer)
            combined_image_writers.append(combined_image_writer)
            md_writers.append(current_md_writer)

        outputs: list[RapidDocOutput | None] = [None] * len(sliced_pdf_bytes_list)
        middle_json_list: list[dict[str, Any] | None] = [None] * len(sliced_pdf_bytes_list)
        finished = [False] * len(sliced_pdf_bytes_list)
        tmp_start_page_id = 0
        batch_idx = 0
        if not pdf_pages_batch:
            pdf_pages_batch = get_processing_window_size(default=64)

        while not all(finished):
            active_indexes = [idx for idx, is_finished in enumerate(finished) if not is_finished]
            active_pdf_bytes_list = [sliced_pdf_bytes_list[idx] for idx in active_indexes]
            active_lang_list = [lang_list[idx] for idx in active_indexes]
            (
                infer_results,
                all_image_lists,
                all_page_dicts,
                final_lang_list,
                ocr_enabled_list,
                file_end_list,
            ) = pipeline_doc_analyze(
                active_pdf_bytes_list,
                active_lang_list,
                parse_method=parse_method,
                formula_enable=formula_enable,
                table_enable=table_enable,
                layout_config=self.layout_config,
                ocr_config=self.ocr_config,
                formula_config=self.formula_config,
                table_config=self.table_config,
                checkbox_config=self.checkbox_config,
                start_page_id=tmp_start_page_id,
                end_page_id=None,
                pdf_pages_batch=pdf_pages_batch,
            )

            for active_idx, model_list in enumerate(infer_results):
                original_idx = active_indexes[active_idx]

                tmp_middle_json = pipeline_result_to_middle_json(
                    model_list,
                    all_image_lists[active_idx],
                    all_page_dicts[active_idx],
                    combined_image_writers[original_idx],
                    final_lang_list[active_idx],
                    ocr_enabled_list[active_idx],
                    formula_enable,
                    ocr_config=self.ocr_config,
                    image_config=self.image_config,
                    batch_idx=batch_idx,
                    pdf_pages_batch=pdf_pages_batch,
                )
                if middle_json_list[original_idx] is None:
                    middle_json_list[original_idx] = tmp_middle_json
                else:
                    middle_json_list[original_idx]["pdf_info"].extend(tmp_middle_json["pdf_info"])

                if file_end_list[active_idx]:
                    outputs[original_idx] = self._build_pipeline_output(
                        name=names[original_idx],
                        pdf_bytes=sliced_pdf_bytes_list[original_idx],
                        middle_json=middle_json_list[original_idx],
                        memory_image_writer=memory_image_writers[original_idx],
                        md_writer=md_writers[original_idx],
                        image_dir_name=image_dir_name,
                        image_output_mode=image_output_mode,
                        f_dump_middle_json=f_dump_middle_json,
                        f_dump_content_list=f_dump_content_list,
                        f_draw_layout_bbox=f_draw_layout_bbox,
                        f_draw_span_bbox=f_draw_span_bbox,
                    )
                    finished[original_idx] = True
                elif not model_list:
                    raise RuntimeError(
                        f"No pages parsed for {names[original_idx]} before reaching the end of the file."
                    )

            tmp_start_page_id += pdf_pages_batch
            batch_idx += 1

        finished_outputs: list[RapidDocOutput] = []
        for output in outputs:
            if output is None:
                raise RuntimeError("Pipeline batch finished without producing all outputs.")
            finished_outputs.append(output)
        return finished_outputs

    def _build_pipeline_output(
        self,
        name: str,
        pdf_bytes: bytes | dict[str, Any],
        middle_json: dict[str, Any],
        memory_image_writer: MemoryDataWriter,
        md_writer: DataWriter | None,
        image_dir_name: str,
        image_output_mode: str,
        f_dump_middle_json: bool,
        f_dump_content_list: bool,
        f_draw_layout_bbox: bool,
        f_draw_span_bbox: bool,
    ) -> RapidDocOutput:
        markdown = pipeline_union_make(
            middle_json["pdf_info"],
            self.make_md_mode,
            image_dir_name,
        )
        content_list_json = pipeline_union_make(
            middle_json["pdf_info"],
            MakeMode.CONTENT_LIST,
            image_dir_name,
        )

        image_bytes_map = dict(memory_image_writer.data)
        logical_image_map = self._build_logical_image_map(
            image_bytes_map,
            image_dir_name=image_dir_name,
        )

        if image_output_mode == "data_uri":
            markdown = self._replace_markdown_images_with_data_uri(
                markdown,
                logical_image_map,
            )

        output = RapidDocOutput(
            markdown=markdown,
            images=image_bytes_map,
            middle_json=middle_json,
            content_list_json=content_list_json,
        )

        self._dump_output_if_needed(
            output=output,
            pdf_bytes=self._extract_pdf_bytes(pdf_bytes),
            name=name,
            md_writer=md_writer,
            f_dump_middle_json=f_dump_middle_json,
            f_dump_content_list=f_dump_content_list,
            f_draw_layout_bbox=f_draw_layout_bbox,
            f_draw_span_bbox=f_draw_span_bbox,
        )
        return output

    def _parse_office(
        self,
        name: str,
        file_bytes: bytes,
        output_dir: str | Path | None,
        image_writer: DataWriter | None,
        md_writer: DataWriter | None,
        image_dir_name: str,
        image_output_mode: str,
        f_dump_middle_json: bool,
        f_dump_content_list: bool,
    ) -> RapidDocOutput:
        if output_dir:
            local_image_dir, local_md_dir = prepare_env(output_dir, name, f"office")
            image_writer, md_writer = FileBasedDataWriter(local_image_dir), FileBasedDataWriter(local_md_dir)
        memory_image_writer, combined_image_writer = self._build_image_writer(
            image_dir_name=image_dir_name,
            extra_image_writer=image_writer,
        )

        middle_json, model_json = office_analyze(
            file_bytes,
            image_writer=combined_image_writer,
        )
        markdown = office_union_make(
            middle_json["pdf_info"],
            self.make_md_mode,
            image_dir_name,
        )
        content_list_json = office_union_make(
            middle_json["pdf_info"],
            MakeMode.CONTENT_LIST,
            image_dir_name,
        )

        image_bytes_map = dict(memory_image_writer.data)
        logical_image_map = self._build_logical_image_map(
            image_bytes_map,
            image_dir_name=image_dir_name,
        )

        if image_output_mode == "data_uri":
            markdown = self._replace_markdown_images_with_data_uri(
                markdown,
                logical_image_map,
            )

        output = RapidDocOutput(
            markdown=markdown,
            images=image_bytes_map,
            middle_json=middle_json,
            content_list_json=content_list_json,
        )

        self._dump_output_if_needed(
            output=output,
            pdf_bytes=file_bytes,
            name=name,
            md_writer=md_writer,
            f_dump_middle_json=f_dump_middle_json,
            f_dump_content_list=f_dump_content_list,
            f_draw_layout_bbox=False,
            f_draw_span_bbox=False,
        )
        return output

    def _dump_output_if_needed(
        self,
        output: RapidDocOutput,
        pdf_bytes: bytes,
        name: str,
        md_writer: DataWriter | None,
        f_dump_middle_json: bool,
        f_dump_content_list: bool,
        f_draw_layout_bbox: bool,
        f_draw_span_bbox: bool,
    ) -> None:
        if md_writer is None:
            return
        if isinstance(pdf_bytes, dict):
            pdf_bytes = pdf_bytes["pdf_bytes"]
        md_writer.write_string(f"{name}.md", output.markdown)
        if f_dump_middle_json:
            md_writer.write_string(
                f"{name}_middle.json",
                json.dumps(output.middle_json, ensure_ascii=False, indent=4),
            )
        if f_dump_content_list:
            md_writer.write_string(
                f"{name}_content_list.json",
                json.dumps(output.content_list_json, ensure_ascii=False, indent=4),
            )
        if f_draw_layout_bbox:
            draw_layout_bbox(output.middle_json["pdf_info"], pdf_bytes, md_writer, f"{name}_layout.pdf")
        if f_draw_span_bbox:
            draw_span_bbox(output.middle_json["pdf_info"], pdf_bytes, md_writer, f"{name}_span.pdf")

    def _build_image_writer(
        self,
        image_dir_name: str,
        extra_image_writer: DataWriter | None,
    ) -> tuple[MemoryDataWriter, DataWriter]:
        memory_image_writer = MemoryDataWriter(parent_dir=image_dir_name)
        return memory_image_writer, FanoutDataWriter(memory_image_writer, extra_image_writer)

    def _normalize_inputs(
        self,
        inputs: list[str | bytes | Path],
    ) -> list[dict[str, Any]]:
        normalized_docs: list[dict[str, Any]] = []
        for index, item in enumerate(inputs):
            doc = self._normalize_single_input(item, index)
            normalized_docs.append(doc)
        return normalized_docs

    def _normalize_single_input(
        self,
        item: str | bytes | Path,
        index: int,
    ) -> dict[str, Any]:
        if isinstance(item, str) and self._is_url(item):
            return self._normalize_url_input(item, index)

        if isinstance(item, (str, Path)):
            path = Path(item)
            if not path.exists():
                raise FileNotFoundError(f"Input file not found: {path}")

            file_suffix = path.suffix.lower().lstrip(".")
            actual_path = path
            if file_suffix in old_office_suffixes:
                actual_path = Path(convert_legacy_office_to_modern(path))
                file_suffix = actual_path.suffix.lower().lstrip(".")

            raw_bytes = actual_path.read_bytes()
            return self._build_normalized_doc(
                raw_bytes=raw_bytes,
                name=actual_path.stem,
                suffix=file_suffix,
                index=index,
            )

        if isinstance(item, bytearray):
            item = bytes(item)

        if isinstance(item, bytes):
            return self._build_normalized_doc(
                raw_bytes=item,
                name=f"document_{index + 1}",
                suffix="",
                index=index,
            )

        raise TypeError(f"Unsupported input type: {type(item)}")

    def _normalize_url_input(
        self,
        url: str,
        index: int,
    ) -> dict[str, Any]:
        response = requests.get(url, timeout=60)
        response.raise_for_status()

        remote_name = self._infer_remote_name(url, index)
        return self._build_normalized_doc(
            raw_bytes=response.content,
            name=Path(remote_name).stem,
            suffix=Path(remote_name).suffix.lower().lstrip("."),
            index=index,
        )

    def _build_normalized_doc(
        self,
        raw_bytes: bytes,
        name: str,
        suffix: str,
        index: int,
    ) -> dict[str, Any]:
        doc_name = name or f"document_{index + 1}"
        file_suffix = suffix.lower().lstrip(".") if suffix else ""
        if not file_suffix:
            file_suffix = guess_suffix_by_bytes(raw_bytes)

        if file_suffix in old_office_suffixes:
            raw_bytes, doc_name, file_suffix = self._convert_legacy_office_bytes(
                raw_bytes=raw_bytes,
                suffix=file_suffix,
                name=doc_name,
            )

        if file_suffix in image_suffixes:
            pdf_bytes = images_bytes_to_pdf_bytes(raw_bytes)
        elif file_suffix in ["pdf", *office_suffixes]:
            pdf_bytes = raw_bytes
        else:
            raise ValueError(f"Unsupported input suffix: {file_suffix}")

        return {
            "name": doc_name,
            "suffix": file_suffix,
            "raw_bytes": raw_bytes,
            "pdf_bytes": pdf_bytes,
        }

    def _convert_legacy_office_bytes(
        self,
        raw_bytes: bytes,
        suffix: str,
        name: str,
    ) -> tuple[bytes, str, str]:
        with tempfile.TemporaryDirectory(prefix="rapid_doc_legacy_") as temp_dir:
            input_path = Path(temp_dir) / f"{Path(name).stem}.{suffix}"
            input_path.write_bytes(raw_bytes)

            converted_path = Path(convert_legacy_office_to_modern(input_path, temp_dir))
            return (
                converted_path.read_bytes(),
                converted_path.stem,
                converted_path.suffix.lower().lstrip("."),
            )

    def _infer_remote_name(self, url: str, index: int) -> str:
        parsed = urlparse(url)
        remote_name = Path(unquote(parsed.path)).name
        if remote_name:
            return remote_name
        return f"document_{index + 1}"

    def _normalize_lang_list(
        self,
        lang: str | list[str] | None,
        doc_count: int,
    ) -> list[str]:
        if lang is None:
            return [self.lang] * doc_count
        if isinstance(lang, str):
            return [lang] * doc_count
        if len(lang) != doc_count:
            raise ValueError("The length of lang list must match the number of inputs.")
        return list(lang)

    def _replace_markdown_images_with_data_uri(
        self,
        markdown: str,
        logical_image_map: dict[str, bytes],
    ) -> str:
        updated_markdown = markdown
        replacements = sorted(logical_image_map.items(), key=lambda item: len(item[0]), reverse=True)
        for logical_ref, image_bytes in replacements:
            data_uri = self._to_data_uri(logical_ref, image_bytes)
            updated_markdown = updated_markdown.replace(logical_ref, data_uri)
        return updated_markdown

    def _build_logical_image_map(
        self,
        image_bytes_map: dict[str, bytes],
        image_dir_name: str,
    ) -> dict[str, bytes]:
        logical_image_map: dict[str, bytes] = {}
        for image_name, image_bytes in image_bytes_map.items():
            logical_ref = f"{image_dir_name}/{image_name}".replace("\\", "/")
            logical_image_map[logical_ref] = image_bytes
            logical_image_map[image_name] = image_bytes
        return logical_image_map

    def _extract_pdf_bytes(self, pdf_bytes: bytes | dict[str, Any]) -> bytes:
        if isinstance(pdf_bytes, dict):
            return pdf_bytes["pdf_bytes"]
        return pdf_bytes

    def _to_data_uri(self, logical_ref: str, image_bytes: bytes) -> str:
        suffix = Path(logical_ref).suffix.lower().lstrip(".") or "png"
        mime_suffix = "jpeg" if suffix == "jpg" else suffix
        encoded = base64.b64encode(image_bytes).decode("utf-8")
        return f"data:image/{mime_suffix};base64,{encoded}"

    def _empty_output(self) -> RapidDocOutput:
        return RapidDocOutput()

    def _validate_image_output_mode(self, image_output_mode: str) -> None:
        if image_output_mode not in {"url", "data_uri"}:
            raise ValueError(
                "image_output_mode only supports 'url' and 'data_uri'."
            )

    def _is_batch_input(self, inputs: Any) -> bool:
        if isinstance(inputs, (bytes, bytearray, str, Path)):
            return False
        return isinstance(inputs, Iterable)

    def _is_url(self, value: str) -> bool:
        try:
            parsed = urlparse(value)
        except Exception:
            return False
        return bool(parsed.scheme and parsed.netloc)


if __name__ == '__main__':

    __dir__ = Path(__file__).resolve().parent.parent
    output_dir = os.path.join(__dir__, "output")

    doc_path_list = [
        __dir__ / "demo/pdfs/示例1-论文模板.pdf",
        __dir__ / "demo/docx/test.docx",
    ]
    engine = RapidDoc()
    outputs = engine(doc_path_list, output_dir=output_dir)
    for output in outputs:
        print(output.markdown)
