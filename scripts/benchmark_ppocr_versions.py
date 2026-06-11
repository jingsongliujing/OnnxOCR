import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import numpy as np

from onnxocr.onnx_paddleocr import ONNXPaddleOcr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = PROJECT_ROOT / "onnxocr" / "models"
TEST_IMAGE_ROOT = PROJECT_ROOT / "onnxocr" / "test_images"
DEFAULT_IMAGES = [
    "715873facf064583b44ef28295126fa7.jpg",
    "12.jpg",
    "1.jpg",
    "french_0.jpg",
    "japan_2.jpg",
    "weixin_pay.jpg",
    "table.jpg",
    "00006737.jpg",
]


def imread(path: Path):
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        raise RuntimeError(f"Empty image file: {path}")
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f"Failed to decode image: {path}")
    return img


def build_model(version: str):
    if version == "ppocrv5":
        return ONNXPaddleOcr(
            use_angle_cls=False,
            use_gpu=False,
            det_model_dir=str(MODEL_ROOT / "ppocrv5" / "det" / "det.onnx"),
            rec_model_dir=str(MODEL_ROOT / "ppocrv5" / "rec" / "rec.onnx"),
            rec_char_dict_path=str(MODEL_ROOT / "ppocrv5" / "ppocrv5_dict.txt"),
            rec_image_shape="3, 48, 320",
            det_limit_side_len=960,
            det_limit_type="max",
            det_db_thresh=0.3,
            det_db_box_thresh=0.6,
            det_db_unclip_ratio=1.5,
            det_db_max_candidates=1000,
        )
    if version in {"ppocrv6_tiny", "ppocrv6_small", "ppocrv6_medium"}:
        return ONNXPaddleOcr(
            use_angle_cls=False,
            use_gpu=False,
            ocr_model_size=version.removeprefix("ppocrv6_"),
        )
    raise ValueError(f"Unsupported version: {version}")


def flatten_result(result):
    if not result or not result[0]:
        return []
    return result[0]


def benchmark_model(model, images, repeats: int):
    records = []
    for image_name, img in images:
        model.ocr(img, cls=False)
        times = []
        last_result = None
        for _ in range(repeats):
            start = time.perf_counter()
            last_result = model.ocr(img, cls=False)
            times.append(time.perf_counter() - start)
        lines = flatten_result(last_result)
        records.append(
            {
                "image": image_name,
                "shape": list(img.shape),
                "text_lines": len(lines),
                "avg_s": statistics.mean(times),
                "min_s": min(times),
                "max_s": max(times),
            }
        )
    return records


def write_markdown(output_path: Path, summary, results):
    lines = [
        "# PP-OCRv5 vs PP-OCRv6 Benchmark",
        "",
        f"- Device: CPUExecutionProvider",
        f"- Repeats after warm-up: {summary['repeats']}",
        f"- Images: {len(summary['images'])}",
        "",
        "| Model | Total avg (s) | Per image avg (s) | Text lines |",
        "| --- | ---: | ---: | ---: |",
    ]
    for model_name, item in summary["models"].items():
        lines.append(
            f"| {model_name} | {item['total_avg_s']:.3f} | {item['per_image_avg_s']:.3f} | {item['text_lines']} |"
        )

    lines.extend(
        [
            "",
            "| Image | Size | PP-OCRv5 avg (s) | PP-OCRv6 tiny avg (s) | PP-OCRv6 small avg (s) | PP-OCRv6 medium avg (s) | tiny/v5 | small/v5 | medium/v5 | v5 lines | tiny lines | small lines | medium lines |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    by_image = {}
    for model_name, rows in results.items():
        for row in rows:
            by_image.setdefault(row["image"], {})[model_name] = row
    for image_name in summary["images"]:
        v5 = by_image[image_name]["ppocrv5"]
        tiny = by_image[image_name]["ppocrv6_tiny"]
        small = by_image[image_name]["ppocrv6_small"]
        medium = by_image[image_name]["ppocrv6_medium"]
        h, w = medium["shape"][:2]
        tiny_ratio = tiny["avg_s"] / v5["avg_s"] if v5["avg_s"] else 0
        small_ratio = small["avg_s"] / v5["avg_s"] if v5["avg_s"] else 0
        medium_ratio = medium["avg_s"] / v5["avg_s"] if v5["avg_s"] else 0
        lines.append(
            f"| {image_name} | {w}x{h} | {v5['avg_s']:.3f} | {tiny['avg_s']:.3f} | {small['avg_s']:.3f} | {medium['avg_s']:.3f} | {tiny_ratio:.2f}x | {small_ratio:.2f}x | {medium_ratio:.2f}x | {v5['text_lines']} | {tiny['text_lines']} | {small['text_lines']} | {medium['text_lines']} |"
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Benchmark PP-OCRv5 and PP-OCRv6 ONNX OCR models.")
    parser.add_argument("--repeats", type=int, default=2, help="Timed repeats after one warm-up run per image.")
    parser.add_argument("--images", nargs="*", default=DEFAULT_IMAGES, help="Image names under onnxocr/test_images.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "output" / "benchmarks"))
    args = parser.parse_args()

    images = [(name, imread(TEST_IMAGE_ROOT / name)) for name in args.images]
    results = {}
    summary = {
        "repeats": args.repeats,
        "images": args.images,
        "models": {},
    }

    for version in ("ppocrv5", "ppocrv6_tiny", "ppocrv6_small", "ppocrv6_medium"):
        model = build_model(version)
        rows = benchmark_model(model, images, repeats=args.repeats)
        results[version] = rows
        total_avg = sum(row["avg_s"] for row in rows)
        summary["models"][version] = {
            "total_avg_s": total_avg,
            "per_image_avg_s": total_avg / len(rows),
            "text_lines": sum(row["text_lines"] for row in rows),
        }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "ppocr_versions_benchmark.json"
    md_path = output_dir / "ppocr_versions_benchmark.md"
    json_path.write_text(json.dumps({"summary": summary, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, summary, results)
    print(md_path)
    print(json_path)


if __name__ == "__main__":
    main()
