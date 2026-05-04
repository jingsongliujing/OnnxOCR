import argparse
import os
import shutil
from pathlib import Path


DEFAULT_MODELSCOPE_REPO_ID = "supersong/onnxocr_model"
DEFAULT_HUGGINGFACE_REPO_ID = "jingsongliu/onnxocr_model"
DEFAULT_MODEL_SUBDIR = "models"

REQUIRED_MODEL_FILES = [
    "ppocrv5/det/det.onnx",
    "ppocrv5/rec/rec.onnx",
    "ppocrv5/cls/cls.onnx",
    "ppocrv5/ppocrv5_dict.txt",
    "license_plate/car_plate_detect.onnx",
    "license_plate/plate_rec.onnx",
    "orientation/rapid_orientation.onnx",
    "layout/layout_cdla.onnx",
    "layout/layout_publaynet.onnx",
    "table/slanet-plus.onnx",
    "rapid_doc/layout/pp_doclayoutv2.onnx",
    "rapid_doc/table/q_cls.onnx",
    "rapid_doc/table/unet.onnx",
    "rapid_doc/table/slanet-plus.onnx",
]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def default_model_dir() -> Path:
    return project_root() / "onnxocr" / "models"


def find_snapshot_model_root(snapshot_dir: Path, model_subdir: str | None) -> Path:
    if model_subdir:
        model_root = snapshot_dir / model_subdir
        if model_root.exists():
            return model_root

    candidates = [
        snapshot_dir / "onnxocr" / "models",
        snapshot_dir / "models",
        snapshot_dir,
    ]
    for candidate in candidates:
        if (candidate / "ppocrv5").exists() or (candidate / "rapid_doc").exists():
            return candidate
    return snapshot_dir


def copy_tree(src: Path, dst: Path, overwrite: bool) -> None:
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.iterdir():
        target = dst / item.name
        if item.is_dir():
            copy_tree(item, target, overwrite=overwrite)
            continue
        if target.exists() and not overwrite:
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(item, target)


def check_required_files(model_dir: Path) -> list[Path]:
    return [model_dir / rel_path for rel_path in REQUIRED_MODEL_FILES if not (model_dir / rel_path).exists()]


def download_from_modelscope(repo_id: str, revision: str | None) -> Path:
    try:
        from modelscope import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "modelscope is not installed. Install it with: pip install modelscope"
        ) from exc

    kwargs = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def download_from_huggingface(
    repo_id: str,
    revision: str | None,
    hf_endpoint: str | None,
) -> Path:
    if hf_endpoint:
        os.environ["HF_ENDPOINT"] = hf_endpoint.rstrip("/")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "huggingface_hub is not installed. Install it with: pip install huggingface_hub"
        ) from exc

    kwargs = {"repo_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    return Path(snapshot_download(**kwargs))


def main() -> None:
    parser = argparse.ArgumentParser(description="Download OnnxOCR models.")
    parser.add_argument(
        "--source",
        choices=("modelscope", "huggingface"),
        default="modelscope",
        help="Download source. Default: modelscope, recommended in mainland China.",
    )
    parser.add_argument(
        "--repo-id",
        default=None,
        help=(
            "Model repository id. Defaults to supersong/onnxocr_model for ModelScope "
            "and jingsongliu/onnxocr_model for HuggingFace."
        ),
    )
    parser.add_argument(
        "--model-subdir",
        default=DEFAULT_MODEL_SUBDIR,
        help="Model directory inside the downloaded snapshot. Default: models.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(default_model_dir()),
        help="Target model directory. Default: onnxocr/models.",
    )
    parser.add_argument("--revision", default=None, help="Optional model repository revision.")
    parser.add_argument(
        "--hf-endpoint",
        default=None,
        help="Optional HuggingFace endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing local model files.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Only check required local model files, without downloading.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()

    if not args.check_only:
        if args.source == "modelscope":
            repo_id = args.repo_id or DEFAULT_MODELSCOPE_REPO_ID
            snapshot_dir = download_from_modelscope(repo_id, args.revision)
        else:
            repo_id = args.repo_id or DEFAULT_HUGGINGFACE_REPO_ID
            snapshot_dir = download_from_huggingface(
                repo_id,
                revision=args.revision,
                hf_endpoint=args.hf_endpoint,
            )
        model_root = find_snapshot_model_root(snapshot_dir, args.model_subdir)
        print(f"Download source: {args.source}")
        print(f"Repository: {repo_id}")
        print(f"Snapshot: {snapshot_dir}")
        print(f"Copy models from: {model_root}")
        print(f"Copy models to: {output_dir}")
        copy_tree(model_root, output_dir, overwrite=args.overwrite)

    missing_files = check_required_files(output_dir)
    if missing_files:
        print("\nMissing model files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        raise SystemExit(1)

    print(f"\nAll required model files are ready in: {output_dir}")


if __name__ == "__main__":
    main()
