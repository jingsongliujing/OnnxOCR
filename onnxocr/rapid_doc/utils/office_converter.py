import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

from loguru import logger


class ConvertToPdfError(Exception):
    def __init__(self, msg):
        self.msg = msg
        super().__init__(self.msg)


EXPORT_FILTERS = {
    "pdf": "pdf",
    "docx": "Office Open XML Text",
    "pptx": "Impress Office Open XML",
    "xlsx": "Calc Office Open XML",
}

LEGACY_TO_MODERN_FORMATS = {
    ".doc": "docx",
    ".ppt": "pptx",
    ".xls": "xlsx",
}


def check_fonts_installed():
    """Check if required Chinese fonts are installed."""
    system_type = platform.system()

    if system_type in ["Windows", "Darwin"]:
        return True

    try:
        output = subprocess.check_output(["fc-list", ":lang=zh"], encoding="utf-8")
        if output.strip():
            return True

        logger.warning(
            "No Chinese fonts were detected, the converted document may not display Chinese content properly."
        )
    except Exception:
        pass

    return False


def get_soffice_command():
    """Return the path to LibreOffice's soffice executable depending on the platform."""
    system_type = platform.system()

    soffice_path = shutil.which("soffice")
    if soffice_path:
        if system_type == "Windows":
            soffice_com = str(Path(soffice_path).with_suffix(".com"))
            if Path(soffice_com).exists():
                return soffice_com
        return soffice_path

    if system_type == "Windows":
        possible_paths = [
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "LibreOffice/program/soffice.com",
            Path(os.environ.get("PROGRAMFILES", "C:/Program Files")) / "LibreOffice/program/soffice.exe",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "LibreOffice/program/soffice.com",
            Path(os.environ.get("PROGRAMFILES(X86)", "C:/Program Files (x86)")) / "LibreOffice/program/soffice.exe",
            Path("C:/Program Files/LibreOffice/program/soffice.com"),
            Path("C:/Program Files/LibreOffice/program/soffice.exe"),
            Path("C:/Program Files (x86)/LibreOffice/program/soffice.com"),
            Path("C:/Program Files (x86)/LibreOffice/program/soffice.exe"),
        ]

        for drive in ["C:", "D:", "E:", "F:", "G:", "H:"]:
            possible_paths.append(Path(f"{drive}/LibreOffice/program/soffice.com"))
            possible_paths.append(Path(f"{drive}/LibreOffice/program/soffice.exe"))

        for path in possible_paths:
            if path.exists():
                return str(path)

        raise ConvertToPdfError(
            "LibreOffice not found. Please install LibreOffice from https://www.libreoffice.org/ "
            "or ensure soffice.exe is in your PATH environment variable."
        )

    possible_paths = [
        "/usr/bin/soffice",
        "/usr/local/bin/soffice",
        "/opt/libreoffice/program/soffice",
        "/Applications/LibreOffice.app/Contents/MacOS/soffice",
    ]

    for path in possible_paths:
        if os.path.exists(path):
            return path

    raise ConvertToPdfError(
        "LibreOffice not found. Please install it:\n"
        "  - Ubuntu/Debian: sudo apt-get install libreoffice\n"
        "  - CentOS/RHEL: sudo yum install libreoffice\n"
        "  - macOS: brew install libreoffice or download from https://www.libreoffice.org/\n"
        "  - Or ensure soffice is in your PATH environment variable."
    )


def _normalize_target_format(target_format):
    normalized = target_format.lower().lstrip(".")
    if normalized not in EXPORT_FILTERS:
        supported = ", ".join(sorted(EXPORT_FILTERS))
        raise ConvertToPdfError(f"Unsupported target format: {target_format}. Supported formats: {supported}")
    return normalized


def _build_convert_to_arg(target_format):
    normalized = _normalize_target_format(target_format)
    export_filter = EXPORT_FILTERS[normalized]
    if export_filter == normalized:
        return normalized
    return f"{normalized}:{export_filter}"


def _expected_output_path(input_path, output_dir, target_format):
    input_file = Path(input_path)
    return Path(output_dir) / f"{input_file.stem}.{_normalize_target_format(target_format)}"


def _build_subprocess_kwargs():
    kwargs = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "timeout": 1200,
    }
    if platform.system() == "Windows" and hasattr(subprocess, "CREATE_NO_WINDOW"):
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def convert_file(input_path, output_dir, target_format):
    """Convert an Office file to the target format and return the converted file path."""
    input_file = Path(input_path)
    output_dir = Path(output_dir)

    logger.info(f"convert_file start: input_path={input_file}, output_dir={output_dir}, target_format={target_format}")

    if not input_file.is_file():
        logger.error(f"convert_file failed: input file does not exist: {input_path}")
        raise FileNotFoundError(f"The input file {input_path} does not exist.")

    output_dir.mkdir(parents=True, exist_ok=True)

    check_fonts_installed()

    soffice_cmd = get_soffice_command()
    convert_to_arg = _build_convert_to_arg(target_format)
    subprocess_kwargs = _build_subprocess_kwargs()

    with tempfile.TemporaryDirectory(prefix="libreoffice-profile-") as profile_dir:
        profile_uri = Path(profile_dir).resolve().as_uri()
        cmd = [
            soffice_cmd,
            f"-env:UserInstallation={profile_uri}",
            "--headless",
            "--nologo",
            "--nodefault",
            "--norestore",
            "--nolockcheck",
            "--convert-to",
            convert_to_arg,
            "--outdir",
            str(output_dir),
            str(input_file),
        ]

        try:
            process = subprocess.run(cmd, **subprocess_kwargs)
        except subprocess.TimeoutExpired as exc:
            stderr = (exc.stderr or b"").decode(errors="ignore").strip()
            stdout = (exc.stdout or b"").decode(errors="ignore").strip()
            details = stderr or stdout or "conversion timed out"
            logger.error(f"convert_file failed: input_path={input_file}, target_format={target_format}, "
                         f"returncode={process.returncode}, details={details}",)
            raise ConvertToPdfError(
                f"LibreOffice convert timed out after {subprocess_kwargs['timeout']} seconds: {details}"
            ) from exc

    if process.returncode != 0:
        stderr = process.stderr.decode(errors="ignore").strip()
        stdout = process.stdout.decode(errors="ignore").strip()
        details = stderr or stdout or "unknown error"
        logger.error(f"convert_file failed: input_path={input_file}, target_format={target_format}, returncode={process.returncode}, details={details}")
        raise ConvertToPdfError(f"LibreOffice convert failed: {details}")

    output_file = _expected_output_path(input_file, output_dir, target_format)
    if not output_file.exists():
        logger.error(f"convert_file failed: output not found after success, input_path={input_file}, expected_output={output_file}")
        raise ConvertToPdfError(f"LibreOffice reported success, but output file was not found: {output_file}")

    logger.info(f"convert_file end: input_path={input_file}, output_file={output_file}, target_format={target_format}")
    return str(output_file)


def convert_file_to_pdf(input_path, output_dir):
    """Convert a single document (ppt, doc, xls, etc.) to PDF."""
    return convert_file(input_path, output_dir, "pdf")


def convert_doc_to_docx(input_path, output_dir):
    """Convert a legacy Word document (.doc) to .docx."""
    return convert_file(input_path, output_dir, "docx")


def convert_ppt_to_pptx(input_path, output_dir):
    """Convert a legacy PowerPoint presentation (.ppt) to .pptx."""
    return convert_file(input_path, output_dir, "pptx")


def convert_xls_to_xlsx(input_path, output_dir):
    """Convert a legacy Excel workbook (.xls) to .xlsx."""
    return convert_file(input_path, output_dir, "xlsx")


def convert_legacy_office_to_modern(input_path, output_dir=None):
    if not output_dir:
        output_dir = tempfile.mkdtemp()
    """Convert .doc/.ppt/.xls files to .docx/.pptx/.xlsx based on the input suffix."""
    input_suffix = Path(input_path).suffix.lower()
    target_format = LEGACY_TO_MODERN_FORMATS.get(input_suffix)
    if not target_format:
        supported = ", ".join(sorted(LEGACY_TO_MODERN_FORMATS))
        raise ConvertToPdfError(
            f"Unsupported legacy Office format: {input_suffix or '<no suffix>'}. Supported inputs: {supported}"
        )

    return convert_file(input_path, output_dir, target_format)


if __name__ == '__main__':
    convert_legacy_office_to_modern(r"D:\file\docx\test.doc",
                        r"D:\file\docx")