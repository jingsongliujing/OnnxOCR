"""OnnxOCR 统一日志模块，基于 loguru。

用法:
    from onnxocr.logger import get_logger
    log = get_logger("predict_det")
    log.info("检测模型加载完成")
"""

import sys
import logging

from loguru import logger

# 移除 loguru 默认 handler，避免重复输出
logger.remove()

# 默认控制台输出格式
_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

# 添加默认控制台 handler
logger.add(
    sys.stderr,
    format=_DEFAULT_FORMAT,
    level="INFO",
    colorize=True,
)


class InterceptHandler(logging.Handler):
    """将标准库 logging 的日志桥接到 loguru。"""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())


# 拦截标准库 logging 的根 logger
logging.basicConfig(handlers=[InterceptHandler()], level=logging.WARNING, force=True)

# 桥接 rapid_layout / rapid_table 的独立 logger
for _name in ("RapidLayout", "RapidTable"):
    _logging_logger = logging.getLogger(_name)
    _logging_logger.handlers = [InterceptHandler()]
    _logging_logger.propagate = False


def get_logger(name: str = "OnnxOCR"):
    """获取带模块标识的 logger。

    Args:
        name: 模块标识，用于区分日志来源，如 "predict_det"、"ocr" 等。

    Returns:
        loguru.Logger: 绑定了模块名称的 logger 实例。
    """
    return logger.bind(name=name)


def add_file_sink(path: str, level: str = "DEBUG", rotation: str = "10 MB"):
    """添加文件日志输出。

    Args:
        path: 日志文件路径。
        level: 文件日志级别，默认 DEBUG。
        rotation: 日志轮转大小，默认 10 MB。
    """
    logger.add(
        path,
        format=_DEFAULT_FORMAT,
        level=level,
        rotation=rotation,
        encoding="utf-8",
    )
