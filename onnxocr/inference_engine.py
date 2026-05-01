from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import onnxruntime


Provider = Union[str, Tuple[str, Dict[str, Any]]]


InferenceSession = onnxruntime.InferenceSession
SessionOptions = onnxruntime.SessionOptions
GraphOptimizationLevel = onnxruntime.GraphOptimizationLevel


def get_available_providers() -> List[str]:
    return onnxruntime.get_available_providers()


def get_device() -> str:
    return onnxruntime.get_device()


def is_session(value: Any) -> bool:
    return isinstance(value, InferenceSession)


def build_providers(
    use_gpu: bool = False,
    gpu_id: int = 0,
    providers: Optional[Sequence[Provider]] = None,
) -> List[Provider]:
    if providers is not None:
        return list(providers)

    if use_gpu:
        return [
            (
                "CUDAExecutionProvider",
                {"cudnn_conv_algo_search": "DEFAULT", "device_id": gpu_id},
            ),
            "CPUExecutionProvider",
        ]
    return ["CPUExecutionProvider"]


def create_session(
    model_path: str,
    providers: Optional[Sequence[Provider]] = None,
    use_gpu: bool = False,
    gpu_id: int = 0,
    sess_options: Optional[SessionOptions] = None,
) -> InferenceSession:
    session_providers = build_providers(
        use_gpu=use_gpu,
        gpu_id=gpu_id,
        providers=providers,
    )
    return InferenceSession(
        model_path,
        sess_options=sess_options,
        providers=session_providers,
    )
