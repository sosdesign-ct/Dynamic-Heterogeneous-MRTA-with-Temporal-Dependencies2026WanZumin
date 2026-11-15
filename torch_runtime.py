from __future__ import annotations

import os
import warnings

import torch


def resolve_torch_device() -> torch.device:
    """Choose runtime device based on MRTA_DEVICE.

    MRTA_DEVICE supports: auto (default), cpu, cuda.

    - auto: probe CUDA and fall back to CPU on incompatibility.
    - cpu: always use CPU.
    - cuda: require CUDA; incompatibility triggers RuntimeError.
    """
    mode = str(os.environ.get("MRTA_DEVICE", "auto")).strip().lower()
    if mode not in {"auto", "cpu", "cuda"}:
        raise ValueError("invalid MRTA_DEVICE, expected one of: auto, cpu, cuda")
    if mode == "cpu":
        return torch.device("cpu")

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            if not torch.cuda.is_available():
                if mode == "cuda":
                    raise RuntimeError("MRTA_DEVICE=cuda but torch.cuda.is_available() is False")
                return torch.device("cpu")
            device = torch.device("cuda")
            probe = torch.tensor([1.0], device=device)
            _ = (probe + 1.0).sum().item()

        for w in caught:
            msg = str(w.message)
            if "is not compatible with the current PyTorch installation" in msg:
                if mode == "cuda":
                    raise RuntimeError(
                        "MRTA_DEVICE=cuda but current torch/CUDA build is incompatible with this GPU: "
                        f"torch={torch.__version__}, cuda_runtime={torch.version.cuda}."
                    )
                print(
                    (
                        "warning: CUDA runtime is reported incompatible with this torch build "
                        f"(torch={torch.__version__}, cuda_runtime={torch.version.cuda}); "
                        "fallback to CPU."
                    ),
                    flush=True,
                )
                return torch.device("cpu")

        return device
    except Exception as exc:  # pragma: no cover - hardware/runtime dependent
        if mode == "cuda":
            raise RuntimeError(f"MRTA_DEVICE=cuda but CUDA probe failed: {exc}") from exc
        print(
            (
                "warning: CUDA probe failed "
                f"(torch={torch.__version__}, cuda_runtime={torch.version.cuda}); "
                f"fallback to CPU. detail={exc}"
            ),
            flush=True,
        )
        return torch.device("cpu")
