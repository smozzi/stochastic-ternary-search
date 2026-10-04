import ctypes
from pathlib import Path
import torch


class DepthScorer:
    def __init__(self, threads=8, incremental=None):
        if type(threads) is not int or not 1 <= threads <= 64:
            raise ValueError("invalid thread count")
        self.threads = threads
        self.incremental = incremental
        self.library = ctypes.CDLL(str(Path(__file__).with_name("_probe.so")))
        self.native = self.library.probe_scores
        self.native.argtypes = [
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.native.restype = ctypes.c_int

    @torch.inference_mode()
    def __call__(self, model, base, candidates, data, k=1):
        x, y = data
        model._check(base)
        model._check(candidates)
        if (
            base.ndim != 1
            or candidates.ndim != 2
            or not 1 <= len(candidates) <= 32
            or model.calibration is None
        ):
            raise ValueError("invalid calibrated candidates")
        if (
            x.dtype != torch.float32
            or x.ndim != 2
            or x.shape[1] != 49
            or y.dtype != torch.int64
            or y.shape != (len(x),)
            or not len(x)
            or bool(((y < 0) | (y >= 10)).any())
        ):
            raise ValueError("invalid dataset")
        sizes = torch.tensor(model.layout.sizes, dtype=torch.int32)
        tensors = (x, y, base, candidates, sizes, model.delta, model.tau)
        if any(v.device.type != "cpu" or not v.is_contiguous() for v in tensors):
            raise ValueError("contiguous CPU tensors required")
        out = torch.empty(len(candidates), dtype=torch.float64)
        inc = len(candidates) > 1 if self.incremental is None else self.incremental
        status = self.native(
            x.data_ptr(),
            y.data_ptr(),
            len(x),
            base.data_ptr(),
            candidates.data_ptr(),
            len(candidates),
            sizes.data_ptr(),
            model.layout.layers,
            model.delta.data_ptr(),
            model.tau.data_ptr(),
            self.threads,
            int(inc),
            out.data_ptr(),
        )
        if status:
            raise RuntimeError(f"native status {status}")
        if not torch.isfinite(out).all():
            raise FloatingPointError("nonfinite network score")
        return out
