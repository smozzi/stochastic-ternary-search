"""Uniform renewed batches via partial Fisher-Yates; reproducible native state."""

import ctypes
from pathlib import Path
import torch


class FastBatchStream:
    def __init__(self, size, batch, seed, threads=8):
        self.size = size
        self.batch = batch
        self.seed = seed
        self.threads = threads
        self.draws = 0
        self.buffers = None
        self.full_ids = None
        if not 1 <= batch <= size or not 1 <= threads <= 64:
            raise ValueError("invalid stream")
        self.lib = ctypes.CDLL(str(Path(__file__).with_name("_batch.so")))
        v = ctypes.c_void_p
        i = ctypes.c_int
        for name, (args, result) in {
            "fb_create": ([i, ctypes.c_uint64], v),
            "fb_destroy": ([v], None),
            "fb_state_size": ([v], i),
            "fb_save": ([v, v], None),
            "fb_load": ([v, v], None),
            "fb_draw": ([v, v, v, i, v, v, v, i], i),
        }.items():
            f = getattr(self.lib, name)
            f.argtypes = args
            f.restype = result
        self.handle = self.lib.fb_create(size, seed)
        if not self.handle:
            raise RuntimeError("native allocation failed")

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.fb_destroy(self.handle)
            self.handle = None

    def set_batch(self, batch):
        if not self.batch <= batch <= self.size:
            raise ValueError("batch must not shrink")
        if batch != self.batch:
            self.buffers = None
        self.batch = batch

    @torch.inference_mode()
    def draw(self, data):
        x, y = data
        if (
            x.shape != (self.size, 49)
            or x.dtype != torch.float32
            or y.shape != (self.size,)
            or y.dtype != torch.int64
            or not x.is_contiguous()
            or not y.is_contiguous()
        ):
            raise ValueError("wrong data")
        self.draws += 1
        if self.batch == self.size:
            if self.full_ids is None:
                self.full_ids = torch.arange(self.size)
            return data, self.full_ids
        if self.buffers is None:
            self.buffers = (
                torch.empty(self.batch, 49),
                torch.empty(self.batch, dtype=torch.int64),
                torch.empty(self.batch, dtype=torch.int64),
            )
        ox, oy, ids = self.buffers
        status = self.lib.fb_draw(
            self.handle,
            x.data_ptr(),
            y.data_ptr(),
            self.batch,
            ox.data_ptr(),
            oy.data_ptr(),
            ids.data_ptr(),
            self.threads,
        )
        if status:
            raise RuntimeError(f"batch draw {status}")
        return (ox, oy), ids

    def state_dict(self):
        state = torch.empty(self.lib.fb_state_size(self.handle), dtype=torch.uint8)
        self.lib.fb_save(self.handle, state.data_ptr())
        return dict(
            size=self.size,
            batch=self.batch,
            seed=self.seed,
            draws=self.draws,
            native=state,
        )

    def load_state_dict(self, s):
        if (s["size"], s["seed"]) != (self.size, self.seed) or s[
            "native"
        ].numel() != self.lib.fb_state_size(self.handle):
            raise ValueError("wrong state")
        self.batch = s["batch"]
        self.draws = s["draws"]
        self.buffers = None
        self.full_ids = None
        self.lib.fb_load(self.handle, s["native"].data_ptr())
