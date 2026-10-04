"""Sparse candidate representation using the unchanged sparse RCT engine."""

import ctypes, bisect
from pathlib import Path
import torch
import copy
from .controller import GridController
from .controller import predict_batch_cost


class Search:
    def __init__(self, base, width, seed, arms, simd=True, precision=31, decay=1.0):
        if precision != 31 or decay != 1.0:
            raise ValueError("invalid confidence settings")
        self.precision = precision
        self.decay = decay
        if not simd:
            raise ValueError("512 scorer requires SIMD")
        self.simd = True
        self.width = width
        self.n = 61 * width + 10
        self.weight_n = 59 * width
        self.leader = base.clone().contiguous()
        self.evidence = torch.zeros(
            self.n if precision else 0,
            dtype={31: torch.int32, 32: torch.float32, 64: torch.float64}[precision],
        )
        self.mass = torch.zeros_like(self.evidence)
        self.lib = ctypes.CDLL(str(Path(__file__).with_name("_search.so")))
        v = ctypes.c_void_p
        i = ctypes.c_int
        d = ctypes.c_double
        defs = {
            "nrct_simd": ([v, i], None),
            "nrct_create": ([i, v, v, v, ctypes.c_uint64, i, d], v),
            "nrct_destroy": ([v], None),
            "nrct_step": ([v, v, v, i, v, v, i, i, v], i),
            "nrct_state_size": ([v], i),
            "nrct_save": ([v, v], None),
            "nrct_load": ([v, v], None),
            "nrct_draw": ([v, i, v], None),
            "nrct_extension_count": ([v], i),
            "nrct_counter": ([v, i, i], ctypes.c_int64),
            "nrct_set_counter": ([v, i, ctypes.c_int64, ctypes.c_int64], None),
        }
        for name, (args, result) in defs.items():
            fn = getattr(self.lib, name)
            fn.argtypes = args
            fn.restype = result
        self.handle = self.lib.nrct_create(
            width,
            self.leader.data_ptr(),
            self.evidence.data_ptr(),
            self.mass.data_ptr(),
            seed,
            precision,
            decay,
        )
        if not self.handle:
            raise RuntimeError("native create failed")
        self.lib.nrct_simd(self.handle, 1)
        self.count_generator = torch.Generator().manual_seed(5_000_000 + seed)
        self.conditional = GridController(arms)
        self.rounds = 0
        self.leader_score = self.best_score = float("-inf")
        self.last_count = 0
        self.last_reward = 0.0
        self.last_cost = 0.0
        self._cdf_anchor = None
        self._out = torch.empty(8, dtype=torch.float64)

    def __del__(self):
        if getattr(self, "handle", None):
            self.lib.nrct_destroy(self.handle)
            self.handle = None

    def state_dict(self):
        state = torch.empty(self.lib.nrct_state_size(self.handle), dtype=torch.uint8)
        self.lib.nrct_save(self.handle, state.data_ptr())
        return dict(
            width=self.width,
            simd=self.simd,
            leader=self.leader.clone(),
            evidence=self.evidence.clone(),
            mass=self.mass.clone(),
            native=state,
            count_rng=self.count_generator.get_state(),
            controller=copy.deepcopy(self.conditional.state_dict()),
            rounds=self.rounds,
            leader_score=self.leader_score,
            best_score=self.best_score,
            last_count=self.last_count,
            last_reward=self.last_reward,
            last_cost=self.last_cost,
            backend="sts-int32-v1",
            precision=self.precision,
            decay=self.decay,
        )

    def load_state_dict(self, s):
        if (
            s.get("backend") != "sts-int32-v1"
            or s.get("precision") != self.precision
            or s.get("decay") != self.decay
        ):
            raise ValueError("wrong backend snapshot")
        if (
            s["width"] != self.width
            or s["simd"] != self.simd
            or s["evidence"].dtype != self.evidence.dtype
        ):
            raise ValueError("wrong snapshot")
        self.leader.copy_(s["leader"])
        self.evidence.copy_(s["evidence"])
        self.mass.copy_(s["mass"])
        self.lib.nrct_load(self.handle, s["native"].data_ptr())
        self.count_generator.set_state(s["count_rng"])
        self.conditional.load_state_dict(s["controller"])
        for k in (
            "rounds",
            "leader_score",
            "best_score",
            "last_count",
            "last_reward",
            "last_cost",
        ):
            setattr(self, k, s[k])
        self._cdf_anchor = None

    @torch.inference_mode()
    def step(self, model, data, threads=8):
        if model.layout.n != self.n or model.layout.layers != 2:
            raise ValueError("wrong layout")
        x, y = data
        if (
            x.dtype != torch.float32
            or y.dtype != torch.int64
            or x.ndim != 2
            or x.shape[1] != 49
            or y.shape != (len(x),)
            or not len(x)
            or not x.is_contiguous()
            or not y.is_contiguous()
        ):
            raise ValueError("wrong batch")
        if self._cdf_anchor != self.conditional.k0:
            self._cdf = self.conditional.probabilities("k").cumsum(0).tolist()
            self._cdf[-1] = 1.0
            self._cdf_anchor = self.conditional.k0
        u = float(torch.rand((), generator=self.count_generator, dtype=torch.float64))
        self.last_count = k = self.conditional.arms[bisect.bisect_right(self._cdf, u)]
        status = self.lib.nrct_step(
            self.handle,
            x.data_ptr(),
            y.data_ptr(),
            len(x),
            model.delta.data_ptr(),
            model.tau.data_ptr(),
            threads,
            k,
            self._out.data_ptr(),
        )
        if status:
            raise RuntimeError(f"native step status {status}")
        before, best, gain, changed, weights, active, extra, selected = (
            self._out.tolist()
        )
        self.leader_score = self.best_score = best
        self.last_reward = gain
        self.last_cost = predict_batch_cost(int(active), bool(extra), len(x))
        self.conditional.record(k, 12, gain, self.last_cost)
        self.rounds += 1
        return dict(
            baseline_score=before,
            selected_score=best,
            gain=gain,
            changed=int(changed),
            weights_changed=int(weights),
            evaluations=1 + int(active) + int(extra),
            active=int(active),
            extra=int(extra),
            k=k,
            cost=self.last_cost,
            selected=int(selected),
        )
