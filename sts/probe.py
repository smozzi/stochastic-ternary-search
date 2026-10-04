import math
import torch


class BatchDoubling:
    def __init__(self, size, initial=32, min_delta=0.0):
        if (
            type(size) is not int
            or type(initial) is not int
            or not 1 <= initial <= size
        ):
            raise ValueError("invalid sizes")
        self.min_delta = float(min_delta)
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("invalid margin")
        self.size = size
        self.batch = initial
        self.previous = None
        self.observations = 0
        self.increases = 0
        self.doublings = 0

    def observe(self, loss):
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("invalid loss")
        old = self.batch
        increase = self.previous is not None and loss > self.previous + self.min_delta
        if increase:
            self.increases += 1
            self.batch = min(self.size, 2 * self.batch)
            if self.batch != old:
                self.doublings += 1
        self.previous = loss
        self.observations += 1
        return dict(
            increased=increase,
            old_batch=old,
            next_batch=self.batch,
            doubled=self.batch != old,
        )

    def state_dict(self):
        return dict(
            size=self.size,
            min_delta=self.min_delta,
            batch=self.batch,
            previous=self.previous,
            observations=self.observations,
            increases=self.increases,
            doublings=self.doublings,
        )

    def load_state_dict(self, st):
        if (st["size"], st["min_delta"]) != (self.size, self.min_delta):
            raise ValueError("incompatible controller")
        if not 1 <= st["batch"] <= self.size:
            raise ValueError("invalid state")
        for key in ("batch", "previous", "observations", "increases", "doublings"):
            setattr(self, key, st[key])


class ProbeTrend:
    """Strict increase between consecutive means of four fixed-probe observations."""

    def __init__(self, size, initial=32, block=4, min_delta=1e-6):
        if type(block) is not int or block < 1:
            raise ValueError("invalid block")
        self.controller = BatchDoubling(size, initial, min_delta)
        self.block = block
        self.pending = []
        self.probes = 0

    @property
    def batch(self):
        return self.controller.batch

    def observe(self, loss):
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("invalid loss")
        self.pending.append(loss)
        self.probes += 1
        if len(self.pending) < self.block:
            return None
        mean = math.fsum(self.pending) / self.block
        self.pending = []
        return dict(self.controller.observe(mean), block_mean=mean)

    def state_dict(self):
        return dict(
            controller=self.controller.state_dict(),
            block=self.block,
            pending=list(self.pending),
            probes=self.probes,
        )

    def load_state_dict(self, st):
        if st["block"] != self.block:
            raise ValueError("incompatible block")
        self.controller.load_state_dict(st["controller"])
        self.pending = list(st["pending"])
        self.probes = st["probes"]


class NestedProbeTrend(ProbeTrend):
    @property
    def active(self):
        return self.batch < self.controller.size

    def observe(self, loss):
        if not self.active:
            raise RuntimeError("probe disabled at full batch")
        event = super().observe(loss)
        if event is not None and event["doubled"]:
            # The sample changes: no comparison of scores from different probe populations.
            self.controller.previous = None
            self.pending = []
        return event


class NestedProbePool:
    """Cache only the current probe, from prefixes of one independent permutation."""

    def __init__(self, data, order):
        self.data = data
        self.order = order
        self.cached_size = 0
        self.cache = None
        if (
            order.dtype != torch.int64
            or len(order) != len(data[1])
            or len(data[0]) != len(order)
        ):
            raise ValueError("incompatible pool")

    def get(self, size):
        if type(size) is not int or not 1 <= size < len(self.order):
            raise ValueError("invalid probe size")
        if size != self.cached_size:
            ids = self.order[:size]
            self.cache = (
                self.data[0][ids].contiguous(),
                self.data[1][ids].contiguous(),
            )
            self.cached_size = size
        return self.cache

    def release(self):
        self.cache = None
        self.cached_size = 0


class RelativeBatchDoubling(BatchDoubling):
    def __init__(self, size, initial=32, min_relative=0.01):
        super().__init__(size, initial, 0.0)
        self.min_relative = float(min_relative)
        if not math.isfinite(self.min_relative) or self.min_relative < 0:
            raise ValueError("invalid relative margin")

    def observe(self, loss):
        loss = float(loss)
        if not math.isfinite(loss) or loss < 0:
            raise ValueError("invalid loss")
        old = self.batch
        increase = self.previous is not None and loss > self.previous * (
            1 + self.min_relative
        )
        if increase:
            self.increases += 1
            self.batch = min(self.size, 2 * self.batch)
            if self.batch != old:
                self.doublings += 1
        self.previous = loss
        self.observations += 1
        return dict(
            increased=increase,
            old_batch=old,
            next_batch=self.batch,
            doubled=self.batch != old,
        )

    def state_dict(self):
        st = super().state_dict()
        st.pop("min_delta")
        return dict(st, min_relative=self.min_relative)

    def load_state_dict(self, st):
        if (st["size"], st["min_relative"]) != (self.size, self.min_relative):
            raise ValueError("incompatible relative controller")
        super().load_state_dict(dict(st, min_delta=0.0))


class RelativeNestedProbeTrend(NestedProbeTrend):
    def __init__(self, size, initial=32, block=4, min_relative=0.01):
        super().__init__(size, initial, block, 0.0)
        self.controller = RelativeBatchDoubling(size, initial, min_relative)
