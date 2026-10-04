import copy
import math
import numpy as np
import pytest
import torch
from sts import initialize


@pytest.fixture
def sample():
    torch.set_num_threads(8)
    g = torch.Generator().manual_seed(737)
    return torch.randn(32, 49, generator=g), torch.randint(10, (32,), generator=g)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            equal(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize("width", [32, 64, 128])
def test_discrete_model_and_replay(sample, width):
    a, model = initialize(737, sample, width)
    assert a.n == 61 * width + 10 and a.evidence.dtype == torch.int32
    assert torch.isfinite(model(sample[0], a.leader)).all()
    delta, tau = model.delta.clone(), model.tau.clone()
    for _ in range(128):
        a.step(model, sample)
    state = copy.deepcopy(a.state_dict())
    b, _ = initialize(737, sample, width)
    b.load_state_dict(state)
    for _ in range(64):
        equal(a.step(model, sample), b.step(model, sample))
    equal(a.state_dict(), b.state_dict())
    assert torch.equal(delta, model.delta) and torch.equal(tau, model.tau)
    assert ((a.leader >= -1) & (a.leader <= 1)).all()


def test_integer_oracle_and_overflow(sample):
    a, _ = initialize(737, sample)
    rng = np.random.default_rng(737)
    e = m = 0
    import ctypes

    fn = a.lib.nrct_update_conf
    fn.argtypes = [ctypes.c_void_p, *([ctypes.c_int] * 4)]
    fn.restype = ctypes.c_double
    for n in range(3000):
        count = int(rng.integers(0, 25))
        inc = int(rng.integers(-count * (24 - count), count * (24 - count) + 1))
        reset = n % 173 == 172
        if reset:
            e = m = 0
        else:
            e += inc
            m += count * (24 - count)
        p = fn(a.handle, 0, inc, count, int(reset))
        assert a.evidence[0].item() == e and a.mass[0].item() == m
        assert math.isclose(
            p,
            0.001 + 0.999 * math.exp(-max(-3, min(3, e / math.sqrt(24 * (m + 72))))),
            rel_tol=1e-14,
        )
    a.lib.nrct_set_counter(a.handle, 0, 2**31 - 1, 2**31 - 1)
    fn(a.handle, 0, 144, 12, 0)
    assert a.lib.nrct_extension_count(a.handle) == 1
    assert a.lib.nrct_counter(a.handle, 0, 0) == 2**31 - 1 + 144
    state = copy.deepcopy(a.state_dict())
    b, _ = initialize(737, sample)
    b.load_state_dict(state)
    equal(state, b.state_dict())
    fn(a.handle, 0, 0, 0, 1)
    assert a.lib.nrct_extension_count(a.handle) == 0
    a.lib.nrct_set_counter(a.handle, 0, 2**63 - 1, 2**63 - 1)
    state = copy.deepcopy(a.state_dict())
    assert math.isnan(fn(a.handle, 0, 144, 12, 0))
    equal(state, a.state_dict())
