import math
import torch
from sts.baselines import (
    BaselineMLP,
    learning_rate,
    ternarize,
    exported_forward,
    AutogradBatchStream,
    optimizer,
    train_batch,
)
from test_reference import equal


def test_schedule_endpoints_and_monotonicity():
    values = [learning_rate(0.001, "cosine", n, 1000) for n in range(1002)]
    assert values[0] == 0.001 and math.isclose(values[1000], 0.00001)
    assert values[1001] == values[1000]
    assert all(x >= y for x, y in zip(values, values[1:]))


def test_quantized_forward_and_identity_surrogate():
    m = BaselineMLP()
    w = torch.tensor([[-2.0, -0.5, 0.0, 0.5, 2.0]], requires_grad=True)
    q = m.weight(w)
    c, s = ternarize(w)
    assert torch.equal(q.detach(), c.float() * s[:, None])
    v = torch.arange(5.0).reshape(1, 5)
    (q * v).sum().backward()
    assert torch.equal(w.grad, v)
    x = torch.randn(37, 49)
    with torch.no_grad():
        assert torch.equal(m(x), exported_forward(x, m.export()))


def test_autograd_buffers_and_optimizer_resume():
    torch.set_num_threads(8)
    gen = torch.Generator().manual_seed(737)
    data = (
        torch.randn(149, 49, generator=gen),
        torch.randint(10, (149,), generator=gen),
    )
    m = BaselineMLP(seed=737)
    o = optimizer(m, 0.001)
    b = AutogradBatchStream(149, 32, 737)
    for _ in range(4):
        (x, y), _ = b.draw(data)
        assert not x.is_inference()
        train_batch(m, o, x, y)
    import copy

    n = BaselineMLP(seed=0)
    n.load_state_dict(m.state_dict())
    no = optimizer(n, 0.001)
    no.load_state_dict(copy.deepcopy(o.state_dict()))
    nb = AutogradBatchStream(149, 32, 737)
    nb.load_state_dict(b.state_dict())
    for _ in range(16):
        (x, y), ids = b.draw(data)
        (xx, yy), ii = nb.draw(data)
        assert torch.equal(ids, ii)
        assert train_batch(m, o, x, y) == train_batch(n, no, xx, yy)
    equal(m.state_dict(), n.state_dict())
    equal(o.state_dict(), no.state_dict())
    equal(b.state_dict(), nb.state_dict())
