import copy, math
import torch
from torch.nn import functional as F
from sts.learned_steps import TernaryQuantizer, LearnedStepMLP as LsqWide
from sts.baselines import exported_forward, optimizer, train_batch
from test_reference import equal as assert_exact


def test_ternary_forward_and_surrogate_gradients():
    w = torch.tensor([[-2.2, -0.8, -0.2, 0.6, 1.4]], requires_grad=True)
    step = torch.tensor([1.0], requires_grad=True)
    q = TernaryQuantizer.apply(w, step)
    assert q.tolist() == [[-1.0, -1.0, 0.0, 1.0, 1.0]]
    (q * torch.arange(1.0, 6.0)).sum().backward()
    assert w.grad.tolist() == [[0.0, 2.0, 3.0, 4.0, 0.0]]
    assert math.isclose(step.grad.item(), 5.8 / math.sqrt(5), rel_tol=1e-6)
    w = torch.tensor([[-1.0, 1.0]], requires_grad=True)
    s = torch.ones(1, requires_grad=True)
    TernaryQuantizer.apply(w, s).sum().backward()
    assert w.grad.tolist() == [[0.0, 0.0]]


def test_export_and_replay():
    torch.set_num_threads(8)
    g = torch.Generator().manual_seed(933)
    x = torch.randn(128, 49, generator=g)
    y = torch.randint(10, (128,), generator=g)
    m = LsqWide(seed=933)
    o = optimizer(m, 0.001)
    for _ in range(16):
        train_batch(m, o, x, y)
        m.project_steps()
    n = LsqWide(seed=0)
    n.load_state_dict(m.state_dict())
    no = optimizer(n, 0.001)
    no.load_state_dict(copy.deepcopy(o.state_dict()))
    for _ in range(16):
        assert train_batch(m, o, x, y) == train_batch(n, no, x, y)
        m.project_steps()
        n.project_steps()
        assert_exact(m.state_dict(), n.state_dict())
        assert_exact(o.state_dict(), no.state_dict())
        with torch.no_grad():
            assert torch.equal(m(x), exported_forward(x, m.export()))
    assert sum(p.numel() for p in m.parameters()) == 3924
    assert (
        (m.export()["hidden"]["codes"] >= -1) & (m.export()["hidden"]["codes"] <= 1)
    ).all()
