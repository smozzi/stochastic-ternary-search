"""Isolated Adam/TWN-like STE benchmark with deterministic update-based schedules."""

import argparse, copy, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from sts.baselines import (
    BaselineMLP as ClassicalWide,
    AutogradBatchStream,
    ternarize,
    exported_forward,
    optimizer,
    train_batch,
    metrics,
    learning_rate as rate,
)
from sts.learned_steps import LearnedStepMLP
from sts.data import load_full


def assert_exact(a, b):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_exact(a[k], b[k])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_exact(x, y)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["adam", "adam_ste"], required=True)
    p.add_argument("--batch", type=int, choices=[128, 5000], required=True)
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--schedule", choices=["constant", "cosine"], required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--quantizer", choices=["twn", "learned"], default="twn")
    p.add_argument("--seconds", type=float, required=True)
    p.add_argument("--horizon", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--commit", default="local")
    p.add_argument("--data", type=Path, default=Path("data"))
    p.add_argument("--pilot", action="store_true")
    a = p.parse_args()
    if a.seconds <= 0 or a.lr <= 0 or a.horizon <= 0:
        p.error("positive budget/rate/horizon required")
    if a.mode == "adam" and a.quantizer != "twn":
        p.error("learned quantizer is only for adam_ste")
    a.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(8)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_full(a.data)
    train = data["train"]

    def fresh():
        model_class = LearnedStepMLP if a.quantizer == "learned" else ClassicalWide
        m = model_class(a.mode, 64, a.seed)
        return (
            m,
            optimizer(m, a.lr),
            AutogradBatchStream(len(train[1]), a.batch, a.seed),
        )

    def snapshot(m, o, b):
        return dict(
            model=copy.deepcopy(m.state_dict()),
            optimizer=copy.deepcopy(o.state_dict()),
            stream=copy.deepcopy(b.state_dict()),
        )

    def update(m, o, b, n):
        lr = rate(a.lr, a.schedule, n - 1, a.horizon)
        for group in o.param_groups:
            group["lr"] = lr
        (x, y), ids = b.draw(train)
        loss = train_batch(m, o, x, y)
        if a.quantizer == "learned":
            m.project_steps()
        return loss, lr

    m, o, b = fresh()
    for n in range(1, 5):
        update(m, o, b, n)
    m, o, b = fresh()
    points = []
    targets = sorted(
        {0.0, a.seconds}
        | {
            t
            for t in [5.0, 10.0, 20.0, 30.0, 60.0, 120.0, 180.0, 240.0]
            if t < a.seconds
        }
    )
    started = time.perf_counter()

    def checkpoint(n, target):
        v = metrics(m, data["validation"])
        state = snapshot(m, o, b)
        points.append(
            dict(
                target=target,
                seconds=time.perf_counter() - started,
                updates=n,
                lr=o.param_groups[0]["lr"],
                validation=v,
                state=state,
            )
        )
        print(
            json.dumps({k: v for k, v in points[-1].items() if k != "state"}),
            flush=True,
        )

    checkpoint(0, targets.pop(0))
    n = 0
    history = []
    while targets:
        if n % 512 == 0:
            replay_start = n
            replay_state = snapshot(m, o, b)
        n += 1
        loss, lr = update(m, o, b, n)
        elapsed = time.perf_counter() - started
        history.append((n, elapsed, loss, lr))
        while targets and elapsed >= targets[0]:
            checkpoint(n, targets.pop(0))
    duration = time.perf_counter() - started
    final = snapshot(m, o, b)
    rm, ro, rb = fresh()
    rm.load_state_dict(replay_state["model"])
    ro.load_state_dict(copy.deepcopy(replay_state["optimizer"]))
    rb.load_state_dict(replay_state["stream"])
    for i in range(replay_start + 1, n + 1):
        loss, lr = update(rm, ro, rb, i)
        assert loss == history[i - 1][2] and lr == history[i - 1][3]
    assert_exact(snapshot(rm, ro, rb), final)
    for point in points:
        m.load_state_dict(point["state"]["model"])
        assert metrics(m, data["validation"]) == point["validation"]
        with torch.inference_mode():
            assert torch.equal(
                m(data["validation"][0]),
                exported_forward(data["validation"][0], m.export()),
            )
        point["train"] = metrics(m, train)
        point["quantization"] = {}
        for name in ["hidden", "output"]:
            w = getattr(m, name).weight.detach()
            if a.quantizer == "learned":
                scale = getattr(m, name + "_step").detach()
                codes = (w / scale[:, None]).clamp(-1, 1).round().to(torch.int8)
            else:
                codes, scale = ternarize(w)
            point["quantization"][name] = dict(
                norm=float(w.norm()),
                max_abs=float(w.abs().max()),
                alpha_mean=float(scale.mean()),
                zero_fraction=float((codes == 0).float().mean()),
                relative_error=float(
                    (w - codes.float() * scale[:, None]).norm() / w.norm()
                ),
            )
    best = min(points, key=lambda x: (x["validation"]["loss"], x["updates"]))
    summary = dict(
        mode=a.mode,
        quantizer=a.quantizer,
        batch=a.batch,
        seed=a.seed,
        lr=a.lr,
        schedule=a.schedule,
        horizon=a.horizon,
        seconds=duration,
        updates=n,
        pilot=a.pilot,
        points=[{k: v for k, v in q.items() if k != "state"} for q in points],
        best={k: v for k, v in best.items() if k != "state"},
    )
    if not a.pilot:
        m.load_state_dict(best["state"]["model"])
        summary["best"]["test"] = metrics(m, data["test"])
        m.load_state_dict(final["model"])
        summary["terminal_test"] = metrics(m, data["test"])
    torch.save(
        dict(
            final=final,
            points=points,
            replay_state=replay_state,
            replay_start=replay_start,
        ),
        a.output / "checkpoints.pt",
    )
    np.savez_compressed(
        a.output / "history.npz",
        values=np.asarray(history),
        columns=["update", "seconds", "batch_loss", "lr"],
    )
    (a.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    package = Path(__file__).resolve().parents[1] / "sts"
    files = [Path(__file__)] + [
        p
        for p in sorted(package.rglob("*"))
        if p.is_file() and p.suffix in [".py", ".json", ".so", ".cpp", ".hpp"]
    ]
    provenance = dict(
        commit=a.commit,
        torch=torch.__version__,
        threads=8,
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()},
        source_sha256={
            q.name: hashlib.sha256(q.read_bytes()).hexdigest() for q in files
        },
        dataset_sha256={
            f"{split}/{j}": hashlib.sha256(t.numpy().tobytes()).hexdigest()
            for split in ["train", "validation", "test"]
            for j, t in enumerate(data[split])
        },
    )
    (a.output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (a.output / "audit.json").write_text(
        json.dumps(
            dict(
                all_passed=True,
                last_block_replay_exact=True,
                all_checkpoint_metrics_recomputed=True,
                inference_export_exact=True,
                test_used_for_tuning=False,
            ),
            indent=2,
        )
        + "\n"
    )
    (a.output / "FINISHED.txt").write_text("complete\n")


if __name__ == "__main__":
    main()
