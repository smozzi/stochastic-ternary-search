"""Small CPU training entry point for the current STS reference."""

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F
from . import initialize
from .batch import FastBatchStream
from .data import load_full
from .probe import NestedProbePool, RelativeNestedProbeTrend
from .scorer import DepthScorer


@torch.inference_mode()
def metrics(model, codes, data):
    x, y = data
    logits = model.forward_many(x, codes[None])[0]
    return dict(
        loss=float(F.cross_entropy(logits, y, reduction="sum").double() / len(y)),
        accuracy=float((logits.argmax(-1) == y).double().mean()),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Gradient-free stochastic ternary training on MNIST 7×7."
    )
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--width", type=int, choices=[32, 64, 128], default=64)
    parser.add_argument("--seed", type=int, default=500)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=600)
    parser.add_argument(
        "--iterations",
        type=int,
        help="Use a deterministic number of rounds instead of a time budget.",
    )
    args = parser.parse_args()
    if (
        not 1 <= args.threads <= 64
        or args.seconds <= 0
        or (args.iterations is not None and args.iterations < 1)
    ):
        parser.error("positive budget and 1–64 threads required")
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    data = load_full(args.data)
    train = data["train"]
    probe_ids = torch.randperm(
        len(train[1]), generator=torch.Generator().manual_seed(13_000_000 + args.seed)
    )
    pool = NestedProbePool(train, probe_ids)
    probe_fn = DepthScorer(args.threads)

    def setup():
        search, model = initialize(args.seed, train, args.width)
        return (
            search,
            model,
            FastBatchStream(len(train[1]), 32, args.seed, args.threads),
            RelativeNestedProbeTrend(len(train[1]), min_relative=0.01),
        )

    def update(search, model, stream, trend, n):
        batch, _ = stream.draw(train)
        used = stream.batch
        info = search.step(model, batch, args.threads)
        if trend.active and n % 16 == 0:
            loss = -float(
                probe_fn(model, search.leader, search.leader[None], pool.get(used), 1)[
                    0
                ]
            )
            event = trend.observe(loss)
            stream.set_batch(trend.batch)
            if event is not None and event["doubled"]:
                print(f"round={n} batch={stream.batch}", flush=True)
            if not trend.active:
                pool.release()
        return info

    # Disposable warmup; its weights and RNG states are not reused.
    with torch.inference_mode():
        warm = setup()
        for n in range(1, 5):
            update(*warm, n)
        del warm
        search, model, stream, trend = setup()
        started = time.perf_counter()
        checkpoints = []
        targets = [
            t
            for t in [5, 10, 20, 30, 60, 120, 180, 240, 360, 480, 600]
            if t < args.seconds
        ] + [args.seconds]

        def validate(target, n):
            value = metrics(model, search.leader, data["validation"])
            point = dict(
                target_seconds=target,
                round=n,
                seconds=time.perf_counter() - started,
                batch=stream.batch,
                validation=value,
            )
            checkpoints.append(point)
            print(json.dumps(point), flush=True)

        validate(0, 0)
        n = 0
        while True:
            n += 1
            update(search, model, stream, trend, n)
            elapsed = time.perf_counter() - started
            if args.iterations is None:
                crossed = []
                while targets and elapsed >= targets[0]:
                    crossed.append(targets.pop(0))
                for target in crossed:
                    validate(target, n)
                done = not targets
            else:
                done = n >= args.iterations
                if done:
                    validate(None, n)
            if done:
                break
        duration = time.perf_counter() - started
        summary = dict(
            seed=args.seed,
            width=args.width,
            threads=args.threads,
            rounds=n,
            seconds=duration,
            final_batch=stream.batch,
            variables=search.n,
            confidence_bytes=8 * search.n,
            extensions_int64=search.lib.nrct_extension_count(search.handle),
            train=metrics(model, search.leader, train),
            validation=checkpoints[-1]["validation"],
            test=metrics(model, search.leader, data["test"]),
            checkpoints=checkpoints,
        )
        torch.save(
            dict(
                search=search.state_dict(),
                model=model.state_dict(),
                calibration=model.calibration,
                stream=stream.state_dict(),
                trend=trend.state_dict(),
                probe_ids=probe_ids,
            ),
            args.output / "checkpoint.pt",
        )
    source = Path(__file__).parent
    provenance = dict(
        torch=torch.__version__,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        source_sha256={
            str(f.relative_to(source)): hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(source.rglob("*"))
            if f.is_file() and f.suffix in (".py", ".cpp", ".hpp", ".json", ".so")
        },
        dataset_sha256={
            f"{split}/{j}": hashlib.sha256(t.numpy().tobytes()).hexdigest()
            for split in ("train", "validation", "test")
            for j, t in enumerate(data[split])
        },
    )
    for name, value in [("summary", summary), ("provenance", provenance)]:
        (args.output / f"{name}.json").write_text(json.dumps(value, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in summary.items() if k != "checkpoints"}, indent=2)
    )


if __name__ == "__main__":
    main()
