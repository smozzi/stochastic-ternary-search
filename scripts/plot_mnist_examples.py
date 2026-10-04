"""Render real 7×7 test digits, sampled by correctness (19 successes / 1 error)."""

import argparse
import gzip
import hashlib
import json
import struct
import urllib.request
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}


def test_data(root):
    root.mkdir(parents=True, exist_ok=True)
    arrays = []
    for name, checksum in FILES.items():
        path = root / name
        if not path.exists():
            tmp = path.with_suffix(".part")
            urllib.request.urlretrieve(
                "https://ossci-datasets.s3.amazonaws.com/mnist/" + name, tmp
            )
            tmp.replace(path)
        if hashlib.md5(path.read_bytes()).hexdigest() != checksum:
            raise ValueError(f"MNIST checksum mismatch: {name}")
        raw = gzip.decompress(path.read_bytes())
        magic, count = struct.unpack(">II", raw[:8])
        if magic == 2051:
            assert struct.unpack(">II", raw[8:16]) == (28, 28)
            image = np.frombuffer(raw[16:], dtype=np.uint8).reshape(count, 7, 4, 7, 4)
            # Same float32 average pooling as the training loader.
            pooled = (image.astype(np.float32) / 255).mean(axis=(2, 4))
            arrays.append(pooled)
        else:
            assert magic == 2049
            arrays.append(np.frombuffer(raw[8:], dtype=np.uint8))
    return arrays


def predict(images, snapshot):
    """Independent NumPy inference; same discrete weights and fixed calibration."""
    c = np.asarray(snapshot["codes"], dtype=np.int8)
    assert len(c) == 3914 and np.isin(c, [-1, 0, 1]).all()
    state = snapshot["model"]
    delta = np.asarray(state["delta"], dtype=np.float32)
    tau = np.asarray(state["tau"], dtype=np.float32)
    levels = np.asarray(state["gain_levels"], dtype=np.float32)
    x = (images.reshape(-1, 49) - np.float32(0.1307)) / np.float32(0.3081)
    w = c[:3136].reshape(64, 49).astype(np.float32)
    v = c[3136:3776].reshape(10, 64).astype(np.float32)
    g = levels[c[3776:3840].astype(np.int64) + 1]
    b = c[3840:3904].astype(np.float32)
    out_b = c[3904:].astype(np.float32)
    u = (x @ w.T + delta[:64] * b) / tau[:64]
    h = (u + u * u) * g
    logits = (h @ v.T + delta[64:] * out_b) / tau[64:]
    assert np.isfinite(logits).all()
    return logits.argmax(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--selection-seed", type=int, default=20261004)
    args = parser.parse_args()
    model_path = ROOT / "results/mnist_example_model.json"
    snapshot = json.loads(model_path.read_text())
    images, labels = test_data(args.data)
    predictions = predict(images, snapshot)
    correct = predictions == labels
    assert int(correct.sum()) == snapshot["test_correct"] == 9393
    assert len(labels) == snapshot["test_count"] == 10000
    rng = np.random.default_rng(args.selection_seed)
    ids = np.concatenate(
        [
            rng.choice(np.flatnonzero(correct), 19, replace=False),
            rng.choice(np.flatnonzero(~correct), 1, replace=False),
        ]
    )
    rng.shuffle(ids)
    assert int(correct[ids].sum()) == 19 and len(set(ids)) == 20
    fig = plt.figure(figsize=(12, 6.4), layout="constrained")
    grid = fig.add_gridspec(4, 5)
    for slot, index in enumerate(ids):
        inner = grid[slot // 5, slot % 5].subgridspec(1, 2, width_ratios=[1, 1.1])
        ax = fig.add_subplot(inner[0])
        ax.imshow(images[index], cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks(np.arange(-0.5, 7, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 7, 1), minor=True)
        ax.grid(which="minor", color="#888888", linewidth=0.25, alpha=0.3)
        ax.tick_params(
            which="both", bottom=False, left=False, labelbottom=False, labelleft=False
        )
        for spine in ax.spines.values():
            spine.set_visible(False)
        text = fig.add_subplot(inner[1])
        text.axis("off")
        text.text(0, 0.69, f"STS: {predictions[index]}", fontsize=12, weight="bold")
        text.text(0, 0.43, f"True: {labels[index]}", fontsize=10)
        text.text(
            0,
            0.18,
            "Correct" if correct[index] else "Error",
            fontsize=10,
            color="#19703b" if correct[index] else "#b52929",
        )
    fig.suptitle(
        "MNIST 7×7 · STS predictions\n"
        "19 correct + 1 error · random selection within each group",
        fontsize=14,
    )
    target = ROOT / "docs/figures/mnist_examples.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=180)
    plt.close(fig)
    report = {
        "purpose": "Illustrative correctness-stratified selection; not an accuracy estimate",
        "selection_seed": args.selection_seed,
        "model_seed": snapshot["seed"],
        "target_seconds": snapshot["target_seconds"],
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "test_accuracy": float(correct.mean()),
        "image_processing": "28×28 /255; 4×4 average pooling; displayed 7×7 with nearest interpolation",
        "selected": [
            {
                "test_index": int(i),
                "true": int(labels[i]),
                "prediction": int(predictions[i]),
                "correct": bool(correct[i]),
            }
            for i in ids
        ],
    }
    (ROOT / "results/mnist_examples.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        f"Rendered 20 real test images (19 correct, 1 error); full-test accuracy {correct.mean():.2%}"
    )


if __name__ == "__main__":
    main()
