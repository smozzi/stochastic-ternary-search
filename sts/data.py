"""Official MNIST IDX files, checksummed; fixed split independent of run seed."""

import gzip
import hashlib
from pathlib import Path
import struct
import urllib.request
import torch

FILES = {
    "train-images-idx3-ubyte.gz": "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    "train-labels-idx1-ubyte.gz": "d53e105ee54ea40749a09fcbcd1e9432",
    "t10k-images-idx3-ubyte.gz": "9fb629c4189551a2d022fa330f9573f3",
    "t10k-labels-idx1-ubyte.gz": "ec29112dd5afa0611ce80d1b7f02629c",
}
BASE = "https://ossci-datasets.s3.amazonaws.com/mnist/"


def load_idx(root, name):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    if not path.exists():
        temporary = path.with_suffix(".part")
        urllib.request.urlretrieve(BASE + name, temporary)
        temporary.replace(path)
    if hashlib.md5(path.read_bytes()).hexdigest() != FILES[name]:
        raise ValueError(f"Checksum mismatch: {path}; remove and download again")
    raw = gzip.decompress(path.read_bytes())
    magic, count = struct.unpack(">II", raw[:8])
    if magic == 2051:
        rows, cols = struct.unpack(">II", raw[8:16])
        return torch.frombuffer(bytearray(raw[16:]), dtype=torch.uint8).reshape(
            count, rows * cols
        )
    if magic == 2049:
        return torch.frombuffer(bytearray(raw[8:]), dtype=torch.uint8).long()
    raise ValueError("Invalid IDX magic")


def nested_indices(labels, per_class=5000):
    if per_class < 20:
        raise ValueError("must retain original 20 examples per class")
    rng = torch.Generator().manual_seed(20260928)
    initial = []
    validation = []
    remaining = []
    for c in range(10):
        candidates = (labels == c).nonzero().flatten()
        if len(candidates) < per_class + 100:
            raise ValueError("not enough examples to preserve held-out validation")
        chosen = candidates[torch.randperm(len(candidates), generator=rng)]
        initial.extend(chosen[:20].tolist())
        validation.extend(chosen[20:120].tolist())
        remaining.append(chosen[120 : 120 + per_class - 20])
    # Original 200 examples retain exactly their former order. Extras interleave classes.
    extra = torch.stack(remaining, dim=1).reshape(-1).tolist()
    return initial + extra, validation


from functools import lru_cache
from torch.nn import functional as F


def normalize_input(x, mode):
    return (x - 0.1307) / 0.3081


@lru_cache(maxsize=1)
def load_full(root):
    raw = load_idx(root, "train-images-idx3-ubyte.gz")
    labels = load_idx(root, "train-labels-idx1-ubyte.gz")
    _, validation = nested_indices(labels)
    mask = torch.ones(len(labels), dtype=torch.bool)
    mask[validation] = False
    indices = mask.nonzero().flatten()

    def prepare(xx):
        return normalize_input(
            F.avg_pool2d(xx.float().reshape(-1, 1, 28, 28) / 255, 4).flatten(1),
            "standard",
        ).contiguous()

    tx = load_idx(root, "t10k-images-idx3-ubyte.gz")
    ty = load_idx(root, "t10k-labels-idx1-ubyte.gz")
    assert len(indices) == 59000 and len(validation) == 1000
    return dict(
        train=(prepare(raw[indices]), labels[indices].contiguous()),
        validation=(prepare(raw[validation]), labels[validation].contiguous()),
        test=(prepare(tx), ty.contiguous()),
        train_indices=indices,
        validation_indices=torch.tensor(validation),
    )
