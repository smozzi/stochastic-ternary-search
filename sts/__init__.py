"""Gradient-free stochastic search in a ternary parameter space."""

from .model import DepthMLP, initial_codes
from .search import Search

__all__ = ["DepthMLP", "Search", "initialize"]


def initialize(seed, train, width=64):
    model = DepthMLP(f"single{width}")
    codes = initial_codes(width, seed)
    model.calibrate(train[0], codes)
    return Search(codes, width, seed, model.layout.arms), model
