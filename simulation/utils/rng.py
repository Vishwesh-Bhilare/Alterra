"""
Seeded RNG management.

Every stochastic component in Alterra (an emitter's parameters, its burst
schedule, sensor noise draws, ...) must pull from an RNG stream that is
independent of every other component's stream, while the whole run stays
reproducible from a single top-level seed. We use numpy's SeedSequence
spawning mechanism for this rather than passing one shared Generator around
(which would make results order-dependent on call sequence).
"""
from __future__ import annotations

import hashlib

import numpy as np


class RNGManager:
    """Spawns independent, reproducible numpy Generators from one root seed."""

    def __init__(self, seed: int):
        self._root = np.random.SeedSequence(seed)
        self._spawn_count = 0

    def spawn(self) -> np.random.Generator:
        """Get a fresh, independent stream. Order-dependent — use spawn_named
        for anything that needs to be stable regardless of call order."""
        child = self._root.spawn(1)[0]
        self._spawn_count += 1
        return np.random.default_rng(child)

    def spawn_named(self, name: str) -> np.random.Generator:
        """Get an independent stream keyed by a stable string name (e.g.
        'emitter_003'). Same name always yields the same stream for a given
        root seed, regardless of call order — use this for anything that
        needs to survive refactors that reorder construction."""
        name_digest = int(hashlib.sha256(name.encode("utf-8")).hexdigest(), 16) % (2**32)
        child = np.random.SeedSequence([self._root.entropy, name_digest])
        return np.random.default_rng(child)
