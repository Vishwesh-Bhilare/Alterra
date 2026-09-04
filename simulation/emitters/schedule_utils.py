"""
Shared scheduling primitives used by multiple emitter types, so burst
behavior stays consistent across fixed/agile/periodic-scan rather than each
subclass reinventing its own on/off model.
"""
from __future__ import annotations

import numpy as np


def two_state_markov_mask(
    rng: np.random.Generator,
    length: int,
    duty_cycle: float,
    mean_on_slots: float,
) -> np.ndarray:
    """
    Generate a boolean activity mask of the given length from a 2-state
    (ON/OFF) Markov chain parameterized by a target duty cycle and mean ON
    burst length, rather than a hardcoded per-slot probability.

    Derivation: for a 2-state chain, mean sojourn time in ON = 1 / p(on->off).
    Stationary P(ON) = p(off->on) / (p(off->on) + p(on->off)). Solving for
    p(off->on) given a target duty cycle D and mean_on_slots L:
        p_on_to_off = 1 / L
        p_off_to_on = D * p_on_to_off / (1 - D)
    """
    duty_cycle = min(max(duty_cycle, 1e-6), 1 - 1e-6)
    mean_on_slots = max(mean_on_slots, 1.0)

    p_on_to_off = 1.0 / mean_on_slots
    p_off_to_on = min(duty_cycle * p_on_to_off / (1.0 - duty_cycle), 1.0)

    mask = np.empty(length, dtype=bool)
    state = bool(rng.random() < duty_cycle)  # sample initial state from stationary dist
    draws = rng.random(length)

    for t in range(length):
        mask[t] = state
        flip_prob = p_on_to_off if state else p_off_to_on
        if draws[t] < flip_prob:
            state = not state

    return mask


def power_schedule_from_mask(
    rng: np.random.Generator,
    mask: np.ndarray,
    power_mean_dbm: float,
    power_jitter_std_db: float,
) -> np.ndarray:
    """Per-slot transmit power (dBm) where active, NaN where inactive."""
    power = np.full(mask.shape, np.nan, dtype=float)
    n_active = int(mask.sum())
    if n_active:
        power[mask] = rng.normal(power_mean_dbm, power_jitter_std_db, size=n_active)
    return power
