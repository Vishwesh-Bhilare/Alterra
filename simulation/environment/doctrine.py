"""
Rule-based scheduling doctrine. Decides which bands are legal to select
next and what dwell to use, given a live mode (EXPLORE / INVESTIGATE /
TRACK / RELOCATE). Ported from model/agents/heuristic_scheduler.py's mode
logic, but returns a *candidate mask* instead of picking one band itself
-- MaskablePPO chooses within that mask. Dwell stays fully rule-based per
the hybrid design: rules control doctrine, ML controls which legal band
is most valuable right now.

CAVEAT (documented, not accidental): sb3-contrib computes per-step
MultiDiscrete masks for band and dwell independently, before the band is
actually chosen -- so dwell below is conditioned on doctrine mode + track
state only, never on which specific band ML ends up picking. Full
band-conditional dwell escalation only applies in TRACK mode, where the
band is already forced and known ahead of time.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.environment.scheduler_insight import BandTrack

MIN_RELOCATE_JUMP = 12
TRACK_MAX_DEPTH = 3
EXPLORE_REASSESS_INTERVAL = 6
RECENT_BANDS_MAXLEN = 8


@dataclass
class DoctrineState:
    """Episode-scoped doctrine bookkeeping. Reads AlterraEnv's existing
    per-band BandTrack dict rather than duplicating it."""
    track_band: int | None = None
    track_depth: int = 0
    track_confirmations: int = 0
    force_relocate: bool = False
    recent_bands: list = field(default_factory=list)
    step_count: int = 0


@dataclass(frozen=True)
class DoctrineDecision:
    mode: str              # "EXPLORE" | "INVESTIGATE" | "TRACK" | "RELOCATE"
    band_mask: np.ndarray  # (num_bands,) bool
    dwell_index: int       # forced -- dwell stays rule-based


def _distance(a: int, b: int) -> int:
    return abs(int(a) - int(b))


def _survey_mask(
    state: DoctrineState, tracks: dict[int, BandTrack], num_bands: int, current_band: int | None
) -> np.ndarray:
    mask = np.zeros(num_bands, dtype=bool)

    unscanned = [b for b in range(num_bands) if b not in tracks]
    if unscanned:
        far = [
            b for b in unscanned
            if current_band is None or _distance(b, current_band) >= MIN_RELOCATE_JUMP
        ]
        pool = far if far else unscanned
        for b in pool[: min(20, len(pool))]:
            mask[b] = True
        if mask.any():
            return mask

    # All bands scanned at least once: rank by staleness, excluding recent
    # trajectory, requiring a real jump -- same fallback as the heuristic.
    score = np.zeros(num_bands, dtype=np.float64)
    for b in range(num_bands):
        t = tracks.get(b)
        last_visit = t.last_visited_t if t and t.last_visited_t is not None else 0
        score[b] = (state.step_count - last_visit) - (t.visit_count * 4.0 if t else 0.0)

    candidates = np.arange(num_bands)
    if current_band is not None:
        far_enough = candidates[np.abs(candidates - current_band) >= MIN_RELOCATE_JUMP]
        if len(far_enough) > 0:
            candidates = far_enough

    non_recent = np.array([b for b in candidates if int(b) not in state.recent_bands])
    if len(non_recent) > 0:
        candidates = non_recent

    top_n = min(12, len(candidates))
    ranked = candidates[np.argsort(score[candidates])[-top_n:]]
    for b in ranked:
        mask[int(b)] = True
    return mask


def _investigate_mask(
    tracks: dict[int, BandTrack], num_bands: int, current_band: int | None, step_count: int
) -> np.ndarray:
    mask = np.zeros(num_bands, dtype=bool)
    hit_bands = [b for b, t in tracks.items() if t.ever_hit]
    if not hit_bands:
        return mask
    if current_band is not None and len(hit_bands) > 1:
        hit_bands = [b for b in hit_bands if b != current_band]
    if not hit_bands:
        return mask

    def score(b: int) -> float:
        t = tracks[b]
        age = max(0, step_count - (t.last_visited_t or 0))
        stale = min(1.0, age / 120.0)
        confirms = min(1.0, t.hit_count / 3.0)
        return 0.40 * confirms + 0.20 * stale + 0.40 * t.confidence

    ranked = sorted(hit_bands, key=score, reverse=True)
    for b in ranked[: min(6, len(ranked))]:
        mask[b] = True
    return mask


def _relocate_mask(state: DoctrineState, num_bands: int, current_band: int | None) -> np.ndarray:
    mask = np.zeros(num_bands, dtype=bool)
    candidates = [
        b for b in range(num_bands)
        if current_band is None or _distance(b, current_band) >= MIN_RELOCATE_JUMP
    ]
    candidates = [b for b in candidates if b not in state.recent_bands] or candidates
    for b in candidates:
        mask[b] = True
    return mask


def decide(
    state: DoctrineState,
    tracks: dict[int, BandTrack],
    num_bands: int,
    current_band: int | None,
    dwell_options: list[int],
) -> DoctrineDecision:
    if state.force_relocate:
        return DoctrineDecision("RELOCATE", _relocate_mask(state, num_bands, current_band), 0)

    if state.track_band is not None and state.track_depth < TRACK_MAX_DEPTH:
        mask = np.zeros(num_bands, dtype=bool)
        mask[state.track_band] = True
        dwell_index = min(1 + state.track_confirmations, len(dwell_options) - 1)
        return DoctrineDecision("TRACK", mask, dwell_index)

    if state.step_count > 0 and state.step_count % EXPLORE_REASSESS_INTERVAL == 0:
        return DoctrineDecision("EXPLORE", _survey_mask(state, tracks, num_bands, current_band), 0)

    if num_bands - len(tracks) > 0:
        return DoctrineDecision("EXPLORE", _survey_mask(state, tracks, num_bands, current_band), 0)

    investigate_mask = _investigate_mask(tracks, num_bands, current_band, state.step_count)
    if investigate_mask.any():
        return DoctrineDecision("INVESTIGATE", investigate_mask, 1)

    return DoctrineDecision("EXPLORE", _survey_mask(state, tracks, num_bands, current_band), 0)


def update_after_step(state: DoctrineState, band: int, any_hit: bool, num_bands: int) -> None:
    if state.track_band == band:
        state.track_depth += 1
        if any_hit:
            state.track_confirmations += 1
    else:
        state.track_band = band if any_hit else None
        state.track_depth = 1 if any_hit else 0
        state.track_confirmations = 1 if any_hit else 0

    state.force_relocate = state.track_band == band and state.track_depth >= TRACK_MAX_DEPTH

    state.recent_bands.append(band)
    if len(state.recent_bands) > RECENT_BANDS_MAXLEN:
        state.recent_bands.pop(0)

    state.step_count += 1
