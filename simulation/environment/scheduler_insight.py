"""
Scheduler explainability layer (Module B): turns the per-band tracking
state the RL loop already maintains into GUI-facing "why did it do that"
data -- priority scores, explore/exploit labels + reasons, and rolling
history. Purely derived from existing state; does not influence the
action, observation, or reward in any way.

`BandTrack` lives here (not in gym_env.py) so both AlterraEnv and this
module can share the one definition without a circular import.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

from simulation.utils.config_loader import AlterraConfig


@dataclass
class BandTrack:
    threat_level: int = 0
    confidence: float = 0.0
    last_hit_t: int | None = None
    ever_hit: bool = False
    hit_count: int = 0
    last_visited_t: int | None = None
    ever_visited: bool = False
    visit_count: int = 0
    last_measured_power_norm: float = 0.0
    power_history: deque = field(default_factory=deque)  # bounded at creation time, see below
    miss_streak: int = 0  # consecutive dwells on this band with no hit; reset to 0 on any hit


@dataclass(frozen=True)
class DecisionExplanation:
    explore_exploit: str  # "EXPLORE" | "EXPLOIT"
    reason: str
    priority_score: float


@dataclass(frozen=True)
class BandPriority:
    band: int
    priority_score: float
    visit_count: int
    hit_count: int
    ever_visited: bool
    ever_hit: bool
    confidence: float
    threat_level: int
    time_since_visit: int | None
    time_since_hit: int | None


@dataclass(frozen=True)
class HistoryEvent:
    t: int
    band: int
    center_freq_hz: float
    dwell_slots: int
    retune_slots: int
    classification_counts: dict[str, int]
    mean_power_dbm: float
    decision: DecisionExplanation
    reward: float


class SchedulerHistory:
    def __init__(self, maxlen: int):
        self._events: deque[HistoryEvent] = deque(maxlen=maxlen)

    def record(self, event: HistoryEvent) -> None:
        self._events.append(event)

    def recent(self, n: int | None = None) -> list[HistoryEvent]:
        events = list(self._events)
        return events[-n:] if n is not None else events

    def recent_hits(self, n: int | None = None) -> list[HistoryEvent]:
        hits = [e for e in self._events if e.classification_counts.get("hit", 0) > 0]
        return hits[-n:] if n is not None else hits


def compute_priority_score(track: BandTrack | None, t: int, config: AlterraConfig) -> float:
    """Mirror image of `_compute_reward`'s own staleness_penalty term: a
    band's "worth revisiting" score rises the longer a previously
    important (threat x confidence) sighting there goes unconfirmed.
    Never-visited bands score max (pure exploration value)."""
    reward_cfg = config.environment.reward
    staleness_norm = reward_cfg.staleness_norm_slots

    if track is None or not track.ever_visited:
        return 1.0

    time_since_visit = t - (track.last_visited_t or 0)
    visit_staleness = min(time_since_visit / staleness_norm, 1.0)

    if not track.ever_hit or track.last_hit_t is None:
        return round(0.5 * visit_staleness, 4)

    threat_by_level = dict(zip(reward_cfg.threat_weight_levels, reward_cfg.threat_weight_values))
    max_threat_weight = max(reward_cfg.threat_weight_values)
    importance = threat_by_level.get(track.threat_level, 1.0) / max_threat_weight * track.confidence

    time_since_hit = t - track.last_hit_t
    hit_staleness = min(time_since_hit / staleness_norm, 1.0)

    score = 0.3 + 0.7 * importance * hit_staleness + 0.2 * visit_staleness
    return round(min(score, 1.0), 4)


def explain_decision(track: BandTrack | None, t: int, config: AlterraConfig) -> DecisionExplanation:
    reward_cfg = config.environment.reward
    staleness_norm = reward_cfg.staleness_norm_slots
    priority = compute_priority_score(track, t, config)

    if track is None or not track.ever_visited:
        return DecisionExplanation("EXPLORE", "Never scanned before", priority)

    if not track.ever_hit or track.last_hit_t is None:
        return DecisionExplanation(
            "EXPLORE", f"No detection yet after {track.visit_count} visit(s)", priority
        )

    time_since_hit = t - track.last_hit_t
    if time_since_hit <= staleness_norm * 0.25 and track.confidence >= 0.5:
        return DecisionExplanation("EXPLOIT", "Recent strong detection", priority)
    if time_since_hit <= staleness_norm:
        return DecisionExplanation("EXPLOIT", "Revisiting a previously active band", priority)
    return DecisionExplanation("EXPLORE", "Re-checking a stale, previously active band", priority)


def compute_band_priorities(
    tracks: dict[int, BandTrack], num_bands: int, t: int, config: AlterraConfig
) -> list[BandPriority]:
    result = []
    for band in range(num_bands):
        track = tracks.get(band)
        score = compute_priority_score(track, t, config)
        if track is None:
            result.append(
                BandPriority(
                    band=band, priority_score=score, visit_count=0, hit_count=0,
                    ever_visited=False, ever_hit=False, confidence=0.0, threat_level=0,
                    time_since_visit=None, time_since_hit=None,
                )
            )
            continue
        time_since_visit = t - track.last_visited_t if track.last_visited_t is not None else None
        time_since_hit = t - track.last_hit_t if track.last_hit_t is not None else None
        result.append(
            BandPriority(
                band=band, priority_score=score, visit_count=track.visit_count,
                hit_count=track.hit_count, ever_visited=track.ever_visited,
                ever_hit=track.ever_hit, confidence=track.confidence,
                threat_level=track.threat_level, time_since_visit=time_since_visit,
                time_since_hit=time_since_hit,
            )
        )
    return result
