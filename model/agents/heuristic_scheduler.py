"""Deterministic cognitive heuristic scheduler for the Alterra demo.

This is intentionally NOT an RL policy.  It operates on the live Alterra
observation stream and demonstrates:

* wide spectrum exploration
* evidence-driven investigation
* adaptive dwell escalation (3 -> 5 -> 8 -> 12)
* limited target tracking
* forced relocation after confirmation
* non-local/direct frequency jumps
* avoidance of repeatedly scanning the same quiet band

The scheduler never reads the simulator truth matrix.  Future emitter activity
is therefore not available to the decision logic.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np


class CognitiveHeuristicScheduler:
    """Memory-based direct-jump scheduler used by the GUI demonstration."""

    def __init__(
        self,
        num_bands: int = 128,
        dwell_options=None,
        seed: int = 0,
    ) -> None:
        self.num_bands = int(num_bands)
        self.dwell_options = [
            int(x) for x in (dwell_options or [3, 5, 8, 12])
        ]
        if not self.dwell_options:
            raise ValueError("dwell_options must not be empty")

        self.rng = np.random.default_rng(seed)
        self.reset(seed)

    def reset(self, seed: int = 0) -> None:
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.step_count = 0
        self.current_band = None
        self.last_band = None

        self.scan_count = np.zeros(self.num_bands, dtype=np.int32)
        self.hit_count = np.zeros(self.num_bands, dtype=np.int32)
        self.miss_count = np.zeros(self.num_bands, dtype=np.int32)
        self.false_alarm_count = np.zeros(self.num_bands, dtype=np.int32)
        self.last_seen = np.full(self.num_bands, -10_000, dtype=np.int32)
        self.last_hit = np.full(self.num_bands, -10_000, dtype=np.int32)

        # Consecutive exploitation of one band.  Once the track budget is
        # exhausted, the next decision MUST relocate.
        self.track_band = None
        self.track_depth = 0
        self.track_confirmations = 0
        self.force_relocate = False

        # Recent trajectory used to stop pathological oscillation.
        self.recent_bands = deque(maxlen=8)

        # Shuffled survey order gives deterministic but non-linear coverage.
        self.explore_order = self.rng.permutation(self.num_bands)
        self.explore_cursor = 0

        # Most recent observation returned by env.step().
        self.last_obs = None

    # ------------------------------------------------------------------
    # Observation handling
    # ------------------------------------------------------------------
    @staticmethod
    def _normalise(values: np.ndarray) -> np.ndarray:
        """Robustly map one feature vector to [0, 1]."""
        x = np.asarray(values, dtype=np.float64)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        if x.size == 0:
            return np.zeros_like(x)

        lo = float(np.percentile(x, 10.0))
        hi = float(np.percentile(x, 90.0))
        if hi - lo < 1e-12:
            return np.full_like(x, 0.5, dtype=np.float64)

        return np.clip((x - lo) / (hi - lo), 0.0, 1.0)

    def set_observation(self, obs: Any) -> None:
        """Store the most recent environment observation."""
        self.last_obs = obs

    def _observation_score(self) -> np.ndarray:
        """Derive a generic per-band evidence score from band_tracks.

        We intentionally do not depend on a specific column ordering.  Each
        feature is normalized across bands and contributes to an aggregate
        'interestingness' score.  This makes the demo resilient to modest
        observation-schema changes across branches.
        """
        score = np.zeros(self.num_bands, dtype=np.float64)
        obs = self.last_obs

        if obs is None:
            return score

        tracks = None
        if isinstance(obs, dict):
            tracks = obs.get("band_tracks")
        else:
            try:
                tracks = obs["band_tracks"]
            except Exception:
                tracks = None

        if tracks is None:
            return score

        try:
            x = np.asarray(tracks, dtype=np.float64)
        except Exception:
            return score

        if x.ndim != 2:
            return score

        # Accept either (bands, features) or (features, bands).
        if x.shape[0] == self.num_bands:
            band_features = x
        elif x.shape[1] == self.num_bands:
            band_features = x.T
        else:
            return score

        usable = min(band_features.shape[1], 8)
        if usable == 0:
            return score

        for col in range(usable):
            score += self._normalise(band_features[:, col])

        score /= float(usable)
        return np.clip(score, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Candidate selection
    # ------------------------------------------------------------------
    def _distance(self, a: int, b: int) -> int:
        return abs(int(a) - int(b))

    def _is_recent(self, band: int) -> bool:
        return int(band) in self.recent_bands

    def _survey_candidate(self) -> int:
        """Choose a band that broadens coverage and creates a real jump."""
        # First preference: never-before-scanned bands.
        for _ in range(self.num_bands):
            band = int(self.explore_order[self.explore_cursor % self.num_bands])
            self.explore_cursor += 1

            if self.scan_count[band] != 0:
                continue
            if self.current_band is not None and self._distance(band, self.current_band) < 12:
                continue
            return band

        # Once coverage is established, choose the stalest candidates while
        # excluding the recent trajectory and enforcing a meaningful jump.
        age = self.step_count - self.last_seen
        score = age.astype(np.float64)
        score += self._observation_score() * 30.0
        score -= self.scan_count.astype(np.float64) * 4.0
        score -= self.false_alarm_count.astype(np.float64) * 1.5

        candidates = np.arange(self.num_bands)
        if self.current_band is not None:
            candidates = candidates[
                np.abs(candidates - self.current_band) >= 12
            ]

        if len(candidates) == 0:
            candidates = np.arange(self.num_bands)

        # Don't immediately bounce among the last few bands.
        non_recent = np.array(
            [b for b in candidates if int(b) not in self.recent_bands],
            dtype=np.int64,
        )
        if len(non_recent) > 0:
            candidates = non_recent

        top_n = min(12, len(candidates))
        ranked = candidates[np.argsort(score[candidates])[-top_n:]]
        return int(self.rng.choice(ranked))

    def _track_score(self, band: int, evidence: np.ndarray) -> float:
        """Score a known band for re-investigation."""
        age = max(0, self.step_count - self.last_seen[band])
        stale = min(1.0, age / 120.0)
        confirms = min(1.0, self.hit_count[band] / 3.0)
        obs = float(evidence[band])
        false_alarm = min(1.0, self.false_alarm_count[band] / 3.0)

        return (
            0.40 * confirms
            + 0.30 * obs
            + 0.20 * stale
            - 0.20 * false_alarm
        )

    def _choose_target(self, evidence: np.ndarray) -> int | None:
        if not np.any(self.hit_count > 0):
            return None

        candidates = np.flatnonzero(self.hit_count > 0)
        if len(candidates) == 0:
            return None

        # Don't keep returning to the exact current target after the tracking
        # budget has expired.
        if self.current_band is not None and len(candidates) > 1:
            candidates = candidates[candidates != self.current_band]

        if len(candidates) == 0:
            return None

        scores = np.array(
            [self._track_score(int(b), evidence) for b in candidates],
            dtype=np.float64,
        )

        # Choose among near-best candidates instead of always taking argmax.
        top_n = min(6, len(candidates))
        ranked = candidates[np.argsort(scores)[-top_n:]]
        return int(self.rng.choice(ranked))

    def _select_band(self) -> tuple[int, str]:
        evidence = self._observation_score()

        if self.force_relocate:
            band = self._survey_candidate()
            return band, "RELOCATE — confirmed target, resume spectrum search"

        # Finish a track only after a bounded number of consecutive dwells.
        if self.track_band is not None and self.track_depth < 3:
            target = int(self.track_band)
            return target, "TRACK — maintain confirmed emitter"

        # Every few decisions, deliberately return to broad search even when a
        # target exists. This prevents one emitter from monopolizing the scan.
        exploration_interval = 6
        if self.step_count > 0 and self.step_count % exploration_interval == 0:
            band = self._survey_candidate()
            return band, "EXPLORE — periodic spectrum reassessment"

        # Early episode: establish broad coverage before exploitation.
        unseen = int(np.count_nonzero(self.scan_count == 0))
        if unseen > 0:
            band = self._survey_candidate()
            return band, "EXPLORE — unseen spectrum region"

        target = self._choose_target(evidence)
        if target is not None:
            return target, "INVESTIGATE — revisit highest-value detection"

        band = self._survey_candidate()
        return band, "EXPLORE — stale/undersampled spectrum region"

    # ------------------------------------------------------------------
    # Dwell selection
    # ------------------------------------------------------------------
    def _select_dwell(self, band: int) -> tuple[int, str]:
        """Select dwell from recent evidence for this specific band."""
        hits = int(self.hit_count[band])
        scans = int(self.scan_count[band])

        if self.track_band == band and self.track_confirmations >= 3:
            idx = min(3, len(self.dwell_options) - 1)
            return idx, "TRACK — maximum dwell for final confirmation"

        if self.track_band == band and self.track_confirmations == 2:
            idx = min(2, len(self.dwell_options) - 1)
            return idx, "TRACK — extended confirmation dwell"

        if self.track_band == band and self.track_confirmations == 1:
            idx = min(1, len(self.dwell_options) - 1)
            return idx, "INVESTIGATE — confirmation dwell"

        if hits >= 2:
            idx = min(1, len(self.dwell_options) - 1)
            return idx, "INVESTIGATE — prior detections present"

        if scans == 0:
            return 0, "EXPLORE — short survey dwell"

        return 0, "EXPLORE — verify before committing"

    # ------------------------------------------------------------------
    # Main step
    # ------------------------------------------------------------------
    def step(self, env):
        """Take one direct band+dwell decision and advance the environment."""
        band, band_reason = self._select_band()
        dwell_idx, dwell_reason = self._select_dwell(band)

        # Guarantee that a relocation is actually non-local whenever possible.
        if self.current_band is not None and band == self.current_band:
            alternatives = [
                b
                for b in range(self.num_bands)
                if abs(b - self.current_band) >= 12
                and b not in self.recent_bands
            ]
            if alternatives:
                evidence = self._observation_score()
                ranked = sorted(
                    alternatives,
                    key=lambda b: (
                        evidence[b]
                        + min(1.0, (self.step_count - self.last_seen[b]) / 120.0)
                        - self.scan_count[b] * 0.01
                    ),
                    reverse=True,
                )
                band = int(ranked[0])
                band_reason = (
                    "RELOCATE — enforced non-local jump"
                    if self.force_relocate
                    else band_reason
                )

        action = np.asarray([band, dwell_idx], dtype=np.int64)
        obs, reward, terminated, truncated, info = env.step(action)
        self.last_obs = obs

        actual_band = int(info.get("band", band))
        actual_dwell = int(
            info.get("dwell_slots", self.dwell_options[dwell_idx])
        )

        any_hit = bool(info.get("any_hit", False))
        any_false_alarm = bool(info.get("any_false_alarm", False))

        # Update memory from the actual environment result.
        self.scan_count[actual_band] += 1
        self.last_seen[actual_band] = int(
            getattr(env, "t", self.step_count)
        )

        if any_hit:
            self.hit_count[actual_band] += 1
            self.last_hit[actual_band] = self.step_count
        elif any_false_alarm:
            self.false_alarm_count[actual_band] += 1
        else:
            self.miss_count[actual_band] += 1

        jump_distance = 0
        if self.last_band is not None:
            jump_distance = self._distance(actual_band, self.last_band)

        # Tracking state: only the same target in consecutive decisions counts
        # toward a confirmation chain.
        if self.track_band == actual_band:
            self.track_depth += 1
            if any_hit:
                self.track_confirmations += 1
        else:
            self.track_band = actual_band if any_hit else None
            self.track_depth = 1 if any_hit else 0
            self.track_confirmations = 1 if any_hit else 0

        # After three target dwells, force the next decision away from it.
        if self.track_band == actual_band and self.track_depth >= 3:
            self.force_relocate = True
        else:
            self.force_relocate = False

        self.recent_bands.append(actual_band)
        self.last_band = actual_band
        self.current_band = actual_band
        self.step_count += 1

        evidence = self._observation_score()
        novelty = 1.0 if self.scan_count[actual_band] == 1 else 0.0
        stale = min(
            1.0,
            max(0, self.step_count - self.last_seen[actual_band]) / 120.0,
        )
        confidence = min(1.0, self.hit_count[actual_band] / 3.0)
        priority = float(
            np.clip(
                0.35 * novelty
                + 0.30 * float(evidence[actual_band])
                + 0.25 * confidence
                + 0.10 * stale,
                0.0,
                1.0,
            )
        )

        if any_hit:
            if self.force_relocate:
                reason = "RELOCATE — target confirmed, force non-local jump"
            elif self.track_confirmations >= 3:
                reason = "TRACK — final confirmation before relocation"
            elif self.track_confirmations == 2:
                reason = "TRACK — repeated detection, extended dwell"
            else:
                reason = "INVESTIGATE — detection confirmed"
        else:
            reason = band_reason
            if band_reason.startswith("EXPLORE"):
                reason = dwell_reason if jump_distance < 12 else band_reason

        info = dict(info)
        info.update(
            {
                "band": actual_band,
                "dwell_slots": actual_dwell,
                "jump_distance": jump_distance,
                "priority_score": priority,
                "decision_reason": reason,
            }
        )

        return obs, float(reward), bool(terminated), bool(truncated), info
