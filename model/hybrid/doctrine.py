"""
Cognitive Doctrine and Action Masking Engine for ALTERA Hybrid Scheduler.
Implements the 4-mode operational state machine:
  - EXPLORE: Under-observed spectrum search
  - INVESTIGATE: Resolving detection evidence uncertainty
  - TRACK: Continuous dwell on active emitter bursts up to a revisit cap
  - RELOCATE: Forced departure from stale/saturated tracks to uncover new signals

Generates candidate action masks M_t in {0, 1}^N and enforces fallback safety.
"""
from __future__ import annotations

from enum import Enum
from typing import Any
import numpy as np


class DoctrineMode(str, Enum):
    EXPLORE = "EXPLORE"
    INVESTIGATE = "INVESTIGATE"
    TRACK = "TRACK"
    RELOCATE = "RELOCATE"


class CognitiveDoctrine:
    """
    Rule-based cognitive doctrine controller.
    Evaluates observation evidence to determine the operational mode and
    constructs the valid-action mask for candidate frequency bands.
    """

    def __init__(
        self,
        num_bands: int = 128,
        max_track_revisits: int = 5,
        investigate_dwell_target: int = 2,
        recent_history_window: int = 10,
    ):
        self.num_bands = num_bands
        self.max_track_revisits = max_track_revisits
        self.investigate_dwell_target = investigate_dwell_target
        self.recent_history_window = recent_history_window

        self.reset()

    def reset(self) -> None:
        """Reset internal state at episode boundary."""
        self.mode = DoctrineMode.EXPLORE
        self.current_band = self.num_bands // 2
        self.tracked_band: int | None = None
        self.track_revisit_count = 0
        self.investigate_count = 0
        self.recent_visited_bands: list[int] = []
        self.relocate_cooldown = 0
        self.consecutive_hits = 0
        self.last_hit = False
        self.t = 0

    def update_state(
        self,
        current_band: int,
        hit: bool,
        measured_power_norm: float = 0.0,
        consecutive_hits: int = 0,
        t: int = 0,
    ) -> DoctrineMode:
        """
        Transitions the doctrine state machine based on the latest dwell outcome.
        """
        self.current_band = current_band
        self.last_hit = hit
        self.consecutive_hits = consecutive_hits
        self.t = t

        self.recent_visited_bands.append(current_band)
        if len(self.recent_visited_bands) > self.recent_history_window:
            self.recent_visited_bands.pop(0)

        if self.relocate_cooldown > 0:
            self.relocate_cooldown -= 1

        # Mode Transition Logic
        if self.mode == DoctrineMode.TRACK:
            if hit:
                self.track_revisit_count += 1
                if self.track_revisit_count >= self.max_track_revisits:
                    # Revisit quota reached -> force relocation
                    self.mode = DoctrineMode.RELOCATE
                    self.relocate_cooldown = 3
                    self.tracked_band = current_band
                    self.track_revisit_count = 0
                else:
                    self.mode = DoctrineMode.TRACK
            else:
                # Signal lost during track -> relocate immediately
                self.mode = DoctrineMode.RELOCATE
                self.relocate_cooldown = 3
                self.tracked_band = current_band
                self.track_revisit_count = 0

        elif self.mode == DoctrineMode.INVESTIGATE:
            if hit:
                # Confirmed signal -> transition to full TRACK
                self.mode = DoctrineMode.TRACK
                self.tracked_band = current_band
                self.track_revisit_count = 1
                self.investigate_count = 0
            else:
                self.investigate_count += 1
                if self.investigate_count >= self.investigate_dwell_target:
                    # Unconfirmed after target tries -> return to explore
                    self.mode = DoctrineMode.EXPLORE
                    self.investigate_count = 0

        elif self.mode == DoctrineMode.RELOCATE:
            if self.relocate_cooldown <= 0:
                self.mode = DoctrineMode.EXPLORE
                self.tracked_band = None
            elif hit:
                # Intercepted a new signal while relocating
                self.mode = DoctrineMode.INVESTIGATE
                self.investigate_count = 1

        elif self.mode == DoctrineMode.EXPLORE:
            if hit or consecutive_hits > 0:
                self.mode = DoctrineMode.INVESTIGATE
                self.investigate_count = 1
                self.tracked_band = current_band
            elif measured_power_norm > 0.4:
                # Elevated power reading even without detection flag
                self.mode = DoctrineMode.INVESTIGATE
                self.investigate_count = 1

        return self.mode

    def generate_action_mask(
        self,
        band_tracks: np.ndarray | None = None,
        receiver_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Constructs the Boolean valid-action mask M_t in {0, 1}^N for candidate bands.
        True indicates the candidate band is valid and eligible for selection.
        """
        mask = np.ones(self.num_bands, dtype=bool)

        if self.mode == DoctrineMode.TRACK and self.tracked_band is not None:
            # In TRACK: prioritize staying on the active band or adjacent agile channels (+/- 1)
            track_mask = np.zeros(self.num_bands, dtype=bool)
            b = self.tracked_band
            track_mask[b] = True
            if b - 1 >= 0:
                track_mask[b - 1] = True
            if b + 1 < self.num_bands:
                track_mask[b + 1] = True
            mask = track_mask

        elif self.mode == DoctrineMode.RELOCATE:
            # In RELOCATE: hard-forbid the recently tracked band to force movement
            if self.tracked_band is not None and 0 <= self.tracked_band < self.num_bands:
                mask[self.tracked_band] = False
            # Also discourage bands dwelled in the last 2 steps
            for b in self.recent_visited_bands[-2:]:
                if 0 <= b < self.num_bands:
                    mask[b] = False

        elif self.mode == DoctrineMode.INVESTIGATE:
            # In INVESTIGATE: focus observation effort within neighborhood (+/- 2 bands)
            investigate_mask = np.zeros(self.num_bands, dtype=bool)
            center = self.current_band
            low = max(0, center - 2)
            high = min(self.num_bands - 1, center + 2)
            investigate_mask[low : high + 1] = True
            mask = investigate_mask

        elif self.mode == DoctrineMode.EXPLORE:
            # In EXPLORE: avoid immediately repeating the exact same band unless no other choice
            if 0 <= self.current_band < self.num_bands:
                # Soft penalty: don't dwell on empty band consecutively
                if not self.last_hit and len(self.recent_visited_bands) > 0:
                    mask[self.current_band] = False

            # If band_tracks observation is available, prioritize under-observed / stale bands
            if band_tracks is not None and band_tracks.shape[0] == self.num_bands:
                # band_tracks[:, 6] is is_unscanned flag in 8-dim tracks, or check visit staleness
                # We do not hard-mask everything, but ensure unscanned/stale bands are kept
                pass

        # Fallback mechanism: Ensure at least one action is valid (relaxed soft constraints)
        if not np.any(mask):
            mask = np.ones(self.num_bands, dtype=bool)
            if self.mode == DoctrineMode.RELOCATE and self.tracked_band is not None:
                mask[self.tracked_band] = False
                if not np.any(mask):
                    mask = np.ones(self.num_bands, dtype=bool)

        return mask

    def select_relative_action(self, target_band: int) -> int:
        """
        Converts an absolute target band into relative step direction index:
          0: STEP DOWN (-1)
          1: STAY (0)
          2: STEP UP (+1)
        """
        if target_band == self.current_band:
            return 1  # STAY
        elif target_band > self.current_band:
            return 2  # STEP UP
        else:
            return 0  # STEP DOWN
