"""
Rule-Based Heuristic Scheduler (Baseline B0 & Teacher for Imitation Learning).
Executes cognitive doctrine decisions (Explore, Investigate, Track, Relocate)
purely using domain rules and priority scoring without learned weights.
"""
from __future__ import annotations

import numpy as np

from model.hybrid.doctrine import CognitiveDoctrine, DoctrineMode
from model.hybrid.dwell_controller import RuleBasedDwellController


class HeuristicScheduler:
    """
    Teacher and Reference Baseline (B0).
    Selects valid actions using cognitive doctrine and priority scoring.
    """

    def __init__(self, num_bands: int = 128, max_track_revisits: int = 5):
        self.num_bands = num_bands
        self.doctrine = CognitiveDoctrine(num_bands=num_bands, max_track_revisits=max_track_revisits)
        self.dwell_controller = RuleBasedDwellController()
        self.sweep_direction = 1  # +1 (sweeping up) or -1 (sweeping down)

    def reset(self) -> None:
        self.doctrine.reset()
        self.sweep_direction = 1

    def select_action(
        self,
        obs: dict[str, np.ndarray],
        last_info: dict | None = None,
    ) -> tuple[int, int, DoctrineMode, np.ndarray]:
        """
        Determines the next band and dwell duration according to doctrine rules.

        Returns:
            (target_band, dwell_idx, mode, mask)
        """
        band_tracks = obs.get("band_tracks")
        receiver = obs.get("receiver")

        # Update doctrine state from last step feedback if available
        if last_info is not None:
            hit = bool(last_info.get("hit", False) or last_info.get("any_hit", False))
            curr_band = int(last_info.get("band", self.doctrine.current_band))
            power_norm = float(last_info.get("mean_power_norm", 0.0))
            consec_hits = int(last_info.get("consecutive_hits", 0))
            t = int(last_info.get("t", 0))
            self.doctrine.update_state(curr_band, hit, power_norm, consec_hits, t)

        mode = self.doctrine.mode
        mask = self.doctrine.generate_action_mask(band_tracks, receiver)
        valid_indices = np.where(mask)[0]

        curr = self.doctrine.current_band

        if mode == DoctrineMode.TRACK:
            # Maintain dwell on the tracked band
            target_band = self.doctrine.tracked_band if self.doctrine.tracked_band is not None else curr
            if target_band not in valid_indices and len(valid_indices) > 0:
                target_band = valid_indices[0]

        elif mode == DoctrineMode.INVESTIGATE:
            # Recheck current band or immediately adjacent band with highest observed power
            target_band = curr
            if target_band not in valid_indices and len(valid_indices) > 0:
                target_band = int(np.random.choice(valid_indices))

        elif mode == DoctrineMode.RELOCATE:
            # Advance sweep direction away from the saturated band
            next_cand = curr + self.sweep_direction * 2
            if next_cand < 0:
                self.sweep_direction = 1
                next_cand = 1
            elif next_cand >= self.num_bands:
                self.sweep_direction = -1
                next_cand = self.num_bands - 2

            if next_cand in valid_indices:
                target_band = next_cand
            elif len(valid_indices) > 0:
                # Pick valid band farthest from the relocated track
                dists = [abs(b - curr) for b in valid_indices]
                target_band = valid_indices[int(np.argmax(dists))]
            else:
                target_band = curr

        else:
            # EXPLORE: Systematic triangular sweep across valid channels
            next_step = curr + self.sweep_direction
            if next_step >= self.num_bands:
                self.sweep_direction = -1
                next_step = self.num_bands - 2
            elif next_step < 0:
                self.sweep_direction = 1
                next_step = 1

            if next_step in valid_indices:
                target_band = next_step
            elif len(valid_indices) > 0:
                # Select valid band closest to current trajectory
                dists = [abs(b - next_step) for b in valid_indices]
                target_band = valid_indices[int(np.argmin(dists))]
            else:
                target_band = curr

        # Ensure target_band is within valid bounds
        target_band = max(0, min(self.num_bands - 1, target_band))

        # Select dwell duration based on mode and evidence
        conf = float(receiver[2]) if receiver is not None and len(receiver) > 2 else 0.0
        pwr = float(receiver[6]) if receiver is not None and len(receiver) > 6 else 0.0
        dwell_slots, dwell_idx = self.dwell_controller.select_dwell(
            mode=mode,
            confidence=conf,
            measured_power_norm=pwr,
            consecutive_hits=self.doctrine.consecutive_hits,
        )

        return target_band, dwell_idx, mode, mask
