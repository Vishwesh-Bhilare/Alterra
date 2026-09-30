"""
Rule-Based Dwell Controller for ALTERA Hybrid Scheduler.
Decouples scan duration from frequency band selection in v1.
Maps operational doctrine mode, signal detection confidence, and measured power
to calibrated dwell duration slots [3, 5, 8, 12].
"""
from __future__ import annotations

from model.hybrid.doctrine import DoctrineMode


class RuleBasedDwellController:
    """
    Selects dwell duration index and slot count according to cognitive doctrine:
      - EXPLORE: 3 slots (fastest sweep rate across wide spectrum)
      - RELOCATE: 3 slots (rapid departure to new frequency region)
      - INVESTIGATE: 5-8 slots (adequate integration time to confirm weak or intermittent pulses)
      - TRACK: 12 slots (maximum duration to capture complete radar pulse trains and PRI)
    """

    def __init__(self, dwell_options: list[int] | None = None):
        if dwell_options is None:
            self.dwell_options = [3, 5, 8, 12]
        else:
            self.dwell_options = dwell_options

    def select_dwell(
        self,
        mode: DoctrineMode,
        confidence: float = 0.0,
        measured_power_norm: float = 0.0,
        consecutive_hits: int = 0,
    ) -> tuple[int, int]:
        """
        Returns:
            (dwell_slots, dwell_option_index)
        """
        if mode == DoctrineMode.TRACK:
            # Full dwell to capture pulse trains, PRI, and deinterleave
            idx = len(self.dwell_options) - 1  # 12 slots (index 3)
        elif mode == DoctrineMode.INVESTIGATE:
            if confidence > 0.6 or consecutive_hits > 0:
                idx = 2  # 8 slots
            else:
                idx = 1  # 5 slots
        elif mode == DoctrineMode.RELOCATE:
            idx = 0  # 3 slots (fast jump)
        else:
            # EXPLORE
            # Quick probe to minimize time spent on empty channels
            idx = 0  # 3 slots

        idx = max(0, min(idx, len(self.dwell_options) - 1))
        return self.dwell_options[idx], idx
