"""
Traditional (non-adaptive) scanning receiver strategies -- models how most
currently-deployed, non-adaptive EW receivers actually operate. Reference:

  S. Apfeld, A. Charlish, W. Koch, "An Adaptive Receiver Search Strategy
  for Electronic Support," Fraunhofer FKIE (IEEE), 2016.

The paper's introduction describes the conventional approach directly: the
frequency range is divided into instantaneous-bandwidth bands "and scanned
sequentially" with a fixed dwell schedule and zero reaction to what's
detected -- that's the SEQUENTIAL mode below. The paper's own "Random"
control baseline -- "selects a random band for each dwell, but makes sure
that each band is selected about the same number of times" -- is the
BALANCED_RANDOM mode below. Both are genuinely non-adaptive: neither
reacts to hits, misses, or signal strength. The point of comparison is how
much an adaptive scheduler improves on either.

NOT implemented from the paper: the antenna-pattern / SNR-time-series
model (Eq. 1) and the adaptive autocorrelation-based strategy itself --
those are a much larger physics change and out of scope here. Our
existing sensor_model.py already has its own internally-consistent (if
simpler) Pd(SNR) model, shared with the RL environment; changing it would
break RL-vs-traditional comparability. Only the two non-adaptive
SCHEDULING strategies are adopted.
"""
from __future__ import annotations

import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.environment.receiver import DwellResult
from simulation.metrics.rollout_metrics import EpisodeMetrics, MetricsTracker

SEQUENTIAL = "sequential"
BALANCED_RANDOM = "balanced_random"
VALID_MODES = (SEQUENTIAL, BALANCED_RANDOM)


class TraditionalScanner:
    """Fixed, non-adaptive band-selection schedule.

    sequential: band 0, 1, 2, ..., num_bands-1, then wraps -- the
    "divided into bands and scanned sequentially" conventional receiver.

    balanced_random: a freshly shuffled permutation of all bands each
    full cycle, so every band is dwelled on exactly once per cycle but in
    random order -- the paper's "Random" control strategy.

    Both dwell a constant `dwell_slots` on each band regardless of
    outcome. Stateful across calls to next_dwell() within one episode --
    call reset() between episodes.
    """

    def __init__(
        self,
        num_bands: int,
        dwell_slots: int,
        mode: str = SEQUENTIAL,
        rng: np.random.Generator | None = None,
    ):
        if mode not in VALID_MODES:
            raise ValueError(f"Unknown traditional scan mode {mode!r}, expected one of {VALID_MODES}")
        self.num_bands = num_bands
        self.dwell_slots = dwell_slots
        self.mode = mode
        self._rng = rng if rng is not None else np.random.default_rng()
        self._next_band = 0
        self._order: np.ndarray | None = None
        self._order_pos = 0

    def reset(self) -> None:
        self._next_band = 0
        self._order = None
        self._order_pos = 0

    def next_dwell(self) -> tuple[int, int]:
        if self.mode == SEQUENTIAL:
            band = self._next_band
            self._next_band = (self._next_band + 1) % self.num_bands
            return band, self.dwell_slots

        # balanced_random
        if self._order is None or self._order_pos >= len(self._order):
            self._order = self._rng.permutation(self.num_bands)
            self._order_pos = 0
        band = int(self._order[self._order_pos])
        self._order_pos += 1
        return band, self.dwell_slots


class TraditionalScanDriver:
    """Drives dwells directly against `env`'s receiver using a
    TraditionalScanner -- bypasses the RL action/observation/reward path
    entirely, since scheduling here is fixed, not learned. Exposes a
    step()/reset() shape close enough to AlterraEnv that a caller (CLI or
    the Qt GUI's PythonBridge) can drive it uniformly alongside the RL
    policy.
    """

    def __init__(
        self,
        env: AlterraEnv,
        mode: str,
        dwell_slots: int | None = None,
        seed: int | None = None,
    ):
        self.env = env
        resolved_dwell = (
            dwell_slots if dwell_slots is not None else env.config.comparison.traditional_scan.dwell_slots
        )
        rng = np.random.default_rng(seed)
        self.scanner = TraditionalScanner(env.config.spectrum.num_bands, resolved_dwell, mode=mode, rng=rng)
        self._t = 0

    def reset(self, seed: int, options: dict | None = None) -> None:
        # `options` (e.g. {"manual_emitters": [...]}) is passed straight
        # through to env.reset() -- lets a caller (PythonBridge) supply a
        # freshly-rerolled custom population, keyed by this same seed,
        # instead of always reusing whatever was built at env-construction
        # time. Defaults to None for full backward compatibility with
        # existing callers (e.g. run_traditional_scan below).
        self.env.reset(seed=seed, options=options)
        self.scanner.reset()
        self._t = 0

    def step(self) -> tuple[DwellResult, bool]:
        band, dwell_slots = self.scanner.next_dwell()
        dwell_result = self.env._receiver.dwell(band, self._t, dwell_slots)  # noqa: SLF001 -- driver/eval-only, same pattern as MetricsTracker.finalize
        self._t = dwell_result.end_t
        truncated = self._t >= self.env.episode_length
        return dwell_result, truncated


def run_traditional_scan(
    env: AlterraEnv,
    seed: int,
    mode: str | None = None,
    dwell_slots: int | None = None,
) -> EpisodeMetrics:
    """Runs one full episode of a traditional scanner against `env`'s
    configured emitter population (rebuilt deterministically from `seed`,
    same as any RL/random run using the same seed), and returns the same
    EpisodeMetrics used to evaluate the RL scheduler, for direct,
    same-episode comparison. `mode`/`dwell_slots` default to whatever is
    configured under `comparison.traditional_scan` if not given.
    """
    resolved_mode = mode if mode is not None else env.config.comparison.traditional_scan.mode
    driver = TraditionalScanDriver(env, mode=resolved_mode, dwell_slots=dwell_slots, seed=seed)
    driver.reset(seed=seed)

    tracker = MetricsTracker()
    truncated = False
    while not truncated:
        dwell_result, truncated = driver.step()
        tracker.record_step(dwell_result, reward=0.0)

    return tracker.finalize(env)
