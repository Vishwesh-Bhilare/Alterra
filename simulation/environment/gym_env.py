"""
Gymnasium environment wrapping the simulation. Action is (band,
dwell_option_index) — MultiDiscrete.

ROOT CAUSE FIX (this version): band_tracks previously only updated on a
hit — a miss left the observation for that band completely unchanged, so
the agent could not distinguish "just checked, empty" from "never
checked." That starved observation, not reward shape or network size, was
why prior training runs converged onto a small fixed set of bands
regardless of per-episode truth (see docs/PROJECT_CONTEXT.md and the
debugging session history). Every dwell now updates a `visited` signal
independent of the `hit` signal — TRACK_FEATURE_DIM went 4 -> 6.
Observation space shape changed: retrain from scratch, do not resume old
checkpoints against this version.

Reward still includes: novelty_bonus (first visit to a band this
episode), info-gain-scaled hit reward (full value for a first-ever
detection or a stale re-confirmation, floored at hit_confirm_floor for
spam-revisiting a just-confirmed band), and a staleness penalty averaged
(not summed) across currently-tracked bands.

KNOWN SIMPLIFICATION: observation "tracks" are indexed by band, not by
deinterleaved emitter identity — swap once model/deinterleaving exists.
"""
from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.emitters import BaseEmitter, build_population
from simulation.environment.receiver import DwellResult, Receiver
from simulation.environment.sensor_model import SensorModel
from simulation.environment.spectrum_world import SpectrumWorld
from simulation.utils.config_loader import AlterraConfig
from simulation.utils.rng import RNGManager

TRACK_FEATURE_DIM = 6  # [threat_norm, confidence, time_since_hit, ever_hit, time_since_visit, ever_visited]
RECEIVER_FEATURE_DIM = 2


@dataclass
class _BandTrack:
    threat_level: int = 0
    confidence: float = 0.0
    last_hit_t: int | None = None
    ever_hit: bool = False
    last_visited_t: int | None = None
    ever_visited: bool = False


class AlterraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, config: AlterraConfig, manual_emitters: list[BaseEmitter] | None = None):
        super().__init__()
        self.config = config
        self._static_manual_emitters = manual_emitters

        self._master_rng = RNGManager(config.rng_seed)
        self._episode_idx = -1

        num_bands = config.spectrum.num_bands
        self._dwell_options = config.environment.dwell_options_slots
        self.action_space = spaces.MultiDiscrete([num_bands, len(self._dwell_options)])
        self.observation_space = spaces.Dict(
            {
                "band_tracks": spaces.Box(
                    low=0.0, high=1.0, shape=(num_bands, TRACK_FEATURE_DIM), dtype=np.float32
                ),
                "receiver": spaces.Box(
                    low=0.0, high=1.0, shape=(RECEIVER_FEATURE_DIM,), dtype=np.float32
                ),
            }
        )

        self._episode_length = config.timing.episode_length_slots

        threat_levels = config.environment.reward.threat_weight_levels
        threat_values = config.environment.reward.threat_weight_values
        self._threat_weight_by_level = dict(zip(threat_levels, threat_values))
        self._max_threat_level = max(threat_levels)

        self._emitters: list[BaseEmitter] = []
        self._spectrum_world: SpectrumWorld | None = None
        self._sensor_model: SensorModel | None = None
        self._receiver: Receiver | None = None
        self._tracks: dict[int, _BandTrack] = {}
        self._visited_bands: set[int] = set()
        self._t = 0
        self._last_band = 0
        self.last_dwell_result: DwellResult | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)

        if seed is not None:
            episode_key = f"seed_{seed}"
        else:
            self._episode_idx += 1
            episode_key = f"episode_{self._episode_idx}"

        episode_seed_rng = self._master_rng.spawn_named(episode_key)
        episode_seed = int(episode_seed_rng.integers(0, 2**31 - 1))
        episode_rng_manager = RNGManager(episode_seed)

        manual_emitters = (options or {}).get("manual_emitters") or self._static_manual_emitters
        if manual_emitters is not None:
            self._emitters = manual_emitters
        else:
            self._emitters = build_population(self.config, episode_rng_manager)

        for emitter in self._emitters:
            emitter.reset(self._episode_length)

        self._spectrum_world = SpectrumWorld(
            emitters=self._emitters,
            num_bands=self.config.spectrum.num_bands,
            episode_length=self._episode_length,
        )
        self._sensor_model = SensorModel(
            config=self.config.sensor,
            rng_manager=episode_rng_manager,
            num_bands=self.config.spectrum.num_bands,
        )
        self._receiver = Receiver(self._spectrum_world, self._sensor_model)

        self._tracks = {}
        self._visited_bands = set()
        self._t = 0
        self._last_band = 0
        self.last_dwell_result = None

        return self._build_observation(), {}

    def step(self, action):
        assert self._receiver is not None, "call reset() before step()"
        band = int(action[0])
        dwell_slots = self._dwell_options[int(action[1])]

        dwell_result = self._receiver.dwell(band, self._t, dwell_slots)
        self.last_dwell_result = dwell_result
        self._t = dwell_result.end_t
        self._last_band = band

        # Reward computed against pre-dwell track state (before
        # _apply_dwell_to_tracks) so it can tell a first-time detection /
        # re-confirmation of a stale track apart from spam-revisiting a
        # band just confirmed.
        reward = self._compute_reward(dwell_result)
        self._apply_dwell_to_tracks(band, dwell_result)

        terminated = False
        truncated = self._t >= self._episode_length
        observation = self._build_observation()
        info = {
            "any_hit": dwell_result.any_hit,
            "any_false_alarm": dwell_result.any_false_alarm,
            "band": band,
            "dwell_slots": dwell_slots,
        }
        return observation, reward, terminated, truncated, info

    def _apply_dwell_to_tracks(self, band: int, dwell_result: DwellResult) -> None:
        # Every dwell updates visited state, regardless of outcome — this
        # is the fix: a miss must still be recorded, or the observation
        # can't distinguish "checked, empty" from "never checked."
        track = self._tracks.setdefault(band, _BandTrack())
        track.ever_visited = True
        track.last_visited_t = self._t

        best_hit = dwell_result.best_hit
        if best_hit is None:
            return
        track.threat_level = best_hit.true_threat_level or track.threat_level
        track.confidence = min(1.0, track.confidence + 0.34)
        track.last_hit_t = self._t
        track.ever_hit = True

    def _compute_reward(self, dwell_result: DwellResult) -> float:
        reward_cfg = self.config.environment.reward
        reward = 0.0

        if dwell_result.band not in self._visited_bands:
            reward += reward_cfg.novelty_bonus
        self._visited_bands.add(dwell_result.band)

        best_hit = dwell_result.best_hit
        if best_hit is not None:
            weight = self._threat_weight_by_level.get(best_hit.true_threat_level, 1.0)
            prior_track = self._tracks.get(dwell_result.band)
            if prior_track is not None and prior_track.ever_hit and prior_track.last_hit_t is not None:
                staleness_norm = reward_cfg.staleness_norm_slots
                time_since_prior = max(self._t - prior_track.last_hit_t, 0.0)
                info_gain_factor = reward_cfg.hit_confirm_floor + (
                    1.0 - reward_cfg.hit_confirm_floor
                ) * min(time_since_prior / staleness_norm, 1.0)
            else:
                info_gain_factor = 1.0
            reward += reward_cfg.hit_reward_base * weight * info_gain_factor
        else:
            reward += reward_cfg.idle_cost

        if dwell_result.any_false_alarm:
            reward += reward_cfg.false_alarm_penalty

        staleness_norm = reward_cfg.staleness_norm_slots
        active_tracks = [
            t for t in self._tracks.values() if t.ever_hit and t.last_hit_t is not None
        ]
        if active_tracks:
            staleness_penalty = 0.0
            for track in active_tracks:
                weight = self._threat_weight_by_level.get(track.threat_level, 1.0)
                time_since = self._t - track.last_hit_t
                staleness = min(time_since / staleness_norm, 1.0)
                staleness_penalty += reward_cfg.staleness_penalty_coeff * weight * staleness
            reward -= staleness_penalty / len(active_tracks)

        return float(reward)

    @property
    def t(self) -> int:
        return self._t

    @property
    def episode_length(self) -> int:
        return self._episode_length

    def _build_observation(self) -> dict[str, np.ndarray]:
        num_bands = self.config.spectrum.num_bands
        band_tracks = np.zeros((num_bands, TRACK_FEATURE_DIM), dtype=np.float32)
        # Default to "fully stale" for both staleness features — a band
        # never in self._tracks (never visited) must not read the same as
        # a band visited at t=0 (staleness=0); both would otherwise be 0.0.
        band_tracks[:, 2] = 1.0
        band_tracks[:, 4] = 1.0
        staleness_norm = self.config.environment.reward.staleness_norm_slots

        for band, track in self._tracks.items():
            if track.ever_hit and track.last_hit_t is not None:
                time_since_hit = self._t - track.last_hit_t
                band_tracks[band, 0] = track.threat_level / self._max_threat_level
                band_tracks[band, 1] = track.confidence
                band_tracks[band, 2] = min(time_since_hit / staleness_norm, 1.0)
                band_tracks[band, 3] = 1.0

            if track.ever_visited and track.last_visited_t is not None:
                time_since_visit = self._t - track.last_visited_t
                band_tracks[band, 4] = min(time_since_visit / staleness_norm, 1.0)
                band_tracks[band, 5] = 1.0

        receiver_features = np.array(
            [
                self._last_band / max(num_bands - 1, 1),
                min(self._t / self._episode_length, 1.0),
            ],
            dtype=np.float32,
        )
        return {"band_tracks": band_tracks, "receiver": receiver_features}
