"""
Gymnasium environment wrapping the simulation. Sole contract surface with
model/agents. Action is (band, dwell_option_index) — MultiDiscrete — so the
scheduler picks both frequency and dwell time per the CORTEX pipeline spec.

Supports both default mode (random population via build_population) and
manual scenario mode (explicit emitters via scenario_builder), selected by
passing `manual_emitters` to reset() or the constructor.

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

TRACK_FEATURE_DIM = 4
RECEIVER_FEATURE_DIM = 2


@dataclass
class _BandTrack:
    threat_level: int = 0
    confidence: float = 0.0
    last_observed_t: int | None = None
    ever_observed: bool = False


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
        self._t = 0
        self._last_band = 0
        self.last_dwell_result: DwellResult | None = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self._episode_idx += 1

        episode_seed_rng = self._master_rng.spawn_named(f"episode_{self._episode_idx}")
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

        self._apply_dwell_to_tracks(band, dwell_result)
        reward = self._compute_reward(dwell_result)

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
        best_hit = dwell_result.best_hit
        if best_hit is None:
            return
        track = self._tracks.setdefault(band, _BandTrack())
        track.threat_level = best_hit.true_threat_level or track.threat_level
        track.confidence = min(1.0, track.confidence + 0.34)
        track.last_observed_t = self._t
        track.ever_observed = True

    def _compute_reward(self, dwell_result: DwellResult) -> float:
        reward_cfg = self.config.environment.reward
        reward = 0.0

        best_hit = dwell_result.best_hit
        if best_hit is not None:
            weight = self._threat_weight_by_level.get(best_hit.true_threat_level, 1.0)
            reward += reward_cfg.hit_reward_base * weight
        else:
            reward += reward_cfg.idle_cost

        if dwell_result.any_false_alarm:
            reward += reward_cfg.false_alarm_penalty

        staleness_norm = reward_cfg.staleness_norm_slots
        for track in self._tracks.values():
            if not track.ever_observed or track.last_observed_t is None:
                continue
            weight = self._threat_weight_by_level.get(track.threat_level, 1.0)
            time_since = self._t - track.last_observed_t
            staleness = min(time_since / staleness_norm, 1.0)
            reward -= reward_cfg.staleness_penalty_coeff * weight * staleness

        return float(reward)

    def _build_observation(self) -> dict[str, np.ndarray]:
        num_bands = self.config.spectrum.num_bands
        band_tracks = np.zeros((num_bands, TRACK_FEATURE_DIM), dtype=np.float32)
        staleness_norm = self.config.environment.reward.staleness_norm_slots

        for band, track in self._tracks.items():
            if not track.ever_observed or track.last_observed_t is None:
                continue
            time_since = self._t - track.last_observed_t
            time_since_norm = min(time_since / staleness_norm, 1.0)
            band_tracks[band, 0] = track.threat_level / self._max_threat_level
            band_tracks[band, 1] = track.confidence
            band_tracks[band, 2] = time_since_norm
            band_tracks[band, 3] = 1.0

        receiver_features = np.array(
            [
                self._last_band / max(num_bands - 1, 1),
                min(self._t / self._episode_length, 1.0),
            ],
            dtype=np.float32,
        )
        return {"band_tracks": band_tracks, "receiver": receiver_features}
