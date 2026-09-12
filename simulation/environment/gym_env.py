"""
Gymnasium environment wrapping the simulation. Action is (band,
dwell_option_index) -- MultiDiscrete.

OBSERVATION UPGRADE (this version): every dwell now writes a normalized
measured-power reading into band_tracks, regardless of hit/miss -- a real
receiver always reports *something* per dwell (noise floor if empty,
signal+noise if occupied). Previously the agent had zero signal-strength
information on a miss. TRACK_FEATURE_DIM 6->7. Observation shape changed
again; retrain from scratch, do not resume prior checkpoints.

Prior fix (kept): visit state tracked independently of hit state, so a
miss still updates "last checked" even without a detection.

KNOWN SIMPLIFICATION: observation "tracks" are indexed by band, not by
deinterleaved emitter identity -- swap once model/deinterleaving exists.
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

TRACK_FEATURE_DIM = 7  # [threat_norm, confidence, time_since_hit, ever_hit, time_since_visit, ever_visited, last_power_norm]
RECEIVER_FEATURE_DIM = 8  # [band_norm, time_norm, last_hit, consec_hits_norm, can_down, can_up, last_power, prev_dir]


@dataclass
class _BandTrack:
    threat_level: int = 0
    confidence: float = 0.0
    last_hit_t: int | None = None
    ever_hit: bool = False
    last_visited_t: int | None = None
    ever_visited: bool = False
    last_measured_power_norm: float = 0.0


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
        self._action_mode = getattr(config.environment, "action_mode", "relative")
        self._step_sizes = getattr(config.environment, "relative_step_sizes", [-1, 0, 1])

        if self._action_mode == "relative":
            self.action_space = spaces.MultiDiscrete([len(self._step_sizes), len(self._dwell_options)])
        else:
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
        self._current_band = 0
        self._consecutive_hits = 0
        self._prev_had_hit = False
        self._prev_action_direction_norm = 0.5
        self._last_measured_power_norm = 0.0
        self.last_dwell_result: DwellResult | None = None

    @property
    def t(self) -> int:
        return self._t

    @property
    def episode_length(self) -> int:
        return self._episode_length

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
        self._current_band = int(episode_seed_rng.integers(0, self.config.spectrum.num_bands))
        self._last_band = self._current_band
        self._consecutive_hits = 0
        self._prev_had_hit = False
        self._prev_action_direction_norm = 0.5
        self._last_measured_power_norm = 0.0
        self.last_dwell_result = None

        return self._build_observation(), {}

    def step(self, action):
        assert self._receiver is not None, "call reset() before step()"
        num_bands = self.config.spectrum.num_bands

        if self._action_mode == "relative":
            direction_idx = int(action[0])
            delta = self._step_sizes[direction_idx]
            target_band = self._current_band + delta
            if target_band < 0:
                band = min(1, num_bands - 1)
                delta = band - self._current_band
                hit_boundary = True
            elif target_band >= num_bands:
                band = max(num_bands - 2, 0)
                delta = band - self._current_band
                hit_boundary = True
            else:
                band = target_band
                hit_boundary = False
            self._prev_action_direction_norm = float(direction_idx) / max(len(self._step_sizes) - 1, 1)
        else:
            band = int(action[0])
            delta = band - self._current_band
            hit_boundary = False
            self._prev_action_direction_norm = 0.5

        dwell_slots = self._dwell_options[int(action[1])]

        dwell_result = self._receiver.dwell(band, self._t, dwell_slots)
        self.last_dwell_result = dwell_result
        self._t = dwell_result.end_t
        self._last_band = self._current_band
        self._current_band = band

        hit = dwell_result.any_hit
        if hit:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        reward = self._compute_reward(dwell_result, delta, hit_boundary)
        self._apply_dwell_to_tracks(band, dwell_result)
        self._prev_had_hit = hit
        self._last_measured_power_norm = self._normalize_power(dwell_result.mean_measured_power_dbm)

        terminated = False
        truncated = self._t >= self._episode_length
        observation = self._build_observation()
        info = {
            "any_hit": dwell_result.any_hit,
            "any_false_alarm": dwell_result.any_false_alarm,
            "band": band,
            "dwell_slots": dwell_slots,
            "delta": delta,
            "consecutive_hits": self._consecutive_hits,
        }
        return observation, reward, terminated, truncated, info

    def _normalize_power(self, power_dbm: float) -> float:
        lo = self.config.sensor.measured_power_norm_min
        hi = self.config.sensor.measured_power_norm_max
        return float(np.clip((power_dbm - lo) / (hi - lo), 0.0, 1.0))

    def _apply_dwell_to_tracks(self, band: int, dwell_result: DwellResult) -> None:
        track = self._tracks.setdefault(band, _BandTrack())
        track.ever_visited = True
        track.last_visited_t = self._t
        track.last_measured_power_norm = self._normalize_power(dwell_result.mean_measured_power_dbm)

        best_hit = dwell_result.best_hit
        if best_hit is None:
            return
        track.threat_level = best_hit.true_threat_level or track.threat_level
        track.confidence = min(1.0, track.confidence + 0.34)
        track.last_hit_t = self._t
        track.ever_hit = True

    def _compute_reward(self, dwell_result: DwellResult, delta: int, hit_boundary: bool) -> float:
        reward_cfg = self.config.environment.reward
        reward = 0.0

        if dwell_result.band not in self._visited_bands:
            reward += reward_cfg.novelty_bonus
        self._visited_bands.add(dwell_result.band)

        best_hit = dwell_result.best_hit
        if best_hit is not None:
            weight = self._threat_weight_by_level.get(best_hit.true_threat_level, 1.0)
            reward += reward_cfg.hit_reward_base * weight
            # Tracking reward for staying on active frequency
            if delta == 0:
                reward += getattr(reward_cfg, "tracking_reward", 6.0) * weight
        else:
            reward += reward_cfg.idle_cost
            # Penalize staying on an empty band to strongly encourage scanning
            if delta == 0:
                reward += getattr(reward_cfg, "empty_stay_penalty", -0.4)
            # Extra penalty for staying on a dead band right after signal ended
            if delta == 0 and self._prev_had_hit:
                reward += getattr(reward_cfg, "signal_lost_penalty", -0.8)

        if hit_boundary:
            reward += getattr(reward_cfg, "boundary_penalty", -0.2)

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

    def _build_observation(self) -> dict[str, np.ndarray]:
        num_bands = self.config.spectrum.num_bands
        band_tracks = np.zeros((num_bands, TRACK_FEATURE_DIM), dtype=np.float32)
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
                band_tracks[band, 6] = track.last_measured_power_norm

        current_band_norm = float(self._current_band) / max(num_bands - 1, 1)
        time_norm = min(float(self._t) / max(self._episode_length, 1), 1.0)
        last_hit_val = 1.0 if (self.last_dwell_result and self.last_dwell_result.any_hit) else 0.0
        consec_hits_val = min(float(self._consecutive_hits) / 10.0, 1.0)
        can_step_down = 1.0 if self._current_band > 0 else 0.0
        can_step_up = 1.0 if self._current_band < num_bands - 1 else 0.0
        last_power_val = float(self._last_measured_power_norm)
        prev_dir_val = float(self._prev_action_direction_norm)

        receiver_features = np.array(
            [
                current_band_norm,
                time_norm,
                last_hit_val,
                consec_hits_val,
                can_step_down,
                can_step_up,
                last_power_val,
                prev_dir_val,
            ],
            dtype=np.float32,
        )
        return {"band_tracks": band_tracks, "receiver": receiver_features}
