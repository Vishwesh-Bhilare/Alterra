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

RECEIVER MODEL UPGRADE (Module A): Receiver now models retune time as
slots consumed before a dwell's detections begin when switching bands,
and exposes each dwell's actual frequency window. Flagged for the PPO
team in docs/model_changes.md -- not yet reflected in reward shaping.

SCHEDULER EXPLAINABILITY (Module B): `BandTrack` now also tracks
visit_count/hit_count/power_history (see scheduler_insight.py). Every
step computes a priority score + EXPLORE/EXPLOIT + reason for the band
just dwelled on (`self.last_decision`), using track state as of *before*
this dwell -- i.e. what the scheduler "knew" when it picked this band --
and logs a rolling history (`self.recent_events()`). Purely derived,
observation/action/reward untouched.

KNOWN SIMPLIFICATION: observation "tracks" are indexed by band, not by
deinterleaved emitter identity -- swap once model/deinterleaving exists.
"""
from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.emitters import BaseEmitter, build_population
from simulation.environment.receiver import DwellResult, Receiver
from simulation.environment.scheduler_insight import (
    BandPriority,
    BandTrack,
    DecisionExplanation,
    HistoryEvent,
    SchedulerHistory,
    compute_band_priorities,
    explain_decision,
)
from simulation.environment.sensor_model import SensorModel
from simulation.environment.spectrum_world import SpectrumWorld
from simulation.utils.config_loader import AlterraConfig
from simulation.utils.rng import RNGManager

TRACK_FEATURE_DIM = 7  # [threat_norm, confidence, time_since_hit, ever_hit, time_since_visit, ever_visited, last_power_norm]
RECEIVER_FEATURE_DIM = 2


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
        self._power_history_len = config.scheduler_insight.power_history_len
        self._event_history_len = config.scheduler_insight.event_history_len

        threat_levels = config.environment.reward.threat_weight_levels
        threat_values = config.environment.reward.threat_weight_values
        self._threat_weight_by_level = dict(zip(threat_levels, threat_values))
        self._max_threat_level = max(threat_levels)

        self._emitters: list[BaseEmitter] = []
        self._spectrum_world: SpectrumWorld | None = None
        self._sensor_model: SensorModel | None = None
        self._receiver: Receiver | None = None
        self._tracks: dict[int, BandTrack] = {}
        self._visited_bands: set[int] = set()
        self._t = 0
        self._last_band = 0
        self.last_dwell_result: DwellResult | None = None
        self.last_decision: DecisionExplanation | None = None
        self._history = SchedulerHistory(maxlen=self._event_history_len)

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
        self._receiver = Receiver(self._spectrum_world, self._sensor_model, self.config)

        self._tracks = {}
        self._visited_bands = set()
        self._t = 0
        self._last_band = 0
        self.last_dwell_result = None
        self.last_decision = None
        self._history = SchedulerHistory(maxlen=self._event_history_len)

        return self._build_observation(), {}

    def step(self, action):
        assert self._receiver is not None, "call reset() before step()"
        band = int(action[0])
        dwell_slots = self._dwell_options[int(action[1])]

        dwell_result = self._receiver.dwell(band, self._t, dwell_slots)
        self.last_dwell_result = dwell_result
        self._t = dwell_result.end_t
        self._last_band = band

        reward = self._compute_reward(dwell_result)

        # Decision explanation uses track state as of BEFORE this dwell's
        # own update below -- i.e. what the scheduler actually knew.
        prior_track = self._tracks.get(band)
        decision = explain_decision(prior_track, self._t, self.config)
        self.last_decision = decision
        self._history.record(
            HistoryEvent(
                t=self._t,
                band=band,
                center_freq_hz=dwell_result.center_freq_hz,
                dwell_slots=dwell_slots,
                retune_slots=dwell_result.retune_slots,
                classification_counts=dwell_result.classification_counts(),
                mean_power_dbm=dwell_result.mean_measured_power_dbm,
                decision=decision,
                reward=reward,
            )
        )

        self._apply_dwell_to_tracks(band, dwell_result)

        terminated = False
        truncated = self._t >= self._episode_length
        observation = self._build_observation()
        info = {
            "any_hit": dwell_result.any_hit,
            "any_false_alarm": dwell_result.any_false_alarm,
            "band": band,
            "dwell_slots": dwell_slots,
            "retune_slots": dwell_result.retune_slots,
            "center_freq_hz": dwell_result.center_freq_hz,
            "decision": decision.explore_exploit,
            "decision_reason": decision.reason,
            "priority_score": decision.priority_score,
        }
        return observation, reward, terminated, truncated, info

    def band_priorities(self) -> list[BandPriority]:
        """Priority/ranking snapshot across all bands, for a GUI priority
        map or ranking table. Computed on demand, not cached per step."""
        return compute_band_priorities(self._tracks, self.config.spectrum.num_bands, self._t, self.config)

    def recent_events(self, n: int | None = None) -> list[HistoryEvent]:
        return self._history.recent(n)

    def recent_hits(self, n: int | None = None) -> list[HistoryEvent]:
        return self._history.recent_hits(n)

    def _normalize_power(self, power_dbm: float) -> float:
        lo = self.config.sensor.measured_power_norm_min
        hi = self.config.sensor.measured_power_norm_max
        return float(np.clip((power_dbm - lo) / (hi - lo), 0.0, 1.0))

    def _apply_dwell_to_tracks(self, band: int, dwell_result: DwellResult) -> None:
        # Every dwell updates visit + measured-power state, regardless of
        # outcome -- a miss is still information.
        if band not in self._tracks:
            self._tracks[band] = BandTrack(power_history=deque(maxlen=self._power_history_len))
        track = self._tracks[band]

        track.ever_visited = True
        track.last_visited_t = self._t
        track.visit_count += 1
        power_norm = self._normalize_power(dwell_result.mean_measured_power_dbm)
        track.last_measured_power_norm = power_norm
        track.power_history.append(power_norm)

        best_hit = dwell_result.best_hit
        if best_hit is None:
            return
        track.threat_level = best_hit.true_threat_level or track.threat_level
        track.confidence = min(1.0, track.confidence + 0.34)
        track.last_hit_t = self._t
        track.ever_hit = True
        track.hit_count += 1

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

        receiver_features = np.array(
            [
                self._last_band / max(num_bands - 1, 1),
                min(self._t / self._episode_length, 1.0),
            ],
            dtype=np.float32,
        )
        return {"band_tracks": band_tracks, "receiver": receiver_features}
