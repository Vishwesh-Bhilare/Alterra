"""
Alterra Gymnasium environment with direct (absolute) frequency selection.

ACTION SPACE
------------
MultiDiscrete([num_bands, num_dwell_options])

    action[0] = absolute band index [0, num_bands-1]
    action[1] = dwell option index (ignored in hybrid mode -- see below)

OBSERVATION SPACE
-----------------
band_tracks: (num_bands, 8)
    [threat_norm, confidence, time_since_hit_norm, ever_hit,
     time_since_visit_norm, scanned_flag, unscanned_flag,
     last_power_norm]

receiver: (10,)
    [current_band_norm, time_norm, last_hit, consecutive_hit_norm,
     at_low_edge, at_high_edge, scanned_ratio, unscanned_ratio,
     last_dwell_norm, steps_since_new_band_norm]

hit_miss_seq: (16, 6)
    [hit, false_alarm, band_norm, dwell_norm, power_norm, time_delta_norm]

Ground-truth emitter identity/threat is never exposed directly in the
observation. It is only used by the simulator for reward/evaluation.

HYBRID DOCTRINE MODE (enable_doctrine=True, opt-in, default False)
--------------------------------------------------------------------
When enabled, simulation.environment.doctrine constrains which bands are
legal each step (EXPLORE/INVESTIGATE/TRACK/RELOCATE) and forces dwell
rule-based, exposed via action_masks() for MaskablePPO. Default
(enable_doctrine=False) is unchanged from every prior version -- action[1]
fully controls dwell, no masking, existing pure-PPO/RecurrentPPO
checkpoints are unaffected. This flag exists specifically so the hybrid
scheduler and the pure-learned scheduler stay comparable, independent
categories per the project's four-way comparison (Traditional / Heuristic
/ Hybrid / Pure-PPO).
"""

from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from simulation.emitters import BaseEmitter, build_population
from simulation.environment.doctrine import DoctrineState, decide, update_after_step
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


TRACK_FEATURE_DIM = 8
RECEIVER_FEATURE_DIM = 10
HIT_MISS_SEQ_LEN = 16
HIT_MISS_FEATURE_DIM = 6


class AlterraEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        config: AlterraConfig,
        manual_emitters: list[BaseEmitter] | None = None,
        enable_doctrine: bool = False,
    ):
        super().__init__()
        self.config = config
        self._static_manual_emitters = manual_emitters
        self._enable_doctrine = enable_doctrine

        self._master_rng = RNGManager(config.rng_seed)
        self._episode_idx = -1

        self.num_bands = config.spectrum.num_bands
        self._dwell_options = config.environment.dwell_options_slots

        self.action_space = spaces.MultiDiscrete(
            [self.num_bands, len(self._dwell_options)]
        )

        self.observation_space = spaces.Dict(
            {
                "band_tracks": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.num_bands, TRACK_FEATURE_DIM),
                    dtype=np.float32,
                ),
                "receiver": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(RECEIVER_FEATURE_DIM,),
                    dtype=np.float32,
                ),
                "hit_miss_seq": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(HIT_MISS_SEQ_LEN, HIT_MISS_FEATURE_DIM),
                    dtype=np.float32,
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
        self._current_band = 0
        self._last_dwell_slots = self._dwell_options[0]
        self._consecutive_hits = 0
        self._steps_since_new_band = 0

        self.last_dwell_result: DwellResult | None = None
        self.last_decision: DecisionExplanation | None = None
        self._history = SchedulerHistory(maxlen=self._event_history_len)
        self._hit_miss_seq: deque = deque(maxlen=HIT_MISS_SEQ_LEN)
        self._doctrine: DoctrineState | None = None

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
            num_bands=self.num_bands,
            episode_length=self._episode_length,
        )
        self._sensor_model = SensorModel(
            config=self.config.sensor,
            rng_manager=episode_rng_manager,
            num_bands=self.num_bands,
        )
        self._receiver = Receiver(
            self._spectrum_world,
            self._sensor_model,
            self.config,
        )

        self._tracks = {}
        self._visited_bands = set()
        self._t = 0

        self._current_band = int(
            episode_rng_manager.spawn_named("start_band").integers(
                0, self.num_bands
            )
        )
        self._last_dwell_slots = self._dwell_options[0]
        self._consecutive_hits = 0
        self._steps_since_new_band = 0

        self.last_dwell_result = None
        self.last_decision = None
        self._history = SchedulerHistory(maxlen=self._event_history_len)
        self._hit_miss_seq = deque(maxlen=HIT_MISS_SEQ_LEN)
        self._doctrine = DoctrineState() if self._enable_doctrine else None

        return self._build_observation(), {}

    def action_masks(self) -> np.ndarray:
        """Concatenated (band_mask ++ dwell_mask), length num_bands +
        len(dwell_options) -- sb3-contrib's MultiDiscrete masking
        convention. Only meaningful when enable_doctrine=True; otherwise
        returns an all-legal mask so accidentally wrapping a non-doctrine
        env with ActionMasker degrades to unconstrained PPO, not an error."""
        if not self._enable_doctrine or self._doctrine is None:
            return np.ones(self.num_bands + len(self._dwell_options), dtype=bool)

        doctrine_decision = decide(
            self._doctrine, self._tracks, self.num_bands, self._current_band, self._dwell_options,
        )
        dwell_mask = np.zeros(len(self._dwell_options), dtype=bool)
        dwell_mask[doctrine_decision.dwell_index] = True
        return np.concatenate([doctrine_decision.band_mask, dwell_mask])

    def step(self, action):
        assert self._receiver is not None, "call reset() before step()"

        band = int(np.clip(int(action[0]), 0, self.num_bands - 1))

        doctrine_decision = None
        if self._enable_doctrine and self._doctrine is not None:
            doctrine_decision = decide(
                self._doctrine, self._tracks, self.num_bands, self._current_band, self._dwell_options,
            )
            # Dwell stays fully rule-based in hybrid mode -- see
            # doctrine.py's docstring for why it can't be conditioned on
            # the specific band ML ends up choosing this step.
            dwell_slots = self._dwell_options[doctrine_decision.dwell_index]
        else:
            dwell_slots = self._dwell_options[int(action[1])]

        prev_band = self._current_band
        prev_t = self._t
        self._current_band = band

        dwell_result = self._receiver.dwell(
            band,
            self._t,
            dwell_slots,
        )
        self.last_dwell_result = dwell_result

        self._t = dwell_result.end_t
        self._last_dwell_slots = dwell_slots

        if band not in self._visited_bands:
            self._steps_since_new_band = 0
        else:
            self._steps_since_new_band += 1

        if dwell_result.any_hit:
            self._consecutive_hits += 1
        else:
            self._consecutive_hits = 0

        reward = self._compute_reward(dwell_result, band)

        prior_track = self._tracks.get(band)
        decision = explain_decision(
            prior_track,
            self._t,
            self.config,
        )
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
        self._push_hit_miss_entry(
            dwell_result,
            band,
            dwell_slots,
            prev_t,
        )

        if self._enable_doctrine and self._doctrine is not None:
            update_after_step(self._doctrine, band, dwell_result.any_hit, self.num_bands)

        terminated = False
        truncated = self._t >= self._episode_length

        observation = self._build_observation()

        info = {
            "any_hit": dwell_result.any_hit,
            "any_false_alarm": dwell_result.any_false_alarm,
            "true_occupied": any(
                d.true_occupied for d in dwell_result.detections
            ),
            "band": band,
            "prev_band": prev_band,
            "jump_distance": abs(band - prev_band),
            "dwell_slots": dwell_slots,
            "retune_slots": dwell_result.retune_slots,
            "center_freq_hz": dwell_result.center_freq_hz,
            "decision": decision.explore_exploit,
            "decision_reason": decision.reason,
            "priority_score": decision.priority_score,
            "doctrine_mode": doctrine_decision.mode if doctrine_decision is not None else None,
        }

        return observation, reward, terminated, truncated, info

    def band_priorities(self) -> list[BandPriority]:
        return compute_band_priorities(
            self._tracks,
            self.num_bands,
            self._t,
            self.config,
        )

    def recent_events(self, n: int | None = None) -> list[HistoryEvent]:
        return self._history.recent(n)

    def recent_hits(self, n: int | None = None) -> list[HistoryEvent]:
        return self._history.recent_hits(n)

    def _normalize_power(self, power_dbm: float) -> float:
        lo = self.config.sensor.measured_power_norm_min
        hi = self.config.sensor.measured_power_norm_max
        return float(np.clip((power_dbm - lo) / (hi - lo), 0.0, 1.0))

    def _apply_dwell_to_tracks(
        self,
        band: int,
        dwell_result: DwellResult,
    ) -> None:
        if band not in self._tracks:
            self._tracks[band] = BandTrack(
                power_history=deque(maxlen=self._power_history_len)
            )

        track = self._tracks[band]

        track.ever_visited = True
        track.last_visited_t = self._t
        track.visit_count += 1

        power_norm = self._normalize_power(
            dwell_result.mean_measured_power_dbm
        )
        track.last_measured_power_norm = power_norm
        track.power_history.append(power_norm)
        self._visited_bands.add(band)

        best_hit = dwell_result.best_hit
        if best_hit is None:
            return

        track.threat_level = (
            best_hit.true_threat_level or track.threat_level
        )
        track.confidence = min(1.0, track.confidence + 0.34)
        track.last_hit_t = self._t
        track.ever_hit = True
        track.hit_count += 1

    def _push_hit_miss_entry(
        self,
        dwell_result: DwellResult,
        band: int,
        dwell_slots: int,
        prev_t: int,
    ) -> None:
        max_dwell = max(self._dwell_options)
        time_delta_norm = min(
            (self._t - prev_t) / max(max_dwell, 1),
            1.0,
        )

        entry = np.array(
            [
                1.0 if dwell_result.any_hit else 0.0,
                1.0 if dwell_result.any_false_alarm else 0.0,
                band / max(self.num_bands - 1, 1),
                dwell_slots / max_dwell,
                self._normalize_power(
                    dwell_result.mean_measured_power_dbm
                ),
                time_delta_norm,
            ],
            dtype=np.float32,
        )
        self._hit_miss_seq.append(entry)

    def _compute_reward(self, dwell_result: DwellResult, band: int) -> float:
        reward_cfg = self.config.environment.reward
        reward = 0.0

        if band not in self._visited_bands:
            reward += reward_cfg.novelty_bonus

        best_hit = dwell_result.best_hit

        if best_hit is not None:
            weight = self._threat_weight_by_level.get(
                best_hit.true_threat_level,
                1.0,
            )
            prior_track = self._tracks.get(band)

            if (
                prior_track is not None
                and prior_track.ever_hit
                and prior_track.last_hit_t is not None
            ):
                staleness_norm = reward_cfg.staleness_norm_slots
                time_since_prior = max(
                    self._t - prior_track.last_hit_t,
                    0.0,
                )
                info_gain_factor = reward_cfg.hit_confirm_floor + (
                    1.0 - reward_cfg.hit_confirm_floor
                ) * min(
                    time_since_prior / staleness_norm,
                    1.0,
                )
            else:
                info_gain_factor = 1.0

            reward += (
                reward_cfg.hit_reward_base
                * weight
                * info_gain_factor
            )
        else:
            reward += reward_cfg.idle_cost

        if dwell_result.any_false_alarm:
            reward += reward_cfg.false_alarm_penalty

        if dwell_result.retune_slots > 0:
            reward += reward_cfg.idle_cost * 0.5 * (
                dwell_result.retune_slots
                / max(self._dwell_options)
            )

        staleness_norm = reward_cfg.staleness_norm_slots
        active_tracks = [
            t
            for t in self._tracks.values()
            if t.ever_hit and t.last_hit_t is not None
        ]

        if active_tracks:
            staleness_penalty = 0.0

            for track in active_tracks:
                weight = self._threat_weight_by_level.get(
                    track.threat_level,
                    1.0,
                )
                time_since = self._t - track.last_hit_t
                staleness = min(
                    time_since / staleness_norm,
                    1.0,
                )
                staleness_penalty += (
                    reward_cfg.staleness_penalty_coeff
                    * weight
                    * staleness
                )

            reward -= staleness_penalty / len(active_tracks)

        return float(reward)

    def _build_observation(self) -> dict[str, np.ndarray]:
        num_bands = self.num_bands

        band_tracks = np.zeros(
            (num_bands, TRACK_FEATURE_DIM),
            dtype=np.float32,
        )

        band_tracks[:, 2] = 1.0
        band_tracks[:, 4] = 1.0
        band_tracks[:, 6] = 1.0

        staleness_norm = self.config.environment.reward.staleness_norm_slots

        for band, track in self._tracks.items():
            if track.ever_hit and track.last_hit_t is not None:
                time_since_hit = self._t - track.last_hit_t
                band_tracks[band, 0] = (
                    track.threat_level / self._max_threat_level
                )
                band_tracks[band, 1] = track.confidence
                band_tracks[band, 2] = min(
                    time_since_hit / staleness_norm,
                    1.0,
                )
                band_tracks[band, 3] = 1.0

            if track.ever_visited and track.last_visited_t is not None:
                time_since_visit = self._t - track.last_visited_t
                band_tracks[band, 4] = min(
                    time_since_visit / staleness_norm,
                    1.0,
                )
                band_tracks[band, 5] = 1.0
                band_tracks[band, 6] = 0.0
                band_tracks[band, 7] = track.last_measured_power_norm

        scanned_ratio = len(self._visited_bands) / num_bands
        max_dwell = max(self._dwell_options)

        receiver_features = np.array(
            [
                self._current_band / max(num_bands - 1, 1),
                min(self._t / self._episode_length, 1.0),
                1.0
                if (
                    self.last_dwell_result is not None
                    and self.last_dwell_result.any_hit
                )
                else 0.0,
                min(self._consecutive_hits / 5.0, 1.0),
                1.0 if self._current_band == 0 else 0.0,
                1.0 if self._current_band == num_bands - 1 else 0.0,
                scanned_ratio,
                1.0 - scanned_ratio,
                self._last_dwell_slots / max_dwell,
                min(self._steps_since_new_band / 50.0, 1.0),
            ],
            dtype=np.float32,
        )

        hit_miss_seq = np.zeros(
            (HIT_MISS_SEQ_LEN, HIT_MISS_FEATURE_DIM),
            dtype=np.float32,
        )

        history = list(self._hit_miss_seq)
        if history:
            hit_miss_seq[-len(history):] = np.stack(history)

        return {
            "band_tracks": band_tracks,
            "receiver": receiver_features,
            "hit_miss_seq": hit_miss_seq,
        }
