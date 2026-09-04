"""
Figures of merit from the problem statement: Pd, Pfa, sensitivity, average
intercept rate, average reward, percentage of correct predictions, average
intercept time error. Computed over a full episode using access to the
env's ground truth (only for evaluation, never fed to the agent).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from simulation.environment.gym_env import AlterraEnv
from simulation.environment.sensor_model import Detection


@dataclass
class EpisodeMetrics:
    probability_of_detection: float
    probability_of_false_alarm: float
    sensitivity: float
    avg_intercept_rate: float
    avg_reward: float
    percent_correct: float
    avg_intercept_time_error_slots: float | None
    steps: int


@dataclass
class MetricsTracker:
    detections: list[Detection] = field(default_factory=list)
    rewards: list[float] = field(default_factory=list)
    first_intercept_t: dict[str, int] = field(default_factory=dict)

    def record_step(self, dwell_result, reward: float) -> None:
        for d in dwell_result.detections:
            self.detections.append(d)
            if d.hit and d.true_emitter_id is not None:
                self.first_intercept_t.setdefault(d.true_emitter_id, d.t)
        self.rewards.append(reward)

    def finalize(self, env: AlterraEnv) -> EpisodeMetrics:
        occupied = [d for d in self.detections if d.true_occupied]
        empty = [d for d in self.detections if not d.true_occupied]

        hits = sum(d.hit for d in occupied)
        pd = hits / len(occupied) if occupied else 0.0

        false_alarms = sum(d.false_alarm for d in empty)
        pfa = false_alarms / len(empty) if empty else 0.0

        low_snr_occupied = [d for d in occupied if (d.estimated_snr_db or 0) < 10.0]
        sensitivity = (
            sum(d.hit for d in low_snr_occupied) / len(low_snr_occupied)
            if low_snr_occupied else pd
        )

        true_negatives = len(empty) - false_alarms
        total = len(self.detections)
        percent_correct = (hits + true_negatives) / total if total else 0.0

        avg_reward = float(np.mean(self.rewards)) if self.rewards else 0.0
        avg_intercept_rate = hits / len(self.rewards) if self.rewards else 0.0

        errors = []
        for emitter in env._emitters:  # ground-truth access, evaluation-only
            true_first_active = next(
                (t for t in range(env._episode_length) if emitter.state_at(t).active), None
            )
            if true_first_active is None:
                continue
            intercepted_t = self.first_intercept_t.get(emitter.emitter_id)
            if intercepted_t is not None:
                errors.append(intercepted_t - true_first_active)
        avg_intercept_time_error = float(np.mean(errors)) if errors else None

        return EpisodeMetrics(
            probability_of_detection=pd,
            probability_of_false_alarm=pfa,
            sensitivity=sensitivity,
            avg_intercept_rate=avg_intercept_rate,
            avg_reward=avg_reward,
            percent_correct=percent_correct,
            avg_intercept_time_error_slots=avg_intercept_time_error,
            steps=len(self.rewards),
        )
