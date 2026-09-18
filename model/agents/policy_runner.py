"""
Unified stepping interface across plain stable_baselines3 PPO,
sb3-contrib RecurrentPPO, and sb3-contrib MaskablePPO (the hybrid
doctrine+ML scheduler). Recurrent policies need LSTM hidden state carried
across predict() calls; maskable policies need the env's current
action_masks() passed in every predict() call. PolicyRunner hides both,
so callers (PythonBridge, comparison.py) use the same
.reset()/.predict(obs, action_masks=...) interface regardless of model
kind -- pass action_masks whenever runner.needs_action_mask is True
(fetched from env.action_masks(); harmless no-op for non-maskable models
if passed anyway, since it's simply ignored).
"""
from __future__ import annotations

import numpy as np

from model.agents import model_registry


class PolicyRunner:
    def __init__(self, model, algo_class: str):
        self.model = model
        self.algo_class = algo_class
        self._lstm_states = None
        self._episode_start = True

    @property
    def needs_action_mask(self) -> bool:
        return self.algo_class == "MaskablePPO"

    def reset(self) -> None:
        self._lstm_states = None
        self._episode_start = True

    def predict(self, obs, action_masks=None):
        if self.algo_class == "RecurrentPPO":
            action, self._lstm_states = self.model.predict(
                obs,
                state=self._lstm_states,
                episode_start=np.array([self._episode_start]),
                deterministic=True,
            )
            self._episode_start = False
            return action

        if self.algo_class == "MaskablePPO":
            action, _ = self.model.predict(
                obs, action_masks=action_masks, deterministic=True
            )
            return action

        action, _ = self.model.predict(obs, deterministic=True)
        return action


def load_model(repo_root: str, model_id: str) -> PolicyRunner:
    path = model_registry.resolve_model_path(repo_root, model_id)
    algo_class = model_registry.get_algo_class(repo_root, model_id)

    if algo_class == "RecurrentPPO":
        try:
            from sb3_contrib import RecurrentPPO
        except ImportError as e:
            raise RuntimeError(
                "sb3-contrib is required to load RecurrentPPO checkpoints. "
                "Install it with `pip install sb3-contrib`."
            ) from e
        model = RecurrentPPO.load(path)
    elif algo_class == "MaskablePPO":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as e:
            raise RuntimeError(
                "sb3-contrib is required to load MaskablePPO checkpoints. "
                "Install it with `pip install sb3-contrib`."
            ) from e
        model = MaskablePPO.load(path)
    else:
        from stable_baselines3 import PPO
        model = PPO.load(path)

    return PolicyRunner(model, algo_class)
