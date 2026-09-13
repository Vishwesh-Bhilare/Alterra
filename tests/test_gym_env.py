import unittest
import torch
import numpy as np
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.lstm_policy import PPOLSTMExtractor


def test_env_observation_and_action_spaces():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    obs, info = env.reset(seed=42)

    assert "band_tracks" in obs
    assert "receiver" in obs
    assert "hit_miss_seq" in obs

    assert obs["band_tracks"].shape == (128, 8)
    assert obs["receiver"].shape == (10,)
    assert obs["hit_miss_seq"].shape == (16, 6)

    # Scanned and unscanned initialization
    assert np.all(obs["band_tracks"][:, 5] == 0.0)  # is_scanned is 0 initially
    assert np.all(obs["band_tracks"][:, 6] == 1.0)  # is_unscanned is 1 initially

    # Receiver scanned/unscanned ratio
    assert obs["receiver"][8] == 0.0  # scanned ratio initially 0
    assert obs["receiver"][9] == 1.0  # unscanned ratio initially 1


def test_step_and_dwell_escalation():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    obs, info = env.reset(seed=42)

    # Perform action: dwell on band 10 with longest dwell option (index 3)
    action = np.array([10, 3])
    next_obs, reward, terminated, truncated, info = env.step(action)

    assert next_obs["band_tracks"][10, 5] == 1.0  # is_scanned on band 10
    assert next_obs["band_tracks"][10, 6] == 0.0  # is_unscanned on band 10 is 0
    assert next_obs["receiver"][8] > 0.0          # scanned_ratio > 0

    # Continuous hit norm in sequence
    assert next_obs["hit_miss_seq"].shape == (16, 6)


def test_lstm_policy_extractor_forward():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    obs, _ = env.reset(seed=42)

    extractor = PPOLSTMExtractor(env.observation_space, lstm_hidden_dim=64, features_dim=256)

    # Convert observation to batch tensor dict
    batch_obs = {
        k: torch.as_tensor(v).unsqueeze(0).float()
        for k, v in obs.items()
    }

    out = extractor(batch_obs)
    assert out.shape == (1, 256)
    assert not torch.isnan(out).any()
