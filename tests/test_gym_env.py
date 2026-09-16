import unittest
import torch
import numpy as np
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.lstm_policy import PPOLSTMExtractor


class TestGymEnv(unittest.TestCase):
    def test_env_observation_and_action_spaces(self):
        config = load_config("configs/default_config.yaml")
        env = AlterraEnv(config)
        obs, info = env.reset(seed=42)

        self.assertIn("band_tracks", obs)
        self.assertIn("receiver", obs)
        self.assertIn("hit_miss_seq", obs)

        self.assertEqual(obs["band_tracks"].shape, (128, 8))
        self.assertEqual(obs["receiver"].shape, (10,))
        self.assertEqual(obs["hit_miss_seq"].shape, (16, 6))

        # Scanned and unscanned initialization
        self.assertTrue(np.all(obs["band_tracks"][:, 5] == 0.0))  # is_scanned is 0 initially
        self.assertTrue(np.all(obs["band_tracks"][:, 6] == 1.0))  # is_unscanned is 1 initially

        # Receiver scanned/unscanned ratio
        self.assertEqual(obs["receiver"][8], 0.0)  # scanned ratio initially 0
        self.assertEqual(obs["receiver"][9], 1.0)  # unscanned ratio initially 1

    def test_step_and_dwell_escalation(self):
        config = load_config("configs/default_config.yaml")
        env = AlterraEnv(config)
        obs, info = env.reset(seed=42)

        # Perform action: dwell with longest dwell option (index 3)
        action = np.array([1, 3])  # stay, longest dwell
        next_obs, reward, terminated, truncated, info = env.step(action)

        current_band = env._current_band
        self.assertEqual(next_obs["band_tracks"][current_band, 5], 1.0)  # is_scanned
        self.assertEqual(next_obs["band_tracks"][current_band, 6], 0.0)  # is_unscanned is 0
        self.assertGreater(next_obs["receiver"][8], 0.0)                 # scanned_ratio > 0

        # Continuous hit norm in sequence
        self.assertEqual(next_obs["hit_miss_seq"].shape, (16, 6))

    def test_lstm_policy_extractor_forward(self):
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
        self.assertEqual(out.shape, (1, 256))
        self.assertFalse(torch.isnan(out).any())


if __name__ == "__main__":
    unittest.main()

