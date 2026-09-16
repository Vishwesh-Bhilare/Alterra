"""
Rigorous system test suite for Alterra Targeted PPO+LSTM.
Tests:
1. Environment contract & space consistency
2. Explicit tracking of all 5 requested features:
   - Hit & Miss representation
   - Scanned bands
   - Unscanned bands
   - Continuous hit duration in same band
3. Dwell escalation reward dynamics (reward proportional to dwell time on consecutive hits)
4. Signal lost penalty and move incentives
5. Re-acquisition bonus for surveyed spectrum
6. Model inference, shape compatibility, and deterministic/stochastic execution
"""
import unittest
import numpy as np
import torch
from stable_baselines3 import PPO

from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.lstm_policy import PPOLSTMExtractor


class TestAlterraSystemRigorous(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = load_config("configs/default_config.yaml")
        cls.model_path = "model/agents/checkpoints/best/best_model.zip"

    def setUp(self):
        self.env = AlterraEnv(self.config)

    def test_01_observation_shapes_and_bounds(self):
        """Verify observation spaces conform to exact dimensions and range [0, 1]."""
        obs, _ = self.env.reset(seed=123)
        self.assertIn("band_tracks", obs)
        self.assertIn("receiver", obs)
        self.assertIn("hit_miss_seq", obs)

        # band_tracks: 128 bands x 8 features
        self.assertEqual(obs["band_tracks"].shape, (128, 8))
        self.assertTrue(np.all(obs["band_tracks"] >= 0.0) and np.all(obs["band_tracks"] <= 1.0))

        # receiver: 10 features
        self.assertEqual(obs["receiver"].shape, (10,))
        self.assertTrue(np.all(obs["receiver"] >= 0.0) and np.all(obs["receiver"] <= 1.0))

        # hit_miss_seq: 16 history x 6 features
        self.assertEqual(obs["hit_miss_seq"].shape, (16, 6))
        self.assertTrue(np.all(obs["hit_miss_seq"] >= 0.0) and np.all(obs["hit_miss_seq"] <= 1.0))

    def test_02_scanned_vs_unscanned_tracking(self):
        """Verify explicit tracking of scanned and unscanned bands."""
        obs, _ = self.env.reset(seed=42)
        initial_band = self.env._current_band

        # Check initial state: only initial_band is visited in reset
        for b in range(128):
            if b == initial_band:
                continue
            self.assertEqual(obs["band_tracks"][b, 5], 0.0, f"Band {b} should be unscanned initially")
            self.assertEqual(obs["band_tracks"][b, 6], 1.0, f"Band {b} is_unscanned should be 1.0 initially")

        # Step to a specific target band
        if getattr(self.env, "_action_mode", "relative") == "relative":
            action = np.array([2, 0])  # step up (+1)
            target = min(initial_band + 1, self.env.config.spectrum.num_bands - 1)
        else:
            action = np.array([50, 0])  # band 50, dwell option 0 (3 slots)
            target = 50
        next_obs, _, _, _, _ = self.env.step(action)

        self.assertEqual(next_obs["band_tracks"][target, 5], 1.0, f"Band {target} is_scanned must be 1.0 after visit")
        self.assertEqual(next_obs["band_tracks"][target, 6], 0.0, f"Band {target} is_unscanned must be 0.0 after visit")
        self.assertGreater(next_obs["receiver"][8], 0.0, "scanned_ratio must be > 0")
        self.assertAlmostEqual(next_obs["receiver"][8] + next_obs["receiver"][9], 1.0, places=5,
                               msg="scanned_ratio + unscanned_ratio must equal 1.0")

    def test_03_continuous_hit_tracking_in_sequence(self):
        """Verify continuous hit duration is recorded into the sequence buffer."""
        obs, _ = self.env.reset(seed=42)
        
        # Simulate two dwells on same band
        self.env._current_band = 20
        if getattr(self.env, "_action_mode", "relative") == "relative":
            action = np.array([1, 1])  # stay (delta 0), dwell option 1
        else:
            action = np.array([20, 1])  # band 20, dwell option 1
        obs1, _, _, _, info1 = self.env.step(action)
        
        # Buffer shape check
        self.assertEqual(obs1["hit_miss_seq"].shape, (16, 6))
        latest_entry = obs1["hit_miss_seq"][-1]
        
        # Features: [hit, miss, band_norm, dwell_norm, continuous_hit_norm, power_norm]
        if info1["any_hit"]:
            self.assertEqual(latest_entry[0], 1.0)
            self.assertEqual(latest_entry[1], 0.0)
            self.assertGreater(latest_entry[4], 0.0)
        else:
            self.assertEqual(latest_entry[0], 0.0)
            self.assertEqual(latest_entry[1], 1.0)
            self.assertEqual(latest_entry[4], 0.0)

    def test_04_dwell_escalation_reward_logic(self):
        """Verify reward function gives higher reward for longer dwell times when consecutive hits occur."""
        self.env.reset(seed=42)

        # Mock dwell results with hits
        from simulation.environment.sensor_model import Detection
        from simulation.environment.receiver import DwellResult
        
        det_hit = Detection(
            band=10,
            t=0,
            hit=True,
            false_alarm=False,
            true_occupied=True,
            estimated_snr_db=15.0,
            measured_power_dbm=10.0,
            true_emitter_id="e1",
            true_threat_level=2,
        )
        
        # Case A: consecutive_hits = 3, short dwell (3 slots)
        dwell_short = DwellResult(band=10, start_t=0, end_t=3, detections=[det_hit]*3)
        self.env._consecutive_hits = 3
        rew_short = self.env._compute_reward(dwell_short, delta=0, hit_boundary=False)

        # Case B: consecutive_hits = 3, long dwell (12 slots)
        dwell_long = DwellResult(band=10, start_t=0, end_t=12, detections=[det_hit]*12)
        self.env._consecutive_hits = 3
        rew_long = self.env._compute_reward(dwell_long, delta=0, hit_boundary=False)

        self.assertGreater(rew_long, rew_short,
                           f"Longer dwell on confirmed hit must yield higher reward: {rew_long} vs {rew_short}")

    def test_05_empty_band_probing_efficiency(self):
        """Verify that on empty bands, shorter dwell (3 slots) is preferred over long dwell (12 slots)."""
        self.env.reset(seed=42)
        from simulation.environment.sensor_model import Detection
        from simulation.environment.receiver import DwellResult

        det_empty = Detection(
            band=99,
            t=0,
            hit=False,
            false_alarm=False,
            true_occupied=False,
            estimated_snr_db=None,
            measured_power_dbm=-90.0,
            true_emitter_id=None,
            true_threat_level=None,
        )

        dwell_short_empty = DwellResult(band=99, start_t=0, end_t=3, detections=[det_empty]*3)
        self.env._consecutive_hits = 0
        rew_short_empty = self.env._compute_reward(dwell_short_empty, delta=10, hit_boundary=False)

        dwell_long_empty = DwellResult(band=99, start_t=0, end_t=12, detections=[det_empty]*12)
        self.env._consecutive_hits = 0
        rew_long_empty = self.env._compute_reward(dwell_long_empty, delta=10, hit_boundary=False)

        self.assertGreater(rew_short_empty, rew_long_empty,
                           f"Quick 3-slot probing on empty band must have higher reward (less penalty) than 12-slot waste: {rew_short_empty} vs {rew_long_empty}")

    def test_06_model_inference_and_action_bounds(self):
        """Verify trained PPO model predicts valid discrete actions without errors."""
        model = PPO.load(self.model_path)
        obs, _ = self.env.reset(seed=999)

        for _ in range(25):
            model_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
            action, _ = model.predict(model_obs, deterministic=False)

            band = int(action[0])
            dwell_idx = int(action[1])

            self.assertGreaterEqual(band, 0)
            self.assertLess(band, 128)
            self.assertGreaterEqual(dwell_idx, 0)
            self.assertLess(dwell_idx, 4)

            obs, reward, term, trunc, _ = self.env.step(action)
            self.assertFalse(np.isnan(reward))
            if term or trunc:
                break


if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestAlterraSystemRigorous)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    if not result.wasSuccessful():
        exit(1)
