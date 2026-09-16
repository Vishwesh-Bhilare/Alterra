import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config, apply_overrides

def test_single_emitter():
    config = load_config("configs/default_config.yaml")
    config = apply_overrides(config, num_emitters=1, episode_length_slots=1000)
    env = AlterraEnv(config)
    model = PPO.load("model/agents/checkpoints/best/best_model.zip")

    for ep in range(3):
        obs, info = env.reset(seed=100 + ep)
        emitter = env._emitters[0]
        bands = getattr(emitter, "band_index", None) or getattr(emitter, "_hop_bands", None)
        print(f"\n================ Episode {ep} ================")
        print(f"Emitter type: {type(emitter).__name__}, active bands: {bands}")
        print(f"Initial receiver band: {env._current_band}")
        
        hit_count = 0
        dwell_count = 0
        tracking_steps = 0
        
        for step in range(120):
            action, _ = model.predict(obs, deterministic=True)
            step_delta = env._step_sizes[int(action[0])]
            dwell_idx = int(action[1])
            dwell_slots = env._dwell_options[dwell_idx]
            
            obs, reward, terminated, truncated, info = env.step(action)
            is_hit = info["any_hit"]
            if is_hit:
                hit_count += 1
                if step_delta == 0:
                    tracking_steps += 1
            
            if is_hit or info["consecutive_hits"] > 0 or step < 10:
                print(f"Step {step:3d} (t={env._t:4d}) | Band: {info['band']:3d} | Delta: {step_delta:+2d} | Dwell: {dwell_slots:2d} | Hit: {is_hit} | Consec: {info['consecutive_hits']:2d} | R: {reward:+.2f}")
            elif step % 20 == 0:
                print(f"Step {step:3d} (t={env._t:4d}) | Band: {info['band']:3d} | Scanning... | R: {reward:+.2f}")
                
            if terminated or truncated:
                break
        print(f"Summary Ep {ep}: Total Steps: {step+1}, Hits: {hit_count}, Tracking steps (stay when hit): {tracking_steps}")

if __name__ == "__main__":
    test_single_emitter()
