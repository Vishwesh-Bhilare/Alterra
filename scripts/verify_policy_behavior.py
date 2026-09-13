import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from model.agents.lstm_policy import PPOLSTMExtractor

def verify():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    model = PPO.load("model/agents/checkpoints/targeted_lstm/best/best_model.zip")

    obs, _ = env.reset(seed=42)
    print(f"--- SIMULATING EPISODE TO VERIFY BEHAVIOR ---")
    
    hits_count = 0
    dwell_stats = {}
    band_history = []
    
    for step in range(100):
        model_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(model_obs, deterministic=True)
        
        band = int(action[0])
        dwell_idx = int(action[1])
        dwell_slots = env._dwell_options[dwell_idx]
        
        prev_band = env._current_band
        obs, reward, term, trunc, info = env.step(action)
        
        hit = info["any_hit"]
        consec = info["consecutive_hits"]
        is_revisit = band in band_history[:-1] if len(band_history) > 1 else False
        band_history.append(band)
        
        dwell_stats[dwell_slots] = dwell_stats.get(dwell_slots, 0) + 1
        if hit:
            hits_count += 1
            
        print(f"Step {step:02d} | Band: {band:03d} (prev: {prev_band:03d}) | Dwell: {dwell_slots:02d} slots ({dwell_slots*10}ms) | Hit: {hit} | ConsecHits: {consec} | Revisit: {is_revisit} | Reward: {reward:+.2f}")
        
        if term or trunc:
            break
            
    print("\n--- SUMMARY OF BEHAVIOR ---")
    print(f"Total steps: {len(band_history)}")
    print(f"Unique bands visited: {len(set(band_history))}")
    print(f"Total Hits: {hits_count}")
    print(f"Dwell distribution: {dwell_stats}")

if __name__ == "__main__":
    verify()
