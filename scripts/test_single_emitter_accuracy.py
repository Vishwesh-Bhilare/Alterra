import numpy as np
from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config, apply_overrides
from simulation.metrics import MetricsTracker

def run_comprehensive_single_emitter_test():
    config = load_config("configs/default_config.yaml")
    config = apply_overrides(config, num_emitters=1, episode_length_slots=1000)
    env = AlterraEnv(config)
    model = PPO.load("model/agents/checkpoints/best/best_model.zip")

    results = []
    print("=" * 80)
    print("           ALTERRA SINGLE-EMITTER TRACKING & ACCURACY TEST")
    print("=" * 80)

    for ep in range(10):
        obs, info = env.reset(seed=200 + ep)
        emitter = env._emitters[0]
        emitter_type = type(emitter).__name__
        
        tracker = MetricsTracker()
        total_hits = 0
        hits_with_track_action = 0
        consecutive_streaks = []
        current_streak = 0
        hit_occurred = False
        steps_to_first_hit = None
        
        for step in range(200):
            action, _ = model.predict(obs, deterministic=True)
            step_delta = env._step_sizes[int(action[0])]
            
            obs, reward, terminated, truncated, info = env.step(action)
            tracker.record_step(env.last_dwell_result, reward)
            is_hit = info["any_hit"]
            
            if is_hit:
                total_hits += 1
                current_streak += 1
                if not hit_occurred:
                    hit_occurred = True
                    steps_to_first_hit = step
                # Check if receiver chooses to stay/track (delta == 0)
                if step_delta == 0:
                    hits_with_track_action += 1
            else:
                if current_streak > 0:
                    consecutive_streaks.append(current_streak)
                    current_streak = 0
            
            if terminated or truncated:
                break
                
        if current_streak > 0:
            consecutive_streaks.append(current_streak)
            
        metrics = tracker.finalize(env)
        max_streak = max(consecutive_streaks) if consecutive_streaks else 0
        lock_ratio = (hits_with_track_action / (total_hits - 1)) if total_hits > 1 else (1.0 if total_hits == 1 else 0.0)
        
        print(f"Test {ep+1:2d} | Emitter: {emitter_type:<18} | 1st Hit Step: {str(steps_to_first_hit):<4} | "
              f"Hits: {total_hits:2d} | Max Streak: {max_streak:2d} | Track/Stay Rate: {lock_ratio*100:5.1f}% | "
              f"Pd: {metrics.probability_of_detection or 0:.2f} | Pfa: {metrics.probability_of_false_alarm or 0:.3f}")
        
        results.append({
            "emitter": emitter_type,
            "hits": total_hits,
            "max_streak": max_streak,
            "lock_ratio": lock_ratio,
            "first_hit": steps_to_first_hit,
            "pd": metrics.probability_of_detection or 0,
        })
        
    avg_hits = np.mean([r["hits"] for r in results])
    locked_tests = [r for r in results if r["hits"] > 0]
    avg_lock_rate = np.mean([r["lock_ratio"] for r in locked_tests]) * 100 if locked_tests else 0
    avg_pd = np.mean([r["pd"] for r in results])
    
    print("-" * 80)
    print(f"Summary: Intercepted in {len(locked_tests)}/10 runs | Avg Hits: {avg_hits:.1f} | Avg Lock/Track Rate: {avg_lock_rate:.1f}% | Avg Pd: {avg_pd:.3f}")
    print("=" * 80)

if __name__ == "__main__":
    run_comprehensive_single_emitter_test()
