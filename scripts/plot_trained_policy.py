from stable_baselines3 import PPO
from simulation.environment import AlterraEnv
from simulation.utils.config_loader import load_config
from simulation.viz.spectrogram_plot import plot_episode
from model.agents.lstm_policy import PPOLSTMExtractor

def main():
    config = load_config("configs/default_config.yaml")
    env = AlterraEnv(config)
    model = PPO.load("model/agents/checkpoints/targeted_lstm/best/best_model.zip")

    obs, _ = env.reset(seed=100)

    dwell_bands, dwell_starts, dwell_ends = [], [], []
    for _ in range(200):
        model_obs = {k: v for k, v in obs.items() if k in model.observation_space.spaces}
        action, _ = model.predict(model_obs, deterministic=True)
        obs, reward, term, trunc, info = env.step(action)
        dr = env.last_dwell_result
        dwell_bands.append(dr.band)
        dwell_starts.append(dr.start_t)
        dwell_ends.append(dr.end_t)
        if term or trunc:
            break

    out_path = "targeted_lstm_waterfall.png"
    plot_episode(env._spectrum_world, dwell_bands, dwell_starts, dwell_ends, out_path)
    print(f"Saved {out_path} with {len(dwell_bands)} dwells")

if __name__ == "__main__":
    main()
