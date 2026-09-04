# Alterra

Smart Scan Strategy for Electronic Warfare — SIH 2026, DRDO PS 26055.

## Layout
- `simulation/` — RF environment, emitters, sensor model, Gymnasium env, metrics. (owner: sim team)
- `model/` — MS-UNet1D perception, deinterleaving, RL agents (PPO / Double DQN via SB3). (owner: model team)
- `interfaces/` — CLI now, web later. Never import `simulation` or `model` internals directly from
  each other — `model.agents` depends on `simulation.environment.gym_env` as its training env, not
  the reverse.
- `configs/` — all tunable ranges/distributions. No magic numbers in code — if it needs a value,
  it goes in a config file and is sampled from a seeded RNG.

## Setup
```bash
python -m venv .venv && source .venv/bin/activate
pip install -e .
```

## Try it
```bash
alterra emitters preview --config configs/default_config.yaml --episode-length 500
```
