# model/

Owner: teammate. Consumes `simulation.environment.gym_env` (coming in Batch 2)
as the Gymnasium training environment.

- `perception/` — MS-UNet1D, raw IQ → PDW extraction
- `deinterleaving/` — PDW stream → per-emitter track association
- `agents/` — PPO (primary) and Double DQN (baseline) via Stable-Baselines3,
  trained against `simulation.environment.gym_env.AlterraEnv`
