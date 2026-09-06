# Alterra — Project Context

Sole reference document for this repo. Read this before touching code — it
supersedes any earlier partial context from chat history. Keep it current.

---

## 1. What this project is

**Alterra** is our SIH (Smart India Hackathon) 2026 project, built against
**DRDO problem statement 26055: "Smart Scan strategy for Electronic
Warfare."** The team's product name (from the SIH idea PPT) is **CORTEX —
Cognitive Observation and Real Time Emitter Exploration**; Alterra is the
team/repo name.

- **GitHub**: https://github.com/Vishwesh-Bhilare/Alterra
- **Organization**: DRDO — Department of Defence Production / IDEX
- **Category**: Software | **Theme**: Robotics and Drones

### 1.1 The problem
Detecting hostile communication/radar signals starts with scanning a wide
frequency spectrum. Sensors have an instantaneous bandwidth an order of
magnitude below total spectrum, so a receiver sweeps bands over time.
"Open loop" strategies (pre-mission data only) waste dwell time on
non-threats instead of prioritizing new/threatening emitters.

**Ask**: a **Smart Scan Strategy** — interception as a two-dimensional
search problem (adjust receiver frequency at the correct time). Required: a
receiver system model fed by a simulated RF environment with ground truth;
predict intercept time/ratio against spatially-scanning and/or
frequency-agile emitters; a robust ML scheduler trained on hits/misses;
work out optimal interception of a periodic-scan emitter; figures of
merit: **Pd, Pfa, sensitivity, avg intercept rate, avg reward/cost, percent
correct predictions, avg intercept time error.**

**Why this shapes how we build it**: no real training data exists — the
simulation *is* the dataset. Hence the hard rule: no hardcoded values in
`simulation/`, only config-driven sampled distributions.

---

## 2. Team & ownership boundary

- **Vishy**: `simulation/` (RF env, emitters, sensor model, gym env,
  metrics, viz, PDW export, dataset generation, CLI) — **and has now also
  trained both `model/deinterleaving` and `model/agents`**, originally the
  teammate's scope, to avoid integration mismatches before handoff.
- **Teammate**: `model/perception` (MS-UNet1D) still open; should review
  and continue iterating on `model/deinterleaving` / `model/agents` now
  that a first trained version of each exists.

**`simulation/` and `model/` still never import each other's internals** —
`model.agents` only depends on `simulation.environment.gym_env.AlterraEnv`.

---

## 3. System architecture

### CORTEX pipeline (SIH idea PPT)

Simulate → Detect (MS-UNet1D → PDWs) → Deinterleave (SEDCAM) →
Assess (LSTM/GRU + DSCAC/CAROTA known/unknown + uncertainty) →
Decide (PPO: band + dwell time) → Act (receiver retunes, loop continues)


### Confirmed workflow diagram (10-step loop, matches gym_env 1:1)

1 EW Simulation → 2 Initialize environment & receiver state →
3 Band-wise state/belief → 4 PPO smart scan scheduler → 5 Select next scan →
6 Receiver collects RF observations → 7 Observe selected band →
8 Evaluate observations (HIT/MISS) → 9 Update band-wise state →
10 Calculate reward & update PPO → (feedback loop to step 3)

Mapping to code: step 3 = `AlterraEnv._build_observation`'s `band_tracks`;
step 4-5 = the PPO action (`MultiDiscrete[band, dwell_idx]`); step 6-7 =
`Receiver.dwell`; step 8 = `Detection.hit`; step 9 =
`_apply_dwell_to_tracks`; step 10 = `_compute_reward` + SB3's own PPO
update. **This confirms band-indexed state (not emitter-indexed) is the
intended design, not just a placeholder** — deinterleaving refines it
later but was never meant to block the RL loop.

> **Open item, still unresolved**: the 13-step technical-approach slide's
> step 12 ("evaluate HIT/MISS/**IG**") implies an information-gain /
> uncertainty term in the reward tied to DSCAC/CAROTA's novelty output.
> Not yet implemented — `_compute_reward` only has
> hit/false-alarm/staleness terms. Revisit once DSCAC/CAROTA produces a
> usable uncertainty signal.

### 4 zones + full tech stack — deployment-phase, not SIH-prototype critical path
Zone 1 Users/External Services, Zone 2 Frontend (EW Control Dashboard),
Zone 3 Backend (API gateway, orchestration, DBs), Zone 4 AI/Intelligence
Layer (the CORTEX pipeline + AI Model Store). Stack: Python/C++,
TensorFlow/PyTorch, SB3+CAROTA, MS-UNet1D/SEDCAM/PDW extraction,
PostgreSQL/Redis/InfluxDB, REST API, Docker/GitHub/Nginx, Qt(C++)+
Plotly/Matplotlib. **Not started** — CLI-first was the agreed plan; GUI
explicitly deferred until simulation + model are validated.

---

## 4. Repository layout (current)

alterra/
├── configs/
│ ├── default_config.yaml
│ ├── deinterleaving_train.yaml
│ └── scenarios/manual_example.yaml
├── simulation/ # DONE (see section 5)
│ ├── emitters/ environment/ metrics/ utils/ viz/
├── model/
│ ├── perception/ # NOT STARTED (teammate — MS-UNet1D)
│ ├── deinterleaving/ # DONE — first trained version (section 6)
│ │ ├── dataset.py model.py losses.py train.py
│ │ └── checkpoints/encoder_epoch{1..10}.pt
│ └── agents/ # DONE — first trained version (section 7)
│ ├── train_ppo.py evaluate.py
│ └── checkpoints/ tb_logs/
├── interfaces/cli/main.py # DONE, in active use
├── interfaces/web/ # empty — future
├── scripts/generate_dataset.py # DONE
├── docs/PROJECT_CONTEXT.md # this file
├── tests/ # NOT YET WRITTEN
└── data/ # generated sim datasets — gitignore

Note: the Turing Synthetic Radar Dataset used for `model/deinterleaving`
training lives **outside this repo** (path passed via `--train-dir`/
`--val-dir`), not under `alterra/`.

---

## 5. Simulation (`simulation/`) — summary, see prior detail preserved below

Two-layer design: Layer A truth engine (`spectrum_world.py`), Layer B
sensor/RF front-end (`sensor_model.py`, SNR-dependent Pd/Pfa). Time in
slots (default 1ms × 2000 = 2s episodes), frequency in bands (default 128
× 70MHz from 1GHz). Three emitter types (Fixed/Agile/PeriodicScan) via
`BaseEmitter`, default (`emitter_factory`) or manual
(`scenario_builder`) construction, both config-driven, no hardcoded
values. Two-state Markov burst model parameterized by duty cycle + mean
burst length. RNG via `RNGManager.spawn_named(<stable-id>)` throughout.

`AlterraEnv`: `MultiDiscrete([num_bands, len(dwell_options)])` action
(band + dwell time together, per the confirmed workflow diagram);
Dict observation (`band_tracks`: threat/confidence/staleness/ever-observed
per band; `receiver`: last band + episode progress); reward =
threat-weighted hit − false-alarm penalty − per-band staleness penalty.
`reset(seed=X)` is now correctly deterministic per seed (fixed a bug where
it wasn't — `gymnasium.utils.env_checker.check_env` passes).

Metrics (`rollout_metrics.py`) compute all 7 problem-statement figures of
merit except the still-open IG/uncertainty term. Viz
(`spectrogram_plot.py`) confirmed visually correct. PDW export
(`pdw_export.py`) is a ground-truth stand-in for the raw-IQ+MS-UNet1D
detection stage (no raw IQ synthesized) — `TOA/PRI/PW/CF/DOA` per pulse,
merged into one time-sorted mixed stream. Batch generation via
`scripts/generate_dataset.py` (truth matrix + PDWs + emitter metadata per
episode + manifest.json).

**YAML gotcha**: no underscore digit separators (`5_000_000` parses as a
string) — use `70e6`/`1.0e9` style; loader explicitly casts every numeric
field rather than trusting `**raw[...]`.

---

## 6. Deinterleaving model (`model/deinterleaving/`) — trained, first version

Trained on the **Turing Synthetic Radar Dataset**, `scan` split (realistic
sweeping receiver; `archive` split has an incompatible schema and is out
of scope, `stare` split is supported by the same code but much heavier per
file).

**Dataset schema** (`data`: (N,5) float32 = `[ToA_us, Freq_MHz, PW_us,
AoA_deg, Amp_dB]`; `labels`: (N,1) int8 = ground-truth per-pulse emitter
cluster id, up to ~90-96 emitters per file).

**Approach**: `PDWWindowDataset` randomly samples fixed-length windows
(default 256 pulses) of consecutive ToA-sorted PDWs per file (loads each
file's full array once — scan files are ~3-40MB, tractable), normalizes
features (ToA→inter-pulse delta in seconds, freq/pw/aoa/amp each scaled to
O(1)). `PDWEncoder` (Transformer, `hidden_dim=64, embed_dim=32,
num_layers=3, num_heads=4` by default) maps each pulse in a window to a
normalized embedding. Trained with **supervised contrastive loss**
(`losses.py`) — pulls same-emitter pulses together, pushes different-
emitter pulses apart, computed per-window using the ground-truth labels.
At eval time, **DBSCAN** clusters the embeddings (matching SEDCAM's own
SEM+CVR-DCM+DBSCAN design), scored against ground truth via ARI/V-measure/
AMI (the challenge's own clustering metrics).

**First trained result** (20 train files, 5 val files, 10 epochs, default
config): loss 5.18→5.01, **val ARI/V-measure/AMI all converged to ~0.74**,
plateauing after epoch ~6. Checkpoints:
`model/deinterleaving/checkpoints/encoder_epoch{1..10}.pt`.

**To do**: scale to more files/epochs, sweep `dbscan_eps`/`min_samples`,
write an inference script that takes a live PDW stream and outputs
per-emitter track assignments — this is what would let `gym_env.py` swap
its band-indexed tracks for real emitter-indexed ones (the documented
"known simplification").

Run: `python -m model.deinterleaving.train --train-dir <scan>/train_scan
--val-dir <scan>/test_scan [--max-train-files N] [--max-val-files N]`.

---

## 7. RL scheduler (`model/agents/`) — trained, first version

`train_ppo.py` trains **PPO** (Stable-Baselines3, `MultiInputPolicy` for
the Dict observation space) directly against `AlterraEnv` — nothing
model-specific about the env, it's the same one `alterra env preview`
uses with a random policy. `evaluate.py` runs a trained checkpoint through
the same `MetricsTracker` as the CLI's `env metrics`, for apples-to-apples
comparison against the random baseline.

**Training runs so far**:
1. Smoke test, 20k timesteps, 2 envs — confirmed the loop works
   end-to-end; policy still near-random (entropy_loss barely moved).
2. Real run, 500k timesteps, 4 envs, GPU — `ep_rew_mean` climbed from
   ~0.5 to ~130-144, `explained_variance` up to 0.77-0.87, entropy_loss
   down from -6.23 to ~-5.3 (policy sharpening).

**Result — PPO (500k) vs. random-policy baseline**, both evaluated over
20 episodes / 200 steps via `MetricsTracker`:

| Metric | Random | PPO (500k) |
|---|---|---|
| Avg intercept rate | ~0.226–0.297 | **1.477** (~5.7×) |
| Avg reward | ~0.152–0.224 | **2.355** (~12×) |
| Avg intercept time error (slots) | ~459–467 | **272.3** (better) |
| Pd | 1.000 | 0.550 |
| Pfa | ~0.021–0.026 | 0.019 |
| Percent correct | ~0.975–0.980 | 0.983 |

PPO clearly learned to revisit active bands far more often and finds
emitters faster (intercept rate, avg reward, intercept time error all
improved substantially). **The Pd drop is not a regression**: Pd is
measured only over bands PPO actually dwelled on while occupied. A random
policy's rare occupied-dwells landed almost entirely on the easy,
high-SNR, high-duty-cycle fixed/periodic emitters. PPO now also chases
weaker agile emitters, pulling more low-SNR encounters into the Pd
denominator even though total hits are up substantially — `sensitivity ==
Pd` here is consistent with that read (most occupied encounters now fall
in the low-SNR bucket used to define `sensitivity`).

**`train_ppo.py` now supports** (added for the "fully train" pass):
periodic checkpointing (`CheckpointCallback`, default every 100k
timesteps), held-out evaluation during training with best-model saving
(`EvalCallback`, default every 50k timesteps, 10 eval episodes,
`best_model.zip` saved separately from the final checkpoint), and
`--resume-from` to continue training from a saved checkpoint.

**To do**:
- Full training run in progress/planned at 2M timesteps — evaluate
  `checkpoints/best/best_model.zip` (not necessarily the final save) once
  done.
- Per-emitter-kind breakdown of hits (is PPO neglecting the
  periodic-scan emitter for easier agile ones?) — not yet measured.
- CARDA and Double DQN baselines still not trained — needed for the
  paper's three-way comparison (CARDA heuristic baseline / Double DQN RL
  baseline / PPO proposed, per the architecture diagram).
- No IG/uncertainty term in the reward yet (see section 3 open item).

Run: `python -m model.agents.train_ppo --timesteps N [--n-envs N]
[--resume-from PATH]` then `python -m model.agents.evaluate --model PATH
--episodes N --steps-per-episode N`.

---

## 8. CLI reference (`interfaces/cli/main.py`, entry point `alterra`)

alterra emitters preview [--config PATH] [--episode-length N]
alterra scenario preview [--config PATH] --scenario PATH [--episode-length N]
alterra env check [--config PATH]
alterra env preview [--config PATH] [--scenario PATH] [--steps N]
alterra env metrics [--config PATH] [--scenario PATH] [--episodes N] [--steps-per-episode N]
alterra env plot [--config PATH] [--scenario PATH] [--steps N] [--out PATH]
alterra pdw preview [--config PATH] [--scenario PATH] [--episode-length N] [--limit N]
alterra pdw export [--config PATH] [--scenario PATH] [--episode-length N] [--out PATH]

Model-side training/eval is invoked directly as Python modules (not yet
wired into the `alterra` CLI): `python -m model.deinterleaving.train ...`,
`python -m model.agents.train_ppo ...`, `python -m model.agents.evaluate ...`.

---

## 9. Status: done / to-do

**Done**: full simulation stack (section 5); first trained deinterleaving
encoder (~0.74 ARI/V-measure/AMI, section 6); first trained PPO scheduler
with a clear, validated improvement over random (section 7); confirmed the
gym env's design matches the team's own workflow diagram exactly.

**To do, roughly in order**:
1. Full 2M-timestep PPO training run, evaluate the eval-selected best
   checkpoint.
2. CARDA (heuristic baseline) and Double DQN implementations, for the
   three-way comparison the paper needs.
3. Deinterleaving inference script + wiring real per-emitter tracks into
   `gym_env.py` (replacing the band-indexed simplification).
4. Decide raw IQ synthesis vs. training MS-UNet1D directly on ground-truth
   PDWs (still-open item from the PDW export batch).
5. Possible IG/uncertainty reward term once DSCAC/CAROTA exists.
6. `tests/` — still nothing written.
7. `interfaces/web/` and the Qt/GUI deployment layer — explicitly
   deferred until 1-4 above are solid.

---

## 10. Conventions for anyone (or any LLM) contributing

- **No hardcoded numeric defaults** in `simulation/` — everything from
  `configs/*.yaml` via `AlterraConfig`, sampled through a named RNG stream.
- **`simulation/` and `model/` never import each other's internals.**
- **Emitter schedules precomputed in `reset()`**, never lazily in
  `state_at(t)`.
- **RNG streams via `RNGManager.spawn_named(<stable-id>)`.**
- **YAML numerics**: no underscore separators; explicit casts in
  `config_loader.py` for new fields.
- **Ground-truth fields** (`true_emitter_id`, `true_threat_level`,
  `true_occupied`, PDW `emitter_id`) are evaluation/metrics/training-only —
  never surfaced in `AlterraEnv`'s observation space.
- File delivery convention: `cat > path << 'EOF' ... EOF` bash heredoc
  blocks (with `mkdir -p` where needed), run/test commands given
  separately from the code block.
