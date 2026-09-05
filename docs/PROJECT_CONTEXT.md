# Alterra — Project Context

Sole reference document for this repo. Read this before touching code — it
supersedes any earlier partial context from chat history. Keep it current:
record architectural decisions and status changes here as they happen.

---

## 1. What this project is

**Alterra** is our SIH (Smart India Hackathon) 2026 project, built against
**DRDO problem statement 26055: "Smart Scan strategy for Electronic
Warfare."** The team's actual product/system name (from the SIH idea PPT)
is **CORTEX — Cognitive Observation and Real Time Emitter Exploration**;
"Alterra" is the team name and repo name.

- **GitHub**: https://github.com/Vishwesh-Bhilare/Alterra
- **Organization**: DRDO — Department of Defence Production / IDEX
- **Category**: Software | **Theme**: Robotics and Drones

### 1.1 The problem

Detecting hostile communication/radar signals starts with scanning a wide
frequency spectrum. Sensors have high sensitivity but an instantaneous
bandwidth an order of magnitude below the total spectrum they must cover,
so a receiver sweeps across bands over time. Existing ("open loop")
strategies use only pre-mission data and prioritize sweeping the whole band
fast — wasting dwell time on non-threatening emitters instead of
prioritizing new/threatening ones.

**Ask**: build a **Smart Scan Strategy** — interception framed as a
two-dimensional search problem (adjusting receiver frequency at the correct
time). Required:
- A **system model of the receiver** fed by a **simulated RF environment**
  with ground truth (transmission/non-transmission) per band per time slot.
- Predict **intercept time** and **interception ratio** against emitters
  that are themselves **spatially scanning** and/or **frequency agile**.
- A **robust ML scheduler**, trained on hits and misses, minimizing
  intercept time and maximizing interception rate.
- Work out how to **optimally intercept a periodic-scan receiver/emitter**.
- Figures of merit: **Pd, Pfa, sensitivity, avg intercept rate, avg
  reward/cost, percent correct predictions, avg intercept time error.**

**Why this matters for how we build it**: there is no real training data —
the simulation *is* the dataset. Every number the RL agent learns from
originates in `simulation/`. This is why "no hardcoded values, only
config-driven sampled distributions" is a hard rule, not a style choice.

---

## 2. Team & ownership boundary

- **Vishy**: `simulation/` — RF environment, emitters, sensor/detection
  model, Gymnasium environment, metrics, viz, PDW export, dataset
  generation, CLI.
- **Teammate**: `model/` — perception (MS-UNet1D), deinterleaving (SEDCAM),
  RL agents (CAROTA, Double DQN, PPO).

**`simulation/` and `model/` never import each other's internals.**
`model.agents` depends only on `simulation.environment.gym_env.AlterraEnv`
as an external Gymnasium environment.

---

## 3. System architecture (from the team's diagrams / SIH idea PPT)

### CORTEX pipeline (from the idea PPT)

Simulate → multi-emitter RF/IQ environment generated as test input
Detect → MS-UNet1D segments raw IQ into pulse activity → PDWs
Deinterleave→ SEDCAM sorts mixed PDWs into per-emitter tracks
Assess → per-emitter LSTM/GRU temporal model; DSCAC/CAROTA decide
known-vs-unknown emitter + how uncertain that judgment is
Decide → PPO uses that uncertainty to pick next band + dwell time
Act → receiver retunes; new observation feeds back into the loop

Innovation pillars: autonomous adaptive scanning without prior
intelligence; closed-loop cognitive scan strategy; unknown-emitter/
novelty-aware scanning.

### 13-step end-to-end workflow (technical approach slide)
EW Simulation → capture mixed RF/IQ → PDW detection → PDW deinterleaving
(SEDCAM) → emitter streams (temporal model LSTM/GRU + representation +
identification/matching) → threat assessment (known/unknown via
DSCAC+CAROTA) → knowledge state (novelty, uncertainty, recency) → PPO
scheduler decides next scan action → select frequency & dwell time →
receiver tunes & observes → collect new RF/IQ from selected band →
evaluate hit/miss/IG & reward → update PPO using reward & experience →
loop continues.

> **Open item, not yet resolved in code**: the pipeline's step 12
> ("evaluate HIT/MISS/**IG**") implies an information-gain / uncertainty
> term in the reward, tied to CORTEX's novelty-aware decision-making. The
> reward implemented in `gym_env.py` (section 5.10) only has
> threat-weighted hit/false-alarm/staleness terms — no explicit IG term
> yet. Revisit once `model`'s DSCAC/CAROTA uncertainty output exists to
> feed it.

### 4 zones (system architecture diagram) — deployment-phase, not SIH prototype critical path
- **Zone 1**: Users & External Services — EW Operator/Analyst; IMD
  Weather/Map/Gov APIs; satellite/threat-intel/frequency DB feeds.
- **Zone 2**: Frontend — EW Control Dashboard (real-time signal view,
  emitter list, map, receiver status, alerts, settings) over HTTPS.
- **Zone 3**: Backend — API Gateway, user/role mgmt, EW data mgmt,
  processing orchestration, receiver control, notifications; PostgreSQL +
  Redis + time-series DB.
- **Zone 4**: AI & Intelligence Layer ("the brain") — the CORTEX pipeline
  above, plus a shared Learning & Policy Update loop and an AI Model Store
  (versioned CAROTA/DQN/PPO models).

### Full tech stack (from diagrams)
Python/C++; TensorFlow/PyTorch; Stable-Baselines3 (PPO, DQN) + CAROTA;
MS-UNet1D, SEDCAM (SEM+CVR-DCM+DBSCAN), PDW extraction (TOA/PRI/PW/CF/DOA);
PostgreSQL/Redis/InfluxDB; REST API/Auth/Notification; Docker/GitHub/Nginx;
Qt (C++) + Plotly/Matplotlib; AI Model Store with versioning.

This repo currently implements only the **Simulate** stage — everything
else in the pipeline (Detect/Deinterleave/Assess/Decide beyond a random
policy) is `model/`'s responsibility, still stubbed.

---

## 4. Repository layout (current)

alterra/
├── configs/
│ ├── default_config.yaml
│ └── scenarios/
│ └── manual_example.yaml
├── simulation/
│ ├── emitters/
│ │ ├── base_emitter.py # DONE
│ │ ├── fixed_emitter.py # DONE
│ │ ├── agile_emitter.py # DONE
│ │ ├── periodic_scan_emitter.py# DONE
│ │ ├── schedule_utils.py # DONE
│ │ ├── emitter_factory.py # DONE (default/random mode)
│ │ └── scenario_builder.py # DONE (manual mode)
│ ├── environment/
│ │ ├── spectrum_world.py # DONE — truth layer
│ │ ├── sensor_model.py # DONE — Pd/Pfa
│ │ ├── receiver.py # DONE — dwell logic
│ │ ├── gym_env.py # DONE — AlterraEnv
│ │ └── pdw_export.py # DONE — ground-truth PDW stream
│ ├── metrics/
│ │ └── rollout_metrics.py # DONE
│ ├── utils/
│ │ ├── rng.py # DONE
│ │ └── config_loader.py # DONE
│ └── viz/
│ └── spectrogram_plot.py # DONE
├── model/ # teammate — all stubbed
│ ├── perception/
│ ├── deinterleaving/
│ └── agents/
├── interfaces/
│ ├── cli/main.py # DONE, in active use
│ └── web/ # empty — future
├── scripts/
│ └── generate_dataset.py # DONE — batch episode export
├── docs/
│ └── PROJECT_CONTEXT.md # this file
├── tests/ # NOT YET WRITTEN
├── data/ # generated datasets — gitignore this
├── pyproject.toml
└── requirements.txt


---

## 5. Simulation design (`simulation/`)

### 5.1 Two-layer design
- **Layer A — truth engine** (`spectrum_world.py`): ground-truth band ×
  time-slot occupancy from emitters' precomputed schedules.
- **Layer B — sensor/RF front-end** (`sensor_model.py`): SNR-dependent
  imperfect detections from Layer A's truth. The RL agent only ever sees
  Layer B's output, never Layer A directly.

### 5.2 Discretization
- **Time**: slots (`timing.slot_duration_s`, default 1ms), episode length
  in slots (`timing.episode_length_slots`, default 2000).
- **Frequency**: bands (`spectrum.num_bands`, default 128), each
  `spectrum.band_bandwidth_hz` wide (default 70 MHz), starting at
  `spectrum.band_start_freq_hz` (default 1 GHz).

### 5.3 Emitter taxonomy
All inherit `BaseEmitter` — schedules precomputed once in `reset()`,
`state_at(t)` is O(1) lookup afterward (deterministic, order-independent).
- **FixedEmitter**: one band, Markov on/off bursting.
- **AgileEmitter**: hops within a randomly (or manually) chosen band subset
  every `hop_dwell_slots`, plus Markov bursting within active hops.
- **PeriodicScanEmitter**: sweeps a contiguous band window, dwelling
  `dwell_slots` per band, cycling continuously — models the "periodic scan
  receiver/emitter" the problem statement explicitly calls out.

Every emitter also carries: threat level (1–3), transmit power (dBm,
jittered per active slot), and now (since the PDW-export batch) **pulse
parameters**: `pri_s`, `pw_s`, `pri_jitter_std_s`, `doa_deg` (section 5.13).

### 5.4 Two-state Markov burst model (`schedule_utils.two_state_markov_mask`)
Parameterized by duty cycle `D` and mean ON burst length `L` (slots):

p(on -> off) = 1 / L
p(off -> on) = D * p(on -> off) / (1 - D)

Initial state sampled from the stationary distribution (`P(on) = D`), so no
cold-start bias across episodes.

### 5.5 RNG management (`utils/rng.py`)
`RNGManager` wraps `numpy.random.SeedSequence`. `spawn_named(name)` gives a
deterministic independent stream keyed by a stable string (SHA-256 of name
+ root entropy) — used for every emitter, population selection, the sensor
model, and per-episode seeding in `gym_env`. This means reordering
construction code never silently changes another component's draws.

### 5.6 Config system (`utils/config_loader.py`, `configs/default_config.yaml`)
Every tunable range lives in YAML → frozen dataclasses. Current top-level
sections: `rng_seed`, `spectrum`, `timing`, `emitters` (population + fixed
+ agile + periodic_scan), `sensor`, `environment` (dwell_options_slots +
reward), `scenario` (manual_scenario_path), `pulse` (pri/pw/jitter/doa).

> **YAML gotcha (bit us once, fixed)**: YAML does **not** support Python
> underscore digit separators (`5_000_000` parses as a *string*, not a
> number). Use plain integers or scientific notation (`70e6`, `1.0e9`) in
> configs, and the loader now explicitly `float()`/`int()`-casts every
> numeric field from `SpectrumConfig`/`TimingConfig` rather than trusting
> `**raw[...]` unpacking, so a bad type fails loud at load time instead of
> silently propagating into a `TypeError` deep in `pdw_export.py`.

`emitter_factory.build_population()` is the only place default-mode
per-emitter parameters are sampled. `scenario_builder.build_manual_population()`
is the manual-mode equivalent, reading explicit specs from a scenario YAML
(`configs/scenarios/manual_example.yaml`) instead — same `BaseEmitter`
output either way, so nothing downstream cares which mode built them.

### 5.7 Manual scenario mode (`emitters/scenario_builder.py`)
Lets you hand-specify emitters (kind, band/hop-bands/sweep, duty cycle,
power, threat, and optionally pri_s/pw_s/doa_deg) via YAML instead of
random sampling. `_ManualAgileEmitter` / `_ManualPeriodicScanEmitter`
subclass the normal emitter classes only to pin an explicit hop-band set /
sweep-start band instead of randomizing it. Any pulse fields omitted from
a manual spec fall back to sampling from `config.pulse` ranges.

### 5.8 Sensor model (`environment/sensor_model.py`)
`SensorModel` samples one noise floor per band at construction (stable for
the episode). `Detection` dataclass fields: `band`, `t`, `hit`,
`false_alarm`, **`true_occupied`** (ground truth — was the band actually
active, independent of detection outcome), `estimated_snr_db`,
`true_emitter_id`, `true_threat_level` (both ground truth, **never** fed to
the RL agent). Pd(SNR) is a logistic curve centered at
`sensor.pd_snr50_db` with steepness `sensor.pd_slope_db`; empty bands get an
independent false-alarm draw at `sensor.pfa_rate`.

### 5.9 Receiver (`environment/receiver.py`)
`Receiver.dwell(band, start_t, dwell_slots)` samples the sensor model
across every slot of the dwell, returns a `DwellResult` (list of
`Detection`s + `any_hit`/`any_false_alarm`/`best_hit` helpers). One dwell =
one RL step.

### 5.10 Gym environment (`environment/gym_env.py`)
`AlterraEnv(config, manual_emitters=None)`.
- **Action space**: `MultiDiscrete([num_bands, len(dwell_options_slots)])`
  — the agent picks **both** band and dwell duration (from a fixed discrete
  set of options in `environment.dwell_options_slots`, e.g.
  `[3, 5, 8, 12]`), matching the CORTEX pipeline's "select frequency and
  dwell time" step. Not a continuous dwell value yet — see open items.
- **Observation**: `Dict` — `band_tracks` (`num_bands × 4`: threat_level
  norm, confidence, time_since_last_observed norm, ever_observed flag) +
  `receiver` (`2`: last_band norm, episode progress norm).
- **Reward** (`_compute_reward`): `hit_reward_base × threat_weight` on a
  hit; `idle_cost` (negative) on a clean miss; `false_alarm_penalty`
  (negative) added on any false alarm in the dwell; per-tracked-band
  staleness penalty (`staleness_penalty_coeff × threat_weight ×
  min(time_since_observed / staleness_norm_slots, 1)`) subtracted every
  step for every band with an active track — this is what should make the
  agent prefer revisiting known threats over blind sweeping.
- **KNOWN SIMPLIFICATION**: `band_tracks` are indexed by band, not by
  deinterleaved emitter identity (`model/deinterleaving` doesn't exist
  yet) — the env currently tracks "something worth revisiting in band b",
  not a true per-emitter track. Swap once SEDCAM output exists; shape is
  written so `model/agents` shouldn't need changes when it does.
- **Determinism fix (bug, fixed)**: `reset(seed=X)` now keys the episode
  RNG stream on the given seed (`f"seed_{X}"`) rather than an incrementing
  call counter, so the same seed always reproduces the same episode
  regardless of how many times `reset()` was previously called —
  required for `gymnasium.utils.env_checker.check_env`'s
  `check_step_determinism`, which now passes.

### 5.11 Metrics (`metrics/rollout_metrics.py`)
`MetricsTracker.record_step(dwell_result, reward)` per step;
`.finalize(env)` computes, using `env._emitters` ground truth
(evaluation-only access, never seen by the agent):
- **Pd** = hits / occupied-band-detections
- **Pfa** = false_alarms / empty-band-detections
- **Sensitivity** = Pd restricted to detections with estimated SNR < 10dB
  (falls back to overall Pd if no low-SNR detections occurred)
- **percent_correct** = (hits + true_negatives) / total detections
- **avg_reward** = mean of per-step rewards
- **avg_intercept_rate** = hits / steps
- **avg_intercept_time_error_slots** = mean, over emitters that were both
  truly active and eventually intercepted, of (first slot they were
  actually hit − first slot they were truly active) — `None` if no
  emitter was ever intercepted in the episode.
CLI's `env metrics` averages these across `--episodes` runs (default 5).

### 5.12 Visualization (`viz/spectrogram_plot.py`)
`plot_episode(spectrum_world, dwell_bands, dwell_starts, dwell_ends,
save_path)` — grey imshow of the full ground-truth matrix, red line
segments overlaid per receiver dwell. Confirmed visually correct: high
duty-cycle emitters show as persistent horizontal bars, dwells mostly miss
them under a random policy (expected).

### 5.13 PDW export layer (`environment/pdw_export.py`)
Stand-in for the raw-IQ + MS-UNet1D detection stage — **no raw IQ samples
are synthesized**; each emitter's own schedule is expanded directly into
its true pulse train. `PulseDescriptorWord` fields: `emitter_id` (ground
truth, not for deinterleaving), `toa_s`, `pw_s`, `pri_s`, `cf_hz`
(band-center frequency via `band_center_hz`), `doa_deg`, `amplitude_dbm`,
`true_band`, `true_threat_level`. `BaseEmitter.generate_pdws()` walks
forward in real time (seconds, via `timing.slot_duration_s`) from t=0,
checking `state_at(slot_idx)` at each step, emitting a pulse when active,
advancing by `pri_s + N(0, pri_jitter_std_s)` each iteration.
`generate_episode_pdws()` merges every emitter's pulses into one
time-sorted mixed stream — the interleaved input SEDCAM deinterleaving is
meant to sort back apart. Config: `pulse.pri_s_range`, `pulse.pw_s_range`,
`pulse.pri_jitter_std_s`, `pulse.doa_deg_range` (sampled per-emitter in
default mode; overridable per-emitter in manual scenarios).

> **Open item**: since there's no raw IQ, `model/perception` (MS-UNet1D)
> currently has nothing to consume unless it's trained directly on these
> ground-truth PDWs (skipping the "detect from IQ" step) or we later add
> actual IQ sample synthesis. Needs a decision with the teammate.

### 5.14 Batch dataset generation (`scripts/generate_dataset.py`)
`python scripts/generate_dataset.py --config PATH --episodes N
[--episode-length N] [--out-dir DIR] [--scenario PATH]`. Per episode,
writes `truth_matrix.npy` (bool, bands × slots), `pdws.jsonl` (one PDW per
line), `emitters.json` (id/kind/threat/pri/pw/doa metadata); writes one
top-level `manifest.json` indexing all episodes. Verified: 5 episodes,
500 slots each, 9–14 emitters and 4k–10k PDWs per episode depending on
randomized population.

---

## 6. CLI reference (`interfaces/cli/main.py`, entry point `alterra`)

alterra emitters preview [--config PATH] [--episode-length N]
alterra scenario preview [--config PATH] --scenario PATH [--episode-length N]
alterra env check [--config PATH]
alterra env preview [--config PATH] [--scenario PATH] [--steps N]
alterra env metrics [--config PATH] [--scenario PATH] [--episodes N] [--steps-per-episode N]
alterra env plot [--config PATH] [--scenario PATH] [--steps N] [--out PATH]
alterra pdw preview [--config PATH] [--scenario PATH] [--episode-length N] [--limit N]
alterra pdw export [--config PATH] [--scenario PATH] [--episode-length N] [--out PATH]

All default `--config` to `configs/default_config.yaml`. `--scenario` swaps
manual mode in for any command; omitted = default random population.

---

## 7. Status: done / to-do

**Done**: emitter taxonomy (fixed/agile/periodic-scan, default + manual
modes), truth layer, sensor model (Pd/Pfa), receiver + dwell logic, Gym
env with MultiDiscrete (band, dwell) actions and threat/staleness-shaped
reward, metrics (all 7 problem-statement figures of merit except a
still-open IG/uncertainty term), waterfall viz, ground-truth PDW export
(TOA/PRI/PW/CF/DOA), batch dataset generation script, full CLI.

**To do**:
- `tests/` — no unit tests written yet (emitter schedule reproducibility,
  Markov mask duty-cycle convergence, gym_env determinism, metrics
  correctness on synthetic fixtures are the priorities).
- Decide raw IQ synthesis vs. teammate consuming ground-truth PDWs
  directly (section 5.13 open item).
- Possible IG/uncertainty reward term once `model`'s DSCAC/CAROTA output
  exists (section 3 open item).
- Continuous (not discretized) dwell-time action, if PPO's "continuous
  actions" ambition from the original tech-stack notes is still wanted.
- `interfaces/web/` — not started, explicitly future (CLI-first was the
  agreed plan).
- `model/` itself — entirely teammate's scope, still stubbed.

---

## 8. Conventions for anyone (or any LLM) contributing

- **No hardcoded numeric defaults** in `simulation/` — every tunable value
  comes from `configs/*.yaml` via `AlterraConfig`, sampled through a named
  RNG stream.
- **`simulation/` and `model/` never import each other's internals.**
- **Emitter schedules are precomputed in `reset()`**, never generated
  lazily in `state_at(t)`.
- **RNG streams are always spawned via `RNGManager.spawn_named(<stable-id>)`.**
- **YAML numeric literals**: no underscore separators — use `70e6` /
  `1.0e9` style, and add explicit type casts in `config_loader.py` for any
  new field rather than trusting bare `**raw[...]` unpacking.
- **Ground-truth fields** (`true_emitter_id`, `true_threat_level`,
  `true_occupied`, anything in `PulseDescriptorWord` except what a real
  receiver could measure) are for evaluation/metrics/PDW-export only —
  never surfaced in `AlterraEnv`'s observation space.
- File delivery convention: `cat > path << 'EOF' ... EOF` bash heredoc
  blocks (with `mkdir -p` where needed), test/run commands given
  separately from the code block.
