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

  
---

## 11. PPO debugging postmortem (resolved)

Three successive training runs converged to a small fixed set of
`(band, dwell)` actions regardless of per-episode truth, despite fixing
the reward's info-gain incentive and widening the policy network. Root
cause: `_apply_dwell_to_tracks` only updated `band_tracks` on a **hit** —
a miss left that band's observation completely unchanged, so the agent
could never distinguish "just checked, empty" from "never checked."
Almost the entire observation was structurally uninformative; no reward
or network fix could work around that.

**Fix**: `band_tracks` now carries visit state independently of hit state
(`TRACK_FEATURE_DIM` 4→6: `[threat_norm, confidence, time_since_hit,
ever_hit, time_since_visit, ever_visited]`) — every dwell updates the
visit fields regardless of outcome. Observation shape changed
(128×4→128×6); old checkpoints are structurally incompatible, retrained
from scratch as `model/agents/checkpoints/visit_fix_v1/`.

**Result** (2M timesteps, ent_coef=0.02, net_arch=[256,256], vs. random
baseline, both over 20 episodes / 200 steps):

| Metric | Random | visit_fix_v1 (deterministic) |
|---|---|---|
| Avg intercept rate | ~0.226–0.297 | **0.642** |
| Avg intercept time error (slots) | ~459–467 | **323.8** |
| Pd | 1.000 | 0.850 |
| Percent correct | ~0.975–0.980 | 0.982 |

`inspect_policy` confirms qualitative fix: 14-21 unique actions per
200-step episode (was 2-5, byte-identical across episodes), leading
action changes per episode (44→40→94 across 3 test episodes) — this is a
policy reading per-episode state, not executing a memorized schedule.
Deterministic reward std (118 on mean 86) reflects genuine episode-to-
episode emitter-population variance (8-15 emitters, randomized), not
policy inconsistency.

**Considered good enough to move forward with.** Lessons for future
debugging: when a policy collapses to a fixed action set independent of
environment state, check the *observation* for missing/asymmetric signal
before tuning reward shape or network size — both are dead ends if the
agent structurally cannot perceive the thing it needs to react to.

**Still open**: CARDA/Double DQN baselines for the paper's three-way
comparison; deinterleaving inference wiring into real per-emitter tracks
(would let `band_tracks` become emitter-tracks); IG/uncertainty reward
term from DSCAC/CAROTA; the `periodic_scan` edge-sampling bias noted
during debugging (middle bands statistically over-covered vs. edges) —
left as-is, plausibly a legitimate modeling choice rather than a bug.


---

## 12. Qt GUI (`interfaces/qt_gui/`) — batch 1+2 done

C++ Qt6 desktop app embedding the Python interpreter directly (pybind11
`scoped_interpreter`) — calls the real `AlterraEnv` + trained PPO
checkpoint live, no reimplementation. `PythonBridge` owns the interpreter
and exposes `reset`/`step`/`currentMetrics`/`truthMatrix`.
`SpectrogramWidget` renders the ground-truth occupancy matrix with the
receiver's actual dwell path overlaid in red and a live time cursor — same
visualization as `simulation/viz/spectrogram_plot.py`, live instead of a
saved PNG. Seed field + Reset Episode lets a specific scenario be pinned
for reliable demos instead of gambling on a random one live.

**Known-good demo seed**: 42 (intercept rate 0.756, Pd 1.0). Pre-test a
few seeds before any live demo and keep 2-3 good ones on hand — performance
varies a lot by episode (randomized 8-15 emitter population), same as the
scheduler's own eval variance documented in section 11.

**Build gotcha (fixed)**: Python 3.14's `PyType_Spec` has a `slots` struct
field; Qt's `slots`/`signals`/`emit` macros collide with it once pybind11
headers are included. Fixed via `QT_NO_KEYWORDS` (use `Q_SLOTS` instead of
`slots` in Qt class declarations) plus a `#pragma push_macro/pop_macro`
guard around the pybind11 includes in `PythonBridge.h` for translation
units that don't set that flag.

---

## 13. Simulation realism upgrade — Module A (config + physical realism)

Config-driven receiver model + real frequency mapping, ahead of the GUI
overhaul (see section 12) and PPO reward/action decisions (see
`docs/model_changes.md`, written for the PPO/agents owner).

- **New `receiver:` config section**: `instantaneous_bandwidth_hz` (B_I,
  decoupled from `spectrum.band_bandwidth_hz` — the latter is now purely
  the spectrum's binning resolution) and `retune_time_s` (T_r).
- **`Receiver` now models retune time**: switching bands costs
  `retune_time_s` (converted to slots) before the new dwell's detections
  begin, consumed from the same slot budget as everything else. Applies
  identically to `TraditionalScanDriver` since both go through
  `Receiver.dwell`. **No reward change accompanies this** — flagged as an
  open decision for the PPO owner in `docs/model_changes.md`.
- **Real frequency mapping**: `SpectrumConfig.band_center_freq_hz(band)` /
  `band_range_hz(band)` / `total_bandwidth_hz()`. `DwellResult` now carries
  `center_freq_hz`, `freq_lo_hz`, `freq_hi_hz` (receiver-window bounds,
  sized by B_I, not the band bin) and `retune_slots` actually consumed.
- **Detection semantics made explicit**: `Detection.classification`
  property → `hit | miss | false_alarm | correct_reject`, derived from
  existing fields, no change to the Pd(SNR) logistic detection model.
  `DwellResult.classification_counts()` added for convenience.
  `SensorModel.detection_threshold_dbm(band)` / new
  `sensor.detection_threshold_db_above_noise` config field give a
  noise-floor-relative threshold for display purposes (does not gate
  `hit` itself).
- **Metrics (`rollout_metrics.py`) reviewed, not changed**: Pd/Pfa/percent
  correct/avg intercept rate were already computed from actual
  detections/false-alarms with valid ranges — no bug found here.

**Not yet done, explicitly deferred to the PPO/agents owner**: whether
retune cost enters the reward function; whether the current 4-option
`dwell_options_slots` menu satisfies "adaptive dwell" or needs to become
richer/continuous. See `docs/model_changes.md` for the full handoff note.

**Next (Module B)**: exposing existing `band_tracks` state (confidence,
staleness, visit history — all already computed) as GUI-facing scheduler
history/priority data; explore/exploit labeling, either heuristic (no PPO
change) or policy-entropy-derived (needs PPO exposure).

---

## 14. Scheduler explainability — Module B

`BandTrack` (moved to new `simulation/environment/scheduler_insight.py`,
imported by `gym_env.py`) now also carries `visit_count`, `hit_count`, and
a bounded `power_history` deque (length from new config
`scheduler_insight.power_history_len`) — purely additive, does not touch
`band_tracks`'s observation encoding.

Every `AlterraEnv.step()` now computes, using track state as of *before*
that step's own update (i.e. what the scheduler actually knew when it
picked the band):

- `compute_priority_score` — a 0-1 "worth revisiting" score, deliberately
  the mirror image of `_compute_reward`'s existing staleness_penalty term
  (unvisited bands score max; visited-with-a-hit bands rise again as that
  sighting goes stale, weighted by threat x confidence).
- `explain_decision` — `EXPLORE`/`EXPLOIT` + a short human-readable reason
  string (e.g. `"Recent strong detection"`, `"Never scanned before"`).

Exposed via `env.last_decision`, `env.step()`'s info dict
(`decision`/`decision_reason`/`priority_score`), `env.band_priorities()`
(full-spectrum ranking snapshot, on demand), and
`env.recent_events()` / `env.recent_hits()` (rolling history, bounded by
new config `scheduler_insight.event_history_len`).

**Heuristic, not policy-derived**: explore/exploit is inferred from
existing `band_tracks` state after the fact, not from the PPO policy's
actual action distribution/entropy — no PPO exposure needed, nothing here
requires a retrain (confirmed no observation/action/reward changes).

**Only wired into `AlterraEnv.step()`** — `TraditionalScanDriver` bypasses
`step()` entirely (calls `Receiver.dwell` directly), so it has no
decision/priority/history data, which is expected: explainability is
scoped to the adaptive scheduler being explained, not the non-adaptive
baseline.

**Next (Module C)**: bridge this + Module A's frequency/threshold data
into `PythonBridge`/`StepResult` so the Qt GUI can actually render it.

---

## 15. Bridge layer — Module C

`PythonBridge` extended to surface Module A + B data to the C++ side, no
Python-side changes needed (everything was already exposed by `DwellResult`/
`AlterraEnv` methods added in Modules A/B).

- **`StepResult`** gained `retuneSlots`, `freqWindow` (`FrequencyWindow`:
  centerHz/loHz/hiHz), `classification` (`ClassificationCounts`:
  hit/miss/falseAlarm/correctReject), and `decision`
  (`SchedulerDecision`: available/exploreExploit/reason/priorityScore).
- **freqWindow/retuneSlots/classification are populated in both RL and
  traditional-scan modes** — they come from `DwellResult` itself
  (Receiver-level, Module A), not from `AlterraEnv.step()`.
- **`decision.available` is only `true` in RL mode** — traditional scan
  bypasses `AlterraEnv.step()` (calls `Receiver.dwell` directly via
  `TraditionalScanDriver`), so there's no scheduler decision to explain,
  by design (matches Module B's own scoping note).
- **New bridge methods**: `bandPriorities()`, `recentEvents(n)`,
  `recentHits(n)` (Module B, on-demand — meaningful only in RL mode, same
  caveat as above), `noiseFloorDbm()` + `detectionThresholdMarginDb()`
  (per-band threshold line = sum of the two), `instantaneousBandwidthHz()`,
  `retuneTimeS()`, `bandBandwidthHz()`, `bandStartFreqHz()`, `numBands()`
  (static per-episode config, fetched once for GUI axis setup rather than
  repeated per step).

**Next (Module D)**: wire all of the above into `MainWindow`/`SpectrogramWidget`
— real frequency axis, receiver-window rendering, legend, decision panel,
structured table, and the extra panels (priority map, recent detections,
scheduler stats) — this is the last module and has no further Python-side
dependencies.

---

## 16. GUI overhaul — Module D (last module)

`MainWindow.ui` restructured: added a "Scheduler Decision" group box
(current band/freq, priority score, EXPLORE/EXPLOIT + color, reason —
shows "N/A (non-adaptive mode)" when `decision.available` is false, i.e.
traditional-scan modes) between the config panel and the spectrogram, and
replaced the standalone log with a `QSplitter` at the bottom:
`eventsTable` (structured `Time | Band | Frequency | Dwell | Signal |
Result | Reward`, rows color-coded by outcome, capped at 500 rows) next to
a `QTabWidget` with **Priority Map** (top 15 bands by
`bandPriorities()`, refreshed every step), **Recent Detections** (last 10
hits via `recentHits(10)`), **Statistics** (running explore/exploit +
hit/miss/false-alarm/correct-reject counts, tracked in `MainWindow`, reset
each episode), and **Log** (same `objectName="log"` as before, so no
logging call sites changed).

`SpectrogramWidget` reworked: real frequency y-axis (`setSpectrumGeometry`,
tick labels via `bandStartFreqHz`/`bandBandwidthHz`, gridlines), dwell
trajectory color-coded by outcome (green hit / orange miss / red false
alarm / grey correct-reject), most recent dwell rendered as a filled
receiver-window rectangle (real frequency interval, not a point — uses
`freq_lo_hz`/`freq_hi_hz`, independent of the band-bin y-ticks), and an
in-widget legend covering all of 2.2's required elements.

**Consolidation choices made**: "Detection Performance" (2.5) folded into
the same Statistics tab as "Scheduler statistics" rather than a 5th tab —
both are small enough to share one view without crowding; the existing
top-of-window `metricsLabel` (Pd/Pfa/intercept rate/avg reward) still
covers the problem-statement figures of merit separately, unchanged.

**All checklist items (1, 2.1-2.5, 3.1-3.5, 3.7, 3.9) are now implemented
end-to-end** except the two items explicitly deferred to the PPO/agents
owner in `docs/model_changes.md` (3.6 adaptive dwell, and whether retune
cost enters the reward) — those remain open, no further simulation/GUI
work is blocked on them.

---

## 17. Scripted test-scenario timing — Module E1

New optional `active_windows: [[start_slot, end_slot], ...]` field on any
manual scenario spec (`fixed`/`agile`/`periodic_scan`), parsed in
`scenario_builder.py` (`_active_windows`) and honored by `FixedEmitter`
and the `_ManualAgileEmitter`/`_ManualPeriodicScanEmitter` wrappers. When
present, it replaces the usual probabilistic `two_state_markov_mask`
(duty_cycle + mean_burst_slots) with an exact, deterministic on/off
schedule via new `schedule_utils.windows_mask` — guarantees an emitter
appears/disappears at a specific mid-episode slot rather than relying on
a probabilistic roll to produce that timing. Backward compatible: absent
field -> unchanged probabilistic behavior, existing manual scenario YAMLs
(e.g. `manual_example.yaml`) untouched.

Band-selection logic (fixed band / hop schedule / sweep schedule) is
untouched — only the active/inactive mask computation branches.

**Next (Module E2)**: the 8 scenario YAML presets, several of which will
use `active_windows` (mid_episode_burst, silent_gap_revisit).

---

## 18. Scenario preset library — Module E2

8 manual scenario YAMLs under `configs/scenarios/`, each targeting one
behavior, for the GUI's scenario dropdown (Module E3):

| File | Category | Tests |
|---|---|---|
| `mid_episode_burst.yaml` | Timing | scripted mid-episode appear/disappear (E1) |
| `silent_gap_revisit.yaml` | Timing | revisit logic after a long staleness gap |
| `dense_congested.yaml` | Density | prioritization under contention |
| `sparse_single_threat.yaml` | Density | clean intercept-time case, near-uncontested |
| `fast_hopping_evasive.yaml` | Evasion | chasing a moving target, not camping |
| `periodic_scan_focus.yaml` | Evasion | PS's explicit periodic-scan-interception ask |
| `high_false_alarm.yaml` | Detection Stress | resistance to false-alarm baiting |
| `known_baseline.yaml` | (top-level) | human-eyeball sanity check, no edge cases |

**New: `config_overrides` in scenario files.** `high_false_alarm.yaml`
needed a config change (sensor.pfa_rate), not just different emitters, so
`scenario_builder.py` gained `load_scenario_overrides(path)` (reads an
optional top-level `config_overrides: {section: {field: value}}` block)
and `apply_scenario_overrides(config, overrides)` (one-level-deep
`dataclasses.replace`). Absent in the other 7 files -- fully backward
compatible, `load_manual_scenario`'s return type/behavior unchanged.

**Next (Module E3)**: nested scenario menu in the GUI (Timing/Density/
Evasion/Detection Stress submenus + Random Population + Known-Analytic
Baseline at top level) + `PythonBridge` wiring to actually load a chosen
scenario's emitters *and* apply its config_overrides before `reset()`.

---

## 19. Scenario menu + bridge wiring — Module E3 (last of the test-case work)

`PythonBridge` gained `setScenario(name)` / `scenarioName()`, backed by a
new shared `rebuildEnv()` (both `reconfigure()` and `setScenario()` now
call it, so the two compose correctly: switching scenarios preserves the
last-applied manual GUI config, and changing episode length/mode
preserves whichever scenario is currently selected). `rebuildEnv()`
reloads config from disk, applies the manual overrides, then -- if
`scenarioName_` is non-empty -- loads
`configs/scenarios/<name>.yaml` via `scenario_builder`
(`load_manual_scenario` + `load_scenario_overrides` +
`apply_scenario_overrides` + `build_manual_population`, using a fresh
`RNGManager(config.rng_seed)`) and passes the resulting emitters into
`AlterraEnv(config, manual_emitters=...)`. Empty name -> unchanged
default-random-population behavior. Constructor now routes through
`rebuildEnv()` too instead of building the env directly, for one code
path.

`MainWindow.ui`: added a `QToolButton` ("scenarioButton",
`InstantPopup`) next to Random Seed. `MainWindow.cpp`'s
`buildScenarioMenu()` constructs the nested `QMenu` tree (Random
Population + Known-Analytic Baseline at top level; Timing/Density/
Evasion/Detection Stress as flyout submenus, one `QAction` per scenario
file from Module E2, `slug` stored via `QAction::setData`).
`onScenarioSelected(QAction*)` (connected to the menu's `triggered`
signal) calls `bridge_->setScenario(slug)`, updates the button label, and
resets the episode -- any scenario load failure is caught and logged
rather than crashing.

**All planned modules (A/B/C/D + E1/E2/E3) are now complete.** Open items
remain only in `docs/model_changes.md` for the PPO/agents owner.

---

## 20. Compose-from-archetypes — Module F (lightweight custom mix, not a full sandbox)

Deliberately scoped down from a free-form emitter editor: reuses the
exact emitter definitions from the 8 scenario presets (Module E2) as a
fixed catalog of 7 named "archetypes" (`fixed_low/medium/high`,
`mid_episode_burst`, `silent_gap_revisit`, `fast_hopper`,
`periodic_scanner`), and lets the user pick counts of each (0-5) plus an
optional false-alarm-rate boost, rather than authoring raw parameters.

`scenario_builder.build_custom_population(archetype_counts,
boost_false_alarm, config, rng_manager)` clones each requested archetype's
template N times, giving every instance a freshly randomized band/hop-set/
sweep-start placement (via a per-instance-named RNG stream, so duplicates
of one archetype don't collide), keeps `active_windows` verbatim on the
two scripted-timing archetypes, and returns `(config, emitters)` -- the
config return matters because `boost_false_alarm` applies the same
`apply_scenario_overrides` mechanism as `high_false_alarm.yaml`.

`PythonBridge::setCustomComposition(counts, boostFalseAlarm)` sets
`scenarioName_ = "__custom__"` and a new `rebuildEnv()` branch calls the
above; `setScenario()`'s existing YAML-loading branch and the default
random-population path are both unchanged (three-way branch:
`"__custom__"` / named scenario / empty).

GUI: a "Custom Mix..." entry (top-level, next to Random Population) opens
a `QDialog` (built inline in `MainWindow.cpp`, no separate `.ui` file --
small enough not to need one) with one spin box per archetype + the
false-alarm checkbox; OK collects non-zero counts and calls
`setCustomComposition`, updates the scenario button label with the total
emitter count, and resets the episode.

**Explicitly not built**: free-form parameter editing (arbitrary band/
threat/duty-cycle/power per emitter), add/remove dynamic rows, or new
archetype authoring from the GUI -- judged not worth the additional GUI
complexity for what it would add on top of the 8 fixed scenarios, per the
scope discussion before starting this module. If a genuine need for
fully custom emitter definitions comes up later, `build_manual_population`
already accepts arbitrary spec dicts -- only the GUI-side authoring form
would need to be built.

---

## 21. Custom Mix rework — per-instance band ranges, menu simplified

Following the scope discussion: the 8 preset scenarios (Timing/Density/
Evasion/Detection Stress submenus + Known-Analytic Baseline) are removed
from the GUI menu -- only **Random Population** and **Custom Mix...**
remain. The scenario YAML files under `configs/scenarios/` are left on
disk untouched (only the GUI menu entries were removed, in case they're
still wanted via a script or later).

`PythonBridge`'s scenario-string mechanism (`setScenario`/`scenarioName_`)
is replaced by a simpler `bool isCustom_` + `setRandomPopulation()` /
`setCustomComposition(requests, boostFalseAlarm)` pair -- there's no
longer a third "named preset" state to track.

**Per-instance band ranges, not per-archetype-type.** Each emitter added
in the Custom Mix dialog gets its own independent `[bandLo, bandHi]`
range (`CustomEmitterRequest`), not a range shared across all copies of
one archetype. `scenario_builder.build_custom_population` now takes a
flat list of `{archetype, band_lo, band_hi}` requests (one per emitter
instance) instead of `{archetype: count}`, and samples that instance's
placement (band / hop-bands / sweep-start, depending on kind) uniformly
within its own range, clamped to the spectrum and to whatever fits (e.g.
an agile emitter's hop-bandset shrinks if the given range is narrower
than its usual size; a periodic scanner's sweep start is clamped so the
sweep still fits inside the spectrum).

**Emitter-count ambiguity resolved by construction, not by syncing two
fields.** Rather than reconciling the Simulation Configuration panel's
"Override emitter count / Exact count" (which only ever applied to the
random-population path) against a separate custom total, the Custom Mix
dialog is now a dynamic per-instance row list (`+ Add Emitter` / `Remove`
per row) -- the number of emitters *is* the number of rows, there is no
second number to disagree with it. `setEmitterCountControlsEnabled(bool)`
disables the config panel's count controls entirely while Custom Mix is
active (re-enabled on switching back to Random Population), so only one
"how many emitters" control is ever live at a time.

Dialog UI: `QScrollArea` containing dynamically added/removed `QFrame`
rows (archetype `QComboBox` + band-range `QSpinBox` pair + Remove
button), all built inline in `onCustomMixRequested()` (no separate `.ui`
file, consistent with the earlier version of this dialog).

---

## 22. "Override emitter count" removed; frequency-based range picker; Random archetype

- **Removed entirely**: the "Override emitter count / Exact count" controls
  in the Simulation Configuration panel, and their wiring
  (`onOverrideEmittersToggled`, `setEmitterCountControlsEnabled`, and the
  `overrideEmitterCount`/`numEmitters` fields' use in `onApplyConfig` --
  `ManualConfig`'s fields themselves are untouched in `PythonBridge.h/.cpp`,
  just never set from the GUI now, so `reconfigure()` always passes the
  struct's defaults). There is now exactly one place to control emitter
  count for the random-population path: the config file's own
  `emitters.population.total_count_range`. `MainWindow.ui`'s `emittersForm`
  (the QFormLayout holding those controls) was removed from `configGroup`.
- **Custom Mix range picker now reads/writes frequency (GHz), not band
  index.** Each row's two `QDoubleSpinBox`es are labeled "Freq range" and
  range over the full spectrum in GHz (derived from
  `bandStartFreqHz_`/`bandBandwidthHz_`, populated each reset from the
  bridge); a local `freqGHzToBand` lambda converts back to a band index
  right before building `CustomEmitterRequest`s, so
  `scenario_builder.build_custom_population`'s band-index contract is
  unchanged -- the unit conversion is purely a GUI-layer concern.
- **New "Random (any type)" archetype.** Selecting it in a row's type
  combo (slug `"random"`) makes `build_custom_population` resolve that
  instance to a uniformly-chosen concrete archetype at build time (new
  `_random_archetype` helper, using that instance's own placement RNG
  stream, so it's deterministic per seed like everything else here). This
  replaces the old "override emitter count -> random population" flow:
  adding N rows all set to "Random" with a wide frequency range now
  reproduces "N random emitters," fully inside Custom Mix.
- **Dialog widened**: `900x460`, `setMinimumWidth(820)` -- previously
  `560x420` was too narrow to show a row's controls without resizing.
