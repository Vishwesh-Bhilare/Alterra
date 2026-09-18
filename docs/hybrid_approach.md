# Alterra — Hybrid Doctrine + MaskablePPO Scheduler

Single reference document for the hybrid scheduler. Supersedes nothing in
`PROJECT_CONTEXT.md` — this is a deep-dive on one subsystem; read
`PROJECT_CONTEXT.md` first for overall project state.

---

## 1. Why this exists

Three prior pure-PPO scheduler generations were built and evaluated
(`visit_fix_v1`, `power_signal_v1`, `lstm_ppo_direct_jump_v1/v2`). The
direct-jump architecture (`MultiDiscrete([128, 4])`, absolute band choice,
no forced local movement) is correct and necessary — the problem statement
explicitly requires non-linear jumps to intercept agile/periodic emitters.
But across every pure-PPO run, one of two failure modes appeared:

- **Collapse**: policy converges onto 2-3 fixed bands regardless of
  per-episode truth (root cause: single-env training + an unwired custom
  feature extractor — both since fixed).
- **Aimless wandering**: once collapse was fixed
  (`lstm_ppo_direct_jump_v2`), the policy explored broadly
  (`jump_distance_mean≈43`, healthy `entropy_loss≈-6.2` throughout
  training) but never learned to convert that exploration into detections
  — `hit_rate` stayed low and *declined* over 3M timesteps (0.0586 →
  0.0508 → 0.0371). `approx_kl`/`clip_fraction` shrinking to near-zero by
  the end confirms the policy stopped updating meaningfully; it was
  exploring well but never settling into exploiting what it found.

**Decision**: rather than keep tuning a fully unconstrained PPO (more
entropy-schedule tuning, more reward shaping — the same class of fix
already tried three times), constrain the *search doctrine* with rules
and let ML solve the *narrower* problem of choosing the best legal
candidate. Doctrine guarantees coverage and prevents pathological
shortcuts by construction; ML only has to learn "which of these ~10-20
legal bands is most valuable right now," which is a much easier credit
assignment problem than "which of 128 bands, unconstrained."

This also keeps the project's four-way comparison intact: **Traditional
scan** (`traditional_scanner.py`) → **Rule-based heuristic**
(`heuristic_scheduler.py`, the doctrine's origin, un-learned) → **Hybrid
doctrine+ML** (this document) → **Pure-learned PPO**
(`lstm_ppo_direct_jump_v2`). Each is a genuinely different point on the
rules-vs-learning spectrum, all measured on the same environment and
metrics.

---

## 2. Architecture
```

```
                AlterraEnv (enable_doctrine=True)
                          │
                          ▼
                 Live observation state
                 (band_tracks, receiver,
                  hit_miss_seq — unchanged)
                          │
          ┌───────────────┴───────────────┐
          │                               │
          ▼                               ▼
```

doctrine.decide(...)              MaskablePPO policy

(simulation/environment/           (sb3-contrib)

doctrine.py)

│                               │

Mode: EXPLORE / INVESTIGATE /              │

TRACK / RELOCATE                     │

│                               │

Produces: band\_mask (bool[128])            │

dwell\_index (forced)             │

│                               │

└───────────► action\_masks() ───┤

(AlterraEnv method)        │

▼

model.predict(obs, action\_masks=mask)

│

▼

action = [band, dwell]

(dwell always overridden by

doctrine's forced dwell\_index

inside AlterraEnv.step())

│

▼

Receiver.dwell(band, ..., dwell)

│

▼

hit / miss / false\_alarm

│

▼

BandTrack updated, DoctrineState updated

│

└──────────► repeat

````

**Key principle**: doctrine controls *what's legal*, MaskablePPO controls
*which legal option is best*. Dwell is always doctrine-forced, never
ML-chosen — see section 5 for why.

---

## 3. Doctrine modes (`simulation/environment/doctrine.py`)

Ported from `model/agents/heuristic_scheduler.py`'s mode-selection logic,
restructured to emit a **candidate mask** instead of directly picking one
band (the heuristic scheduler still exists, unmodified, as its own
standalone baseline for the comparison — doctrine.py is a separate,
mask-only extraction of the same ideas).

Priority order, evaluated fresh every `step()`:

1. **RELOCATE** (`DoctrineState.force_relocate == True`): fires
   immediately after a TRACK sequence completes (see mode 3). Mask =
   every band at least `MIN_RELOCATE_JUMP` (12) bands from
   `current_band`, excluding the last 8 visited bands
   (`RECENT_BANDS_MAXLEN`). Forces a genuine non-local jump away from a
   just-confirmed target. Dwell forced to index 0 (shortest).

2. **TRACK** (`track_band is not None and track_depth < TRACK_MAX_DEPTH`
   (3)): mask = exactly one band (`track_band`) — no choice, MaskablePPO
   simply confirms. Dwell escalates with `track_confirmations`: index
   `min(1 + confirmations, 3)`, i.e. longer dwell on each consecutive
   confirmed hit, capped at the longest option. This is the "harvest
   maximum pulses to classify the threat" behavior.

3. **EXPLORE (periodic reassessment)**
   (`step_count % EXPLORE_REASSESS_INTERVAL == 0`, interval=6): forces a
   survey candidate even if a known target exists, so one emitter can
   never monopolize the whole episode. Mask from `_survey_mask` (below).
   Dwell forced to index 0.

4. **EXPLORE (coverage)** (`num_bands - len(tracks) > 0`, i.e. any band
   never yet visited this episode): mask from `_survey_mask`. This is
   what guarantees `band_coverage == 1.0` — the doctrine will not allow
   exploitation to begin until every band has been visited at least once.

5. **INVESTIGATE** (all bands visited, at least one has `ever_hit`): mask
   = up to 6 best-scoring previously-hit bands, ranked by
   `0.40×confirms + 0.20×staleness + 0.40×confidence`. Dwell forced to
   index 1 (short-medium).

6. **EXPLORE (fallback)**: if INVESTIGATE finds no candidates (no band
   has ever been hit yet, despite full coverage), falls back to
   `_survey_mask` again.

### `_survey_mask` candidate generation

Two-phase:
- **Phase 1**: any never-visited band, preferring ones ≥12 bands from
  `current_band` (ensures early-episode coverage sweeps are real jumps,
  not a slow crawl). Returns up to 20 candidates.
- **Phase 2** (once every band has been visited at least once): ranks by
  `staleness_score = (step_count − last_visited_t) − visit_count × 4.0`,
  restricted to bands ≥12 away from `current_band` and not in the last 8
  visited, top 12 candidates.

### State updated after every step (`update_after_step`)

- If the just-dwelled band equals `track_band`: increment `track_depth`;
  increment `track_confirmations` only if this dwell was also a hit.
- Else: start a new track if this dwell was a hit (`track_band = band,
  depth=1, confirmations=1`), or clear tracking if it wasn't.
- `force_relocate` set true exactly when `track_band == band and
  track_depth >= 3` — i.e. RELOCATE fires the step *after* TRACK's third
  dwell on the same band, forcing mode 1 next step.
- `recent_bands` (max 8) and `step_count` updated unconditionally.

---

## 4. Action masking mechanism

Uses **sb3-contrib's `MaskablePPO`** — not a custom policy-gradient
implementation. This was a deliberate choice: SB3 already solves "let the
network output logits over the full action space, then zero out
illegal-action probability mass before sampling" correctly and
efficiently; reimplementing it would have been pure risk for no benefit.

- `AlterraEnv.action_masks()` returns a single concatenated boolean array
  of length `num_bands + len(dwell_options)` (sb3-contrib's convention
  for `MultiDiscrete` masking — one flat array covering both sub-spaces
  in order). Band portion = `doctrine.decide(...).band_mask`. Dwell
  portion = one-hot at the doctrine-forced `dwell_index` (i.e. dwell is
  never actually a free choice for MaskablePPO either — see section 5).
- **When `enable_doctrine=False`** (the default, unchanged from every
  prior scheduler generation): `action_masks()` returns an all-`True`
  mask. This means wrapping a non-doctrine env with sb3-contrib's
  `ActionMasker` degrades gracefully to unconstrained PPO rather than
  erroring — a safety property, not something to rely on operationally
  (hybrid training always uses `enable_doctrine=True`).
- Training wraps each env with `ActionMasker(env, mask_fn)` where
  `mask_fn` just calls `env.action_masks()`.
- Inference (`PolicyRunner`, `comparison.py`, GUI) fetches
  `env.action_masks()` fresh every step and passes it to
  `model.predict(obs, action_masks=mask)`.

---

## 5. Dwell: fully rule-based, not ML-chosen — and why

The doctrine forces `dwell_index` in every mode (RELOCATE→0, TRACK→escalating
1-3, EXPLORE→0, INVESTIGATE→1). MaskablePPO's dwell-portion mask is
therefore always a one-hot — the network technically outputs dwell
logits, but only one option is ever legal, so it has no real dwell
decision to make.

**This is intentional, per the original hybrid design proposal**: "I'd
initially keep dwell rule-based... ML focuses on the harder problem:
where should the receiver go next." Simple, well-understood
confidence-based dwell rules (short dwell to verify, escalating dwell on
confirmed tracks) don't need learning — they're not the part of the
problem that was failing.

**Documented caveat**: because sb3-contrib computes the full
`MultiDiscrete` mask (band + dwell) in one call to `action_masks()`,
*before* MaskablePPO has chosen which specific band it will pick, the
dwell decision cannot be conditioned on which particular legal band ends
up selected — only on the current doctrine mode and track state. In
practice this only matters in TRACK mode (dwell escalation is precise
there, since the band is already forced and known). In EXPLORE/INVESTIGATE,
dwell is a fixed short value regardless of which of the several candidate
bands MaskablePPO ends up choosing among.

**Future extension** (not yet built): dwell could become genuinely
band-conditional by computing the mask in two passes — first constrain +
sample the band, then compute a band-specific dwell mask — but this
requires either a custom prediction loop (bypassing
`MaskablePPO.predict`'s single-call convention) or restructuring the
action space itself. Not pursued in this version; current dwell rules are
adequate and untangling this wasn't necessary for the hit-rate problem
being solved.

---

## 6. Reward function: unchanged

The hybrid scheduler uses the exact same `_compute_reward` as every prior
pure-PPO generation (novelty bonus, threat-weighted hit reward with
info-gain scaling, false-alarm penalty, averaged staleness penalty — see
`PROJECT_CONTEXT.md` section on reward history for full derivation).
**Nothing about the reward was changed for the hybrid approach** — the
entire fix here is action-space constraint, not reward shaping. This is
itself informative: it means the reward function was never the actual
problem in the "aimless wandering" failure mode; the unconstrained
128-way choice was.

---

## 7. Files added / changed

**New:**
- `simulation/environment/doctrine.py` — `DoctrineState`, `DoctrineDecision`,
  `decide()`, `update_after_step()`. Pure functions/dataclasses, no
  dependency on gym/RL libraries — could be unit-tested standalone.
- `model/agents/train_hybrid.py` — `MaskablePPO` training script,
  `ActionMasker`-wrapped parallel envs, `MaskableEvalCallback` for
  held-out best-checkpoint selection.
- `model/agents/evaluate_hybrid.py` — 50-episode evaluation reporting the
  usual figures of merit plus doctrine mode distribution (the diagnostic
  that actually distinguishes this scheduler's behavior from prior runs).
- `docs/hybrid_approach.md` — this document.

**Changed:**
- `simulation/environment/gym_env.py` — `AlterraEnv.__init__` gains
  `enable_doctrine: bool = False`. `reset()` creates a `DoctrineState`
  when enabled. New `action_masks()` method. `step()` branches: if
  doctrine enabled, calls `doctrine.decide()`, uses its forced
  `dwell_slots`, calls `doctrine.update_after_step()` after the dwell;
  otherwise behavior is **byte-identical** to the pre-hybrid version
  (`action[1]` fully controls dwell, no masking). `info` dict gains
  `"doctrine_mode"` key (`None` when doctrine disabled).
- `model/agents/model_registry.py` — `VALID_ALGO_CLASSES` gains
  `"MaskablePPO"`.
- `model/agents/policy_runner.py` — `PolicyRunner` gains
  `needs_action_mask` property and an `action_masks` parameter on
  `predict()`; `load_model()` branches to `sb3_contrib.MaskablePPO.load()`
  for that algo class.
- `simulation/environment/comparison.py` — job dicts gain an optional
  `"hybrid": bool` key. `_run_rl` builds the env with
  `enable_doctrine=hybrid`, fetches+passes `action_masks()` every step
  when `runner.needs_action_mask` is true. **Required** for any
  MaskablePPO job — omitting it means the env has no real doctrine state
  and the mask is all-legal, silently degrading the hybrid model to
  unconstrained behavior.

**Unchanged, confirmed compatible:**
- `simulation/environment/scheduler_insight.py` — `BandTrack` dataclass
  reused as-is by `doctrine.py` (imported, not duplicated).
- `model/agents/heuristic_scheduler.py` — still exists as the standalone
  rule-only baseline (no ML), untouched.
- Reward function, observation space, sensor model, emitter models —
  fully unchanged.

---

## 8. How to train and evaluate

```bash
# From repo root, with PYTHONPATH=. (or pip install -e . active)

# Train
python model/agents/train_hybrid.py \
  --total-timesteps 2000000 \
  --n-envs 8 \
  --device cuda \
  --n-steps 4096 \
  --batch-size 2048 \
  --n-epochs 5

# Evaluate (50 fresh held-out episodes + doctrine mode breakdown)
python model/agents/evaluate_hybrid.py \
  --model model/agents/checkpoints/hybrid_maskable_v1/best/best_model.zip \
  --config configs/default_config.yaml \
  --episodes 50
```

`--n-steps`/`--batch-size`/`--n-epochs` defaults (2048/2048/5) already
reflect the GPU-utilization fix discovered while training the pure-PPO
`v2` run (small batches starve the GPU with launch-overhead-bound
micro-batches; large batches actually saturate it) — no need to
re-discover that here.

To register a trained hybrid checkpoint for the GUI/comparison harness:

```python
from model.agents import model_registry
model_registry.register_model(
    repo_root=".",
    source_path="model/agents/checkpoints/hybrid_maskable_v1/best/best_model.zip",
    label="Hybrid Doctrine v1",
    algo_class="MaskablePPO",
)
```

---

## 9. First result (2M timesteps, `hybrid_maskable_v1`)

50-episode held-out evaluation via `evaluate_hybrid.py`:

| Metric | Pure-PPO `v2` (prior) | Hybrid `v1` |
|---|---|---|
| Hit rate | 0.0371–0.0586 (declining) | **0.1995** |
| Band coverage | variable | **1.0000** (every episode) |
| False-alarm rate | 0.107–0.139 | **0.0645** |
| Mean jump distance | ~43 bands | 35.0 bands |
| Reward mean | 271–306 (training rollout) | 548.8 ± 205.6 (held-out eval) |
| Mean hits/episode | not tracked this way | 82.1 |

Doctrine mode distribution across evaluation:

| Mode | Share |
|---|---|
| INVESTIGATE | 40.4% |
| EXPLORE | 38.6% |
| TRACK | 14.0% |
| RELOCATE | 6.9% |

This is the first scheduler generation across the whole project that
clearly beats pure PPO on hit rate while *also* guaranteeing full
spectrum coverage — the exact combination the problem statement asks for
("rapidly sweep the entire band... [without losing] time to
nonthreatening emitters by not giving time to new or threatening ones").
The mode distribution is itself evidence of working as designed: no
single mode dominates, TRACK+RELOCATE together (21%) show real target
engagement is happening (not just endless exploration), and EXPLORE+INVESTIGATE
(79%) show the doctrine is still prioritizing coverage/discovery over
over-committing to any one target.

---

## 10. Ideal outcome / what this is ultimately for

The intended final deliverable is the **four-way comparison table**
(`comparison.py`, already built to support exactly this) run on identical
seeded episodes:

1. **Traditional** (sequential or balanced-random scan, no intelligence)
2. **Heuristic** (`heuristic_scheduler.py`, rules only, no ML)
3. **Hybrid** (this document — rules constrain, MaskablePPO chooses)
4. **Pure PPO** (`lstm_ppo_direct_jump_v2`, fully learned, no rules)

against the problem statement's required figures of merit (Pd, Pfa,
sensitivity, avg intercept rate, avg reward/cost, percent correct
predictions, avg intercept time error — `simulation/metrics/rollout_metrics.py`).

The expected/hoped-for narrative, consistent with results so far:
Traditional < Pure-PPO (PPO should beat blind sweeping on intercept
rate/time, per the original problem statement's whole premise) but
Pure-PPO's real weakness is *consistency* (declining hit rate, incomplete
coverage) — and Hybrid should land at or above Heuristic's reliability
*while* exceeding it on adaptability (ML choosing among legal candidates
should outperform the heuristic's own fixed candidate-scoring formula,
since MaskablePPO can in principle learn subtler value signals than the
hand-tuned `0.40×confirms + 0.20×staleness + 0.40×confidence` weights).
If Hybrid doesn't clearly beat Heuristic on the actual figures of merit,
that's a real, useful negative result worth reporting honestly — it would
mean the constrained action space did all the work and the ML layer
adds cost without adding value, which is exactly the kind of finding this
four-way structure is designed to surface either way.

---

## 11. Known limitations / explicitly deferred

- Dwell is not band-conditional within a single mask call (section 5) —
  deferred, not a blocker for current results.
- `doctrine.py`'s constants (`MIN_RELOCATE_JUMP=12`, `TRACK_MAX_DEPTH=3`,
  `EXPLORE_REASSESS_INTERVAL=6`, `RECENT_BANDS_MAXLEN=8`) are ported
  directly from `heuristic_scheduler.py`'s hand-tuned values, not
  re-derived or swept for the hybrid context specifically. Worth a
  sensitivity sweep before treating current results as fully tuned.
- No four-way comparison run has been executed yet — section 10's table
  is the next concrete step, not yet produced.
- GUI (`interfaces/qt_gui/`) does not yet expose a way to run a
  registered `MaskablePPO` model in doctrine mode — `PolicyRunner`
  supports it, but `PythonBridge`/`MainWindow` wiring for the "hybrid"
  toggle has not been added.