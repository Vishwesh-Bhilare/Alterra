
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
