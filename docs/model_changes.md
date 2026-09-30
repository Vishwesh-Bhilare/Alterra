# Model / PPO Team Notes — Simulation Changes (Module A)

Message for whoever owns `model/agents` (PPO scheduler) and reward design.
These are environment-semantics changes, not just config additions — please
read before your next training run.

## 1. Retune time is now real (needs your attention)

`Receiver` now charges `config.receiver.retune_time_s` (converted to
slots) whenever the agent's chosen band differs from the previous dwell's
band. That cost is consumed as slots *before* the new dwell's detections
begin — it eats directly into the episode's fixed slot budget
(`episode_length_slots`), the same way `TraditionalScanDriver` is affected
too.

**Effect on training**: switching bands is no longer free. An agent that
hops bands every step now gets fewer total dwells per episode than one
that revisits the same band repeatedly. This changes the effective
cost-benefit of exploration vs. exploitation, but **`_compute_reward` has
not been touched** — there's no explicit retune penalty or bonus, it's a
pure environment-timing effect right now.

**Action needed from you**: decide whether reward should explicitly
account for retune cost (e.g. a small penalty proportional to
`info["retune_slots"]`, now returned in `step()`'s info dict), or whether
letting it show up only via reduced total-dwell-count is sufficient.
Recommend a short retrain + `inspect_policy` check either way — this
changes the environment dynamics old checkpoints were trained under.

`config.receiver.retune_time_s` defaults to `0.001` (1ms → 0 slots at the
default 10ms `slot_duration_s`, i.e. **currently a no-op** at default
config). It only starts affecting timing once set high enough to round to
≥1 slot. Flagging now so it's not a surprise later when someone tunes it
for realism.

## 2. Dwell is still fixed-menu, not continuously adaptive

`config.environment.dwell_options_slots: [3, 5, 8, 12]` is unchanged — the
agent already picks from 4 discrete dwell lengths per action
(`MultiDiscrete([num_bands, len(dwell_options)])`), so dwell was never a
single fixed value. If "genuinely adaptive dwell" (task item 3.6) means
something beyond this — e.g. continuous dwell time, or dwell chosen as a
function of live per-band confidence rather than just being one of the
agent's own action dimensions — that's an action-space change and squarely
your call. No simulation change has been made preempting this; let us know
if/how you want the action space adjusted.

## 3. Detection outcome semantics — now explicit (informational, no action needed)

`Detection.classification` (in `sensor_model.py`) now names every outcome
as one of `hit / miss / false_alarm / correct_reject`, computed directly
from the existing `hit`/`false_alarm`/`true_occupied` fields you already
had — no change to the Pd(SNR) logistic model or what counts as a hit.
`DwellResult.classification_counts()` gives you a per-dwell tally if
useful for logging/debugging. Purely additive, nothing to retrain over.

## 4. Frequency + receiver window — GUI/display only, no action needed

`band → real frequency` mapping (`SpectrumConfig.band_center_freq_hz` /
`band_range_hz`) and a receiver instantaneous-bandwidth window
(`config.receiver.instantaneous_bandwidth_hz`, independent of the
spectrum's per-band bin width) are new, exposed on `DwellResult`
(`center_freq_hz`, `freq_lo_hz`, `freq_hi_hz`). This is for the GUI's axes
and receiver-window visualization — doesn't touch the observation space,
action space, or reward at all.

---

**Summary of what needs a decision from you**: (1) whether retune cost
should enter the reward explicitly, (2) whether the current 4-option
dwell menu satisfies "adaptive dwell" or needs to become continuous/richer.
Everything else in this batch is additive/display-layer and doesn't
require a retrain on its own.
