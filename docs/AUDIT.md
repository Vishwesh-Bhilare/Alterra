# Alterra — Pre-Polish Audit

Full-codebase review pass (GUI, PythonBridge, simulation/, model/agents/,
configs, README) done from an uploaded snapshot, not a live checkout —
line numbers below are approximate; re-check against the actual file
before patching. Ordered by priority: fix P0 before anything else, P1
before demo, P2/P3 are polish, P4 is scope/reporting, not code.

---

## P0 — Crash bugs

### P0.1 — `PythonBridge::step()` Traditional branch: calling an already-evaluated property
`interfaces/qt_gui/src/PythonBridge.cpp`, `step()`, Traditional-mode branch:

```cpp
r.hit = dwellResult.attr("any_hit")().cast<bool>();
r.falseAlarm = dwellResult.attr("any_false_alarm")().cast<bool>();
```

`any_hit`/`any_false_alarm` are `@property` on `DwellResult`
(`simulation/environment/receiver.py`) — `.attr("any_hit")` already
evaluates the property and returns a plain Python `bool`. The trailing
`()` then tries to call that `bool`, raising `TypeError: 'bool' object
is not callable` → `py::error_already_set`. Confirmed by `gym_env.py`
itself, which always accesses these bare (`dwell_result.any_hit`, no
parens) throughout `step()`.

**Effect**: the very first `Step`/`Start` click in either Traditional
scheduler mode throws.

**Fix**: drop the trailing `()` on both lines — `.attr("any_hit").cast<bool>()`.

### P0.2 — No exception handling around `reset()`/`step()` in the two hottest call sites
`interfaces/qt_gui/src/MainWindow.cpp`:
- `onResetEpisode()` calls `bridge_->reset(seed)` with no try/catch.
- `doStep()` calls `bridge_->step()` with no try/catch.

Every other `PythonBridge` call site in this file (`onImportModel`,
`onRlModelComboChanged`, `onCustomMixTriggered`, `onRunComparison`,
`preloadModel`) already wraps in `try { ... } catch (const
std::exception& e) { QMessageBox::critical(...); }`. These two don't,
and an exception escaping a Qt slot during the event loop is undefined
behavior (`std::terminate` in practice) — so P0.1, or any other
exception from `reset()`/`step()` (a malformed scenario file, an
incompatible model), crashes the whole app instead of showing a dialog.

**Fix**: wrap both call sites the same way the others already are.

---

## P1 — Behavioral regressions / correctness

### P1.1 — `DwellOutcome::CorrectReject` is dead code
`PythonBridge::step()` only ever assigns `Hit`, `FalseAlarm`, or `Miss`:

```cpp
if (r.hit) r.dwellOutcome = DwellOutcome::Hit;
else if (r.falseAlarm) r.dwellOutcome = DwellOutcome::FalseAlarm;
else r.dwellOutcome = DwellOutcome::Miss;
```

This collapses "true miss" (occupied, not detected) and "correct
reject" (empty, correctly reported nothing) into the same "Miss"
bucket — every genuinely-empty band now shows orange "Miss" in the
spectrogram, events table, and legend, even though
`DwellResult.classification_counts()` (which `gym_env.py`'s own
`HistoryEvent` already uses) correctly distinguishes all four outcomes.

**Fix**: read `dwellResult.attr("classification_counts")()` (already
exposed) in both `step()` branches and pick the outcome from that dict
instead of the two booleans.

### P1.2 — `PythonBridge::loadModel()` silently resets to seed 0
```cpp
void PythonBridge::loadModel(...) {
    ...
    hasModel_ = true;
    reset(0);   // <-- hardcoded, ignores whatever seed the GUI is on
}
```
Called from `onRlModelComboChanged()`. Switching the RL Model dropdown
resets the running episode to seed 0 regardless of the Seed spinbox's
current value, and nothing re-syncs the spinbox afterward — the
displayed seed and the actual episode seed silently disagree.

**Fix**: either don't auto-reset on model change (let the next
Start/Step/Reset pick it up), or reset using the GUI's current seed
value (needs a seed getter threaded through, or have `MainWindow` call
`onResetEpisode()` itself right after `loadModel()` instead of
`PythonBridge` resetting internally).

### P1.3 — `gui_comparison.run_full_comparison()` is dead code with a misleading docstring
Its docstring claims `cli_main.py`'s `alterra env compare` still uses
it — it doesn't; that command has its own inline traditional-vs-random
comparison logic and never imports `gui_comparison` at all. Nothing
in the codebase calls `run_full_comparison`.

**Fix**: either delete it, or actually wire it into the CLI if an
"everything registered" comparison mode is wanted there too. Low
priority — just don't leave the misleading comment.

---

## P2 — UX / polish

### P2.1 — Doctrine mode isn't in the structured UI
`StepResult.doctrineMode` is populated correctly, but `MainWindow.cpp`
only surfaces it in the plain-text Log line (`mode=%9`). The Decision
strip (`currentBandLabel`/`decisionValueLabel`/`reasonValueLabel`) has
no doctrine-mode field at all — for the hybrid scheduler, EXPLORE vs.
INVESTIGATE vs. TRACK vs. RELOCATE is the actual differentiator being
demoed, and it's currently invisible unless you're reading the log feed.

**Suggested fix**: add a `doctrineModeLabel` to the Decision strip,
shown only when `!r.doctrineMode.empty()` (hidden/blank for
non-hybrid runs).

### P2.2 — `CustomMixDialog`'s default row band range doesn't adapt to `numBands_`
`addRow(const std::string& archetype = "random", int bandLo = 0, int
bandHi = 127)` — the `127` default is hardcoded, not derived from the
`numBands_` passed to the dialog's constructor. Currently harmless
(default config has 128 bands, and Qt clamps the spinbox value to its
range anyway), but fragile if `num_bands` is ever changed in config.

**Suggested fix**: default `bandHi` to `numBands_ - 1` in the
constructor's call to `addRow()`, not in the header's default
parameter.

---

## P3 — README / documentation misalignment

The project README (the "Alterra — Cognitive Smart Scan Scheduler"
doc) describes a system that in several places doesn't match the
actual code. Judges/readers will hit these directly if they follow the
Quick Start section:

1. **Action space**: README says `Direction: Step Down/Stay/Step Up`
   (relative). Actual `gym_env.py`: `action[0]` is an *absolute* band
   index — the docstring literally says "direct (absolute) frequency
   selection." Whole "Operational Scan Doctrine" diagram's dir=1/Δ=0
   framing doesn't match the real action encoding.
2. **Checkpoint path**: README's Quick Start uses
   `model/agents/checkpoints/best/best_model.zip` (flat). Actual
   `train_hybrid.py`/`train_ppo.py` save under
   `model/agents/checkpoints/<run_name>/best/best_model.zip`
   (run-name-scoped).
3. **GUI launch command is missing a required argument.** README:
   `./interfaces/qt_gui/build/alterra_gui configs/default_config.yaml
   model/agents/checkpoints/best/best_model.zip` (2 args). Actual
   `main.cpp`: `preloadModel(argv[2], argv[3])` only fires when `argc >
   3`, i.e. needs config + model path + algo class (3 args) — the
   README's example silently loads no model at all.
4. **References to files not present anywhere in what's been
   reviewed**: `run.sh`, `scripts/train_smart_ppo_lstm.py`,
   `scripts/test_all_scenarios.py`, `tests/test_system_rigorous.py`.
   If these don't exist, the Quick Start's test/train commands fail
   outright. If they're meant to exist, they need to be written.
5. **CMake example is macOS/Homebrew-specific**
   (`/opt/homebrew/opt/qtbase`) — inconsistent with the KDE/Linux
   workflow used throughout this project's actual development. Not
   wrong, just worth a second (Linux) example or a note.

**Fix**: either update the README to match the real system, or treat
it as a spec and build the missing pieces — needs a decision, not just
an edit, since #4 implies real work (a test suite + a training script
alias) if the README is meant to be accurate as written.

---

## P4 — Still-open scope items (not bugs, carried from PROJECT_CONTEXT.md)

- `tests/` has no actual test files anywhere in what's been reviewed —
  flagged as empty as far back as the original PPO debugging postmortem,
  still true now.
- Original three-way comparison plan (CAROTA / Double DQN / PPO) has
  become, in practice, a four-way one (Traditional / Heuristic / Hybrid
  MaskablePPO / pure PPO or RecurrentPPO) — `heuristic_scheduler.py`
  looks like CAROTA's intended replacement, but a Double-DQN baseline
  was never built; only PPO-family models exist. Worth confirming this
  is an intentional scope change before writing it up as "the" comparison
  in a report, since it's a different claim than the original plan.
- No full hybrid-model training run + `evaluate_hybrid.py` results are
  in hand yet (per `hybrid_approach.md`, only early/short-run numbers
  exist) — needed before any comparison numbers go in a report or demo.

---

## Suggested fix order

1. P0.1 + P0.2 together (same file, same sitting — the crash and the
   missing safety net that would have contained it).
2. P1.1 (classification fix) — touches the same `step()` function as P0.1.
3. P1.2 (seed desync) — small, isolated.
4. P2.1 (doctrine mode in the UI) — now meaningful once P0/P1 are solid.
5. P1.3, P2.2 — quick cleanups, any time.
6. P3 — README, whenever there's a documentation pass; decide "fix docs"
   vs. "build the missing pieces" first.
7. P4 — not code — training run + a documented decision on the
   comparison scope, ahead of any report/demo claims.
