~
❯ cd Projects/Alterra/

~/Projects/Alterra hybrid-scheduler*
❯ source .venv/bin/activate.fish

~/Projects/Alterra hybrid-scheduler*
.venv ❯ bash -c 'python3 - << \'EOF\'
        path = "docs/gui_model_merge_status.md"
        s = open(path).read()

        old = """## 6. Decisions (resolved)

### 6.1 CLI filename — RESOLVED
`cli_main.py` is canonical. Delete `interfaces/cli/main.py`. Update
`pyproject.toml`:

```toml
[project.scripts]
alterra = "interfaces.cli.cli_main:cli"
```

### 6.2 Heuristic scheduler on Comparison page — RESOLVED
Add a checkbox (`includeHeuristicCheck`, checked by default, matching the
pattern of `includeSequentialCheck`/`includeBalancedRandomCheck`) to
`MainWindow.ui`'s `modelsGroup`.

Update `comparisonDescriptionLabel`'s text to mention the Heuristic
scheduler alongside Adaptive RL and the two Traditional baselines.

`gui_comparison.run_full_comparison()`'s rewrite (task 6) takes an
`include_heuristic: bool` alongside the other selections.

### 6.3 Custom Mix scenario UI — RESOLVED
No existing dialog exists. Build it from scratch.

New task 8a: create a `QDialog` (`CustomMixDialog`, own `.ui` or hand-built)
reachable from `scenarioButton`'s popup menu alongside the static preset list.

Contents:

- Repeatable "emitter request" row
- Archetype dropdown populated from `scenario_builder`'s archetype catalog
  keys plus `"random"`
- `band_lo` / `band_hi` spinboxes
- "Add Emitter" button
- "Remove" button per row
- `boost_false_alarm` checkbox
- OK / Cancel buttons

On accept, call a new
`PythonBridge::buildCustomScenario(...)` wrapping
`scenario_builder.build_custom_population()`, then behave like any other
scenario selection and apply the scenario on the next reset.

Add a Python-side helper, e.g.
`build_custom_scenario_emitters(config, repo_root, requests,
boost_false_alarm)`, so `PythonBridge` remains a thin wrapper and the
scenario/model logic stays in Python.
