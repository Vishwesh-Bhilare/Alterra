# Turing Synthetic Radar Dataset — For Alterra Evaluation

## What is This Dataset?

The **Turing Synthetic Radar Dataset** is a collection of **radar pulse descriptor word (PDW) measurements** from simulated radar emitters. It contains synthetic RF signal observations collected by a scanning receiver under various receiver configurations and emitter populations.

Think of it as: **"Ground-truth labeled radar pulses in a simulated RF environment"**

---

## Dataset Structure

### 1. **Train / Test / Validation Split**
```
train_0.h5       →  6,503 samples  (7 transmitter types)
validation_0.h5  →  42,207 samples (63 transmitter types)
test_0.h5        →  29,748 samples (88 transmitter types)
```

**Progressive complexity:** Training uses fewer emitter types; test/validation expose the model to diverse, unseen emitters.

### 2. **Five Input Features Per Pulse**

Each radar pulse is described by **5 numerical measurements**:

| Feature | Units | Meaning |
|---------|-------|---------|
| **UTCTime** | seconds | Time of observation (absolute timestamp) |
| **RF** | MHz | Observed radio frequency (center frequency) |
| **PulseWidth** | microseconds | Duration of the detected pulse |
| **AOA** | degrees | Angle of arrival (direction of incoming signal) |
| **PA** | dBm | Pulse amplitude (received signal strength) |

These are **standard PDW features** used in Electronic Warfare receiver systems.

### 3. **Classification Labels**

Each pulse is labeled with its **emitter class** (0–6).

Example from training data:
```
Class 0: 363 samples  (5.6%)   ← Less common emitter type
Class 1: 188 samples  (2.9%)
Class 2: 4,234 samples (65.1%)  ← Most common
Class 3: 1,343 samples (20.7%)
Class 4: 216 samples  (3.3%)
Class 5: 158 samples  (2.4%)
Class 6: 1 sample     (0.0%)   ← Rare edge case
```

**Imbalanced distribution models real-world conditions** where some emitters are encountered more frequently.

### 4. **Emitter Metadata**

Each transmitter in the dataset is fully characterized as a JSON object. Example structure:

```json
{
  "function": "transmitter_57",
  "frequency_config": {
    "freq_mode": "FixedMultiSimultaneous",
    "freq_grp_size": 5,
    "frequencies": [5200.5, 5210.3, ...]
  },
  "scan_mode": "FixedScan",
  "dwell_time_ms": 12.5,
  "power_dbm": -30.0,
  "pdw_generation_rate": 500,
  ...
}
```

**This tells you:** transmitter identity, frequency hopping behavior, dwell patterns, transmit power, and signal generation parameters.

### 5. **Receiver Configuration**

Metadata for how signals were collected:

| Parameter | Example | Meaning |
|-----------|---------|---------|
| `freq_range_mhz` | [1000, 18000] | Receiver tuning range |
| `bandwith_mhz` | 50 | Instantaneous measurement bandwidth |
| `dwell_centres_mhz` | [1500, 2000, ...] | Frequency bands receiver scanned |
| `dwell_times_s` | [0.1, 0.15, ...] | How long at each frequency |
| `sensitivity_dbm` | -90 | Receiver noise floor |
| `freq_noise_scale_mhz` | 0.5 | Measurement uncertainty |
| `scan_mode` | "Systematic" / "RandomScan" | Scanning strategy |

---

## Configuration Files (Different Receiver Scenarios)

The dataset includes **multiple receiver configurations**, showing how performance varies with **different scanning strategies**:

```
config_0.h5   →  13,929 samples   (2 transmitters)   [Simple scenario]
config_19.h5  →  34,044 samples   (28 transmitters)
config_40.h5  →  85,151 samples   (54 transmitters)
config_55.h5  → 1,867,780 samples (49 transmitters)  [Most complex]
config_82.h5  →  844,952 samples  (52 transmitters)
```

**Purpose:** Evaluate receiver scheduling across diverse RF environments and scanning configurations.

---

## How Does This Help Alterra?

### 1. **Emitter Classification (Perception Layer)**

Your **MS-UNet1D + PDW extraction pipeline** can use this data to learn:
- Which combination of (UTC, RF, PulseWidth, AOA, PA) identifies emitter type
- Robustness to measurement noise
- Handling of multi-emitter scenarios

**Use case:** Train a classifier to identify `Class 2` (aggressive emitter) vs. `Class 0` (benign), enabling threat prioritization.

### 2. **Receiver Scheduling (RL Agent Training)**

The **receiver configuration metadata** tells the RL agent:
- What frequencies were scanned
- How long the receiver dwelled at each band
- Whether hits/misses occurred

This simulates the feedback loop:
```
Receiver scans Band 5200 MHz
         ↓
Detects pulses → PDW extraction
         ↓
Classify as "Transmitter_57" (Class 3)
         ↓
RL agent learns: "Band 5200 MHz is active; prioritize it in next scan"
```

### 3. **Training Different Algorithms**

- **CAROTA (baseline):** Use receiver config to compute priority scores heuristically
- **Double DQN:** State = emitter classifications; Action = next band to scan; Reward = detection rate
- **PPO (proposed):** Learn adaptive scan policies that minimize intercept time

### 4. **Evaluation Metrics**

The dataset provides ground truth for:
- **Probability of Detection (Pd):** What % of emitter transmissions were detected?
- **Probability of False Alarm (Pfa):** Spurious detections?
- **Intercept Time:** Time from episode start to first detection of each emitter
- **Intercept Rate:** Total emitters successfully detected / total emitters present

---

## How to Explain This During Evaluation

### **30-Second Elevator Pitch**

> "The Turing Synthetic Radar Dataset provides 78,000+ labeled radar pulse measurements across 88 unique emitter types. Each pulse is described by 5 features (time, frequency, pulse width, angle, power) that a receiver collects when scanning the RF spectrum. We use this to train emitter classifiers and benchmark receiver scheduling algorithms—comparing how quickly different scan strategies detect hostile signals."

### **1-Minute Breakdown**

1. **What:** Simulated radar pulses from a synthetic RF environment
2. **Why:** DRDO problem requires training without prior intelligence; simulation provides labeled ground truth
3. **Structure:** 6K–42K pulses per split, 88 emitter types, 5 PDW features, multiple receiver configs
4. **Application:** Feeds emitter classifier → provides feedback to RL scheduler → measures intercept performance
5. **Realism:** Includes measurement noise, frequency agility, variable transmit power—mimics operational conditions

### **Visual Explanation for Slides**

```
┌─────────────────────────────────────────────────────┐
│      Turing Synthetic Radar Dataset                 │
├─────────────────────────────────────────────────────┤
│                                                     │
│  RF Environment (Simulated)                        │
│  ├─ 88 Emitter Types                              │
│  ├─ Frequency: 1–18 GHz                           │
│  ├─ Varying Power & Modulation                    │
│  └─ Time: 0–5 minutes                             │
│                                                    │
│           ↓                                        │
│                                                    │
│  Receiver Collects Pulses                         │
│  ├─ UTC Time              ↓                       │
│  ├─ RF (observed freq)    Receiver Scans          │
│  ├─ Pulse Width           Different Bands         │
│  ├─ Angle of Arrival      in Sequence             │
│  └─ Power (amplitude)                            │
│                                                    │
│           ↓                                        │
│                                                    │
│  Dataset: 78,000+ Labeled Pulses                 │
│  ├─ Train: 6.5K (7 emitter types)               │
│  ├─ Val:  42.2K (63 emitter types)              │
│  └─ Test: 29.7K (88 emitter types)              │
│                                                    │
│           ↓                                        │
│                                                    │
│  Alterra Processing                              │
│  ├─ Perception: Classify emitter from PDW        │
│  ├─ Scheduling: Decide next frequency to scan    │
│  └─ Metrics: Measure intercept speed & rate      │
│                                                    │
└─────────────────────────────────────────────────────┘
```

### **Key Talking Points**

1. **Imbalanced class distribution** (65% Class 2, 3% others)
   - "Realistic: some emitters are more common than others"

2. **Progressive complexity** (train → val → test)
   - "Training on 7 emitters, evaluated on 88 unknown types"

3. **Multiple receiver configurations**
   - "Tests how the scheduler adapts to different scanning strategies"

4. **Synthetic but realistic**
   - "Includes measurement noise, frequency agility, angle-of-arrival errors—mirrors actual EW scenarios"

5. **Closed feedback loop**
   - "Each detection → classification → triggers RL agent to re-prioritize next scan frequency"

---

## Dataset Statistics Summary

| Metric | Value |
|--------|-------|
| **Total pulses** | 78,458 |
| **Train samples** | 6,503 |
| **Validation samples** | 42,207 |
| **Test samples** | 29,748 |
| **Features per pulse** | 5 (PDW descriptors) |
| **Emitter types in test** | 88 |
| **Receiver configs** | 5 (variants) |
| **Total data size** | ~3 MB |
| **Frequency range** | 1–18 GHz |

---

## Recommended Usage During Presentation

### **Slide 1: Dataset Overview**
- Show file structure (train/val/test split)
- Highlight: "78K realistic radar pulses, 88 emitter types"

### **Slide 2: PDW Features**
- Table of 5 features (UTC, RF, PW, AOA, PA)
- Explain each in 1 sentence
- Note: "Standard EW receiver output"

### **Slide 3: Emitter Diversity**
- Bar chart of class distribution (imbalanced)
- Note: "Real-world distribution: some threats more common"

### **Slide 4: Data Flow**
- Show: Receiver → PDW → Classification → Scheduling Agent → Next Action
- Emphasize closed loop: "Agent learns from hit/miss patterns"

### **Slide 5: Receiver Configurations**
- Table of 5 configs with transmitter counts and sample sizes
- Explain: "Tests adaptation to different EW scenarios"

---

## Quick Reference for Q&A

**Q: Why multiple config files instead of one dataset?**
A: Each config represents a different receiver scanning strategy (dwell times, frequency ordering, bandwidth). We evaluate whether the RL scheduler performs consistently across scenarios.

**Q: Why is the test set so different (88 types vs. 7 in train)?**
A: Deliberate generalization test. The model trains on known emitters but must identify unseen types—realistic for operational EW.

**Q: How does this relate to the DRDO problem?**
A: The problem asks for a "smart scan strategy." This dataset provides the observational data (PDWs) and emitter ground truth. Our scheduler learns to minimize intercept time by predicting which frequency bands are most likely to have active emitters.

**Q: Is this data realistic?**
A: Synthetic but realistic. It includes measurement noise (frequency/timing errors), power variation, frequency agility, and angle-of-arrival uncertainty—matching actual EW receiver constraints.

---

## Next Steps for Evaluation

When presenting, connect each dataset component to your architecture:

1. **Perception Layer** → Train MS-UNet1D + PDW extractor on this data
2. **Deinterleaving** → Use emitter metadata to validate SEDCAM clustering
3. **RL Scheduler** → Use receiver config feedback to train CAROTA/DQN/PPO
4. **Metrics** → Report Pd, Pfa, intercept time, intercept rate from validation data

