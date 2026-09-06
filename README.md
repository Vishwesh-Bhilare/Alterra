# Alterra — CORTEX Smart Scan Scheduler

> **Cognitive Electronic Warfare Scanning Receiver & AI Frequency Tracking**  
> *Smart India Hackathon (SIH) 2026 — DRDO Problem Statement 26055*

---

## 🛰️ Overview

**Alterra** is an advanced cognitive electronic warfare (EW) receiver scheduling and signal intelligence platform. Traditional scanning receivers sweep radar frequency bands blindly using fixed or linear patterns, often missing agile, frequency-hopping, and low-probability-of-intercept (LPI) emitters.

Alterra solves this by pairing a high-fidelity RF spectrum simulator with a **PPO + RNN Cognitive Reinforcement Learning scheduler** and a real-time **C++20 / Qt6 Waterfall Spectrogram Desktop Interface**:
- **RNN Hit/Miss Sequence Modeling**: Stores the temporal sequence of historical dwell outcomes `[hit, miss, band, dwell, power]`. A PyTorch Recurrent Neural Network (RNN) analyzes the pulse arrival timing and past intercepts to identify emitter burst rhythms.
- **PPO Smart Scheduling**: Depending on the RNN temporal sequence embedding, PPO decides where to search next (`STAY`, `STEP UP`, `STEP DOWN`), predicting where agile and periodic radar pulses will appear next.
- **Dynamic Frequency Tracking**: Rather than camping on a single band or blindly jumping across channels, the receiver utilizes a relative directional action space with boundary reflection.
- **Closed-Loop Signal Lock & Track**: When an emitter pulse is intercepted, the receiver automatically locks onto the frequency band and tracks the signal throughout its pulse burst.
- **Intelligent Sweep on Signal Loss**: When an emitter hops or goes silent, the scheduler smoothly transitions to directional search guided by historical sequence patterns.
- **Real-Time ESM Metrics**: Measures Probability of Detection ($P_d = 100\%$), Probability of False Alarm ($P_{fa} \le 2\%$), and Average Intercept Rate ($>90\%$).
- **Radar Pulse Deinterleaving**: Integrates with the Turing Synthetic Radar Dataset (TSRD) for pulse descriptor word (PDW) extraction and emitter deinterleaving.

---

## 🏛️ System Architecture

```
                                  +-----------------------------+
                                  |   configs/default_config    |
                                  +--------------+--------------+
                                                 |
                    +----------------------------v-----------------------------+
                    |                 Layer A: SpectrumWorld                   |
                    |   - Fixed Emitters (Continuous / Pulsed)                 |
                    |   - Frequency-Hopping Agile Emitters                     |
                    |   - Periodic Radar Scanning Sweepers                     |
                    +----------------------------+-----------------------------+
                                                 | Ground-Truth Occupancy
                    +----------------------------v-----------------------------+
                    |                 Layer B: SensorModel                     |
                    |   - Thermal Noise Floor (-95 to -85 dBm)                 |
                    |   - SNR-dependent Detection Probability Pd(SNR)          |
                    |   - False Alarm Rate (Pfa) & Reading Noise               |
                    +----------------------------+-----------------------------+
                                                 | Dwell Result / Power (dBm)
                    +----------------------------v-----------------------------+
                    |               Layer C: Alterra Gymnasium Env             |
                    |   - Hit/Miss History Buffer: [T x 5] Rolling Sequence    |
                    |   - Spectrum Tracks: [128 x 7]                           |
                    |   - Receiver State: [8 dims]                             |
                    +----------------------------+-----------------------------+
                                                 | Observations
                    +----------------------------v-----------------------------+
                    |                Layer D: PPO + RNN Agent                  |
                    |                                                          |
                    |  hit_miss_seq [T x 5] ---> nn.RNN() ---> h_n [64-dim]    |
                    |  band_tracks [128x7]  ---> Flatten/Linear                |
                    |  receiver    [8-dim]  ---> Linear                        |
                    |                                 |                        |
                    |                        [Fused 256-dim State]             |
                    |                                 |                        |
                    |                 +---------------+---------------+        |
                    |                 |                               |        |
                    |           Policy (Actor)                  Value (Critic) |
                    |       [Direction: -1, 0, +1]               V(s) estimate |
                    |       [Dwell: 3, 5, 8, 12]                               |
                    +--------------+-----------------------------+-------------+
                                   |                             |
                                   | Actions                     | Python Bridge
                                   v                             v
                    +----------------------------+ +-----------------------------+
                    |     RF Environment Step    | |   C++20 / Qt6 Desktop GUI   |
                    |     - Tune RF Front-End    | |   - Live Waterfall Display  |
                    |     - Collect Dwell Hits   | |   - ESM Detection Metrics   |
                    |     - Push to RNN Buffer   | |   - Interactive Controls    |
                    +----------------------------+ +-----------------------------+
```

---

## 📁 Repository Layout

- **`simulation/`** — Core RF simulation engine:
  - `emitters/`: Emitter population generator, fixed/agile/periodic-scan emitters, and TSRD dataset adapter.
  - `environment/`: Spectrum world, sensor front-end model, scanning receiver, Gymnasium environment (`gym_env.py`), and PDW exporter.
  - `metrics/`: Rollout tracking metrics ($P_d$, $P_{fa}$, intercept rates).
  - `utils/`: Structured config loaders and seeded RNG managers.
- **`interfaces/`**:
  - `cli/`: Click-based `alterra` CLI for headless simulations, metrics evaluation, and waterfall plots.
  - `qt_gui/`: Modern C++20 / Qt6 desktop GUI with embedded Python runtime bridge for real-time visualization.
- **`model/`**:
  - `agents/`: PPO scheduler training (`train_ppo.py`), evaluation (`evaluate.py`), policy inspection, and saved checkpoints.
  - `deinterleaving/`: Transformer encoder, datasets, and loss functions for radar pulse deinterleaving.
- **`configs/`** — Tunable YAML configurations (timing, spectrum bandwidth, sensor noise, emitter populations).
- **`run.sh`** — Comprehensive pipeline execution shell script.

---

## 🚀 Complete Setup Guide (macOS & Linux)

### 🍏 macOS Setup

#### 1. Install System Prerequisites
Open Terminal and install build essentials, CMake, Qt 6, and Python 3.13 via Homebrew:
```bash
# Install Xcode Command Line Tools
xcode-select --install

# Install Homebrew (if not already installed)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install CMake, Qt 6, and Python 3.13
brew install cmake qt@6 python@3.13
```

#### 2. Clone & Setup Python Environment
```bash
git clone https://github.com/Vishwesh-Bhilare/Alterra.git
cd Alterra
git checkout PPO+RNN

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 3. Build & Launch GUI
```bash
# Automated 1-command build & launch:
./run.sh --gui

# Or manual CMake build:
cmake -B interfaces/qt_gui/build -S interfaces/qt_gui \
  -DCMAKE_PREFIX_PATH="$(brew --prefix qt@6 2>/dev/null || brew --prefix qt);$(.venv/bin/python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build interfaces/qt_gui/build
./interfaces/qt_gui/build/alterra_gui configs/default_config.yaml model/agents/checkpoints/best/best_model.zip
```

---

### 🐧 Linux Setup (Ubuntu / Debian / Fedora / Arch)

#### 1. Install System Prerequisites

**Ubuntu / Debian / Linux Mint:**
```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  cmake \
  qt6-base-dev \
  qt6-base-dev-tools \
  libgl1-mesa-dev \
  python3-dev \
  python3-venv \
  git
```

**Fedora / RHEL / Rocky Linux:**
```bash
sudo dnf install -y \
  gcc-c++ \
  cmake \
  qt6-qtbase-devel \
  mesa-libGL-devel \
  python3-devel \
  git
```

**Arch Linux / Manjaro:**
```bash
sudo pacman -S --needed \
  base-devel \
  cmake \
  qt6-base \
  python \
  git
```

#### 2. Clone & Setup Python Environment
```bash
git clone https://github.com/Vishwesh-Bhilare/Alterra.git
cd Alterra
git checkout PPO+RNN

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

#### 3. Build & Launch GUI
```bash
# Automated 1-command build & launch:
./run.sh --gui

# Or manual CMake build:
cmake -B interfaces/qt_gui/build -S interfaces/qt_gui \
  -DCMAKE_PREFIX_PATH="$(.venv/bin/python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"
cmake --build interfaces/qt_gui/build
./interfaces/qt_gui/build/alterra_gui configs/default_config.yaml model/agents/checkpoints/best/best_model.zip
```

> **Note for Headless Linux / Servers**:  
> If running on a remote cloud instance or headless container without a display, you can run all simulations, training, and metrics headlessly using the CLI:
> ```bash
> alterra env check --config configs/default_config.yaml
> ./run.sh --sim
> ./run.sh --train-rl 200000
> ```

### 2. Automated Pipeline Runner (`run.sh`)

Alterra includes an automated runner script [`run.sh`](file:///Users/anuragpatil/Alterra/run.sh) to execute end-to-end tasks:

```bash
# Launch the C++ Qt6 GUI desktop application
./run.sh --gui

# Run complete pipeline demo (simulation suite + PDW export + dataset gen)
./run.sh

# Fast simulation test with waterfall plot generation
./run.sh --sim

# Launch TensorBoard training dashboard (http://localhost:6006)
./run.sh --tb

# Quick 1,000-step RL smoke test
./run.sh --smoke-rl

# Train PPO for 200,000 steps
./run.sh --train-rl 200000

# View all options
./run.sh --help
```

---

## 🖥️ C++20 / Qt6 GUI Desktop Application

The desktop application provides a real-time waterfall spectrogram showing emitter activity and the live scanning trajectory of the PPO scheduler.

### Features
- **Waterfall Spectrogram Display**: High-resolution visualization of all 128 frequency bands over time with active radar signals shown as white pulses.
- **Live AI Scan Trajectory**: Red receiver line tracking active emitter pulses and stepping directionally (`+1` / `-1` / `0`).
- **Interactive Controls**: Play, Pause, Step-Once, Speed Slider, and arbitrary Seed selector.
- **Telemetry & Live Metrics**: Dynamic readout of $P_d$, $P_{fa}$, intercept rate, instantaneous reward, and measured signal strength in dBm.

### Manual Build & Run

```bash
# Configure and build with CMake
cmake -B interfaces/qt_gui/build -S interfaces/qt_gui \
  -DCMAKE_PREFIX_PATH="/opt/homebrew/opt/qtbase;$(.venv/bin/python -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(pwd)/.venv/bin/python"

cmake --build interfaces/qt_gui/build

# Launch
./interfaces/qt_gui/build/alterra_gui configs/default_config.yaml model/agents/checkpoints/best/best_model.zip
```

> **Note for VS Code / Antigravity IDE Users**:  
> CMake automatically generates `compile_commands.json` symlinked to the project root. If you see IDE language server squigglies, run **`Cmd+Shift+P` -> `clangd: Restart language server`**.

---

## 🧠 PPO Smart Scan Scheduler

### Directional Action Space
Instead of selecting an absolute band index $\in [0, 127]$, the policy outputs relative frequency shifts:
- `0: STAY (delta = 0)` — Lock receiver LO on the current frequency to follow active pulses.
- `1: STEP UP (delta = +1)` — Step to higher adjacent frequency band.
- `2: STEP DOWN (delta = -1)` — Step to lower adjacent frequency band.
- Paired with configurable dwell duration options: `[3, 5, 8, 12]` slots.

### Reward Shaping
- **Hit Base Reward**: $+10.0 \times \text{Threat Level}$
- **Tracking Bonus**: $+6.0 \times \text{Threat Level}$ when maintaining dwell on an active emitter (`delta == 0`).
- **Empty Stay Penalty**: $-0.4$ for idling on an empty band, motivating directional sweeping.
- **Signal Lost Penalty**: $-0.8$ for staying on a frequency band after the emitter has ceased transmitting.
- **Boundary Bounce**: Automatic reflection at band $0$ and band $127$ back into the spectrum.

### Hit/Miss Sequence Modeling with RNN

The scheduler leverages a recurrent neural network (`nn.RNN`) over the historical sequence of dwell events to achieve smart scan scheduling:
- **Rolling Hit/Miss Sequence Buffer**: Tracks the last $T=16$ dwells across 5 features:
  - `hit`: Binary indicator of emitter pulse detection ($1.0$ if detected, $0.0$ otherwise).
  - `miss`: Binary indicator of empty dwell ($1.0$ if missed, $0.0$ otherwise).
  - `band`: Normalized frequency band index in $[0, 1]$.
  - `dwell`: Normalized dwell duration in $[0, 1]$.
  - `power`: Received RF power in $[0, 1]$.
- **Recurrent Feature Extractor (`PPORNNExtractor`)**: The PyTorch RNN encodes the temporal sequence into a 64-dimensional temporal state $h_n$, capturing pulse repetition rhythms and dwell success history.
- **Smart Scheduling Decision**: The Actor-Critic networks combine $h_n$ with receiver telemetry and spectrum tracks into a 256-dimensional representation, deciding whether to `STAY` (track active emitter burst) or step `UP` / `DOWN` (directional search toward predicted pulse arrivals).

### Training the PPO + RNN Scheduler

```bash
# Multi-core training with PPO + RNN policy
python -m model.agents.train_ppo \
  --config configs/default_config.yaml \
  --timesteps 100000 \
  --n-envs 4 \
  --ent-coef 0.03 \
  --rnn-hidden-dim 64 \
  --out model/agents/checkpoints/best/best_model.zip \
  --tensorboard-log model/agents/tb_logs
```

### Evaluating Checkpoints & Benchmark Results

```bash
# Benchmark metrics over 10 episodes (200 steps per episode)
python -m model.agents.evaluate \
  --config configs/default_config.yaml \
  --model model/agents/checkpoints/best/best_model.zip \
  --episodes 10 \
  --steps-per-episode 200
```

**Empirical Performance (Deterministic Rollouts):**
| Metric | Baseline Linear Scan | PPO + RNN Smart Scheduler |
| :--- | :--- | :--- |
| **Probability of Detection ($P_d$)** | $52.4\%$ | **$100.0\%$** |
| **False Alarm Rate ($P_{fa}$)** | $6.8\%$ | **$2.0\%$** |
| **Average Intercept Rate** | $28.1\%$ | **$90.3\%$** |
| **Percent Correct Classification** | $84.2\%$ | **$98.3\%$** |
| **Mean Episode Total Reward** | $\sim 140$ | **$1084.91$** (Max $1988.89$) |

---

## 🛠️ CLI Command Reference (`alterra`)

The headless CLI tool provides complete control over simulation, environment verification, and data export.

### 1. Environment Verification & Metrics
```bash
# Run Gymnasium check_env compliance suite
alterra env check --config configs/default_config.yaml

# Live rollout preview of steps in terminal
alterra env preview --config configs/default_config.yaml --steps 50

# Benchmark detection metrics across multiple seeds
alterra env metrics --config configs/default_config.yaml --episodes 10 --steps-per-episode 200

# Generate an offline waterfall plot PNG
alterra env plot --config configs/default_config.yaml --steps 200 --out episode_waterfall.png
```

### 2. Emitters & Scenario Previews
```bash
# Inspect randomly sampled emitter population
alterra emitters preview --config configs/default_config.yaml --episode-length 500

# Inspect a manual scenario definition
alterra scenario preview --config configs/default_config.yaml --scenario configs/scenarios/manual_example.yaml
```

### 3. Pulse Descriptor Words (PDWs)
```bash
# Stream PDWs to stdout (ToA, Frequency, Pulse Width, Power, DOA)
alterra pdw preview --config configs/default_config.yaml --episode-length 500 --limit 20

# Export full episode PDWs to JSONL
alterra pdw export --config configs/default_config.yaml --episode-length 2000 --out pdws.jsonl
```

---

## 📡 Turing Synthetic Radar Dataset (TSRD) Integration

Alterra includes a dedicated adapter in [`simulation/emitters/tsrd_adapter.py`](file:///Users/anuragpatil/Alterra/simulation/emitters/tsrd_adapter.py) to ingest pulse trains from the Turing Synthetic Radar Dataset:

```python
from simulation.utils.config_loader import load_config
from simulation.emitters.tsrd_adapter import build_tsrd_population

cfg = load_config("configs/default_config.yaml")
emitters = build_tsrd_population(
    file_path="/path/to/turing_synthetic_radar_data/scan/train_scan/config_0.h5",
    config=cfg,
    max_emitters=10,
)
```

---

## 📄 License & Attribution

Developed for **Smart India Hackathon (SIH) 2026** under **DRDO Problem Statement 26055: Smart Scan Strategy for Electronic Warfare Receiver**.
Released under the [MIT License](LICENSE).
