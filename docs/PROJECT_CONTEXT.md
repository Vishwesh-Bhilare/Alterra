# Alterra

### Smart Scan Strategy for Electronic Warfare

**Alterra** is an AI-driven Electronic Support (ES) receiver scheduling system developed for **Smart India Hackathon 2026**, targeting **DRDO Problem Statement 26055: "Smart Scan strategy for Electronic Warfare."**

The project focuses on intelligently scheduling a receiver across a wide frequency spectrum to reduce intercept time and increase the probability of detecting relevant emitters.

---

## Overview

Modern Electronic Warfare receivers may need to monitor a spectrum that is significantly wider than their instantaneous bandwidth. As a result, the receiver must continuously scan different frequency bands over time.

Traditional open-loop scanning strategies generally rely on prior intelligence and attempt to cover the spectrum as quickly as possible. This can waste valuable dwell time on inactive or low-priority emitters.

Alterra treats interception as a **two-dimensional search problem**:

* **Frequency:** Which band should the receiver observe?
* **Time:** When should it observe that band?

The system learns from previous **hits and misses** to dynamically prioritize receiver actions.

### Core objectives

* Minimize emitter intercept time.
* Maximize interception rate.
* Handle frequency-agile emitters.
* Handle periodic-scanning emitters.
* Account for imperfect sensor detections.
* Learn receiver scheduling policies from simulated observations.
* Compare heuristic and reinforcement-learning approaches.

---

## System Architecture

```text
                    ┌─────────────────────────────┐
                    │      RF Environment         │
                    │                             │
                    │ Fixed / Agile / Periodic    │
                    │ Scan Emitters               │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │     Spectrum World          │
                    │                             │
                    │ Ground-truth band × time    │
                    │ occupancy                   │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │       Sensor Model           │
                    │                             │
                    │ SNR → Pd / Pfa → Detection  │
                    └──────────────┬──────────────┘
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │    Receiver / Gymnasium      │
                    │          Environment         │
                    │                             │
                    │ Observation → Action →      │
                    │ Reward                     │
                    └──────────────┬──────────────┘
                                   │
                  ┌────────────────┼────────────────┐
                  │                │                │
                  ▼                ▼                ▼
             ┌────────┐      ┌──────────┐      ┌───────┐
             │ CAROTA │      │ Double   │      │  PPO  │
             │Baseline│      │   DQN    │      │Proposed│
             └────────┘      └──────────┘      └───────┘
                  │                │                │
                  └────────────────┼────────────────┘
                                   ▼
                    ┌─────────────────────────────┐
                    │      Performance Metrics     │
                    │                             │
                    │ Pd / Pfa / Intercept Rate   │
                    │ Reward / Intercept Time     │
                    └─────────────────────────────┘
```

---

## Signal Processing Pipeline

The eventual intelligence pipeline is designed around:

```text
RF / IQ Signals
       │
       ▼
  MS-UNet1D
       │
       ▼
Pulse Segmentation
       │
       ▼
   PDW Extraction
       │
       ├── TOA
       ├── PRI
       ├── PW
       ├── CF
       └── DOA
       │
       ▼
   SEDCAM
       │
       ├── SEM
       ├── CVR-DCM
       └── DBSCAN
       │
       ▼
 Deinterleaved Emitters
       │
       ▼
 Emitter State Construction
       │
       ▼
 Intelligent Scheduling
```

---

## Scheduling Algorithms

Alterra compares three scheduling approaches.

| Algorithm      | Type                   | Purpose                           |
| -------------- | ---------------------- | --------------------------------- |
| **CAROTA**     | Heuristic              | Classical priority-based baseline |
| **Double DQN** | Reinforcement Learning | Value-based RL baseline           |
| **PPO**        | Reinforcement Learning | Proposed adaptive scheduler       |

### CAROTA

CAROTA provides a non-RL baseline based on priority and receiver constraints.

### Double DQN

Double DQN formulates receiver band selection as a discrete-action reinforcement-learning problem and provides a stronger learned baseline.

### PPO

PPO is the proposed scheduler. It learns a policy for dynamically selecting receiver actions based on emitter state and previous observations.

All approaches are evaluated using the same simulated environment and performance metrics.

---

## Simulation

The simulation is a core part of Alterra rather than simply a testing utility.

Because the problem assumes that no prior intelligence about emitters is available, the simulation acts as the **synthetic data-generation layer** from which the learning system obtains its experience.

### Two-layer simulation architecture

#### Layer A: Environment Truth

The environment maintains the ground truth for:

* Emitter identity
* Frequency band
* Transmission state
* Transmit power
* Time slot

This produces a band × time representation of the RF environment.

#### Layer B: Sensor Model

The sensor model converts ground truth into imperfect receiver observations.

It models effects such as:

* Signal-to-noise ratio
* Probability of detection (`Pd`)
* Probability of false alarm (`Pfa`)
* Power variation
* Missed detections

The scheduler therefore does **not** receive perfect ground truth.

---

## Emitter Models

Three emitter behaviors are currently implemented.

### Fixed Emitter

A fixed emitter remains on a particular frequency band while its transmission activity follows a stochastic ON/OFF process.

```text
Band
 │
 ├──────────────────────────────
 │       ON       OFF     ON
 │     ██████     ██     █████
 │
 └──────────────────────────────► Time
```

### Agile Emitter

An agile emitter selects a set of frequency bands and hops between them during an episode.

```text
Frequency
   ▲
   │        █
   │              █
   │  █
   │                    █
   │          █
   └──────────────────────────► Time
```

The hop dwell duration and number of hop bands are sampled from configuration-defined distributions.

### Periodic Scan Emitter

A periodic scan emitter sweeps through a contiguous frequency window and dwells on each band before moving to the next.

```text
Frequency
   ▲
   │       ┌─────────┐
   │      /           │
   │     /             │
   │    /               │
   │   /                 │
   └──────────────────────────► Time
```

This emitter is particularly important because the DRDO problem explicitly requires investigation of optimal interception of periodic-scan receivers/emitters.

---

## Burst Model

Emitter activity is generated using a two-state Markov process.

The process is parameterized by:

* Target duty cycle `D`
* Mean ON burst length `L`

The transition probabilities are:

```text
p(ON → OFF) = 1 / L

p(OFF → ON) = D × p(ON → OFF) / (1 - D)
```

The initial state is sampled from the stationary distribution, avoiding a systematic OFF-state bias at the beginning of episodes.

---

## Reproducibility

Alterra uses independent deterministic random-number streams through `numpy.random.SeedSequence`.

Emitter streams are identified using stable names such as:

```text
emitter_population_selection
fixed_003
agile_002
periodic_scan_001
```

This means that adding or reordering components does not unnecessarily change the random sequences used by existing components.

Emitter schedules are also **precomputed during `reset()`** rather than generated inside `state_at()`.

Therefore:

```text
reset()
   │
   ▼
Generate complete episode schedule
   │
   ▼
state_at(t)
   │
   ▼
O(1) deterministic lookup
```

---

## Configuration

Alterra follows a strict **configuration-driven design**.

Tunable simulation parameters belong in:

```text
configs/default_config.yaml
```

Examples include:

* Number of frequency bands
* Bandwidth
* Starting frequency
* Episode length
* Slot duration
* Emitter population
* Threat-level distribution
* Frequency-hopping ranges
* Burst parameters
* Power ranges
* Sensor parameters

The simulation code should not contain hardcoded tunable numeric defaults.

---

## Repository Structure

```text
alterra/
│
├── configs/
│   └── default_config.yaml
│
├── simulation/
│   ├── emitters/
│   │   ├── base_emitter.py
│   │   ├── fixed_emitter.py
│   │   ├── agile_emitter.py
│   │   ├── periodic_scan_emitter.py
│   │   └── schedule_utils.py
│   │
│   ├── environment/
│   │   ├── spectrum_world.py
│   │   ├── sensor_model.py
│   │   ├── receiver.py
│   │   └── gym_env.py
│   │
│   ├── metrics/
│   │
│   ├── utils/
│   │   ├── rng.py
│   │   └── config_loader.py
│   │
│   └── viz/
│
├── model/
│   ├── perception/
│   │   ├── MS-UNet1D
│   │   └── PDW extraction
│   │
│   ├── deinterleaving/
│   │   └── SEDCAM
│   │
│   └── agents/
│       ├── CAROTA
│       ├── Double DQN
│       └── PPO
│
├── interfaces/
│   ├── cli/
│   └── web/
│
├── docs/
│   └── PROJECT_CONTEXT.md
│
├── scripts/
├── tests/
├── pyproject.toml
└── requirements.txt
```

---

## Current Development Status

### Phase 1: Simulation and Data Generation

| Component                     | Status      |
| ----------------------------- | ----------- |
| Configuration system          | Done        |
| RNG management                | Done        |
| Base emitter                  | Done        |
| Fixed emitter                 | Done        |
| Agile emitter                 | Done        |
| Periodic scan emitter         | Done        |
| Emitter population generation | Done        |
| CLI emitter preview           | Done        |
| Spectrum world                | In progress |
| Sensor model                  | In progress |
| Receiver                      | In progress |
| Gymnasium environment         | In progress |
| Metrics                       | Planned     |
| Visualization                 | Planned     |

### Phase 2: AI / Signal Processing

Planned components:

* MS-UNet1D pulse segmentation
* PDW extraction
* SEDCAM deinterleaving
* Emitter state construction
* CAROTA
* Double DQN
* PPO

---

## CLI

The current CLI can preview generated emitter populations:

```bash
alterra emitters preview
```

A custom configuration can be supplied with:

```bash
alterra emitters preview --config configs/default_config.yaml
```

An episode length can also be overridden:

```bash
alterra emitters preview --episode-length 2000
```

---

## Performance Metrics

The system will evaluate scheduling strategies using the figures of merit specified by the problem statement:

* **Probability of Detection (`Pd`)**
* **Probability of False Alarm (`Pfa`)**
* **Sensitivity**
* **Average Intercept Rate**
* **Average Reward / Cost**
* **Percentage of Correct Predictions**
* **Average Intercept Time Error**

Metrics will support both:

1. **Online evaluation** during RL training
2. **Offline evaluation** for algorithm comparison and experimental results

---

## Development Principles

### 1. No hardcoded simulation defaults

All tunable simulation parameters must come from configuration.

### 2. Deterministic randomness

Use `RNGManager.spawn_named()` for independently reproducible stochastic components.

### 3. Precompute emitter schedules

Emitter schedules are generated during `reset()` and subsequently accessed through deterministic lookups.

### 4. Separate truth from observation

The RL agent must never receive the environment's ground truth directly.

```text
Ground Truth
     │
     ▼
Sensor Model
     │
     ▼
Imperfect Observation
     │
     ▼
RL Agent
```

### 5. Keep simulation and model independent

`simulation/` must never import internals from `model/`.

The model side consumes the simulation through the Gymnasium environment interface.

### 6. Reproducible experiments

Random seeds, configuration files, model versions, and evaluation metrics should be recorded for experiments.

---

## Technology Stack

| Area                   | Technology                  |
| ---------------------- | --------------------------- |
| Language               | Python, C++                 |
| Deep Learning          | PyTorch, TensorFlow         |
| Reinforcement Learning | Stable-Baselines3           |
| Environment            | Gymnasium                   |
| Signal Processing      | MS-UNet1D, SEDCAM           |
| Database               | PostgreSQL, Redis, InfluxDB |
| Backend                | REST API                    |
| Visualization          | Plotly, Matplotlib          |
| GUI                    | Qt                          |
| Deployment             | Docker, Nginx               |
| Version Control        | Git / GitHub                |

---

## Development Roadmap

```text
                    ┌────────────────────┐
                    │ 1. Requirements    │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 2. Simulation      │
                    │    & Data Gen      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 3. Signal          │
                    │    Processing      │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 4. AI Models      │
                    │ CAROTA / DQN / PPO│
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 5. Integration     │
                    │    & Testing       │
                    └─────────┬──────────┘
                              │
                              ▼
                    ┌────────────────────┐
                    │ 6. Deployment      │
                    │    & Monitoring    │
                    └────────────────────┘
```

---

## Research Goal

The central research question is:

> **Can an adaptive machine-learning receiver scheduler intercept dynamic and frequency-agile emitters more quickly and reliably than conventional scanning strategies?**

The experimental evaluation will compare CAROTA, Double DQN, and PPO under identical simulated RF scenarios.

Particular attention will be given to:

* Periodic scanning behavior
* Frequency agility
* Threat prioritization
* Sensor uncertainty
* Missed detections
* False alarms
* Receiver scanning constraints

---

## Project Context

For detailed architectural decisions, implementation constraints, simulation assumptions, and development status, see:

```text
docs/PROJECT_CONTEXT.md
```

This document is intended to remain the project's single source of truth for architectural context.

---

## Team

**Alterra**
Smart India Hackathon 2026
DRDO Problem Statement 26055

### Responsibilities

* **Simulation / Environment:** Vishy
* **Perception / Deinterleaving / RL:** Team member

The ownership boundary is intentionally maintained so that the simulation and ML pipeline can be developed and tested independently.

