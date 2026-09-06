# Alterra — GUI Evaluation Screenshots

This directory contains evaluation comparison screenshots captured from the C++20 / Qt6 Desktop GUI (`alterra_gui`) on **Seed 32707**.

---

## 1. Baseline PPO (`1_ppo_only.png`)

![PPO Only](1_ppo_only.png)

- **Model**: Baseline PPO (Standard Multi-Input Actor-Critic without recurrent sequence modeling)
- **Metrics on Seed 32707**:
  - **Probability of Detection ($P_d$)**: `1.000`
  - **False Alarm Rate ($P_{fa}$)**: `0.022`
  - **Intercept Rate**: `0.990`
  - **Average Reward**: `7.423`
- **Behavior**: Exhibits localized frequency drifting with erratic, jittery search steps across channels when signal is lost.

---

## 2. PPO + RNN Smart Scheduler (`2_ppo_rnn.png`)

![PPO + RNN](2_ppo_rnn.png)

- **Model**: PPO + RNN (`PPORNNExtractor` with historical hit/miss rolling sequence buffer)
- **Metrics on Seed 32707**:
  - **Probability of Detection ($P_d$)**: `1.000`
  - **False Alarm Rate ($P_{fa}$)**: `0.020`
  - **Intercept Rate**: `1.090` *(+10.1% higher than baseline PPO)*
  - **Average Reward**: `10.809` *(+45.6% higher than baseline PPO)*
- **Behavior**: Systematic triangular sweep-and-lock pattern across the spectrum channels, utilizing temporal pulse arrival memory from the recurrent neural network to intercept active signals and maintain track.

---

## 📊 Comparative Analysis: Why PPO + RNN is Superior

| Evaluation Metric | Baseline PPO (`1_ppo_only.png`) | PPO + RNN (`2_ppo_rnn.png`) | Performance Delta |
| :--- | :--- | :--- | :--- |
| **Average Reward** | `7.423` | **`10.809`** | **+45.6% higher 🚀** |
| **Intercept Rate** | `0.990` | **`1.090`** | **+10.1% higher 📈** |
| **False Alarm Rate ($P_{fa}$)** | `0.022` | **`0.020`** | **Lower (Better)** |
| **Detection Probability ($P_d$)** | `1.000` | **`1.000`** | **100% (Complete intercept)** |

### Key Reasons for Superiority:
1. **Temporal Memory (Non-Markovian Radar Pulses)**: Radar emitters operate with pulse repetition intervals (PRI) and periodic scans. Stateless PPO only observes instantaneous dwell results and cannot distinguish between an empty channel and a pause between pulses. PPO + RNN maintains a temporal hidden state ($h_n \in \mathbb{R}^{64}$) of the last 16 dwells to model pulse arrival rhythms.
2. **Systematic Sweep-and-Lock**: As seen in `2_ppo_rnn.png`, PPO + RNN avoids localized trapping. It performs a steady, triangular frequency sweep until a signal is intercepted, locks onto the burst, and resumes sweeping smoothly upon burst completion.
3. **Noise and False Alarm Suppression**: By cross-referencing instantaneous power with recent dwell sequences, the RNN differentiates one-off thermal noise spikes from persistent pulse trains, reducing false alarms to 2.0%.
