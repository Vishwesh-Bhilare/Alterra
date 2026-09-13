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

## 3. PPO + LSTM Smart Scheduler (`3_ppo_lstm.jpg`)

![PPO + LSTM](3_ppo_lstm.jpg)

- **Model**: PPO + LSTM (`PPOLSTMExtractor` with gated memory cell state and LayerNorm feature balancing)
- **Metrics on Seed 32707**:
  - **Probability of Detection ($P_d$)**: `1.000`
  - **False Alarm Rate ($P_{fa}$)**: `0.024`
  - **Intercept Rate**: **`1.523`** *(+53.8% vs Baseline PPO, +39.7% vs PPO+RNN)*
  - **Average Reward**: **`16.949`** *(+128.3% vs Baseline PPO, +56.8% vs PPO+RNN)*
- **Behavior**: Wide-band continuous triangular spectrum sweeping across the full 128 channels with smooth edge reflection and rapid emitter burst lock-on.

---

## 📊 Comparative Analysis: PPO vs PPO+RNN vs PPO+LSTM

| Evaluation Metric | Baseline PPO (`1_ppo_only.png`) | PPO + RNN (`2_ppo_rnn.png`) | PPO + LSTM (`3_ppo_lstm.jpg`) | Performance Delta vs Baseline |
| :--- | :--- | :--- | :--- | :--- |
| **Average Reward** | `7.423` | `10.809` | **`16.949`** | **+128.3% higher 🚀** |
| **Intercept Rate** | `0.990` | `1.090` | **`1.523`** | **+53.8% higher 📈** |
| **False Alarm Rate ($P_{fa}$)** | `0.022` | `0.020` | `0.024` | **Low & Controlled** |
| **Detection Probability ($P_d$)** | `1.000` | `1.000` | **`1.000`** | **100% (Complete intercept)** |

### Key Architectural Strengths of PPO + LSTM:
1. **Long-Range Gated Temporal Memory**: LSTM memory cells ($c_t$) preserve pulse interval histories over longer dwell spans without vanishing gradients, allowing the agent to anticipate periodic and agile emitter returns.
2. **LayerNorm Feature Balancing**: Balances the scale between bounded LSTM hidden activations and unbounded compressed spectrum track embeddings.
3. **Continuous Triangular Sweep & Boundary Reflection**: Eliminates localized channel trapping and boundary oscillations, ensuring full 128-channel spectrum coverage.

