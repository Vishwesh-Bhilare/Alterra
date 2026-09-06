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
- **Behavior**: Exhibits localized frequency drifting with occasional random search steps across channels.

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
