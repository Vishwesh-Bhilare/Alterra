# ALTERRA Cognitive Scheduler: Complete Mathematical Formulation & Metrics Reference

**Project**: ALTERRA — Smart Scan Strategy for Electronic Warfare (DRDO PS 26055, SIH 2026)  
**Document Version**: 1.0  
**Scope**: Complete mathematical equations, loss functions, action masking rules, neural network architectures, and evaluation metrics used during model training and validation.

---

## 1. System Architecture Overview

The ALTERRA scheduler operates on the design principle:  
> **"Rules control the doctrine; Machine Learning controls decisions within that doctrine."**

At each decision epoch $t$:
1. **Observation Processing**: The agent observes RF environment tracks $X_t \in \mathbb{R}^{N \times D_{\text{track}}}$, receiver state $r_t \in \mathbb{R}^{D_{\text{rec}}}$, and historical sequence buffer $S_t \in \mathbb{R}^{L \times D_{\text{seq}}}$.
2. **Cognitive Doctrine State Machine**: Categorizes operation into modes $\mathcal{M}_t \in \{\text{EXPLORE}, \text{INVESTIGATE}, \text{TRACK}, \text{RELOCATE}\}$.
3. **Candidate Action Masking**: Generates a binary mask $M_t \in \{0, 1\}^N$ restricting valid frequency bands.
4. **Neural Utility Estimation**: A sequence backbone (RNN, LSTM, GRU, or Transformer) evaluates candidate utility logits $z_t \in \mathbb{R}^N$.
5. **Dwell Duration Selection**: A calibrated rule-based dwell controller maps confidence and doctrine mode to discrete dwell durations $\tau_t \in \{3, 5, 8, 12\}$ slots.
6. **Environment Action Execution**: Steps relative direction $a_t \in \{-1, 0, +1\}$ or absolute channel band.

---

## 2. RF Spectrum & Environment Formulation

### 2.1 Spectrum Discretization
The RF spectrum of total bandwidth $B_{\text{total}}$ is partitioned into $N = 128$ uniform contiguous channels:
$$f_c(b) = f_{\min} + \left(b + \frac{1}{2}\right) \Delta f, \quad b \in \{0, 1, \dots, N-1\}$$
where $\Delta f = \frac{B_{\text{total}}}{N}$ is the channel width (e.g. $70 \text{ MHz}$ for a $1.0 - 9.96 \text{ GHz}$ coverage).

### 2.2 Relative Action Space
In relative scanning mode, the action space is discrete:
$$a_t = [\delta_t, d_t] \in \{0, 1, 2\} \times \{0, 1, 2, 3\}$$
* $\delta_t \in \{0, 1, 2\}$ maps to relative frequency step $\Delta b \in \{-1, 0, +1\}$:
  $$\Delta b = \begin{cases} -1, & \delta_t = 0 \text{ (Step Down)} \\ 0, & \delta_t = 1 \text{ (Stay / Dwell)} \\ +1, & \delta_t = 2 \text{ (Step Up)} \end{cases}$$
* Next band transition with reflective boundaries:
  $$b_{t+1} = \min\Big(\max(b_t + \Delta b, 0), N - 1\Big)$$
* $d_t \in \{0, 1, 2, 3\}$ maps to dwell durations $\tau \in \{3, 5, 8, 12\}$ time slots ($1 \text{ slot} = 20 \ \mu\text{s}$).

---

## 3. Cognitive Doctrine & Action Masking

### 3.1 Doctrine Mode State Machine

The cognitive state machine manages transitions based on detection history:
$$\mathcal{M}_{t+1} = f_{\text{doctrine}}(\mathcal{M}_t, \text{hit}_t, \text{consec}_t, P_t, \text{revisits}_t)$$

1. **EXPLORE $\to$ INVESTIGATE**:
   Triggered on pulse detection or elevated power:
   $$\mathcal{M}_{t+1} = \text{INVESTIGATE} \quad \text{if} \quad \text{hit}_t = 1 \quad \lor \quad P_t > 0.4$$
2. **INVESTIGATE $\to$ TRACK**:
   Triggered if signal is re-confirmed within dwell window ($k \le 2$):
   $$\mathcal{M}_{t+1} = \text{TRACK} \quad \text{if} \quad \text{hit}_t = 1$$
3. **TRACK $\to$ RELOCATE**:
   Triggered if target signal is lost or revisit quota $K_{\max} = 5$ is exhausted:
   $$\mathcal{M}_{t+1} = \text{RELOCATE} \quad \text{if} \quad \text{hit}_t = 0 \quad \lor \quad \text{revisits}_t \ge K_{\max}$$
4. **RELOCATE $\to$ EXPLORE**:
   Triggered once cooldown period $\tau_{\text{relocate}} = 3$ expires without re-acquisition.

### 3.2 Candidate Action Mask Formulation

The Boolean action mask $M_t(b) \in \{0, 1\}$ for band $b \in \{0, \dots, N-1\}$ is defined by operational doctrine:

$$M_t(b) = \begin{cases}
\mathbb{I}\Big(b \in \{b^* - 1, b^*, b^* + 1\}\Big), & \mathcal{M}_t = \text{TRACK} \text{ (Active target } b^*\text{)} \\
\mathbb{I}\Big(|b - b_t| \le 2\Big), & \mathcal{M}_t = \text{INVESTIGATE} \\
1 - \mathbb{I}\Big(b = b^* \lor b \in \mathcal{H}_{\text{recent}}^{[-2:]}\Big), & \mathcal{M}_t = \text{RELOCATE} \\
1 - \mathbb{I}\Big(b = b_t \land \text{hit}_t = 0\Big), & \mathcal{M}_t = \text{EXPLORE}
\end{cases}$$

**Fallback Safety Invariant**:
If all bands are pruned, the mask falls back to uniform coverage:
$$M_t(b) = 1 \quad \forall b \in \{0, \dots, N-1\} \quad \text{if} \quad \sum_{j=0}^{N-1} M_t(j) = 0$$

---

## 4. Neural Network Architectures & Forward Passes

The modular `HybridScorer` processes four feature streams and fuses them into a 128-band utility logit head.

```
Observation Stream:
  1. Sequence Buffer S_t (16 x 6)   --> Temporal Backbone (RNN / LSTM / GRU / Transformer) --> h_seq  (64)
  2. Band Tracks X_t (128 x 8)      --> Spectrum Compressor (Linear + LayerNorm + ReLU)    --> h_spec (128)
  3. Receiver + Mode r_t, m_t (14)  --> State Encoder (Linear + LayerNorm + ReLU)          --> h_rec  (32)
                                                                                                 |
Fused Representation: h_fuse = Trunk([h_seq, h_spec, h_rec]) in R^256 ---------------------------+
  |--> Masked Utility Head: z_t(b) in R^128 (with -inf invalid masking)
  |--> Value Critic Head:   V(s_t) in R^1
```

### 4.1 Sequence Backbones

#### A. LSTM Backbone (`HybridScorer[LSTM]`)
Given input sequence $x_\tau \in \mathbb{R}^6$ for $\tau = 1, \dots, L$:
$$\begin{aligned}
i_\tau &= \sigma\left(W_{ii} x_\tau + b_{ii} + W_{hi} h_{\tau-1} + b_{hi}\right) \\
f_\tau &= \sigma\left(W_{if} x_\tau + b_{if} + W_{hf} h_{\tau-1} + b_{hf}\right) \\
g_\tau &= \tanh\left(W_{ig} x_\tau + b_{ig} + W_{hg} h_{\tau-1} + b_{hg}\right) \\
o_\tau &= \sigma\left(W_{io} x_\tau + b_{io} + W_{ho} h_{\tau-1} + b_{ho}\right) \\
c_\tau &= f_\tau \odot c_{\tau-1} + i_\tau \odot g_\tau \\
h_\tau &= o_\tau \odot \tanh(c_\tau)
\end{aligned}$$
Final sequence embedding: $h_{\text{seq}} = h_L \in \mathbb{R}^{64}$.

#### B. GRU Backbone (`HybridScorer[GRU]`)
$$\begin{aligned}
z_\tau &= \sigma\left(W_{iz} x_\tau + b_{iz} + W_{hz} h_{\tau-1} + b_{hz}\right) \quad &\text{(Update gate)} \\
r_\tau &= \sigma\left(W_{ir} x_\tau + b_{ir} + W_{hr} h_{\tau-1} + b_{hr}\right) \quad &\text{(Reset gate)} \\
n_\tau &= \tanh\left(W_{in} x_\tau + b_{in} + r_\tau \odot (W_{hn} h_{\tau-1} + b_{hn})\right) \quad &\text{(Candidate hidden)} \\
h_\tau &= (1 - z_\tau) \odot n_\tau + z_\tau \odot h_{\tau-1}
\end{aligned}$$
Final sequence embedding: $h_{\text{seq}} = h_L \in \mathbb{R}^{64}$.

#### C. Elman RNN Backbone (`HybridScorer[RNN]`)
$$h_\tau = \tanh\left(W_{ih} x_\tau + b_{ih} + W_{hh} h_{\tau-1} + b_{hh}\right)$$
Final sequence embedding: $h_{\text{seq}} = h_L \in \mathbb{R}^{64}$.

#### D. Transformer Backbone (`HybridScorer[Transformer]`)
Input projection with learned positional embedding:
$$\mathbf{E}_0 = W_{\text{proj}} X_{\text{seq}} + \mathbf{P}, \quad \mathbf{E}_0 \in \mathbb{R}^{L \times 64}$$
Multi-Head Self-Attention ($H = 4$ heads, $d_k = 16$):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$
$$\mathbf{E}'_l = \text{LayerNorm}\left(\mathbf{E}_{l-1} + \text{MHA}(\mathbf{E}_{l-1})\right)$$
$$\mathbf{E}_l = \text{LayerNorm}\left(\mathbf{E}'_l + \text{FFN}(\mathbf{E}'_l)\right)$$
Final sequence embedding: $h_{\text{seq}} = \mathbf{E}_{L}[-1, :] \in \mathbb{R}^{64}$.

---

### 4.2 Feature Compression & Fusion Trunk

1. **Spectrum Tracks Compression**:
   $$h_{\text{spec}} = \text{ReLU}\left(\text{LayerNorm}\left(W_{\text{tr}} \text{vec}(X_t) + b_{\text{tr}}\right)\right), \quad h_{\text{spec}} \in \mathbb{R}^{128}$$
2. **Receiver & Mode Encoder**:
   $$h_{\text{rec}} = \text{ReLU}\left(\text{LayerNorm}\left(W_{\text{rec}} [r_t \,\|\, m_t] + b_{\text{rec}}\right)\right), \quad h_{\text{rec}} \in \mathbb{R}^{32}$$
3. **Nonlinear Fusion Trunk**:
   $$h_{\text{fuse}} = \text{MLP}_{256}\left([h_{\text{seq}} \,\|\, h_{\text{spec}} \,\|\, h_{\text{rec}}]\right) \in \mathbb{R}^{256}$$

---

### 4.3 Masked Utility Head & Action Selection

Raw logits across all 128 candidate channels:
$$u(b) = W_{\text{util}}^{(b)} h_{\text{fuse}} + b_{\text{util}}^{(b)}, \quad b \in \{0, \dots, N-1\}$$

**Invalid Action Suppression (-$\infty$ Masking)**:
$$\tilde{u}(b) = \begin{cases} u(b), & M_t(b) = 1 \\ -\infty, & M_t(b) = 0 \end{cases}$$

**Masked Categorical Policy Distribution**:
$$\pi_\theta(b \mid s_t, M_t) = \frac{\exp(\tilde{u}(b))}{\sum_{k=0}^{N-1} \exp(\tilde{u}(k))}$$

* **Deterministic Action (Inference / Benchmark)**:
  $$b^* = \arg\max_{b \in \{0, \dots, N-1\}} \tilde{u}(b)$$
* **Stochastic Action (Exploration / Sampling)**:
  $$b^* \sim \text{Categorical}(\pi_\theta(\cdot \mid s_t, M_t))$$

Relative environment step selection:
$$\delta_t = \begin{cases} 0 \text{ (Step Down)}, & b^* < b_{\text{curr}} \\ 1 \text{ (Stay)}, & b^* = b_{\text{curr}} \\ 2 \text{ (Step Up)}, & b^* > b_{\text{curr}} \end{cases}$$

---

## 5. Supervised Imitation Learning Formulation

### 5.1 Masked Cross-Entropy Loss
The model is trained on expert trajectories generated by the heuristic teacher doctrine:
$$\mathcal{L}_{\text{imitation}}(\theta) = -\frac{1}{B} \sum_{i=1}^B \sum_{b=0}^{N-1} y_{\text{teacher}}^{(i)}(b) \log \pi_\theta(b \mid s_t^{(i)}, M_t^{(i)})$$
where $y_{\text{teacher}}^{(i)}$ is the one-hot encoded teacher band selection.

### 5.2 Cosine Annealing Learning Rate Schedule
Learning rate $\eta_e$ at epoch $e \in \{1, \dots, E_{\max}\}$:
$$\eta_e = \eta_{\min} + \frac{1}{2}(\eta_{\max} - \eta_{\min}) \left(1 + \cos\left(\frac{e}{E_{\max}} \pi\right)\right)$$
where $\eta_{\max} = 5 \times 10^{-4}$ and $\eta_{\min} = 1 \times 10^{-5}$.

### 5.3 Gradient Norm Clipping
$$\mathbf{g} \leftarrow \begin{cases} \mathbf{g}, & \|\mathbf{g}\|_2 \le \gamma_{\text{clip}} \\ \gamma_{\text{clip}} \frac{\mathbf{g}}{\|\mathbf{g}\|_2}, & \|\mathbf{g}\|_2 > \gamma_{\text{clip}} \end{cases} \quad (\gamma_{\text{clip}} = 1.0)$$

---

## 6. Reinforcement Learning / PPO Formulation

For comparative PPO models (`PPO`, `PPO+RNN`, `PPO+LSTM`, `PPO+Transformer`), the objective is:

### 6.1 Probability Ratio & Clipped Surrogate Objective
$$r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}$$
$$L^{\text{CLIP}}(\theta) = \hat{\mathbb{E}}_t \left[ \min\left(r_t(\theta)\hat{A}_t, \, \text{clip}(r_t(\theta), 1 - \epsilon, 1 + \epsilon)\hat{A}_t\right) \right]$$
where $\epsilon = 0.2$ is the clipping parameter.

### 6.2 Generalized Advantage Estimation (GAE)
$$\hat{A}_t^{\text{GAE}(\gamma, \lambda)} = \sum_{l=0}^\infty (\gamma \lambda)^l \delta_{t+l}^V$$
$$\delta_t^V = r_t + \gamma V(s_{t+1}) - V(s_t)$$
where $\gamma = 0.99$ and $\lambda = 0.95$.

### 6.3 Total PPO Loss
$$\mathcal{L}_{\text{PPO}}(\theta, \phi) = -L^{\text{CLIP}}(\theta) + c_1 \hat{\mathbb{E}}_t \left[ (V_\phi(s_t) - R_t)^2 \right] - c_2 \hat{\mathbb{E}}_t \left[ \mathcal{H}(\pi_\theta(\cdot \mid s_t)) \right]$$
where $c_1 = 0.5$ (value loss coefficient) and $c_2 = 0.01$ (entropy bonus coefficient).

---

## 7. Environment Reward Structure Formulation

At each dwell step, the scalar reward $R_t$ awarded to the agent is:

$$R_t = R_{\text{hit}} + R_{\text{track}} + R_{\text{novelty}} - C_{\text{dwell}} - C_{\text{switch}} - C_{\text{empty}} - C_{\text{lost}} - C_{\text{boundary}} - C_{\text{FA}} - C_{\text{stale}}$$

$$\begin{aligned}
R_{\text{hit}} &= R_{\text{base}} \cdot w_{\text{threat}}(\text{emitter}) \cdot \mathbb{I}(\text{hit}_t = 1) \quad &(R_{\text{base}} = 10.0, \, w \in \{1, 2, 4\}) \\
R_{\text{track}} &= R_{\text{cont}} \cdot \mathbb{I}(\text{consecutive\_hits} \ge 2) \quad &(R_{\text{cont}} = 6.0) \\
R_{\text{novelty}} &= R_{\text{new}} \cdot \mathbb{I}(b_t \notin \mathcal{V}_{\text{visited}}) \quad &(R_{\text{new}} = 1.0) \\
C_{\text{dwell}} &= c_d \cdot \tau_t \quad &(c_d = 0.1 \text{ per slot}) \\
C_{\text{switch}} &= c_s \cdot |b_t - b_{t-1}| \quad &(c_s = 0.05 \text{ retune penalty}) \\
C_{\text{empty}} &= c_e \cdot \mathbb{I}(\Delta b = 0 \land \text{hit}_t = 0) \quad &(c_e = 0.8 \text{ staying on dead band}) \\
C_{\text{lost}} &= c_l \cdot \mathbb{I}(\text{hit}_{t-1} = 1 \land \text{hit}_t = 0) \quad &(c_l = 0.8 \text{ dropped track}) \\
C_{\text{boundary}} &= c_b \cdot \mathbb{I}(\text{hit\_boundary}) \quad &(c_b = 2.0 \text{ illegal boundary step}) \\
C_{\text{FA}} &= c_{\text{fa}} \cdot \mathbb{I}(\text{false\_alarm}_t = 1) \quad &(c_{\text{fa}} = 0.5) \\
C_{\text{stale}} &= c_{\text{stale}} \cdot \frac{t - t_{\text{last\_visit}}(b_t)}{\tau_{\text{norm}}} \quad &(c_{\text{stale}} = 0.02, \, \tau_{\text{norm}} = 200)
\end{aligned}$$

---

## 8. The Seven Evaluation Figures of Merit

These seven metrics correspond exactly to Section 3 of `ALTERA_Hybrid_Scheduler_Implementation_and_Metric_Validation.docx` and are computed in `simulation/metrics/rollout_metrics.py`.

### 8.1 Metric 1 — Probability of Detection ($P_D$)
$$P_D = P(\text{declare signal} \mid H_1) = \frac{\text{TP}}{\text{TP} + \text{FN}}$$
* $\text{TP}$: Dwell slots where channel was physically occupied ($H_1$) and receiver declared a valid detection.
* $\text{FN}$: Dwell slots where channel was physically occupied ($H_1$) but receiver missed it.

### 8.2 Metric 2 — Probability of False Alarm ($P_{FA}$)
$$P_{FA} = P(\text{declare signal} \mid H_0) = \frac{\text{FP}}{\text{FP} + \text{TN}}$$
* $\text{FP}$: Dwell slots on empty/noise-only channels ($H_0$) where thermal noise crossed the detection threshold.
* $\text{TN}$: Dwell slots on empty channels ($H_0$) where no signal was declared.

### 8.3 Metric 3 — Receiver Sensitivity ($P_{\min}$)
Physical thermal-noise floor expression:
$$P_{\min, \text{dBm}} = -174 \text{ dBm/Hz} + 10 \log_{10}(B_{\text{Hz}}) + \text{NF}_{\text{dB}} + \text{SNR}_{\min, \text{dB}}$$
* In simulation evaluation, sensitivity is quantified as the **Low-SNR Detection Efficiency** (detection probability on weak signals with $\text{SNR} < 10 \text{ dB}$):
$$\text{Sensitivity} = \frac{\sum_{i \in \text{occupied}, \, \text{SNR}_i < 10\text{dB}} \text{hit}_i}{\sum_{i \in \text{occupied}, \, \text{SNR}_i < 10\text{dB}} 1}$$

### 8.4 Metric 4 — Average Interception Rate ($R_I$)
$$R_I = \frac{N_{\text{intercepts}}}{T_{\text{observed}}} = \frac{\sum_{k=1}^K \text{hit}_k}{\sum_{k=1}^K \tau_k}$$
Across $M$ evaluation episodes:
$$\overline{R}_I = \frac{1}{M} \sum_{m=1}^M \frac{N_{I, m}}{T_m}$$

### 8.5 Metric 5 — Average Episodic Return ($\overline{G}$)
$$G_m = \sum_{t=0}^{T_m - 1} \gamma^t R_{m, t} \quad (\gamma = 1.0 \text{ for fixed-horizon scheduling})$$
$$\overline{G} = \frac{1}{M} \sum_{m=1}^M G_m$$

### 8.6 Metric 6 — Classification Accuracy ($\text{Acc}$)
$$\text{Accuracy} = \frac{\text{TP} + \text{TN}}{\text{TP} + \text{TN} + \text{FP} + \text{FN}} \times 100\%$$

### 8.7 Metric 7 — Interception Time Error / TTFI Error ($\text{MAE}_{\text{time}}$)
$$\text{MAE}_{\text{time}} = \frac{1}{|\mathcal{E}_{\text{intercepted}}|} \sum_{e \in \mathcal{E}_{\text{intercepted}}} \left| t_{\text{first\_intercept}}(e) - t_{\text{activation}}(e) \right|$$
where $t_{\text{activation}}(e)$ is the slot when emitter $e$ first transmitted, and $t_{\text{first\_intercept}}(e)$ is the first slot the receiver intercepted it. Unintercepted emitters are recorded as misses.

---

## 9. Summary Mapping: Code Implementation to Mathematical Symbols

| Symbol / Equation | Code Location | Class / Function |
| :--- | :--- | :--- |
| $\mathcal{M}_t, M_t(b)$ | `model/hybrid/doctrine.py` | `CognitiveDoctrine.update_state()`, `.generate_action_mask()` |
| $\tau_t, d_t$ | `model/hybrid/dwell_controller.py` | `RuleBasedDwellController.select_dwell()` |
| $h_{\text{seq}}, h_{\text{spec}}, h_{\text{fuse}}$ | `model/hybrid/scorers.py` | `HybridScorer.forward()` |
| $\pi_\theta(b \mid s_t, M_t)$ | `model/hybrid/scorers.py` | `HybridScorer.predict_action()` |
| $\mathcal{L}_{\text{imitation}}(\theta)$ | `training/train_imitation.py` | `train_epoch()`, `HybridScorer.compute_masked_imitation_loss()` |
| $a_t = [\delta_t, d_t]$ | `model/hybrid/policy.py` | `HybridPolicy.predict()` |
| $P_D, P_{FA}, R_I, \text{Acc}$ | `simulation/metrics/rollout_metrics.py` | `MetricsTracker.finalize()` |
| $R_t$ (Scalar Reward) | `simulation/environment/gym_env.py` | `AlterraEnv._compute_reward()` |
