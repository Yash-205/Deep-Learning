# Lecture 15: Gated RNNs — LSTM, GRU & Bidirectional RNNs

> **Worksheet File**: [Open Lecture 15 Interactive Worksheet](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_15_Gated_RNNs/Lecture_15_Gated_RNNs.html)  
> **Core Theme**: Overcoming the vanilla RNN memory bottleneck through additive gradient highways (LSTM), streamlined gating (GRU), and future context awareness (Bidirectional RNNs).

---

## 1. The Basics: Why Vanilla RNNs Forget

In a vanilla RNN, the hidden state update is strictly multiplicative:
$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b), \quad \hat{y}_t = \text{head}(h_t)$$
* When backpropagating through time, every step multiplies the gradient by recurrent weights and the derivative of $\tanh$.
* As established in Lecture 14, multiplying numbers smaller than 1 across 20 steps causes the gradient to vanish to near zero ($0.8^{20} \approx 0.0115$). The model completely forgets early clues.
* **The Core Insight of Gating:** To remember long-term clues, we need an **additive cell state highway** where information can flow across time without repeated matrix multiplications.

*(Note on word representations: A word ID is a categorical integer label, not a numerical magnitude. An embedding layer maps discrete word IDs into learned continuous feature vectors $x_t \in \mathbb{R}^d$).*

---

## 2. Long Short-Term Memory (LSTM)

Introduced by Hochreiter & Schmidhuber (1997), the LSTM divides memory into two distinct paths:
1. **Cell State ($c_t$):** The long-term memory "highway". Information travels down this line with mostly linear, additive operations.
2. **Hidden State ($h_t$):** The short-term working memory, filtered and exposed to the outside world.

```
                   c_{t-1} ───────[ ⊗ ]──────────────────[ ⊕ ]──────────────▶ c_t
                                    ▲                     ▲
                                    │ f_t                 │ (i_t ⊗ g_t)
                                  ┌───┐                 ┌───┐
                    h_{t-1} ──┬───┤ σ │   ┌───┐  ┌────┐ │ ⊗ │          ┌────┐
                              │   └───┘   │ σ │  │tanh│ └───┘    ┌─────┤tanh│
                              │           └───┘  └────┘          │     └────┘
                              │             ▲      ▲             ▼       ▲
                              │      i_t ───┘      └── g_t      [ ⊗ ]    │
                              │                                  ▲       │
                              │                              o_t │       │
                              │                                ┌───┐     │
                              │ ───────────────────────────────┤ σ │     │
                              │                                └───┘     │
                              │                                          ▼
                              └────────────────────────────────────────▶ h_t
                     x_t ─────┴────────────────────────────────────────┘
```

### The Equations of the LSTM Cell
At each step $t$, the input $x_t$ and previous hidden state $h_{t-1}$ are concatenated:
$$a_t = [h_{t-1} \,;\, x_t]$$

The cell then computes **three gates** (using Sigmoid $\sigma \in [0, 1]$) and **one candidate state** (using $\tanh \in [-1, 1]$):

1. **Forget Gate ($f_t$):** Decides what fraction of past cell memory to keep:
   $$f_t = \sigma(W_f a_t + b_f)$$
   * $f_t \approx 1 \implies$ Keep all past memory.
   * $f_t \approx 0 \implies$ Wipe clean (e.g. at sentence boundaries).
2. **Input Gate ($i_t$):** Decides which values will be updated with new evidence:
   $$i_t = \sigma(W_i a_t + b_i)$$
3. **Candidate Memory ($g_t$):** Computes new candidate values to write:
   $$g_t = \tanh(W_g a_t + b_g)$$
4. **Cell State Update ($c_t$):** Additive combination:
   $$c_t = f_t \odot c_{t-1} + i_t \odot g_t$$
   *(where $\odot$ is element-wise multiplication)*.
5. **Output Gate ($o_t$):** Decides which parts of the cell state to reveal:
   $$o_t = \sigma(W_o a_t + b_o)$$
6. **Hidden State Emission ($h_t$):**
   $$h_t = o_t \odot \tanh(c_t)$$

---

## 3. The Memory Highway: How Additive Paths Prevent Decay

In a vanilla RNN, the backward signal decays exponentially as a power law (e.g., $0.8^L$).

In an LSTM, the backward derivative flowing directly along the cell state is governed by the forget gate:
$$\text{Direct cell-state path} = \prod_{k=1}^L f_k$$

* If the model learns to keep the forget gate open ($f_k \approx 1.0$), the product across $L$ steps is:
  $$f^L = (1.0)^L = \mathbf{1.0}$$
* **The Breakthrough:** The gradient travels back across dozens or hundreds of steps with **zero exponential decay**! The network can preserve a clue from step 1 and use it at step 50.

---

## 4. Gated Recurrent Unit (GRU): One Notebook, Two Decisions

Proposed by Cho et al. (2014), the GRU streamlines the LSTM:
* It eliminates the separate cell state $c_t$, storing all memory in a single hidden state $h_t$.
* It uses only **2 gates**: **Reset Gate ($r_t$)** and **Update Gate ($z_t$)**.

### The GRU Equations:
1. **Reset Gate ($r_t$):** Decides how much of past memory $h_{t-1}$ to expose when computing candidate memory:
   $$r_t = \sigma(W_r [h_{t-1} \,;\, x_t] + b_r)$$
2. **Update Gate ($z_t$):** Acts simultaneously as both forget and input gate:
   $$z_t = \sigma(W_z [h_{t-1} \,;\, x_t] + b_z)$$
3. **Candidate Memory ($\tilde{h}_t$):**
   $$\tilde{h}_t = \tanh(W_x x_t + W_h (r_t \odot h_{t-1}) + b)$$
4. **Interpolated State Update ($h_t$):**
   $$h_t = z_t \odot h_{t-1} + (1 - z_t) \odot \tilde{h}_t$$
   * Notice the balance: If $z_t = 1$, the state is completely kept ($h_t = h_{t-1}$). If $z_t = 0$, the state is completely overwritten by candidate $\tilde{h}_t$.

### Exact Numerical Walkthrough (From the Worksheet)
Suppose input $x = 0.2$, old state $h = 0.8$, reset gate $r = 0.5$, weights are $1$, and bias is $0$:
1. Reset masked memory: $r \odot h = 0.5 \times 0.8 = 0.4$.
2. Candidate before activation: $x + 1 \times (0.4) = 0.2 + 0.4 = 0.6$.
3. Candidate state: $\tilde{h} = \tanh(0.6) \approx \mathbf{0.537}$.
4. At update gate $z = 0.75$:
   $$h_t = (0.75 \times 0.8) + ((1 - 0.75) \times 0.537) = 0.6 + (0.25 \times 0.537) = 0.6 + 0.13425 \approx \mathbf{0.734}$$

---

## 5. Bidirectional RNNs: Seeing the Future

### Why Unidirectional Models Fall Short
A standard left-to-right RNN processes token $t$ knowing only past tokens $1 \dots t-1$.
* *Example:* "The bank was muddy after the heavy rain."
  When reading the word "bank", a forward RNN cannot know whether it refers to a river bank or a financial bank until later words arrive.

### The Bidirectional Architecture
A Bidirectional RNN runs **two independent recurrent layers** over the same sequence:
1. **Forward RNN ($\vec{F}$):** Reads left-to-right ($t = 1 \dots T$):
   $$\vec{h}_t = \vec{F}(x_t, \vec{h}_{t-1})$$
2. **Backward RNN ($\overleftarrow{F}$):** Reads right-to-left ($t = T \dots 1$):
   $$\overleftarrow{h}_t = \overleftarrow{F}(x_t, \overleftarrow{h}_{t+1})$$
3. **Concatenation:** At every step $t$, the two states are joined:
   $$u_t = [\vec{h}_t \,;\, \overleftarrow{h}_t] \in \mathbb{R}^{2H} \implies \hat{y}_t = \text{head}(u_t)$$

### Latency & The Causal Boundary (Exam Warning!)
* **When to Use:** Offline / non-causal tasks where the entire sentence is already available at once (e.g. sentiment classification, named entity recognition, translation encoding).
* **When NOT to Use:** Real-time speech streaming or autoregressive language generation (generating the next word), because future tokens $t+1, t+2$ do not yet exist!
