# Lecture 17: Encoder–Decoder Architecture & Attention Mechanisms

> **Worksheet File**: [Open Lecture 17 Interactive Worksheet](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_17_Encoder_Decoder_Attention/Lecture_17_Encoder_Decoder_Attention.html)  
> **Course Progression**: Lecture 16 (Text Representations & Embeddings) &rarr; **Lecture 17 (Seq2Seq & Attention)** &rarr; Module 7

---

## 1. The Basics: The Closed Notebook Analogy

Imagine being shown a complex sentence:
> *"On Tuesday, Mira sent the small blue parcel to Arun, not to Leela."*

Now, close the notebook. Can you answer from memory:
* Who must *not* receive the parcel?
* What color was the parcel?

When translating between languages (e.g. English $\to$ French), word counts do not match:
$$\text{"I love you"} \quad (3 \text{ words}) \quad \longrightarrow \quad \text{"je t'aime"} \quad (2 \text{ words} + \langle\text{EOS}\rangle)$$
* We cannot align inputs and outputs step-by-step on a single clock.
* The model must read the complete source sentence first, store its meaning, and then generate the target translation word by word.

```
       Source Sequence (Clock 1)                       Target Sequence (Clock 2)
  [ x_1 ] ──▶ [ x_2 ] ──▶ [ x_3 ]                  [ y_1 ] ──▶ [ y_2 ] ──▶ <EOS>
         \     |     /                                    ▲          ▲
          ▼    ▼    ▼                                     │          │
     ======================                          ======================
     ENCODER HIDDEN STATES                             DECODER GENERATION
     ======================                          ======================
```

---

## 2. The Classic Encoder–Decoder Architecture

Introduced by Sutskever et al. (2014) and Cho et al. (2014), the network consists of two recurrent networks:

### 1. The Encoder (Source Clock: $i = 1, \dots, n$)
The encoder reads source tokens $x_1, \dots, x_n$ one by one, updating its recurrent state:
$$h_i = \text{Encoder}(h_{i-1}, E_x(x_i))$$
Where:
* $E_x(x_i) \in \mathbb{R}^d$ is the input word embedding.
* $h_i \in \mathbb{R}^{H_e}$ is the encoder hidden state.
* In a simple recurrent cell: $h_i = \tanh(W_x E_x(x_i) + W_h h_{i-1} + b)$.
* After reading all $n$ tokens, the final hidden state $h_n$ serves as the fixed **summary context vector**:
  $$c = h_n$$

### 2. The Linear Bridge ($W_{\text{bridge}}$)
If the encoder hidden dimension $H_e$ differs from the decoder hidden dimension $H_d$, a trainable linear bridge transforms the summary vector into the initial decoder state $s_0$:
$$s_0 = \text{bridge}(c) = \tanh(W_{\text{bridge}} c + b_{\text{bridge}})$$
* **Dimension Rule**: If $H_e = 4$ and $H_d = 3$, matrix $W_{\text{bridge}}$ has shape:
  $$\mathbf{H_d \times H_e = 3 \times 4} \quad (\text{exactly } 3 \text{ rows})$$

### 3. The Decoder (Target Clock: $t = 1, \dots, m$)
The decoder generates target tokens autoregressively until it emits an end-of-sequence token $\langle\text{EOS}\rangle$:
$$s_t = \text{Decoder}(s_{t-1}, E_y(y_{t-1}))$$
$$p(y_t \mid y_{<t}, x) = \text{softmax}(W_o s_t + b_o)$$
* **Token Lengths Can Differ**: Three source tokens do *not* require three target tokens. The decoder continues generating until $\langle\text{EOS}\rangle$ is emitted or a maximum length limit is reached.

### 4. Training with Teacher Forcing
During training, the ground-truth target sequence is known.
* To predict token $y_t$, the decoder is fed the **previous ground-truth target** $y_{t-1}$ as input (shifted by one step).
* **Exam Check**: When training the model to predict `"t'aime"` in `"je t'aime"`, the input fed to the decoder is `"je"`. Feeding `"t'aime"` directly to predict `"t'aime"` would leak the answer!
* At inference (test time), ground truth is unavailable, so the decoder feeds its own previous prediction $\hat{y}_{t-1}$ as the next input.

---

## 3. The Information Bottleneck Problem

In the classic architecture, the entire source sentence must be compressed into a **single fixed-size vector**:
$$c = h_n \in \mathbb{R}^{H_e}$$

```
                          THE FIXED-VECTOR BOTTLENECK
       Source length: 50 words               Fixed context vector
       [ x_1 ] ──▶ ... ──▶ [ x_50 ] ──────▶ [ c ] ∈ R^8  (Only 8 numbers!)
                                               │
                                               ▼
                                      Must recall every detail
                                      to generate target words
```

### Key Exam Questions on Bottleneck Behavior
1. **Length Scaling**: If we increase source length from $5$ to $50$ tokens while keeping hidden width $H = 8$, how many numbers remain in the final encoder summary $h_n$?
   * *Answer*: Exactly **8 coordinates**. Running more recurrent steps does not increase the hidden state dimension or add new parameter banks.
2. **Gradient & Information Decay**: In a simple scalar recurrence $h_i = \tanh(x_i + r h_{i-1})$, the gradient of the final state with respect to the initial state is:
   $$\frac{\partial h_n}{\partial h_1} = \prod_{i=2}^n r (1 - h_i^2)$$
   Multiplying factors smaller than 1 across 50 steps causes early details to be overwritten and forgotten.
3. **The Solution**: Instead of forcing the decoder to rely on a single closed notebook $c = h_n$, we retain **all intermediate encoder states** $\{h_1, h_2, \dots, h_n\}$ in memory and allow the decoder to look back at them dynamically at every target step.

---

## 4. The Shared 4-Step Attention Recipe

At every target decoding step $t$, the decoder computes a customized dynamic context vector $c_t$ using a 4-step recipe:

```
Encoder States:    h_1           h_2           h_3          ...   h_n
                    │             │             │                  │
   Step 1: Score    ├── Query q_t ├── Query q_t ├── Query q_t      ├── Query q_t
                    ▼             ▼             ▼                  ▼
   Alignment:      e_{t,1}       e_{t,2}       e_{t,3}      ...   e_{t,n}
                    │             │             │                  │
   Step 2: Softmax  └─────────────┼─────────────┼──────────────────┘
                                  ▼
   Weights:        α_{t,1}       α_{t,2}       α_{t,3}      ...   α_{t,n}  (Σ α = 1)
                    │             │             │                  │
   Step 3: Mix      └─────────────┴──────┬──────┴──────────────────┘
                                         ▼
   Dynamic Context:                     c_t = Σ_i α_{t,i} h_i
                                         │
   Step 4: Predict                       ▼
                                Combine c_t and s_t ──▶ Softmax ──▶ p(y_t)
```

### The 4 Equations
1. **Compute Alignment Scores**:
   $$e_{ti} = \text{score}(q_t, h_i)$$
   Where $q_t$ is the decoder query state and $h_i$ is the $i$-th encoder hidden state.
2. **Compute Attention Weights (Softmax over source positions)**:
   $$\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{j=1}^n \exp(e_{tj})}$$
   * The weights $\alpha_{ti} \in [0, 1]$ form a valid probability distribution: $\sum_{i=1}^n \alpha_{ti} = 1$.
3. **Compute Dynamic Context Vector**:
   $$c_t = \sum_{i=1}^n \alpha_{ti} h_i$$
   $c_t$ is a weighted average of source representations, retaining width $H_e$.
4. **Predict Next Target Token**:
   Combine $c_t$ with the decoder state to generate vocabulary logits.

---

## 5. Worked Numerical Examples (From HTML Worksheet)

### Example 1: Context Vector Computation (Check 7)
* Attention weights: $\alpha = [0.25, 0.75]^T$
* Encoder hidden states: $h_1 = [2, 0]^T, \; h_2 = [0, 4]^T$
* Calculate context vector $c$:
  $$c = \alpha_1 h_1 + \alpha_2 h_2 = 0.25 \begin{bmatrix} 2 \\ 0 \end{bmatrix} + 0.75 \begin{bmatrix} 0 \\ 4 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 0 \end{bmatrix} + \begin{bmatrix} 0 \\ 3.0 \end{bmatrix} = \begin{bmatrix} 0.5 \\ 3.0 \end{bmatrix}$$
  *The first coordinate is $0.5$, and the second coordinate is **3.0**.*

### Example 2: Equal Scores Distribution (Check 8)
* Three source tokens have identical alignment scores: $e = [s, s, s]$.
* Attention weights via softmax:
  $$\alpha_i = \frac{e^s}{e^s + e^s + e^s} = \frac{e^s}{3 e^s} = \frac{1}{3} \approx 0.3333$$
  *Each source position receives exactly $1/3$ weight. Common score shifts do not change the proportions.*

### Example 3: Full Softmax & Context Calculation (From HTML Theory Fold 2)
* Alignment scores: $e = [1, 0, 1]^T$
* Exponentials: $\exp([1, 0, 1]) = [2.718, 1.000, 2.718]^T$
* Sum of exponentials: $\sum = 2.718 + 1.000 + 2.718 = 6.437$
* Attention weights:
  $$\alpha_1 = \frac{2.718}{6.437} \approx 0.422, \quad \alpha_2 = \frac{1.000}{6.437} \approx 0.155, \quad \alpha_3 = \frac{2.718}{6.437} \approx 0.422$$
* Encoder states: $h_1 = [1, 0]^T, \; h_2 = [0, 1]^T, \; h_3 = [1, 1]^T$
* Context vector coordinates:
  * Coordinate 1: $0.422(1) + 0.155(0) + 0.422(1) = 0.844 \approx 0.845$
  * Coordinate 2: $0.422(0) + 0.155(1) + 0.422(1) = 0.577 \approx 0.578$
  $$c \approx [0.845, 0.578]^T$$

### Example 4: Attention Matrix Properties (Check 9)
In an attention weight matrix $A \in \mathbb{R}^{m \times n}$ ($m$ target steps, $n$ source positions):
* **Rows sum to 1**: Every row $t$ is normalized by softmax across all source positions: $\sum_{i=1}^n A_{ti} = 1$.
* **Columns DO NOT need to sum to 1**: Different target steps can attend to the same source word (e.g. both target step 2 and step 3 can place high weight on source word 1).

---

## 6. Bahdanau Additive Attention (Bahdanau et al., 2015)

Bahdanau attention computes alignment scores using a small feed-forward neural network (MLP) with a $\tanh$ non-linearity.

### The Mathematical Formulation
$$e_{ti} = v_a^T \tanh(W_q s_{t-1} + W_h h_i + b_a)$$
$$\alpha_t = \text{softmax}(e_t)$$
$$c_t = \sum_{i=1}^n \alpha_{ti} h_i$$
$$s_t = \text{Decoder}(s_{t-1}, E_y(y_{t-1}), c_t)$$

### Matrix Dimensions & Parameter Shapes (Check 10 & 12)
Let:
* Decoder hidden width: $H_d = 3$
* Encoder hidden width: $H_e = 4$
* Attention alignment layer width: $A = 5$

Shapes required:
* $W_q \in \mathbb{R}^{A \times H_d} = \mathbf{5 \times 3}$ ($5$ rows, projects decoder query to dimension $5$)
* $W_h \in \mathbb{R}^{A \times H_e} = \mathbf{5 \times 4}$ ($5$ rows, projects encoder state to dimension $5$)
* $b_a \in \mathbb{R}^A = \mathbf{5 \times 1}$
* $v_a \in \mathbb{R}^A = \mathbf{5 \times 1}$ (compresses the 5-dimensional vector down to a scalar score)
* **Context Vector Width**: Despite scoring width $A = 5$, the context vector $c_t$ has dimension $\mathbf{H_e = 4}$. The attention weights multiply the *original* encoder annotations $h_i \in \mathbb{R}^4$.

### Timing Convention
* In classic Bahdanau attention, the query is the **previous decoder state** $s_{t-1}$.
* The context $c_t$ is calculated first, and then fed directly into the recurrent cell to produce current state $s_t$.

### Worked Additive Scoring Example (From HTML Fold 3)
* Parameters: $W_q = W_h = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}, b_a = \begin{bmatrix} 0 \\ 0 \end{bmatrix}, v_a = \begin{bmatrix} 1 \\ -1 \end{bmatrix}$
* Query: $q = s_{t-1} = [1, 0]^T$
* For source $h_1 = [1, 0]^T$:
  $$\text{sum} = W_q q + W_h h_1 = \begin{bmatrix} 1 \\ 0 \end{bmatrix} + \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 2 \\ 0 \end{bmatrix}$$
  $$\tanh([2, 0]^T) = [\tanh(2), \tanh(0)]^T \approx [0.964, 0]^T$$
  $$e_1 = v_a^T [0.964, 0]^T = 1(0.964) + (-1)(0) = 0.964$$
* For source $h_2 = [0, 1]^T$:
  $$\text{sum} = [1, 0]^T + [0, 1]^T = [1, 1]^T$$
  $$e_2 = v_a^T [\tanh(1), \tanh(1)]^T = \tanh(1) - \tanh(1) = 0.000$$
* For source $h_3 = [1, 1]^T$:
  $$\text{sum} = [1, 0]^T + [1, 1]^T = [2, 1]^T$$
  $$e_3 = 1(\tanh(2)) - 1(\tanh(1)) \approx 0.9640 - 0.7616 \approx 0.202$$

---

## 7. Luong Multiplicative Attention (Luong et al., 2015)

Luong attention simplifies scoring by replacing the additive MLP with matrix multiplications.

### Two Scoring Functions
1. **Dot Product Scoring** (Requires $H_d = H_e$):
   $$e_{ti} = s_t^T h_i$$
2. **General Scoring** (Allows $H_d \neq H_e$, learns matrix $W_a \in \mathbb{R}^{H_d \times H_e}$):
   $$e_{ti} = s_t^T W_a h_i$$

### Attentional Hidden State & Prediction
In Luong's formulation:
1. Update decoder state first: $s_t = \text{Decoder}(s_{t-1}, E_y(y_{t-1}))$.
2. Compute attention using **current decoder state** $s_t$ as query.
3. Compute dynamic context $c_t = \sum_i \alpha_{ti} h_i$.
4. Combine context and state into an **attentional hidden state** $\tilde{s}_t$:
   $$\tilde{s}_t = \tanh(W_c [c_t \,;\, s_t])$$
5. Output prediction:
   $$p(y_t) = \text{softmax}(W_o \tilde{s}_t + b_o)$$

### Worked General Scoring Calculation (Check 13 & Fold 4)
Let:
* Query: $q = s_t = [1, 1]^T$
* Weight matrix: $W_a = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix}$
* Source states: $h_1 = [1, 0]^T, \; h_2 = [0, 1]^T, \; h_3 = [1, 1]^T$

Compute $W_a h_i$:
* $W_a h_1 = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 0 \end{bmatrix} = \begin{bmatrix} 1 \\ 0 \end{bmatrix}$
* $W_a h_2 = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 0 \\ 1 \end{bmatrix} = \begin{bmatrix} 0 \\ 2 \end{bmatrix}$
* $W_a h_3 = \begin{bmatrix} 1 & 0 \\ 0 & 2 \end{bmatrix} \begin{bmatrix} 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 1 \\ 2 \end{bmatrix}$

Compute General Scores $e_i = q^T (W_a h_i)$:
* $e_1 = [1, 1] \begin{bmatrix} 1 \\ 0 \end{bmatrix} = 1(1) + 1(0) = 1$
* $e_2 = [1, 1] \begin{bmatrix} 0 \\ 2 \end{bmatrix} = 1(0) + 1(2) = 2$  *(Check 13 answer)*
* $e_3 = [1, 1] \begin{bmatrix} 1 \\ 2 \end{bmatrix} = 1(1) + 1(2) = 3$

*Compare with Raw Dot Product ($q^T h_i$)*:
* $e_{\text{dot}, 1} = [1, 1] \cdot [1, 0] = 1$
* $e_{\text{dot}, 2} = [1, 1] \cdot [0, 1] = 1$
* $e_{\text{dot}, 3} = [1, 1] \cdot [1, 1] = 2$
* *Core Insight*: In raw dot product, $h_1$ and $h_2$ were tied at score 1. In general scoring, the learned weight $W_a$ multiplied the second coordinate by 2, ranking $h_2$ higher than $h_1$.

---

## 8. Bahdanau vs. Luong: Comparison & Architectural Differences

| Feature | Bahdanau Additive Attention | Luong Multiplicative Attention |
| :--- | :--- | :--- |
| **Scoring Formula** | $e_{ti} = v_a^T \tanh(W_q s_{t-1} + W_h h_i + b_a)$ | Dot: $s_t^T h_i$ <br> General: $s_t^T W_a h_i$ |
| **Query State** | **Previous** decoder state $s_{t-1}$ | **Current** decoder state $s_t$ |
| **Computation Order** | Query $\to$ Attention $\to$ Context $c_t$ $\to$ Recurrent Step $s_t$ | Recurrent Step $s_t$ $\to$ Query $\to$ Context $c_t$ $\to$ Attentional $\tilde{s}_t$ |
| **Combining Context** | $c_t$ fed directly into decoder recurrent cell | $[c_t \,;\, s_t]$ merged into attentional state $\tilde{s}_t = \tanh(W_c [c_t ; s_t])$ |
| **Learned Scoring Params** | $W_q, W_h, b_a, v_a$ (MLP layers) | None in Dot; $W_a$ in General |

---

## 9. Engineering Realities: Masking & Interpreting Heatmaps

### Handling Source Padding Positions (Check 16)
When batching sentences of different lengths, shorter sentences are padded with a `<PAD>` token:
* Source: `["The", "blue", "parcel", "<PAD>", "<PAD>"]`
* The decoder must not pay attention to `<PAD>` tokens.
* **Padding Mask**: Before computing softmax, set the pre-softmax alignment score of all padding tokens to $-\infty$ (or a large negative number like $-10^9$):
  $$e_{ti} = -\infty \quad \text{for } i \in \text{padding}$$
  $$\exp(-\infty) = 0 \implies \alpha_{ti} = 0$$
* The valid tokens are renormalized by softmax to sum to $1.0$.

### Interpreting Attention Heatmaps (Check 18)
An attention matrix can be plotted as a 2D heatmap showing alignment weights:
* If the heatmap shows a clean diagonal line:
  * **What it proves**: For these specific inputs, each target step places high attention weight on the corresponding source token position.
  * **What it does NOT prove**: It does **not** prove human-like reasoning, and does **not** prove that every generated translation is linguistically correct. Heatmaps visualize model computation, not truth.

---

## 10. Summary Checklist & Exam Must-Knows

- [ ] **Variable Length**: Source and target lengths can differ; the decoder emits tokens until $\langle\text{EOS}\rangle$.
- [ ] **Teacher Forcing**: During training, feed the previous *ground-truth* token $y_{t-1}$ to predict $y_t$.
- [ ] **Linear Bridge**: Maps encoder context to decoder initial state: $W_{\text{bridge}} \in \mathbb{R}^{H_d \times H_e}$ has $H_d$ rows.
- [ ] **Bottleneck**: Increasing source length from 5 to 50 still leaves exactly $H$ numbers in final state $h_n$.
- [ ] **Context Vector Calculation**: For $\alpha = [0.25, 0.75]$ and states $[[2, 0], [0, 4]]$, context is $c = [0.5, 3.0]^T$.
- [ ] **Equal Scores**: Three equal scores yield weight $1/3$ each; common score offsets don't change softmax weights.
- [ ] **Matrix Sums**: Attention rows sum to 1 (normalized over source); columns do not have to sum to 1.
- [ ] **Bahdanau Dimensions**: For $H_d = 3, H_e = 4, A = 5$, matrix $W_h$ has shape $5 \times 4$ ($5$ rows); context vector $c_t$ width remains $H_e = 4$.
- [ ] **Timing Comparison**: Bahdanau queries with previous state $s_{t-1}$; Luong queries with current state $s_t$.
- [ ] **Luong General Score**: For $q = [1, 1], h = [0, 1], W_a = [[1, 0], [0, 2]]$, score $q^T W_a h = 2$.
- [ ] **Masking Padding**: Masked padding tokens receive an attention weight of exactly $0$.
- [ ] **Heatmaps**: A diagonal heatmap reflects weight allocation during computation, not human reasoning or proof of correctness.
