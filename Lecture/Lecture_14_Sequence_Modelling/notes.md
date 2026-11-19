# Lecture 14: Sequence Modelling — Remember What Came Before

> **Worksheet File**: [Open Lecture 14 Interactive Worksheet](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_14_Sequence_Modelling/Lecture_14_Sequence_Modelling.html)  
> **Core Theme**: Why standard feed-forward networks fail on sequences, how hidden state memory processes dynamic lengths, time unrolling, and the mathematical roots of vanishing/exploding gradients.

---

## 1. The Basics: Why Sequential Data is Different

### The Motivating Hook: The Word "ORCHID"
Imagine someone shows you the letters **O-R-C-H-I-D** one by one. If you only look at one letter at a time without remembering what came before, the final letter "D" tells you nothing about the word. To understand sequences, a model must maintain a **running memory** that updates as new tokens arrive.

### Why Standard Neural Networks Fail
1. **Dynamic Length ($T$):** Standard MLPs expect fixed-size inputs (e.g., a vector of exactly 100 numbers). Natural language sentences and audio clips vary in length from 2 words to 200 words.
2. **Order is Evidence:** In a sequence $X = (x_1, x_2, \dots, x_T)$, order carries meaning:
   $$\text{"not bad, quite good"} \quad \neq \quad \text{"not good, quite bad"}$$
3. **Failure of Order-Blind Summaries:** An average pooling operation:
   $$m = \frac{1}{T} \sum_{t=1}^T x_t$$
   is **permutation-invariant**. Reordering words yields the exact same average vector $m$, completely destroying grammar, word order, and negation.

---

## 2. Sequence Tasks & Topologies

Depending on what we want to predict, sequence models are configured into three primary problem types:

```
1. Sequence Label (Many-to-One):        2. Aligned Output (Many-to-Many):        3. Next-Token (Autoregressive):
      ŷ = g(h_T)                               ŷ_1   ŷ_2   ŷ_T                         x_2   x_3   x_{T+1}
          ▲                                     ▲     ▲     ▲                           ▲     ▲       ▲
          │                                     │     │     │                           │     │       │
       [ h_T ]                               [ h_1][ h_2][ h_T ]                     [ h_1][ h_2]  [ h_T ]
          ▲                                     ▲     ▲     ▲                           ▲     ▲       ▲
          │                                     │     │     │                           │     │       │
    x_1 ... x_T                                x_1   x_2   x_T                         x_1   x_2     x_T
(e.g., Sentiment Analysis)            (e.g., Part-of-Speech Tagging)           (e.g., Language Generation)
```

1. **Sequence Label:** $\hat{y} = g(h_T)$. The model reads the entire sequence and outputs a single label from the final hidden state $h_T$ (e.g., classifying a movie review as positive or negative).
2. **Aligned Output:** $\hat{y}_t = g(h_t)$. The model predicts a label at *every* step $t$ matching the input length (e.g., labelling each word as a noun, verb, or adjective).
3. **Next-Token Prediction:** $p(x_{t+1} \mid x_1, \dots, x_t)$. The model predicts the next word given the past context.

### Tensor Layouts in Code
In deep learning libraries, a batch of sequences of length $T$ with feature dimension $d$ and batch size $B$ is typically stored as:
* **Batch-first:** $B \times T \times d$ (common in PyTorch standard modules)
* **Time-first:** $T \times B \times d$ (often used in optimized recurrent loops)

---

## 3. The Recurrent Update & Hidden State Memory

### The Core Update Formula
At each time step $t$, the Recurrent Neural Network (RNN) updates its internal memory vector $h_t \in \mathbb{R}^H$ by combining the new input $x_t \in \mathbb{R}^d$ with the previous memory $h_{t-1} \in \mathbb{R}^H$:
$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b_h)$$
$$z_t^{\text{out}} = W_y h_t + b_y$$

* $W_x$: Input-to-hidden weight matrix (shape $H \times d$).
* $W_h$: Hidden-to-hidden recurrent transition matrix (shape $H \times H$).
* $b_h$: Hidden bias vector (shape $H$).
* $W_y$: Hidden-to-output matrix (shape $C \times H$).
* $b_y$: Output bias vector (shape $C$).
* $\tanh$: Squeezes activations smoothly into $[-1, 1]$, keeping values bounded.

### Exact Numerical Walkthrough (From the Worksheet)
Suppose feature size $d = 3$, hidden size $H = 2$, input $x_t = [1, 0, 0]^T$, previous state $h_{t-1} = [0.2, -0.1]^T$, and biases are zero ($b_h = [0, 0]^T$):
* Input matrix: $W_x = \begin{bmatrix} 0.5 & 0.0 & 0.0 \\ -0.3 & 0.0 & 0.0 \end{bmatrix} \implies W_x x_t = \begin{bmatrix} 0.5 \\ -0.3 \end{bmatrix}$
* Recurrent matrix: $W_h = \begin{bmatrix} 0.8 & 0.1 \\ -0.2 & 0.7 \end{bmatrix} \implies W_h h_{t-1} = \begin{bmatrix} 0.8(0.2) + 0.1(-0.1) \\ -0.2(0.2) + 0.7(-0.1) \end{bmatrix} = \begin{bmatrix} 0.15 \\ -0.11 \end{bmatrix}$
* Sum before activation: $\begin{bmatrix} 0.5 + 0.15 \\ -0.3 - 0.11 \end{bmatrix} = \begin{bmatrix} 0.65 \\ -0.41 \end{bmatrix}$
* New hidden state:
  $$h_t = \begin{bmatrix} \tanh(0.65) \\ \tanh(-0.41) \end{bmatrix} \approx \begin{bmatrix} 0.57167 \\ -0.38847 \end{bmatrix}$$

---

## 4. Unrolling in Time & Parameter Counting

### Unrolling: One Rule Across Time
When we unroll an RNN across $T$ steps:
$$h_1 = f_\theta(x_1, h_0), \quad h_2 = f_\theta(x_2, h_1), \quad h_3 = f_\theta(x_3, h_2), \dots$$
* **Crucial Rule:** The exact same weight matrices $W_x, W_h, W_y$ are reused at every time step $t=1 \dots T$.
* The number of parameters **does not depend on the sequence length $T$**!

### Exact Parameter Counting Formula
Total parameters $P$ for an RNN cell with input dimension $d$, hidden state $H$, and output classes $C$:
$$P = Hd + H^2 + H + CH + C$$

#### Worked Example from Worksheet ($d = 3, H = 2, C = 2$):
| Weight / Tensor | Math Shape | Dimensions ($d=3, H=2, C=2$) | Number of Parameters |
| :--- | :---: | :---: | :---: |
| **Input Weights $W_x$** | $H \times d$ | $2 \times 3$ | **6** |
| **Recurrent Weights $W_h$** | $H \times H$ | $2 \times 2$ | **4** |
| **State Bias $b_h$** | $H$ | $2$ | **2** |
| **Output Weights $W_y$** | $C \times H$ | $2 \times 2$ | **4** |
| **Output Bias $b_y$** | $C$ | $2$ | **2** |
| **Total Parameters** | | | **18** |

### Gradient Accumulation Across Time
Because parameters are shared across steps, the total gradient for any shared weight $\theta$ is the sum of its contributions across all time steps:
$$\nabla_\theta \mathcal{L} = \sum_{t=1}^T \nabla_{\theta}^{(t)} \mathcal{L}$$

---

## 5. Long Dependencies & The Vanishing / Exploding Gradient Problem

### The Mathematical Cause
To compute how an early hidden state $h_0$ influenced a late state $h_L$, we use the chain rule across the temporal Jacobian matrices $J_t = \frac{\partial h_t}{\partial h_{t-1}}$:
$$\frac{\partial h_L}{\partial h_0} = J_L \cdot J_{L-1} \cdots J_2 \cdot J_1 = \prod_{t=1}^L J_t$$

### The Scalar Demonstration (From the Worksheet)
Consider a simple linear scalar RNN where $h_t = w h_{t-1}$:
$$\frac{\partial h_L}{\partial h_0} = w^L$$

1. **Vanishing Gradient ($w < 1$, e.g. $w = 0.8$ across $L = 20$ steps):**
   $$\frac{\partial h_{20}}{\partial h_0} = (0.8)^{20} \approx \mathbf{0.011529} \quad (\approx 1.15\% \text{ of original signal})$$
   * *Consequence:* By step 20, the gradient has vanished by ~99%. Errors at step 20 cannot send any learning signal back to step 0. The network suffers from short-term memory loss.
2. **Exploding Gradient ($w > 1$, e.g. $w = 1.2$ across $L = 20$ steps):**
   $$\frac{\partial h_{20}}{\partial h_0} = (1.2)^{20} \approx \mathbf{38.3376} \quad (> 3800\% \text{ growth})$$
   * *Consequence:* Gradients blow up exponentially, causing numerical overflow and training collapse.

### With Non-Linear $\tanh$
For $h_t = \tanh(w h_{t-1})$:
$$J_t = w \cdot (1 - h_t^2)$$
Because the derivative of $\tanh$ is $(1 - h_t^2) \le 1$, whenever neurons saturate ($|h_t| \approx 1$), the derivative approaches $0$, accelerating vanishing gradients even further.

---

## 6. What Helps — And What Remains Unresolved?

The worksheet provides a direct comparison of partial remedies:

| Remedy | What It Changes | What It Leaves Unresolved |
| :--- | :--- | :--- |
| **Gradient Clipping** | Rescales the gradient vector if its norm exceeds threshold $\theta$. | Prevents exploding gradients, but **cannot restore a vanished gradient**. |
| **Truncated BPTT** | Splits the sequence into smaller windows (e.g. 20 steps) for backprop. | Speeds up training, but **discards dependencies longer than the window**. |
| **Orthogonal Initialization** | Initializes $W_h$ such that its eigenvalues have magnitude 1. | Stabilizes linear signal propagation, but **non-linear activations ($\tanh$) still saturate**. |

### The Bridge to Lecture 15:
To genuinely solve vanishing gradients over long horizons, we cannot rely on multiplicative recurrent updates. We need **gated architectures (LSTM & GRU)** that provide a linear, additive error highway!
