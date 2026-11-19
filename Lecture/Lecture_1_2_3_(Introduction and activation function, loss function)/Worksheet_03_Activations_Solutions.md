# Deep Learning — Worksheet 03: ReLU, Leaky ReLU, ELU, Softmax, and Loss Functions

**Complete Solutions, Step-by-Step Numerical Walkthroughs & Comprehensive Guide**

---

## Part 1: Reference Formulas

- **Gradient Descent Update Rule**:
  $$
  w_{new} = w_{old} - \alpha (\hat{y} - y) f'(z) x \quad \text{where } z = wx + b
  $$
- **Sigmoid**:
  $$
  \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z)), \quad \sigma'_{max} = 0.25
  $$
- **Tanh**:
  $$
  \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}, \quad \tanh'(z) = 1 - \tanh^2(z), \quad \tanh'_{max} = 1.00
  $$

---

## Part 2: Sigmoid’s Real Failure — The Vanishing Gradient

### Numerical Demonstration of Severe Saturation

**Given Setup**: $x = 1, w = 6, b = 0, y = 0, \alpha = 1, e^{-6} = 0.00247875$.

1. **Step 1 — Weighted Sum**:

   $$
   z = wx + b = (6)(1) + 0 = \mathbf{6}
   $$
2. **Step 2 — Activation**:

   $$
   \hat{y} = \sigma(6) = \frac{1}{1 + e^{-6}} = \frac{1}{1 + 0.00247875} = \frac{1}{1.00247875} = \mathbf{0.99752738}
   $$
3. **Step 3 — Error**:

   $$
   \hat{y} - y = 0.99752738 - 0 = \mathbf{0.99752738} \quad (\mathbf{99.75\%} \text{ of maximum possible mistake})
   $$
4. **Step 4 — Derivative**:

   $$
   1 - \sigma(6) = 1 - 0.99752738 = \mathbf{0.00247262}
   $$

   $$
   \sigma'(6) = \sigma(6)(1 - \sigma(6)) = (0.99752738)(0.00247262) = \mathbf{0.00246651}
   $$
5. **Step 5 — Gradient**:

   $$
   \frac{\partial L}{\partial w} = (\hat{y} - y) \sigma'(z) x = (0.99752738)(0.00246651)(1) = \mathbf{0.00246041}
   $$
6. **Step 6 — Update**:

   $$
   w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} = 6 - (1)(0.00246041) = \mathbf{5.99753959}
   $$

---

### 2.1 The Vanishing Gradient Contradiction

| Metric                                                                                                 | Value                                               |
| :----------------------------------------------------------------------------------------------------- | :-------------------------------------------------- |
| **Error $(\hat{y} - y)$** | **$0.99752738$** *(Almost the largest mistake possible)* |                                                     |
| **Step Actually Taken**                                                                          | **$0.00246041$** *(Almost zero movement)* |

#### Fill-in Answers:

- The error is nearly the largest it can be ($0.99752738$), yet the step taken is only **$0.00246041$**.
- The factor responsible is **$\sigma'(z)$ (the derivative of the activation function)**, whose value here is **$0.00246651$**.
- Reading Figure 2, the reason that factor is so small is **the sigmoid function saturates (flattens out) for large $|z|$, so its slope approaches zero ($\sigma'(z) \approx 0$)**.

---

## Part 3: Tanh Does Not Fix the Vanishing Gradient

### Analysis of Derivative Profiles

- Sigmoid: $\sigma'(z) = \frac{e^{-z}}{(1+e^{-z})^2}$
- Tanh: $\tanh'(z) = 1 - \tanh^2(z) = \text{sech}^2(z)$
- **Crossing Point**: $\sigma'(z)$ and $\tanh'(z)$ intersect at **$z \approx 1.663$**.

#### Fill-in Answers:

1. Near $z = 0$, the steeper derivative belongs to **$\tanh$** ($\tanh'(0) = 1.00$ vs. $\sigma'(0) = 0.25$).
2. The two derivative curves cross at $z = \mathbf{1.663}$.
3. Beyond the crossing point ($|z| > 1.663$), the curve closer to zero belongs to **$\tanh$** (e.g., at $z = 6$, $\tanh'(6) \approx 0.000045 < \sigma'(6) \approx 0.002467$).
4. At $z = 6$, both derivatives are effectively **zero**.

> **Root Cause**: The **S-shape itself is the problem**. Any saturating S-shaped function is flat at both extremities, causing gradients to vanish for large activations regardless of rescaling.

---

## Part 4: ReLU (Rectified Linear Unit)

### 4.1 Definition & Derivative

$$
f(z) = \max(0, z) = \begin{cases} z & \text{if } z > 0 \\ 0 & \text{if } z \le 0 \end{cases}
$$

$$
f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0 & \text{if } z < 0 \end{cases} \quad (f'(0) \text{ is undefined, taken as } 0 \text{ in libraries})
$$

- **Constant Gradient**: For all $z > 0$, $f'(z) = 1$ constantly. It never shrinks, whether $z = 6$ or $z = 600$.

---

### 4.2 Architectural Placement: Why ReLU Cannot Be in the Final Layer

> **Question**: *Why can the final layer of a binary classifier not use ReLU?*
>
> **Answer**: ReLU outputs are unbounded ($[0, \infty)$), meaning $\text{ReLU}(6) = 6$, which is an uncalibrated score rather than a bounded binary decision or probability in $[0, 1]$. Therefore, ReLU is used in hidden layers, while the final layer uses **Sigmoid** (for binary) or **Softmax** (for multi-class).

---

### 4.3 Numerical Walkthrough on the Positive Side

**Given Setup** *(Identical to Part 2)*: $x = 1, w = 6, b = 0, y = 0, \alpha = 1$.

1. **Weighted Sum**: $z = (6)(1) + 0 = \mathbf{6}$
2. **Activation**: $\hat{y} = \max(0, 6) = \mathbf{6}$
3. **Error**: $\hat{y} - y = 6 - 0 = \mathbf{6}$
4. **Derivative**: $f'(6) = \mathbf{1}$
5. **Gradient**: $\frac{\partial L}{\partial w} = (\hat{y} - y) f'(z) x = (6)(1)(1) = \mathbf{6}$
6. **Update**: $w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} = 6 - (1)(6) = \mathbf{0}$
7. **Verification**: With $w_{new} = 0 \implies z = (0)(1) + 0 = 0 \implies \hat{y} = \max(0, 0) = 0 \implies \text{Error} = \mathbf{0}$.

#### Step Size Comparison: ReLU vs. Sigmoid

| Activation        | Derivative$f'(6)$ | Step Size Taken |           Problem Solved?           |
| :---------------- | :-----------------: | :-------------: | :---------------------------------: |
| **Sigmoid** |   $0.00246651$   | $0.00246041$ |      No (Negligible movement)      |
| **ReLU**    |   $1.00000000$   | $6.00000000$ | **Yes (Converged in 1 step)** |

$$
\frac{\text{ReLU Step}}{\text{Sigmoid Step}} = \frac{6}{0.00246041} \approx \mathbf{2{,}439\times \text{ larger}}
$$

---

## Part 5: ReLU on the Negative Side — The "Dying ReLU" Problem

### Numerical Walkthrough

**Given Setup**: $x = 1, w = -3, b = 0, y = 1, \alpha = 1$.

1. **Weighted Sum**: $z = wx + b = (-3)(1) + 0 = \mathbf{-3}$
2. **Activation**: $\hat{y} = \max(0, -3) = \mathbf{0}$
3. **Error**: $\hat{y} - y = 0 - 1 = \mathbf{-1}$
4. **Derivative**: $f'(-3) = \mathbf{0}$
5. **Gradient**: $\frac{\partial L}{\partial w} = (-1)(0)(1) = \mathbf{0}$
6. **Update**: $w_{new} = -3 - (1)(0) = \mathbf{-3}$ *(No change!)*

---

### 5.1 Analysis of the Dead Perceptron

- **Next Update**: $z = -3$, $\text{Gradient} = \mathbf{0}$.
- **Effect of $\alpha = 1{,}000{,}000$**: The step size remains **$0$**, because $(1{,}000{,}000) \times 0 = \mathbf{0}$.
- **Conclusion**: When $z < 0$, ReLU's gradient is **identically zero**. If a neuron is pushed into the negative regime for all dataset points, it can **never recover**; it is permanently "dead".

---

## Part 6: A Tempting Wrong Answer — The Identity Function

### 6.1 The Proposal: $f(z) = z$ Everywhere

- If $f(z) = z$, then $f'(z) = 1$ everywhere, eliminating dead neurons.

### 6.2 Why the Identity Function Fails

- Stacking multiple linear layers with identity activations collapses mathematically into a single linear transformation:
  $$
  \hat{y} = W_3(W_2(W_1 x + b_1) + b_2) + b_3 = W_{eff} x + b_{eff}
  $$
- A straight line can only produce flat hyperplanes. It **cannot create non-linear decision boundaries** and fails completely on non-linearly separable datasets (e.g., XOR).
- **Fill-in**: *The identity function fails because* **a composition of linear functions is strictly linear; stacking layers without non-linear activations cannot learn complex curved boundaries.**

---

## Part 7: Leaky ReLU

### 7.1 Definition & Derivative

$$
f(z) = \begin{cases} z & \text{if } z > 0 \\ 0.01z & \text{if } z \le 0 \end{cases} \quad\implies\quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ 0.01 & \text{if } z < 0 \end{cases}
$$

- **Checks**:
  - Compared with Identity: Slope is $0.01$ on the left and $1.0$ on the right, creating a **bend at $z=0$** (non-linear).
  - Compared with ReLU: On the negative side, derivative is **$0.01 \neq 0$**, preventing neurons from dying.

---

### 7.2 Numerical Walkthrough

**Given Setup**: $x = 1, w = -3, b = 0, y = 1, \alpha = 1$.

1. **Weighted Sum**: $z = (-3)(1) + 0 = \mathbf{-3}$
2. **Activation**: $\hat{y} = 0.01(-3) = \mathbf{-0.03}$
3. **Error**: $\hat{y} - y = -0.03 - 1 = \mathbf{-1.03}$
4. **Derivative**: $f'(-3) = \mathbf{0.01}$
5. **Gradient**: $\frac{\partial L}{\partial w} = (-1.03)(0.01)(1) = \mathbf{-0.0103}$
6. **Update**: $w_{new} = -3 - (1)(-0.0103) = \mathbf{-2.9897}$

#### Two Weaknesses of Leaky ReLU:

1. The negative output was only **$-0.03$** (barely different from zero), causing the weight to update by only $0.0103$ ("crawling").
2. At $z = 0$, the derivative has a jump discontinuity from **$0.01$ to $1.00$** (non-smooth).

---

## Part 8: ELU (Exponential Linear Unit)

### 8.1 Definition & Smooth Derivative

$$
f(z) = \begin{cases} z & \text{if } z > 0 \\ e^z - 1 & \text{if } z \le 0 \end{cases} \quad\implies\quad f'(z) = \begin{cases} 1 & \text{if } z > 0 \\ e^z & \text{if } z < 0 \end{cases}
$$

- **Properties**:
  - As $z \to -\infty$, $e^z \to 0 \implies f(z) \to \mathbf{-1}$ (bounded negative saturation floor).
  - At $z = 0$: $f'(0^+) = 1.0$ and $f'(0^-) = e^0 = 1.0$. The derivative is **continuous everywhere**.

---

### 8.2 Numerical Walkthrough

**Given Setup**: $x = 1, w = -3, b = 0, y = 1, \alpha = 1, e^{-3} = 0.049787$.

1. **Weighted Sum**: $z = (-3)(1) + 0 = \mathbf{-3}$
2. **Activation**: $\hat{y} = e^{-3} - 1 = 0.049787 - 1 = \mathbf{-0.950213}$
3. **Error**: $\hat{y} - y = -0.950213 - 1 = \mathbf{-1.950213}$
4. **Derivative**: $f'(-3) = e^{-3} = \mathbf{0.049787}$
5. **Gradient**: $\frac{\partial L}{\partial w} = (-1.950213)(0.049787)(1) = \mathbf{-0.097095}$
6. **Update**: $w_{new} = -3 - (1)(-0.097095) = \mathbf{-2.902905}$

---

### 8.3 Verdict on the Negative Side ($z = -3$)

| Activation           | Output$\hat{y}$ | Derivative$f'(-3)$ | Gradient$\frac{\partial L}{\partial w}$ |  $w_{new}$  | Verdict                           |
| :------------------- | :---------------: | :------------------: | :---------------------------------------: | :-----------: | :-------------------------------- |
| **ReLU**       |       $0$       |        $0$        |                   $0$                   | $-3.000000$ | **Dead**                    |
| **Identity**   |      $-3$      |        $1$        |                  $-4$                  | $+1.000000$ | Alive, but network remains linear |
| **Leaky ReLU** |     $-0.03$     |       $0.01$       |                $-0.0103$                | $-2.989700$ | Alive, but crawling slowly        |
| **ELU**        |   $-0.950213$   |     $0.049787$     |               $-0.097095$               | $-2.902905$ | **Alive, moving properly**  |

$$
\frac{\text{ELU Step}}{\text{Leaky ReLU Step}} = \frac{0.097095}{0.0103} \approx \mathbf{9.4\times \text{ larger}}
$$

- **Fill-in**: *ELU takes a step about **$9.4$** times larger than Leaky ReLU on the same problem, because its negative-side output is **$-0.950213$** rather than nearly zero, and its derivative there is **$0.049787$** rather than a small fixed constant.*

---

## Part 9: Comprehensive Comparison

### 9.1 Mathematical Summary

| Function             |                Formula$f(z)$                |             Derivative$f'(z)$             |     Output Range     | Max$f'(z)$ |
| :------------------- | :--------------------------------------------: | :-----------------------------------------: | :-------------------: | :----------: |
| **Threshold**  |   $1 \text{ if } z \ge 0, \text{ else } 0$   |       $0 \quad (\forall z \neq 0)$       |     $\{0, 1\}$     |    $0$    |
| **Sigmoid**    |            $\frac{1}{1 + e^{-z}}$            |        $\sigma(z)(1 - \sigma(z))$        |      $(0, 1)$      |   $0.25$   |
| **Tanh**       |     $\frac{e^z - e^{-z}}{e^z + e^{-z}}$     |             $1 - \tanh^2(z)$             |      $(-1, 1)$      |   $1.00$   |
| **ReLU**       |                 $\max(0, z)$                 |  $1 \text{ if } z > 0, \text{ else } 0$  |    $[0, \infty)$    |   $1.00$   |
| **Leaky ReLU** |  $z \text{ if } z > 0, \text{ else } 0.01z$  | $1 \text{ if } z > 0, \text{ else } 0.01$ | $(-\infty, \infty)$ |   $1.00$   |
| **ELU**        | $z \text{ if } z > 0, \text{ else } e^z - 1$ | $1 \text{ if } z > 0, \text{ else } e^z$ |   $(-1, \infty)$   |   $1.00$   |

---

### 9.2 Practical Application Guide

| Function             | Primary Advantage                                                    | Main Disadvantage                                    | Typical Real-World Use Case                           |
| :------------------- | :------------------------------------------------------------------- | :--------------------------------------------------- | :---------------------------------------------------- |
| **Threshold**  | Simple binary output                                                 | Derivative is 0; untrainable                         | Historical (1958 Rosenblatt Perceptron)               |
| **Sigmoid**    | Directly outputs valid probability                                   | Severe vanishing gradient; max$f' = 0.25$          | Binary classification output layer; LSTM gating units |
| **Tanh**       | Zero-centered; steeper than sigmoid                                  | Saturates and vanishes for$|z| > 1.663$            | RNN / LSTM hidden state updates                       |
| **ReLU**       | No vanishing gradient for$z > 0$; extremely fast compute           | Dying ReLU for$z < 0$; unbounded                   | Default hidden layer in CNNs & MLPs                   |
| **Leaky ReLU** | Never dies permanently; very fast                                    | Heuristic fixed negative slope; non-smooth jump at 0 | GANs (Generative Adversarial Networks)                |
| **ELU**        | Smooth everywhere; non-zero negative outputs                         | Requires computing exponential$e^z$                | Deep architectures prone to dying ReLU                |
| **Softmax**    | Converts raw score vector to normalized probabilities ($\sum = 1$) | Defined across layer vector, not single scalar       | Multi-class classification output layer               |

---

## Part 10: Beyond Binary — The Softmax Activation

### 10.1 Definition & Properties

For a final layer of $n$ classes with logits $z = [z_1, z_2, \dots, z_n]^T$:

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^n e^{z_j}}
$$

- **Two Invariants**:
  1. $0 < p_i < 1$ for all $i$.
  2. $\sum_{i=1}^n p_i = 1$.

---

### 10.2 Worked Numerical (3 Classes: Cat, Dog, Bird)

**Given Logits**: $z_1 = 2.0$ (Cat), $z_2 = 1.0$ (Dog), $z_3 = 0.1$ (Bird).

1. **Step 1 — Exponentiate Each Score**:
   - $e^{z_1} = e^{2.0} = \mathbf{7.389056}$
   - $e^{z_2} = e^{1.0} = \mathbf{2.718282}$
   - $e^{z_3} = e^{0.1} = \mathbf{1.105171}$
2. **Step 2 — Sum of Exponentials**:
   $$
   \sum_{j=1}^3 e^{z_j} = 7.389056 + 2.718282 + 1.105171 = \mathbf{11.212509}
   $$
3. **Step 3 — Compute Probabilities ($p_i = \frac{e^{z_i}}{\sum e^{z_j}}$)**:
   - $p_1 (\text{Cat}) = \frac{7.389056}{11.212509} = \mathbf{0.659001} \quad (65.90\%)$
   - $p_2 (\text{Dog}) = \frac{2.718282}{11.212509} = \mathbf{0.242433} \quad (24.24\%)$
   - $p_3 (\text{Bird}) = \frac{1.105171}{11.212509} = \mathbf{0.098566} \quad (9.86\%)$
4. **Step 4 — Sum Check**:
   $$
   0.659001 + 0.242433 + 0.098566 = \mathbf{1.000000} \quad \checkmark
   $$
5. **Step 5 — Final Decision**:
   - Class **Cat** wins with **$65.9\%$ confidence**.

---

### 10.3 Reduction: Softmax with 2 Classes is Sigmoid

For $n = 2$ with scores $z_1 = z$ and $z_2 = 0$:

$$
\text{softmax}(z_1) = \frac{e^z}{e^z + e^0} = \frac{e^z}{e^z + 1}
$$

Divide numerator and denominator by $e^z$:

$$
\frac{e^z / e^z}{(e^z + 1) / e^z} = \frac{1}{1 + e^{-z}} = \mathbf{\sigma(z)}
$$

> **Conclusion**: Binary Sigmoid classification is a mathematically exact special case of multi-class Softmax.

---

### 10.4 Architectural Summary & Check Question

| Task Type                                            | Recommended Final Layer Activation          | Output Layer Width |
| :--------------------------------------------------- | :------------------------------------------ | :----------------: |
| **Binary Classification**                      | **Sigmoid** ($\sigma$)              |     $1$ unit     |
| **Multi-Class Classification** ($K$ classes) | **Softmax**                           |    $K$ units    |
| **Hidden Layers**                              | **ReLU** (or Leaky ReLU / ELU / GELU) |     Arbitrary     |

> **Question**: *A network classifies images into 10 digits ($0$ to $9$). How many units are in the final layer, and which activation function do they use?*
>
> **Answer**:
>
> - **Number of units in the final layer**: **$10$ units** (one per digit class).
> - **Activation function**: **Softmax** (to produce a probability distribution summing to $1.0$).

---

## Part 11: Loss Functions (Complete Official Section from Worksheet)

### 11.1 The Loss We Have Been Using

Every update so far has used one loss:

$$
L = \frac{1}{2}(\hat{y} - y)^2
$$

- That was a design choice, not a fundamental law. The loss measures how bad a prediction is, and different tasks require different metrics. Over a dataset of $n$ examples, we average across all examples.
- **Core Principle**: *The activation function decides what the network can output. The loss function decides how we score it. They must always be chosen as a pair.*

---

### 11.2 Mean Squared Error (MSE)

$$
\text{MSE} = \frac{1}{n}\sum_{i=1}^n (\hat{y}_i - y_i)^2 = \frac{1}{n}\sum_{i=1}^n e_i^2
$$

*Square each error, then average. Squaring removes the sign and magnifies large errors.*

#### Worked Activity: Given Four Predictions ($n = 4$)

| Example$i$ | Prediction$\hat{y}_i$ | Target$y_i$ | Error$e_i = \hat{y}_i - y_i$ |  Squared Error$e_i^2$  | Absolute Error$|e_i|$ |
| :----------: | :---------------------: | :-----------: | :----------------------------: | :----------------------: | :---------------------: |
| **1** |          $2$          |     $2$     |             $0$             | $0$ | **$0$** |                        |
| **2** |          $4$          |     $5$     |             $-1$             | $1$ | **$1$** |                        |
| **3** |          $6$          |     $5$     |             $1$             | $1$ | **$1$** |                        |
| **4** |         $12$         |     $6$     |             $6$             | $36$ | **$6$** |                        |

- **Step 1 (Add Squared Errors)**:
  $$
  \sum e_i^2 = 0 + 1 + 1 + 36 = \mathbf{38}
  $$
- **Step 2 (Divide by $n = 4$)**:
  $$
  \text{MSE} = \frac{38}{4} = \mathbf{9.5}
  $$

> **Key Observation**: The fourth example alone contributes $36$ out of the total $38$, which is **$94.7\%$ of the entire loss**. One single bad prediction (outlier) dominates the training signal.

---

### 11.3 Mean Absolute Error (MAE)

$$
\text{MAE} = \frac{1}{n}\sum_{i=1}^n |\hat{y}_i - y_i| = \frac{1}{n}\sum_{i=1}^n |e_i|
$$

*Take the magnitude of each error, then average. No squaring, so large errors are not disproportionately magnified.*

- **Step 1 (Add Absolute Errors)**:
  $$
  \sum |e_i| = 0 + 1 + 1 + 6 = \mathbf{8}
  $$
- **Step 2 (Divide by $n = 4$)**:
  $$
  \text{MAE} = \frac{8}{4} = \mathbf{2.0}
  $$

---

### The Effect of an Outlier: Comparing All 4 vs. First 3 Examples

| Dataset Subset                               |                         MSE                         |                   MAE                   |
| :------------------------------------------- | :--------------------------------------------------: | :-------------------------------------: |
| **All four examples**                  |                     $9.500000$                     |              $2.000000$              |
| **First three only (without outlier)** |       $\frac{0+1+1}{3} = \mathbf{0.666667}$       | $\frac{0+1+1}{3} = \mathbf{0.666667}$ |
| **Growth Factor with Outlier**         | **$14.25\times$** | **$3.00\times$** |                                        |

#### ✏️ Fill-in Answers:

- With no outlier, the two losses are **identical (both $0.666667$)**.
- Adding one outlier multiplied MSE by **$14.25\times$**, but MAE by only **$3.00\times$**.

> **Guideline**:
>
> - Use **MSE** when large errors genuinely matter more and severe penalties are desired.
> - Use **MAE** when data contains noisy outliers that should not dominate training.

---

## 11.4 Binary Cross Entropy (BCE)

MSE and MAE measure geometric distance. For classification, the output is a calibrated probability in $(0, 1)$, requiring an information-theoretic log loss:

$$
\text{BCE} = -\Big[y \ln(\hat{y}) + (1 - y)\ln(1 - \hat{y})\Big]
$$

Since target $y \in \{0, 1\}$, only one term survives:

- If $y = 1$: $\text{BCE} = -\ln(\hat{y})$
- If $y = 0$: $\text{BCE} = -\ln(1 - \hat{y})$

*(BCE pairs canonically with **Sigmoid** in the final layer, ensuring $0 < \hat{y} < 1$ so logarithms are strictly defined).*

---

### Numerical: Confident & Right vs. Confident & Wrong ($y = 1$)

| Prediction$\hat{y}$ |   Qualitative Assessment   |   $\text{BCE} = -\ln(\hat{y})$   |   $\text{MSE} = (\hat{y} - 1)^2$   |
| :-------------------: | :-------------------------: | :---------------------------------: | :-----------------------------------: |
|  $\hat{y} = 0.900$  |      Confident & Right      | $-\ln(0.900) = \mathbf{0.105361}$ |  $(0.9 - 1)^2 = \mathbf{0.010000}$  |
|  $\hat{y} = 0.100$  |            Wrong            | $-\ln(0.100) = \mathbf{2.302585}$ |  $(0.1 - 1)^2 = \mathbf{0.810000}$  |
|  $\hat{y} = 0.010$  |         Very Wrong         | $-\ln(0.010) = \mathbf{4.605170}$ | $(0.01 - 1)^2 = \mathbf{0.980100}$ |
|  $\hat{y} = 0.001$  | Extremely Confident & Wrong | $-\ln(0.001) = \mathbf{6.907755}$ | $(0.001 - 1)^2 = \mathbf{0.998001}$ |

#### ✏️ Fill-in Answers:

- As $\hat{y} \to 0$ with true target $y = 1$, BCE **grows without limit ($\to \infty$)**, while MSE **approaches $1.0$ (flattens off)**.

---

### Why This Matters: The Payoff of the Whole Worksheet

Consider the saturated perceptron from Part 2 ($x = 1, w = 6, b = 0, y = 0, \alpha = 1$):

- Pre-activation $z = 6$
- Prediction $\hat{y} = \sigma(6) = 0.99752738$
- Derivative $\sigma'(6) = 0.00246651$

#### 1. Under MSE:

- Gradient:
  $$
  \frac{\partial L_{MSE}}{\partial w} = (\hat{y} - y)\sigma'(z)x = (0.99752738)(0.00246651)(1) = \mathbf{0.00246041}
  $$
- Parameter Update:
  $$
  w_{new} = 6 - (1)(0.00246041) = \mathbf{5.99753959} \quad (\text{Step size } = \mathbf{0.00246041})
  $$

#### 2. Under BCE (Derivative Factor Cancels Exactly!):

$$\frac{\partial L_{BCE}}{\partial w} = (\hat{y} - y) \cdot x$$

- Gradient:
  $$\frac{\partial L_{BCE}}{\partial w} = (0.99752738 - 0)(1) = \mathbf{0.99752738}$$
- Parameter Update:
  $$w_{new} = 6 - (1)(0.99752738) = \mathbf{5.00247262} \quad (\text{Step size } = \mathbf{0.99752738})$$

---

### 📐 Step-by-Step Mathematical Derivation of the BCE Gradient Cancellation

**How do we get $\frac{\partial L_{\text{BCE}}}{\partial w} = (\hat{y} - y) \cdot x$?**

The forward chain of operations is:
$$w \xrightarrow{\text{linear}} z = wx + b \xrightarrow{\text{Sigmoid}} \hat{y} = \sigma(z) \xrightarrow{\text{BCE Loss}} L_{\text{BCE}} = -\Big[y \ln(\hat{y}) + (1 - y) \ln(1 - \hat{y})\Big]$$

By the Chain Rule:
$$\frac{\partial L_{\text{BCE}}}{\partial w} = \underbrace{\frac{\partial L_{\text{BCE}}}{\partial \hat{y}}}_{\text{Link 1}} \cdot \underbrace{\frac{\partial \hat{y}}{\partial z}}_{\text{Link 2}} \cdot \underbrace{\frac{\partial z}{\partial w}}_{\text{Link 3}}$$

#### Step 1: Differentiate Loss with respect to Prediction $\hat{y}$ (Link 1)
Using $\frac{d}{du}\ln(u) = \frac{1}{u}$ and $\frac{d}{du}\ln(1-u) = -\frac{1}{1-u}$:
$$\frac{\partial L_{\text{BCE}}}{\partial \hat{y}} = -\left[ \frac{y}{\hat{y}} - \frac{1 - y}{1 - \hat{y}} \right] = -\left[ \frac{y(1 - \hat{y}) - (1 - y)\hat{y}}{\hat{y}(1 - \hat{y})} \right]$$

Expanding the numerator:
$$y(1 - \hat{y}) - (1 - y)\hat{y} = y - y\hat{y} - \hat{y} + y\hat{y} = y - \hat{y}$$

Therefore:
$$\frac{\partial L_{\text{BCE}}}{\partial \hat{y}} = -\left[\frac{y - \hat{y}}{\hat{y}(1 - \hat{y})}\right] = \mathbf{\frac{\hat{y} - y}{\hat{y}(1 - \hat{y})}}$$

#### Step 2: Differentiate Sigmoid Output with respect to $z$ (Link 2)
$$\frac{\partial \hat{y}}{\partial z} = \sigma'(z) = \sigma(z)(1 - \sigma(z)) = \mathbf{\hat{y}(1 - \hat{y})}$$

#### Step 3: Combine Link 1 and Link 2 (The Algebraic Cancellation)
$$\frac{\partial L_{\text{BCE}}}{\partial z} = \frac{\partial L_{\text{BCE}}}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \left[ \frac{\hat{y} - y}{\cancel{\hat{y}(1 - \hat{y})}} \right] \cdot \cancel{\hat{y}(1 - \hat{y})} = \mathbf{(\hat{y} - y)}$$

#### Step 4: Differentiate $z$ with respect to Weight $w$ (Link 3)
$$\frac{\partial z}{\partial w} = \frac{\partial}{\partial w}(wx + b) = \mathbf{x}$$

#### Multiplying All Three Links:
$$\boxed{\frac{\partial L_{\text{BCE}}}{\partial w} = (\hat{y} - y) \cdot x}$$

---

### Step Size Comparison: MSE vs. BCE

| Loss Function |                  Step Size Taken                  | Multiplier Ratio |
| :------------ | :------------------------------------------------: | :--------------: |
| **MSE** |                   $0.00246041$                   |   $1\times$   |
| **BCE** | $0.99752738$ | **$405.43\times$ larger** |                  |

$$
\text{Step Ratio} = \frac{0.99752738}{0.00246041} \approx \mathbf{405.43} = \frac{1}{\sigma'(6)}
$$

> 🌟 **Key Takeaway**: The activation function did not change; only the loss function did. **Choosing BCE instead of MSE completely removes the vanishing gradient problem at the output layer.**

---

## 11.5 Categorical Cross Entropy (CCE)

Extends Cross-Entropy from binary to multi-class classification, pairing with **Softmax**:

$$
\text{CCE} = -\sum_{i=1}^K y_i \ln(p_i)
$$

For one-hot target vectors ($y_i = 1$ for true class, $0$ elsewhere), all zero terms vanish:

$$
\text{CCE} = -\ln(p_{\text{correct class}})
$$

---

### Worked Numerical (Using Part 10 Softmax Probabilities)

**Given**: $p_1 = 0.659001$ (Cat), $p_2 = 0.242433$ (Dog), $p_3 = 0.098566$ (Bird).
*(Useful values: $\ln(0.659001) = -0.417030$, $\ln(0.098566) = -2.317029$)*.

1. **Case A (Target is Cat: $y = [1, 0, 0]$)**:
   $$
   \text{CCE} = -[(1)\ln(0.659001) + (0) + (0)] = -\ln(0.659001) = \mathbf{0.417030} \quad (\text{Low loss, model was right})
   $$
2. **Case B (Target is Bird: $y = [0, 0, 1]$)**:
   $$
   \text{CCE} = -[(0) + (0) + (1)\ln(0.098566)] = -\ln(0.098566) = \mathbf{2.317029} \quad (\text{High loss, model was wrong})
   $$

#### ✏️ Fill-in Answers:

- The network made the identical prediction vector in both cases, yet the loss was **$0.417030$** when it was right and **$2.317029$** when it was wrong.
- The loss depends **only on the probability allocated to the correct class**.
- With two classes, CCE reduces exactly to BCE (just as Softmax reduces to Sigmoid).

---

## 11.6 Comprehensive Guide: Which Loss Pairs with Which Final Layer

| Task                                              | Final Activation                                                      | Canonical Loss Function                   | Mathematical / Practical Reason                                                          |
| :------------------------------------------------ | :-------------------------------------------------------------------- | :---------------------------------------- | :--------------------------------------------------------------------------------------- |
| **Predicting a continuous number**          | **None** (Linear), or **ReLU** (if strictly non-negative) | **Mean Squared Error (MSE)**        | Large errors should be punished quadratically more.                                      |
| **Predicting a number (Data has outliers)** | **None** (Linear)                                               | **Mean Absolute Error (MAE)**       | Stops one anomalous outlier point from dominating gradient updates.                      |
| **Binary Classification**                   | **Sigmoid** ($\sigma$)                                        | **Binary Cross Entropy (BCE)**      | Removes the$\sigma'(z)$ factor, preventing vanishing gradients at the output.          |
| **Multi-Class Classification**              | **Softmax**                                                     | **Categorical Cross Entropy (CCE)** | Directly evaluates and optimizes the log probability assigned to the ground-truth class. |

---

## Summary of All Four Loss Functions

| Loss Function                             |                      Formula                      | Key Characteristic                                                                          |
| :---------------------------------------- | :------------------------------------------------: | :------------------------------------------------------------------------------------------ |
| **Mean Squared Error (MSE)**        |      $\frac{1}{n}\sum (\hat{y}_i - y_i)^2$      | Magnifies large errors quadratically.                                                       |
| **Mean Absolute Error (MAE)**       |      $\frac{1}{n}\sum \|\hat{y}_i - y_i\|$      | Treats all error distances linearly in proportion.                                          |
| **Binary Cross Entropy (BCE)**      | $-\Big[y \ln \hat{y} + (1-y)\ln(1-\hat{y})\Big]$ | Unbounded logarithmic penalty for confident mistakes; eliminates output vanishing gradient. |
| **Categorical Cross Entropy (CCE)** |               $-\sum y_i \ln p_i$               | Multi-class extension of BCE; scores the probability of the correct class.                  |

---

### ❓ Final Worksheet Question

> **Question**: *A network classifies chest X-rays as normal or pneumonia. Name the final-layer activation, the number of output units, and the loss function.*
>
> **Answer**:
>
> - **Final-Layer Activation**: **Sigmoid** ($\sigma$)
> - **Number of Output Units**: **$1$ unit** (representing $P(\text{pneumonia})$)
> - **Loss Function**: **Binary Cross Entropy (BCE)**
