# Deep Learning — Worksheet 06: Optimisers With Memory

**Complete Solutions, Mathematical Derivations & Diagnostic Guide**

---

## Overview & Progression of Optimizers

Plain Stochastic Gradient Descent (SGD) reacts only to the current mini-batch gradient and retains zero memory across steps. Modern optimizers build stateful memories to improve convergence speed, stability, and navigation through ill-conditioned curvature.

```
       Plain SGD (Reacts to 1 batch, stateless)
          │
          ▼
       EWMA (Exponentially Weighted Moving Average)
          │
    ┌─────┴─────────────────────────┐
    ▼                               ▼
Direction Memory (1st Moment)   Scale Memory (2nd Moment)
    │                               │
    ├─► Momentum (Velocity)         ├─► AdaGrad (Cumulative sum g²)
    │                               │
    └─► NAG (Look-ahead evaluation) └─► RMSProp (Fading EWMA g²)
    │                               │
    └───────────────┬───────────────┘
                    ▼
           Adam (1st + 2nd Moment + Bias Correction)
                    │
                    ▼
           AdamW (Decoupled Weight Decay)
```

---

## Part 0: The Baseline — SGD Reacts to One Estimate

### 0.1 Update Rule

$$
\theta_{t+1} = \theta_t - \alpha g_t \quad \text{where } g_t = \nabla_\theta L(\theta_t)
$$

- **Analogy**: Steering a boat by looking only at the newest surface ripple.
- **Limitation**: Highly sensitive to mini-batch noise; zig-zags violently across ravine walls with high curvature.

---

#### 🔮 Opening Prediction

> **Question**: *Three mini-batches give gradients $+4, +4, -1$. Should a sensible optimiser fully reverse direction after the final batch?*
>
> **Answer**: **No.** The two successive $+4$ gradients establish a strong, persistent downhill trend. A single noisy mini-batch with gradient $-1$ should slow down the movement slightly, but should not immediately reverse the overall travel direction.

---

#### 🔮 Prediction 0.1

> **Question**: *With $\theta_t = 2.0, \alpha = 0.1, g_t = 3.0$, does $\theta$ increase or decrease? What information would vanilla SGD retain for the next step?*
>
> **Answer**:
>
> - **Direction**: $\theta_{t+1} = 2.0 - 0.1(3.0) = 2.0 - 0.3 = \mathbf{1.7}$ (**$\theta$ decreases**).
> - **Retained Memory**: Vanilla SGD retains **zero information / state** for the next step (it is completely stateless).

---

## Part 1: The First Memory — EWMA

### 1.1 Exponentially Weighted Moving Average Rule

$$
s_t = \beta s_{t-1} + (1 - \beta) x_t, \quad 0 \le \beta < 1
$$

- **Unrolled Form**:
  $$
  s_t = (1 - \beta) x_t + \beta(1 - \beta) x_{t-1} + \beta^2(1 - \beta) x_{t-2} + \dots + \beta^t s_0
  $$
- **Effective Memory Horizon**: Approximately $\frac{1}{1 - \beta}$ steps.
  - $\beta = 0.9 \implies \sim 10$ steps memory.
  - $\beta = 0.99 \implies \sim 100$ steps memory.

---

#### ❓ Question Before Calculation (EWMA Influence)

> **Question**: *Given $s_0 = 0, \beta = 0.8, x_1 = 5, x_2 = 1$. Which sample should influence $s_2$ more: $x_1 = 5$ or $x_2 = 1$? State the coefficient that supports your claim.*
>
> **Answer**: **$x_2 = 1$ has greater influence.**
>
> - Weight of newest observation $x_2$: $(1 - \beta) = \mathbf{0.20}$
> - Weight of older observation $x_1$: $\beta(1 - \beta) = 0.8 \times 0.2 = \mathbf{0.16}$
> - Since $0.20 > 0.16$, newer observations carry strictly larger weights.

---

### 1.2 & 1.3 EWMA Calculation Ladder

**Given Setup**: $s_0 = 0, \beta = 0.8, x_1 = 5, x_2 = 1$. Using $s_t = 0.8 s_{t-1} + 0.2 x_t$:

- **Step 1 (Initial memory)**: $s_0 = \mathbf{0}$
- **Step 2 (First update)**: $s_1 = 0.8(0) + 0.2(5) = \mathbf{1.0}$
- **Step 3 (Second update)**: $s_2 = 0.8(1.0) + 0.2(1) = 0.8 + 0.2 = \mathbf{1.0}$
- **Step 4 (Direction Check)**:
  - Newer influence ($x_2$): $0.20$
  - Older influence ($x_1$): $0.16$

---

#### ✏️ Practice: Make a New Memory

Given $s_0 = 0, \beta = 0.6, x_1 = 4, x_2 = 2$:

1. **Step 1 (Calculate $s_1$)**:
   $$
   s_1 = 0.6(0) + (1 - 0.6)(4) = 0 + 0.4(4) = \mathbf{1.6}
   $$
2. **Step 2 (Calculate $s_2$)**:
   $$
   s_2 = 0.6(1.6) + 0.4(2) = 0.96 + 0.80 = \mathbf{1.76}
   $$
3. **Step 3**: *Explain in one line why $s_2$ is not simply the arithmetic average of 4 and 2 (which is 3.0).*
   - **Answer**: $s_2 = 1.76 \neq 3.0$ because EWMA weights recent observations more heavily than older ones and is also scaled down by the initial zero starting state ($s_0 = 0$).

---

## Part 2: Momentum Uses Remembered Direction

### 2.1 SGD with Momentum Formulation

$$
v_t = \beta v_{t-1} + (1 - \beta) g_t
$$

$$
\theta_{t+1} = \theta_t - \alpha v_t
$$

- **Physical Analogy**: A heavy ball rolling down a contoured hill. Small high-frequency lateral oscillations cancel out, while low-frequency consistent gradients accumulate velocity.

---

#### 🔮 Prediction 2.2: Momentum Direction

> **Question**: *Given $v_0 = 0, \theta_0 = 2.00, \beta = 0.9, \alpha = 0.1, g_1 = 4, g_2 = -2$. Will the second velocity $v_2$ be positive, negative, or zero? Justify without calculating.*
>
> **Answer**: **Positive.** The first update builds a positive velocity ($v_1 = 0.4$). At the second step, $90\%$ of that positive velocity is retained ($0.9 \times 0.4 = 0.36$), which outweighs the opposing negative gradient's $10\%$ contribution ($0.1 \times (-2) = -0.20$).

---

### 2.3 Momentum Calculation Ladder

**Given**: $v_0 = 0, \theta_0 = 2.00, \beta = 0.9, \alpha = 0.1, g_1 = 4, g_2 = -2$.

1. **Step 1 (Old velocity contribution)**: $\beta v_0 = 0.9(0) = \mathbf{0}$
2. **Step 2 (New gradient contribution)**: $(1 - \beta) g_1 = 0.1(4) = \mathbf{0.4}$
3. **Step 3 (First velocity)**: $v_1 = 0 + 0.4 = \mathbf{0.4}$
4. **Step 4 (Parameter update)**: $\theta_1 = 2.00 - 0.1(0.4) = 2.00 - 0.04 = \mathbf{1.96}$
5. **Step 5 (Opposing second gradient)**:
   - $v_2 = 0.9(0.4) + 0.1(-2) = 0.36 - 0.20 = \mathbf{0.16}$
   - $\theta_2 = 1.96 - 0.1(0.16) = 1.96 - 0.016 = \mathbf{1.944}$
6. **Step 6 (Direction Check)**:
   - Even though $g_2 = -2$ points uphill/opposite, the remembered velocity $v_2 = +0.16$ is positive. Therefore $\theta$ still **decreases**, but more slowly.

---

#### ✏️ Practice: Step 3 ($g_3 = -2$, from $v_2 = 0.16, \theta_2 = 1.944$)

- **Compute $v_3$**:
  $$
  v_3 = 0.9(0.16) + 0.1(-2) = 0.144 - 0.200 = \mathbf{-0.056}
  $$
- **Compute $\theta_3$**:
  $$
  \theta_3 = 1.944 - 0.1(-0.056) = 1.944 + 0.0056 = \mathbf{1.9496}
  $$
- **Interpretation**: The persistent negative gradients have finally overcome the previous positive momentum, reversing parameter movement.

---

## Part 3: NAG Checks Where It Is Heading

### 3.1 Nesterov Accelerated Gradient (NAG) Formulation

1. **Look-ahead Prediction**: $\theta_{look} = \theta_t - \alpha \beta v_t$
2. **Evaluate Gradient at Look-ahead**: $g_{look} = \nabla L(\theta_{look})$
3. **Update Velocity**: $v_{t+1} = \beta v_t + (1 - \beta) g_{look}$
4. **Update Parameter**: $\theta_{t+1} = \theta_t - \alpha v_{t+1}$

---

#### 🔮 Prediction 3.2: NAG Evaluation Point

> **Question**: *Which labelled point is used to calculate the NAG gradient? Explain why it is not the current point.*
>
> **Answer**:
>
> - **Point Used**: The **look-ahead point** $\theta_{look} = \theta_t - \alpha \beta v_t$.
> - **Explanation**: It evaluates the gradient where the existing momentum is about to carry the parameter. If the objective surface curves sharply upward ahead, NAG detects the opposing slope early and applies "early braking" before overshooting the valley bottom.

---

### 3.3 NAG Calculation Ladder

**Given Setup**: $\theta_t = 2.000, v_t = 0.500, \alpha = 0.1, \beta = 0.9, \nabla L(\theta_{look}) = 3.000$.

- **Step 1 (Predict Look-ahead)**:
  $$
  \theta_{look} = 2.000 - 0.1(0.9)(0.500) = 2.000 - 0.045 = \mathbf{1.955}
  $$
- **Step 2 (Two terms for corrected velocity)**:
  $$
  \beta v_t = 0.9(0.500) = 0.450, \quad (1 - \beta)\nabla L(\theta_{look}) = 0.1(3.000) = 0.300
  $$
- **Step 3 (Add terms)**:
  $$
  v_{t+1} = 0.450 + 0.300 = \mathbf{0.750}
  $$
- **Step 4 (Real parameter update)**:
  $$
  \theta_{t+1} = 2.000 - 0.1(0.750) = 2.000 - 0.075 = \mathbf{1.925}
  $$
- **Step 5 (Direction Check)**:
  - The current velocity moved the look-ahead point downhill; the gradient sampled there corrected the next velocity.

---

#### ✏️ Practice Activity

Given $\theta_t = 1.500, v_t = 0.200, \alpha = 0.1, \beta = 0.8, \nabla L(\theta_{look}) = -1.000$:

1. $\theta_{look} = 1.500 - 0.1(0.8)(0.200) = 1.500 - 0.016 = \mathbf{1.484}$
2. $v_{t+1} = 0.8(0.200) + (1 - 0.8)(-1.000) = 0.160 - 0.200 = \mathbf{-0.040}$
3. $\theta_{t+1} = 1.500 - 0.1(-0.040) = 1.500 + 0.004 = \mathbf{1.504}$

---

## Part 4: Adaptive Scale — AdaGrad vs. RMSProp

### 4.1 AdaGrad: Cumulative Squared Gradients

$$
r_t = r_{t-1} + g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{r_t} + \varepsilon} g_t
$$

---

#### ❓ Question Before Calculation (AdaGrad Coordinate Rescaling)

> **Question**: *Given $r_{t-1} = (0,0), g_t = (4,1)$, and ignore $\varepsilon$. Will the first coordinate retain its raw size 4 after normalisation? Why or why not?*
>
> **Answer**: **No.** Dividing by $\sqrt{r_t} = \sqrt{16} = 4$ normalizes the effective step size along the first coordinate to $\frac{4}{4} = \mathbf{1}$. AdaGrad rescales coordinates adaptively by their historical magnitude.

---

### 4.2 AdaGrad Calculation Ladder

**Given**: $r_{t-1} = (0,0), g_t = (4,1)$, ignore $\varepsilon$.

1. **Step 1 (Square gradient coordinates)**: $g_t^2 = (4^2, 1^2) = \mathbf{(16, 1)}$
2. **Step 2 (Update accumulator)**: $r_t = (0,0) + (16,1) = \mathbf{(16, 1)}$
3. **Step 3 (Square-root scale)**: $\sqrt{r_t} = (\sqrt{16}, \sqrt{1}) = \mathbf{(4, 1)}$
4. **Step 4 (Divide raw gradient by scale)**: $\frac{g_t}{\sqrt{r_t}} = \left(\frac{4}{4}, \frac{1}{1}\right) = \mathbf{(1, 1)}$
5. **Step 5 (Interpret)**:
   - The coordinate with raw gradient $4$ takes an effective direction of $1$; the coordinate with raw gradient $1$ also takes $1$.
   - **Drawback**: $r_t$ monotonically grows without bound ($\lim_{t \to \infty} r_t = \infty$), causing the effective learning rate $\frac{\alpha}{\sqrt{r_t}}$ to permanently shrink to zero ("learning dies").

#### ✏️ Practice: The Next Identical Gradient

Given $r_t = (16, 1), g_{t+1} = (4, 1)$:

- $r_{t+1} = (16, 1) + (16, 1) = \mathbf{(32, 2)}$
- Scaled direction:
  $$
  \frac{g_{t+1}}{\sqrt{r_{t+1}}} = \left(\frac{4}{\sqrt{32}}, \frac{1}{\sqrt{2}}\right) = \left(\frac{4}{4\sqrt{2}}, \frac{1}{\sqrt{2}}\right) = \left(\frac{1}{\sqrt{2}}, \frac{1}{\sqrt{2}}\right) \approx \mathbf{(0.7071, 0.7071)}
  $$

---

### 4.3 RMSProp: Fading Squared Gradients

$$
s_t = \rho s_{t-1} + (1 - \rho) g_t^2, \quad \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{s_t} + \varepsilon} g_t
$$

---

#### 🔮 Prediction 4.3: RMSProp vs. AdaGrad

> **Question / Prediction**: *Given $s_{t-1} = (16,1), g_t = (4,1), \rho = 0.9$, and ignore $\varepsilon$. In this first RMSProp step, $s_t$ happens to equal $(16,1)$. Does that mean RMSProp is identical to AdaGrad? Explain using what happens to a very old squared gradient.*
>
> **Answer**: **No, RMSProp is fundamentally different from AdaGrad.**
>
> - In **AdaGrad**, past squared gradients are accumulated permanently without decay ($r_t = \sum g_i^2$).
> - In **RMSProp**, past squared gradients are multiplied by $\rho^k$ at each subsequent step $k$, causing old gradients to fade exponentially. This prevents the accumulator from growing to infinity and allows the learning rate to adaptively recover.

---

### 4.4 RMSProp Calculation Ladder

**Given**: $s_{t-1} = (16, 1), g_t = (4, 1), \rho = 0.9$, ignore $\varepsilon$.

1. **Step 1 (Square incoming gradient)**: $g_t^2 = (4^2, 1^2) = \mathbf{(16, 1)}$
2. **Step 2 (Keep 90% of old scale)**: $\rho s_{t-1} = 0.9(16, 1) = \mathbf{(14.4, 0.9)}$
3. **Step 3 (Add 10% of new squared gradient)**: $(1 - \rho) g_t^2 = 0.1(16, 1) = \mathbf{(1.6, 0.1)}$
   - Combined $s_t = (14.4, 0.9) + (1.6, 0.1) = \mathbf{(16, 1)}$
4. **Step 4 (Normalise gradient)**: $\frac{g_t}{\sqrt{s_t}} = \left(\frac{4}{\sqrt{16}}, \frac{1}{\sqrt{1}}\right) = \mathbf{(1, 1)}$

#### ✏️ Practice: Changed Recent History

Given $s_{t-1} = (4, 9), g_t = (2, 3), \rho = 0.5$:

- $s_t = 0.5(4, 9) + 0.5(2^2, 3^2) = (2, 4.5) + (2, 4.5) = \mathbf{(4, 9)}$
- Normalised gradient: $\frac{g_t}{\sqrt{s_t}} = \left(\frac{2}{\sqrt{4}}, \frac{3}{\sqrt{9}}\right) = \mathbf{(1, 1)}$

> **Step 5 Question**: *Explain why this method can recover from an old high-gradient period whereas AdaGrad cannot.*
>
> **Answer**: Because RMSProp uses an EWMA with decay parameter $\rho < 1.0$. When the model enters a flat region with small gradients, the accumulator $s_t$ decays downwards, causing the effective step size $\frac{\alpha}{\sqrt{s_t}}$ to expand again. AdaGrad's accumulator $r_t$ is monotonically non-decreasing and cannot forget past large gradients.

---

## Part 5: Adam Keeps Two Memories

### 5.1 Formulation & Direction vs. Scale

- **First Moment (Direction Memory)**:
  $$
  m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9)
  $$
- **Second Moment (Scale Memory)**:
  $$
  v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)
  $$
- **Bias Correction**:
  $$
  \hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
  $$
- **Parameter Update Rule**:
  $$
  \theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t
  $$

---

#### ❓ Question Before Calculation: Sign of the Gradient

> **Question**: *Which state should retain the sign of the gradient: $m_t$ or $v_t$? Why?*
>
> **Answer**: **$m_t$ (the first moment) must retain the sign of the gradient.**
>
> **Reasoning**:
>
> 1. **$m_t$ is the Direction Memory**: It tracks the exponentially decaying average of the raw signed gradients $g_t$. The sign of $m_t$ tells the optimizer whether to step forward or backward in parameter space ($-\alpha \hat{m}_t$).
> 2. **$v_t$ is the Scale / Magnitude Memory**: It tracks the moving average of squared gradients $g_t^2$. Because every squared term $g_t^2 \ge 0$, $v_t$ is strictly non-negative and is placed under a square root ($\sqrt{v_t}$) in the denominator to normalize coordinates and adjust the step size, discarding directional sign information entirely.

---

### 5.2 Why Adam Corrects Its Early Memories (Bias Correction)

Both $m_t$ and $v_t$ begin at zero ($m_0 = 0, v_0 = 0$). While convenient, this causes an artificial early bias: before enough gradients have arrived, each moving average contains too much zero ("imaginary zero marks").

$$
\mathbb{E}[m_t] = \mathbb{E}\left[(1 - \beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i\right] = g \cdot (1 - \beta_1^t)
$$

Dividing by $(1 - \beta_1^t)$ and $(1 - \beta_2^t)$ removes this start-up artifact.

---

#### 🔮 Prediction 5.2: Bias Correction Scaling Effect

> **Question / Prediction**: *At $t = 1, \beta_1 = 0.9$, and raw $m_1 = 0.4$. Is dividing by $1 - 0.9$ going to make the estimate larger or smaller? Why is that the desired correction?*
>
> **Answer**:
>
> - **Effect**: Dividing by $(1 - 0.9) = 0.1$ makes the estimate **larger** ($\frac{0.4}{0.1} = \mathbf{4.0}$).
> - **Why this is desired**: At $t=1$, $m_1 = 0.9(0) + 0.1(4.0) = 0.4$. The estimate $0.4$ is artificially deflated ($10\times$ too small) because $90\%$ of the initial moving average was contributed by the arbitrary zero initialization $m_0 = 0$. Dividing by $(1 - 0.9) = 0.1$ scales the estimate back up to $4.0$, restoring the true magnitude of the incoming gradient.

---

### 5.3 Adam Calculation Ladder

**Given**: $m_0 = 0, v_0 = 0, g_1 = 4, \beta_1 = 0.9, \beta_2 = 0.999, \alpha = 0.01$, ignore $\varepsilon$.

1. **Step 1 (First moment - direction)**: $m_1 = 0.9(0) + 0.1(4) = \mathbf{0.4}$
2. **Step 2 (Second moment - scale)**: $v_1 = 0.999(0) + 0.001(4^2) = 0.001(16) = \mathbf{0.016}$
3. **Step 3 (Bias-correct first moment)**: $\hat{m}_1 = \frac{0.4}{1 - 0.9^1} = \frac{0.4}{0.1} = \mathbf{4.0}$
4. **Step 4 (Bias-correct second moment)**: $\hat{v}_1 = \frac{0.016}{1 - 0.999^1} = \frac{0.016}{0.001} = \mathbf{16.0}$
5. **Step 5 (Normalised direction)**: $\frac{\hat{m}_1}{\sqrt{\hat{v}_1}} = \frac{4.0}{\sqrt{16.0}} = \frac{4}{4} = \mathbf{1.0}$
6. **Step 6 (Parameter movement)**: $\Delta \theta_1 = 0.01(1.0) = 0.01 \implies \theta_1 = \theta_0 - \mathbf{0.01}$

#### ✏️ Short Continuation: Step 2 ($g_2 = 2$)

- $m_2 = 0.9(0.4) + 0.1(2) = 0.36 + 0.20 = \mathbf{0.56}$
- $v_2 = 0.999(0.016) + 0.001(2^2) = 0.015984 + 0.004 = \mathbf{0.019984}$

---

## Part 6: AdamW — Decoupled Weight Decay

### 6.1 Why Decouple Weight Decay from Adam?

- **L2 Regularization in Standard Adam**:
  Adds $\lambda \theta_t$ to the gradient: $g_t \leftarrow g_t + \lambda \theta_t$.
  When $g_t$ is scaled by $\frac{1}{\sqrt{v_t}}$, weights with large gradients receive *less* weight decay, while weights with small gradients receive *more* weight decay.
- **AdamW Solution**:
  Applies weight decay directly to the parameter **independently** of the gradient-scaling term:
  $$
  \theta_{t+1} = (1 - \alpha \lambda) \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \varepsilon} \hat{m}_t
  $$

---

#### 🔮 Prediction 6.1: Direction & Decay Independence

> **Question / Prediction**: *If the normalised Adam direction is negative, does $-\alpha(\text{negative})$ increase or decrease $\theta$? Does decay still point toward zero?*
>
> **Answer**:
>
> 1. **Adaptive Term ($-\alpha \times \text{negative}$)**: Subtracting a negative number yields a positive addition, which **increases $\theta$** (moves $\theta$ upward).
> 2. **Decay Term ($(1 - \alpha \lambda)\theta_t$)**: Weight decay **still shrinks $\theta$ toward zero** (reduces positive $\theta$, increases negative $\theta$), operating completely independently of the adaptive gradient direction.

---

### 6.2 AdamW Calculation Ladder

**Given**: $\theta_t = 1.00, \alpha = 0.01, \lambda = 0.1$, normalized Adam direction $\frac{\hat{m}_t}{\sqrt{\hat{v}_t}} = 0.50$.

1. **Step 1 (Decay-Only Step)**:
   $$
   (1 - \alpha \lambda) \theta_t = (1 - 0.01 \times 0.1)(1.00) = (1 - 0.001)(1.00) = \mathbf{0.999}
   $$
2. **Step 2 (Adaptive Move)**:
   $$
   \alpha \times 0.50 = 0.01 \times 0.50 = \mathbf{0.005}
   $$
3. **Step 3 (Combine Operations)**:
   $$
   \theta_{t+1} = 0.999 - 0.005 = \mathbf{0.994}
   $$

---

#### 🧭 Direction Check

> **Check**: *If the normalised Adam direction is negative, the adaptive term $-\alpha(\text{negative})$ moves $\theta$ upward. Weight decay still shrinks $\theta$ toward zero.*

---

#### 🧩 Choose the State (Summary of Memory Models)

| Optimizer          | State Maintained                      | Functional Purpose                                                  |
| :----------------- | :------------------------------------ | :------------------------------------------------------------------ |
| **EWMA**     | $s_t$                               | Fading memory of one quantity                                       |
| **Momentum** | $v_t$                               | Fading direction history (velocity)                                 |
| **NAG**      | $v_t, \theta_{look}$                | Direction plus a look-ahead gradient measurement                    |
| **AdaGrad**  | $r_t$                               | Cumulative squared-gradient scale                                   |
| **RMSProp**  | $s_t$                               | Fading squared-gradient scale                                       |
| **Adam**     | $m_t, v_t$                          | Direction memory and scale memory combined (with bias correction)   |
| **AdamW**    | $m_t, v_t, (1-\alpha\lambda)\theta$ | Adam direction and scale memories plus separate parameter shrinkage |

---

## Part 7: Choose the Right Kind of Memory

### 7.1 Optimizer Decision Map

```
                       What is the training challenge?
                                     │
         ┌───────────────────────────┼───────────────────────────┐
         ▼                           ▼                           ▼
  Noisy mini-batches,         Different coordinate         Need explicit
consistent direction           gradient scales              weight decay
         │                           │                           │
  Momentum / NAG              RMSProp / Adam                   AdamW
```

### 7.2 Read the Diagnoses

- **Noisy direction**: Smooth direction with **Momentum**.
- **Overshoot risk**: Sample gradient at a prediction with **NAG**.
- **Uneven coordinates**: Use per-coordinate scale with **AdaGrad** or **RMSProp**.
- **Adaptive baseline**: Combine both direction and scale memories with **Adam**.
- **Adaptive plus regularisation**: Use **AdamW**.

---

#### ❓ Scenario Question

> **Scenario**: *A model has noisy mini-batch gradients, layers with very different scales, and a need for explicit weight decay. Recommend a starting optimiser and justify it using two states or update terms.*
>
> **Recommendation**: **AdamW**
>
> **Justification**:
>
> 1. **First Moment ($m_t$)**: Smooths out noisy mini-batch gradients via momentum-like averaging.
> 2. **Second Moment ($v_t$)**: Automatically scales updates coordinate-wise, handling disparate layer gradient magnitudes.
> 3. **Decoupled Decay Term ($(1 - \alpha \lambda)\theta_t$)**: Applies pure, scale-independent weight regularization directly to parameters.

---

## Part 8: Practice, Retrieval & Subjective Solutions

### 8.1 Match the Following

|      #      | Optimizer |    Match    | Description                            |
| :---------: | :-------- | :---------: | :------------------------------------- |
| **1** | AdaGrad   | **C** | Squared gradients accumulated forever  |
| **2** | RMSProp   | **B** | Fading squared-gradient scale          |
| **3** | Momentum  | **D** | Smoothed direction carried forward     |
| **4** | NAG       | **A** | Gradient measured at a predicted point |
| **5** | AdamW     | **E** | Separate weight decay                  |

---

### 8.2 Multiple Choice Questions (MCQs)

1. **MCQ 1**: Increasing EWMA $\beta$ gives:
   - **Answer**: **B. smoother memory** (longer memory horizon $\frac{1}{1-\beta}$).
2. **MCQ 2**: Momentum stores:
   - **Answer**: **B. a gradient average** (first-moment velocity vector).
3. **MCQ 3**: NAG measures its gradient:
   - **Answer**: **B. at look-ahead** ($\theta_{look} = \theta_t - \alpha \beta v_t$).
4. **MCQ 4**: AdaGrad’s denominator can grow forever because it uses:
   - **Answer**: **B. cumulative $g^2$** ($r_t = r_{t-1} + g_t^2$).
5. **MCQ 5**: RMSProp differs by using:
   - **Answer**: **A. fading $g^2$ scale** (exponentially decaying average).
6. **MCQ 6**: Adam’s signed direction state is:
   - **Answer**: **B. $m_t$** (the first raw moment of gradients).
7. **MCQ 7**: AdamW decouples:
   - **Answer**: **B. decay and adaptive update** (separating parameter shrinkage from gradient scale).

---

### 8.3 Evidence Check (Two Detailed Proofs)

- **Selected MCQ 3 (NAG)**:

  - *Option*: **B. at look-ahead**
  - *Mathematical Rule*: $v_{t+1} = \beta v_t + (1-\beta)\nabla L(\theta_t - \alpha \beta v_t)$.
  - *Evidence*: NAG computes $\theta_{look} = \theta_t - \alpha \beta v_t$ first, and evaluates the gradient $\nabla L(\theta_{look})$ at that anticipated future coordinate to apply early braking.
- **Selected MCQ 7 (AdamW)**:

  - *Option*: **B. decay and adaptive update**
  - *Mathematical Rule*: $\theta_{t+1} = (1 - \alpha \lambda)\theta_t - \alpha \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \varepsilon}$.
  - *Evidence*: The shrinkage term $(1 - \alpha \lambda)\theta_t$ is applied directly to $\theta_t$, isolated from the adaptive denominator $\sqrt{\hat{v}_t}$.

---

### 8.4 Subjective Questions

#### Q1. EWMA 3-Step Unrolling

> **Question**: *Unroll EWMA one more step and state the coefficient of $x_{t-2}$.*
>
> **Derivation**:
>
> $$
> s_t = \beta s_{t-1} + (1 - \beta) x_t
> $$
>
> Substitute $s_{t-1} = \beta s_{t-2} + (1 - \beta) x_{t-1}$:
>
> $$
> s_t = \beta[\beta s_{t-2} + (1 - \beta) x_{t-1}] + (1 - \beta) x_t = \beta^2 s_{t-2} + \beta(1 - \beta) x_{t-1} + (1 - \beta) x_t
> $$
>
> Substitute $s_{t-2} = \beta s_{t-3} + (1 - \beta) x_{t-2}$:
>
> $$
> s_t = \beta^3 s_{t-3} + \mathbf{\beta^2(1 - \beta)} x_{t-2} + \beta(1 - \beta) x_{t-1} + (1 - \beta) x_t
> $$
>
> **Answer**: The coefficient of $x_{t-2}$ is **$\mathbf{\beta^2(1 - \beta)}$**.

---

#### Q2. Momentum Resistance to Gradient Reversal

> **Question**: *Explain why an opposing gradient may not immediately reverse Momentum.*
>
> **Answer**:
> The velocity $v_t = \beta v_{t-1} + (1 - \beta) g_t$ is a running weighted average of past gradients. If a strong directional momentum has accumulated ($v_{t-1} \gg 0$), a single opposing gradient ($g_t < 0$) is multiplied by only $(1 - \beta) \approx 0.1$. The retained past momentum $\beta v_{t-1} \approx 0.9 v_{t-1}$ dominates the sum, reducing the magnitude of velocity without flipping its sign.

---

#### Q3. NAG's Four Ordered Actions in Words

> **Question**: *Write NAG’s four ordered actions in words.*
>
> **Answer**:
>
> 1. **Predict**: Project a look-ahead parameter location using the current velocity ($\theta_{look} = \theta_t - \alpha \beta v_t$).
> 2. **Measure**: Compute the loss gradient at that projected look-ahead location ($\nabla L(\theta_{look})$).
> 3. **Correct**: Update the velocity by combining the retained momentum with the look-ahead gradient ($v_{t+1} = \beta v_t + (1-\beta)\nabla L(\theta_{look})$).
> 4. **Step**: Update the actual model parameters by taking a step along the newly corrected velocity ($\theta_{t+1} = \theta_t - \alpha v_{t+1}$).

---

#### Q4. Contrast AdaGrad and RMSProp

> **Question**: *Contrast AdaGrad and RMSProp using their squared-gradient histories.*
>
> **Answer**:
>
> - **AdaGrad** maintains a monotonic cumulative sum $r_t = \sum_{\tau=1}^t g_\tau^2$. Since every new squared gradient is added permanently, the denominator $\sqrt{r_t}$ grows indefinitely, causing the effective step size to shrink to zero and permanently stopping learning early.
> - **RMSProp** maintains an exponentially weighted moving average $s_t = \rho s_{t-1} + (1 - \rho) g_t^2$ ($\rho \in (0, 1)$). Old squared gradients fade geometrically by a factor of $\rho$ per step, enabling the denominator to decrease when gradients become smaller, allowing the optimizer to adaptively recover its step size.

---

#### Q5. Decoupling in AdamW

> **Question**: *Explain why AdamW’s decay is separate from Adam’s normalised direction.*
>
> **Answer**:
> In standard Adam with L2 regularization, the weight decay gradient $\lambda \theta$ is added directly into the gradient $g_t$, which then gets divided by $\sqrt{\hat{v}_t}$. Consequently, weights that have large gradient histories receive disproportionately weakened decay, while weights with sparse/infrequent gradients receive overly strong decay.
>
> **AdamW decouples weight decay** by applying $(1 - \alpha \lambda)\theta_t$ directly to the parameters. This ensures that every parameter decays proportionally to its own magnitude, restoring the true theoretical purpose of weight regularization while allowing Adam's adaptive mechanism to focus purely on scale-invariant directional optimization.
