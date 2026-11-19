 

# Deep Learning — Worksheet 05: Gradient Descent & Learning-Rate Control

**Complete Solutions & Detailed Conceptual Guide**

---

## Overview & Core Concepts

- **Gradient Descent Objective**: Optimize parameters $\theta$ to minimize a scalar loss function $L(\theta)$ by taking steps proportional to the negative gradient:
  $$
  \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
  $$
- **Key Hyperparameters**:
  - $\alpha > 0$: **Learning Rate (Step Size)** — dictates the distance traversed along the descent direction.
  - $\nabla L(\theta_t)$: **Gradient Vector** — dictates the instantaneous direction of steepest ascent (so $-\nabla L$ is steepest descent).

---

## Part 1: Finding Downhill

### 1.1 The Idea Behind Gradient Descent

- **Analogy**: Walking down a foggy hill while blindfolded. You evaluate the local slope beneath your feet and step downhill.
- **Update Rule**:
  $$
  \theta_{t+1} = \theta_t - \alpha \nabla L(\theta_t)
  $$

---

### 1.2 Gradient Descent from Scratch

#### Worked Example Review

- **Loss Function**: $L(w) = (w - 3)^2$
- **Gradient**: $\frac{dL}{dw} = 2(w - 3)$
- **Initial Setup**: $w_0 = 0.0$, $\alpha = 0.25$
  - Step 0: $g_0 = 2(0 - 3) = -6$, $w_1 = 0 - 0.25(-6) = 1.5$, $L(w_1) = (1.5 - 3)^2 = 2.25$
  - Step 1: $g_1 = 2(1.5 - 3) = -3$, $w_2 = 1.5 - 0.25(-3) = 2.25$, $L(w_2) = (2.25 - 3)^2 = 0.5625$

---

#### ✏️ Boxed Task: Complete Two More Updates

Given $w_2 = 2.25$ and $\alpha = 0.25$:

1. **Step 2 ($t=2$ to $t=3$)**:

   - Compute Gradient:
     $$
     g_2 = 2(w_2 - 3) = 2(2.25 - 3) = 2(-0.75) = \mathbf{-1.5}
     $$
   - Update Parameter:
     $$
     w_3 = w_2 - \alpha g_2 = 2.25 - 0.25(-1.5) = 2.25 + 0.375 = \mathbf{2.625}
     $$
   - Compute Loss:
     $$
     L(w_3) = (2.625 - 3)^2 = (-0.375)^2 = \mathbf{0.140625} \quad \left(\frac{9}{64}\right)
     $$
2. **Step 3 ($t=3$ to $t=4$)**:

   - Compute Gradient:
     $$
     g_3 = 2(w_3 - 3) = 2(2.625 - 3) = 2(-0.375) = \mathbf{-0.75}
     $$
   - Update Parameter:
     $$
     w_4 = w_3 - \alpha g_3 = 2.625 - 0.25(-0.75) = 2.625 + 0.1875 = \mathbf{2.8125}
     $$
   - Compute Loss:
     $$
     L(w_4) = (2.8125 - 3)^2 = (-0.1875)^2 = \mathbf{0.03515625} \quad \left(\frac{9}{256}\right)
     $$

> **Question**: *Is $L(w_3)$ smaller than $L(w_2)$? Explain in one sentence.*
>
> **Answer**: **Yes, $L(w_3) = 0.140625$ is smaller than $L(w_2) = 0.5625$** because gradient descent takes a step in the negative gradient direction with an appropriately small learning rate ($\alpha = 0.25 < 1.0$), ensuring a monotonic decrease in the quadratic loss towards the global minimum at $w^* = 3$.

---

#### 💻 Boxed Task: Turn the Mathematics into a Training Loop

Complete the three missing lines in the Python snippet:

```python
w = 0.0
alpha = 0.25

for step in range(4):
    loss = (w - 3.0) ** 2               # Line 1: Measure loss
    grad = 2.0 * (w - 3.0)              # Line 2: Measure slope
    w = w - alpha * grad                # Line 3: Step downhill
    print(f"Step {step}: w = {w:.4f}, Loss = {loss:.6f}")
```

**Output Trace**:

- `Step 0: w = 1.5000, Loss = 9.000000`
- `Step 1: w = 2.2500, Loss = 2.250000`
- `Step 2: w = 2.6250, Loss = 0.562500`
- `Step 3: w = 2.8125, Loss = 0.140625`

---

#### ❓ Deep-Dive Question

> **Question**: *If you changed $\alpha$ from $0.25$ to $1.2$, what would you expect to happen near $w = 3$?*
>
> **Mathematical Derivation**:
> The recurrence relation for $w$ is:
>
> $$
> w_{t+1} = w_t - \alpha \cdot 2(w_t - 3)
> $$
>
> Subtract $3$ from both sides:
>
> $$
> (w_{t+1} - 3) = (w_t - 3) - 2\alpha(w_t - 3) = (1 - 2\alpha)(w_t - 3)
> $$
>
> For $\alpha = 1.2$:
>
> $$
> (1 - 2\alpha) = 1 - 2(1.2) = 1 - 2.4 = -1.4
> $$
>
> Therefore:
>
> $$
> (w_{t+1} - 3) = (-1.4)(w_t - 3)
> $$
>
> **Conclusion**: Because $|1 - 2\alpha| = 1.4 > 1$, the error oscillates with alternating signs and **grows exponentially** in magnitude by $1.4\times$ at every step. Instead of converging, the parameter will **overshoot the minimum at $w = 3$ and diverge to $\pm \infty$**.

---

## Part 2: Three Ways to Ask the Data

### 2.1 Full-Batch (Vanilla) Gradient Descent

- **Concept**: Averages the gradient across all $N$ training examples before making one update:
  $$
  \nabla L(\theta) = \frac{1}{N} \sum_{i=1}^N \nabla L_i(\theta), \quad \theta \leftarrow \theta - \alpha \nabla L(\theta)
  $$
- **Updates per epoch**: Exactly $1$.

#### ✏️ Boxed Task: Fill in the Consequence

> Full-batch GD has an **accurate** (smooth/exact) gradient. For $N = 50{,}000$ examples, each update requires seeing **50,000** examples. This makes each update **expensive**.

---

### 2.2 Stochastic Gradient Descent (SGD)

- **Concept**: Randomly selects $1$ single example $i$ per step and updates immediately:
  $$
  \theta \leftarrow \theta - \alpha \nabla L_i(\theta)
  $$
- **Updates per epoch**: Exactly $N$.
- **Characteristics**: Highly stochastic/noisy path; very cheap per step; noise provides an exploratory mechanism to escape narrow saddle points/local minima.

---

### 2.3 Mini-Batch Gradient Descent

- **Concept**: Computes the gradient over a mini-batch of size $B$ (e.g., $B = 32, 64, 128, 256$):
  $$
  \hat{\nabla} L(\theta) = \frac{1}{B} \sum_{i \in \mathcal{B}} \nabla L_i(\theta), \quad \theta \leftarrow \theta - \alpha \hat{\nabla} L(\theta)
  $$
- **Updates per epoch**: $\frac{N}{B}$.

---

#### ✏️ Comparison Table of Gradient Estimators

| Method                            | Examples per Update (\(B\)) | Gradient Noise                                           | Updates per Epoch | Hardware Parallelism                                              |
| --------------------------------- | :-------------------------: | -------------------------------------------------------- | :---------------: | ----------------------------------------------------------------- |
| **Vanilla / Full-Batch GD** |   \(N\) (entire dataset)   | **Low** (exact gradient over the dataset)          |       \(1\)       | Limited by memory; high memory demand and possible Out Of Memory. |
| **True SGD**                |            \(1\)            | **High** (large variance)                          |       \(N\)       | Limited vectorization; less efficient GPU utilization             |
| **Mini-Batch GD**           |    \(B\) (e.g., 32–512)    | **Moderate** (balance between noise and stability) |     \( N/B )     | **High** (vectorized operations and GPU acceleration)       |

**Note:** Here, \(N\) is the number of training examples, and \(B\) is the mini-batch size. The number of mini-batch updates per epoch is technically \(\lceil N/B \rceil\) when the final, smaller batch is retained; it is \(N/B\) when \(B\) divides \(N\) exactly.

---

#### ❓ Scenario Question

> **Question**: *You have a GPU and 1,000,000 images. Which method is the practical default, and why?*
>
> **Answer**: **Mini-batch Gradient Descent** (with a batch size such as $B = 64, 128,$ or $256$).
>
> **Detailed Justification**:
>
> 1. **Hardware Efficiency (GPU Parallelism)**: GPUs are massively parallel processors designed for matrix multiplications. Processing a mini-batch of 64–256 images saturates GPU memory bandwidth and tensor cores simultaneously. True SGD (batch size 1) leaves GPU compute units idle and spends almost all time on memory latency/overhead.
> 2. **Memory Feasibility**: Loading 1,000,000 high-resolution images into GPU VRAM at once causes an Out-Of-Memory (OOM) crash, rendering Full-Batch GD computationally impossible.
> 3. **Convergence Speed**: Full-batch GD only performs 1 parameter update per full dataset pass (1 update per 1M images). Mini-batch GD performs $\frac{1,000,000}{128} \approx 7,812$ parameter updates per epoch, converging orders of magnitude faster in wall-clock time while maintaining low gradient variance.

---

## Part 3: Choosing the Size of Every Step

### 3.1 Fixed Learning Rate Diagnostics

| Observation                                                       | Diagnosis             | Learning Rate State                                                       |
| :---------------------------------------------------------------- | :-------------------- | :------------------------------------------------------------------------ |
| Loss falls very slowly for many epochs.                           | **Stagnation**  | $\alpha$ is **too low** (overly conservative steps).              |
| Loss goes down overall, but bounces up and down near the minimum. | **Oscillation** | $\alpha$ is **slightly too high** (overshooting the basin walls). |
| Loss rises rapidly; values may become`NaN` or `Inf`.          | **Divergence**  | $\alpha$ is **much too high** (unstable positive feedback loop).  |

---

### 3.2 Practical LR Selection: The Learning-Rate Range Test (LR Finder)

- Developed by Leslie Smith.
- **Procedure**: Start from an extremely small learning rate ($\approx 10^{-7}$) and exponentially increase it to a large value ($\approx 10^1$) across a few hundred mini-batches while tracking training loss.
- **Rule of Thumb**: Pick the base learning rate where the loss decreases most steeply, typically one order of magnitude below the point where the loss begins to diverge or plateau.

---

## Part 4: Accelerate, Then Brake (Learning Rate Scheduling)

### 4.1 Match Each Schedule to Its Behaviour

| Schedule                    |    Code    | Exact Behaviour Description                                                                                                                                |
| :-------------------------- | :---------: | :--------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Step decay**        | **E** | **A staircase of sudden reductions** at pre-specified epoch milestones (e.g., multiply by $0.1$ every 30 epochs).                                  |
| **Exponential decay** | **C** | **A smooth percentage-style fall** ($\alpha_t = \alpha_0 \cdot \gamma^t$).                                                                         |
| **Cosine decay**      | **D** | **A smooth soft landing shaped like half a cosine** curve ($\alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_0 - \alpha_{min})(1 + \cos(\pi t / T))$). |
| **Warm-up**           | **A** | **Starts near zero and rises for a short initial period** to prevent violent early updates from chaotic initial gradients.                           |
| **Reduce on plateau** | **B** | **Reduces only after validation performance stops improving** (monitors validation loss with patience parameter).                                    |

---

#### ❓ Scenario Question

> **Question**: *Your validation loss has not improved for five epochs, but you do not know how many total epochs you will train. Which schedule is a natural fit? Why?*
>
> **Answer**: **Reduce on Plateau (`ReduceLROnPlateau`)**.
>
> **Reasoning**:
>
> - Pre-scheduled decays (such as Step Decay, Cosine Annealing, or Polynomial Decay) are *time-driven* and require pre-specifying the total epoch budget $T$ or hardcoded milestones.
> - **Reduce on Plateau** is purely *data-driven and dynamic*. It continuously tracks validation loss; when progress stalls for a patience window of 5 epochs, it automatically multiplies the learning rate by a decay factor (e.g., $0.1$ or $0.5$) without needing to know the total duration of training.

---

## Part 5: Practice & Assessment

### 5.1 Match the Following

|      #      | Term                        |    Match    | Description                                      |
| :---------: | :-------------------------- | :---------: | :----------------------------------------------- |
| **1** | Vanilla Gradient Descent    | **C** | Uses all examples before updating.               |
| **2** | Stochastic Gradient Descent | **D** | Updates after one randomly selected example.     |
| **3** | Mini-batch Gradient Descent | **A** | Uses a small subset to estimate the gradient.    |
| **4** | Fixed Learning Rate         | **B** | One step size is used throughout training.       |
| **5** | Reduce on Plateau           | **E** | Lowers the rate when validation progress stalls. |

---

### 5.2 Multiple Choice Questions (MCQs)

1. **MCQ 1**: If the gradient is positive, $\theta \leftarrow \theta - \alpha \nabla L$ moves $\theta$ in which direction?

   - **Answer**: **B. negative**
   - *Explanation*: $\nabla L > 0 \implies -\alpha \nabla L < 0$, so $\theta$ decreases.
2. **MCQ 2**: Which method uses all $N$ training examples for one gradient estimate?

   - **Answer**: **A. Vanilla GD**
   - *Explanation*: Vanilla / Full-batch GD computes $\frac{1}{N} \sum_{i=1}^N \nabla L_i$.
3. **MCQ 3**: With $N = 1{,}000$ and $B = 50$, a mini-batch epoch contains:

   - **Answer**: **B. 20 updates**
   - *Explanation*: $\text{Updates per epoch} = \frac{N}{B} = \frac{1000}{50} = 20$.
4. **MCQ 4**: Which gradient estimate is usually noisiest?

   - **Answer**: **C. true SGD**
   - *Explanation*: A single sample estimate has maximal sample variance compared to mini-batch or full-batch averages.
5. **MCQ 5**: A loss that explodes upward most strongly suggests:

   - **Answer**: **B. $\alpha$ is too high**
   - *Explanation*: Excessively large step size creates unstable overshooting and divergent numerical instability.
6. **MCQ 6**: What is the main purpose of mini-batches?

   - **Answer**: **B. balance cost and gradient noise**
   - *Explanation*: Mini-batches provide vectorizable GPU compute efficiency while reducing gradient variance to an acceptable level.
7. **MCQ 7**: What does warm-up do?

   - **Answer**: **A. begins with a smaller rate and increases it**
   - *Explanation*: Linearly or non-linearly ramps up $\alpha$ during the initial iterations to stabilize random initial weights.
8. **MCQ 8**: Which schedule reacts to validation loss rather than a predefined epoch?

   - **Answer**: **C. reduce on plateau**
   - *Explanation*: It is adaptive/metric-based rather than step-count/time-based.
9. **MCQ 9**: Why might a rate that was useful early become harmful late?

   - **Answer**: **A. gradients become smaller near the minimum**
   - *Explanation*: As the parameters approach the basin of attraction, a large step size causes the optimizer to bounce across the steep ravine walls instead of settling into the exact minimum.
10. **MCQ 10**: Cosine decay is best described as:

    - **Answer**: **B. a smooth soft landing**
    - *Explanation*: It follows a smooth cosine curve from $\alpha_{max}$ down to $\alpha_{min} \approx 0$.

---

### 5.3 Subjective Questions

#### Q1. Parameter Update Calculation

> **Question**: *For $L(w) = (w - 4)^2$, start at $w_0 = 0$ with $\alpha = 0.25$. Compute $g_0$ and $w_1$. Show the substitution.*
>
> **Solution**:
>
> 1. **Derivative / Gradient Function**:
>    $$
>    g(w) = \frac{dL}{dw} = 2(w - 4)
>    $$
> 2. **Evaluate at $w_0 = 0$**:
>    $$
>    g_0 = 2(0 - 4) = 2(-4) = \mathbf{-8}
>    $$
> 3. **Parameter Update Substitution**:
>    $$
>    w_1 = w_0 - \alpha g_0 = 0 - 0.25(-8) = 0 + 2.0 = \mathbf{2.0}
>    $$
> 4. **Verification of Loss Reduction**:
>    $$
>    L(w_0) = (0 - 4)^2 = 16, \quad L(w_1) = (2 - 4)^2 = 4 \quad (\text{Loss decreased from } 16 \to 4)
>    $$

---

#### Q2. Optimizer Selection at Scale

> **Question**: *You are training on 2 million images with a GPU. Choose full-batch GD, true SGD, or mini-batch GD, and justify your choice using both cost and noise.*
>
> **Solution**:
>
> - **Chosen Method**: **Mini-batch Gradient Descent** (with batch size $B \in [64, 512]$).
> - **Cost Justification**:
>   - *Full-Batch GD*: Impossible because $2{,}000{,}000$ images cannot fit in GPU VRAM (OOM error), and calculating gradients for all 2M images before a single update is computationally inefficient.
>   - *True SGD ($B=1$)*: Highly inefficient on GPU architecture due to memory transfer latency and inability to utilize parallel CUDA/Tensor cores.
>   - *Mini-batch GD*: Maximizes hardware throughput via parallel tensor matrix multiplications, enabling high samples-per-second processing.
> - **Noise Justification**:
>   - *True SGD*: Noise variance is excessively high, leading to erratic wandering and poor final convergence.
>   - *Full-Batch GD*: Lacks stochastic exploration and can easily get trapped in sharp suboptimal saddle points or local minima.
>   - *Mini-batch GD*: Averages away destructive variance while retaining sufficient stochasticity to generalize well and escape saddle points.

---

#### Q3. Dual Nature of SGD Noise

> **Question**: *Explain how noisy SGD updates can sometimes be useful early in training, but troublesome near a minimum.*
>
> **Solution**:
>
> - **Early in Training (Helpful)**: The stochastic noise introduces random perturbations in the trajectory. This enables the optimizer to "jump over" shallow local minima, bypass saddle points, and explore broader, flatter regions of the parameter space that yield superior generalization.
> - **Late in Training / Near a Minimum (Troublesome)**: Near the bottom of a basin of attraction, the true batch gradient approaches zero ($\nabla L \to 0$), but individual sample gradients remain non-zero ($\nabla L_i \neq 0$). The remaining variance causes the parameters to bounce randomly around the minimum in a limit cycle rather than settling precisely into the optimal point, unless the learning rate is scheduled to decay.

---

#### Q4. Stagnation Diagnosis & Intervention

> **Question**: *Training and validation loss both fall very slowly. Diagnose the learning-rate issue and state one intervention.*
>
> **Solution**:
>
> - **Diagnosis**: **Stagnation due to an overly small learning rate ($\alpha$)** or entering a flat loss plateau where gradient magnitudes are tiny.
> - **Interventions**:
>   1. **Increase the Learning Rate**: Multiply $\alpha$ by a factor of $3\times$ to $10\times$.
>   2. **Run an LR Range Test**: Perform a learning rate finder sweep to empirically identify the optimal learning rate.
>   3. **Adopt Momentum or Adaptive Optimizers**: Use Momentum, Adam, or AdamW, which amplify progress along flat dimensions through accumulated velocity.

---

#### Q5. Learning Rate Schedule for Unstable Early Updates

> **Question**: *A model has unstable early updates and a known training budget. Propose a schedule and explain each phase.*
>
> **Solution**:
>
> - **Proposed Schedule**: **Linear Warm-up followed by Cosine Annealing Decay** (Warm-up Cosine Schedule).
> - **Detailed Phase Explanation**:
>   ```
>   Learning Rate (α)
>     ^
>     |          /‾‾‾‾\
>   α_max|         /      \
>     |        /        \
>     |       /           \
>     |      /             \
>     |     /               \
>   α_min|____/                 \____
>     +----+--------------------+----> Training Steps / Epochs
>       Warm-up      Cosine Decay
>       (5-10%)        (90-95%)
>   ```
>   1. **Phase 1 — Linear Warm-Up (First 5–10% of training budget)**:
>      - *Mechanism*: $\alpha_t$ ramps up linearly from $\approx 0$ to $\alpha_{max}$.
>      - *Purpose*: At initialization, random weights produce erratic, massive gradients. A small initial $\alpha$ prevents destructive early updates, allowing weights and batch normalization / layer statistics to stabilize.
>   2. **Phase 2 — Cosine Annealing Decay (Remaining 90–95% of training budget)**:
>      - *Mechanism*: $\alpha_t$ smoothly decreases from $\alpha_{max}$ down to $\alpha_{min} \approx 0$ following the upper half of a cosine wave:
>        $$
>        \alpha_t = \alpha_{min} + \frac{1}{2}(\alpha_{max} - \alpha_{min})\left(1 + \cos\left(\frac{\pi(t - T_{warm})}{T - T_{warm}}\right)\right)
>        $$
>      - *Purpose*: Provides sustained high learning rates during mid-training for rapid exploration, followed by a gentle deceleration to allow parameters to settle smoothly and precisely into a sharp minimum.
