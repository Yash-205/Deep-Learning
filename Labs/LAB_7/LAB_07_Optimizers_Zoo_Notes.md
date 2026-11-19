# Lab 07: The Optimizer Zoo — From SGD to Adam
**Notebook Reference:** `Lab_7(Optimizers_Zoo).ipynb`  
**Focus:** Ill-Conditioned Anisotropic Ravines, Loss Landscapes, First & Second Moment Dynamics, Saddle Point Escape, and PyTorch `torch.optim` Mechanics.

---

## 1. Mathematical Objective: The Anisotropic Ravine

Optimization in deep learning navigates loss surfaces with severely ill-conditioned curvature—steep in some parameter directions and flat in others. In this lab, all optimizers are systematically benchmarked on the 2D **anisotropic quadratic ravine** function:

$$f(w_0, w_1) = 0.5 \cdot (0.1 w_0^2 + 2.0 w_1^2)$$

### Analytical Gradients & Curvature
$$\nabla f(w_0, w_1) = \begin{bmatrix} \frac{\partial f}{\partial w_0} \\ \frac{\partial f}{\partial w_1} \end{bmatrix} = \begin{bmatrix} 0.1 w_0 \\ 2.0 w_1 \end{bmatrix}$$

$$\text{Hessian Matrix } H = \nabla^2 f = \begin{bmatrix} 0.1 & 0 \\ 0 & 2.0 \end{bmatrix}$$

- **Condition Number ($\kappa$):** $\kappa = \frac{\lambda_{\max}}{\lambda_{\min}} = \frac{2.0}{0.1} = 20$.
- **Axis Characteristics:**
  - $w_0$ (Gentle / Shallow Axis): Curvature $\lambda_1 = 0.1$. The gradient $\nabla_{w_0} f = 0.1 w_0$ is tiny, requiring large or accelerated steps to make progress.
  - $w_1$ (Steep / Canyon Axis): Curvature $\lambda_2 = 2.0$ ($20\times$ steeper). The gradient $\nabla_{w_1} f = 2.0 w_1$ is massive, causing standard gradient descent to oscillate violently across canyon walls.
- **Global Minimum:** $(w_0^*, w_1^*) = (0.0, 0.0)$ with $f(0, 0) = 0.0$.
- **Starting Position:** $w_{\text{start}} = [-4.5, 4.0]$.

```text
   ┌─────────────────────────────────────────────────────────────┐
   │                    THE EVOLUTIONARY PATH                    │
   │                                                             │
   │   Plain SGD ────────► SGD + Momentum ────────► NAG          │
   │   (Oscillates)        (Builds speed)           (Brakes)     │
   │        │                                        │           │
   │        ▼                                        ▼           │
   │   AdaGrad   ────────► RMSProp        ────────► ADAM         │
   │   (Stalls out)        (Adapts strides)         (The King)   │
   │                                                             │
   └─────────────────────────────────────────────────────────────┘
```

---

## 2. Cell-by-Cell & Line-by-Line Code Breakdown

---

### Cell 3: Environment Setup, Ravine Loss, and Simulation Runner (Code)

```python
import torch
import numpy as np
import matplotlib.pyplot as plt

# Fix random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Define the ravine loss function
def ravine_loss(w):
    return 0.5 * (0.1 * w[0]**2 + 2.0 * w[1]**2) # w is a tensor of shape (2,)

# Helper function to run native PyTorch optimizers
def track_optimizer(optimizer_cls, opt_kwargs, start_pos=[-4.5, 4.0], steps=50):
    # Create the parameter to optimize
    w = torch.tensor(start_pos, dtype=torch.float32, requires_grad=True)

    # Initialize PyTorch optimizer
    optimizer = optimizer_cls([w], **opt_kwargs)

    trajectory = [w.detach().clone().numpy()]
    loss_history = []

    for _ in range(steps):
        optimizer.zero_grad()
        loss = ravine_loss(w)
        loss.backward()
        optimizer.step()

        trajectory.append(w.detach().clone().numpy())
        loss_history.append(loss.item())

    return np.array(trajectory), loss_history

# Visualizer for 2D Contour & Trajectories
def plot_trajectories(trajectories, title):
    x = np.linspace(-5.0, 5.0, 300)
    y = np.linspace(-5.0, 5.0, 300)
    X, Y = np.meshgrid(x, y)
    Z = 0.5 * (0.1 * X**2 + 2.0 * Y**2)

    plt.figure(figsize=(9, 6))
    plt.contour(X, Y, Z, levels=35, cmap="Blues_r", alpha=0.6)

    for name, traj in trajectories.items():
        plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=3, label=name, lw=2)

    plt.scatter([0], [0], color="red", marker="*", s=250, label="Global Minimum (0,0)", zorder=10)
    plt.title(title, fontsize=13, fontweight='bold')
    plt.xlabel("$w_0$ (Gentle Axis)", fontsize=11)
    plt.ylabel("$w_1$ (Steep Axis)", fontsize=11)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
```

#### Line-by-Line Explanation
1. `import torch`, `import numpy as np`, `import matplotlib.pyplot as plt`: Imports core scientific computing, automatic differentiation, and 2D contour visualization modules.
2. `torch.manual_seed(42)`, `np.random.seed(42)`: Fixes pseudo-random number generation across CPU tensor operations and NumPy array initializations.
3. `def ravine_loss(w):`: Defines scalar objective taking 1D tensor `w` of length 2.
   - `return 0.5 * (0.1 * w[0]**2 + 2.0 * w[1]**2)`: Evaluates anisotropic quadratic loss using PyTorch tensor arithmetic preserving backward computational graph.
4. `def track_optimizer(optimizer_cls, opt_kwargs, start_pos=[-4.5, 4.0], steps=50):`: Parameterized testing harness that optimizes parameter tensor `w` over `steps` iterations using any PyTorch optimizer class.
   - `w = torch.tensor(start_pos, dtype=torch.float32, requires_grad=True)`: Creates root leaf parameter tensor requiring gradient computation initialized at $[-4.5, 4.0]$.
   - `optimizer = optimizer_cls([w], **opt_kwargs)`: Instantiates optimizer with parameter list `[w]` and unpacked hyperparameter dictionary `opt_kwargs` (e.g., `{'lr': 0.08, 'momentum': 0.9}`).
   - `trajectory = [w.detach().clone().numpy()]`: Records initial parameter vector detached from autograd graph as a NumPy float array.
   - `loss_history = []`: Records step-by-step loss values.
   - `for _ in range(steps):`: Executes optimization loop.
     - `optimizer.zero_grad()`: Flushes `.grad` attribute of tensor `w` to zero to prevent accumulation across iterations.
     - `loss = ravine_loss(w)`: Performs forward pass computing loss scalar tensor.
     - `loss.backward()`: Computes $\nabla_w f = [\frac{\partial f}{\partial w_0}, \frac{\partial f}{\partial w_1}]$ via reverse-mode autograd and populates `w.grad`.
     - `optimizer.step()`: Updates parameter `w` in-place according to optimizer's specific internal state equations.
     - `trajectory.append(w.detach().clone().numpy())`: Logs updated 2D coordinate.
     - `loss_history.append(loss.item())`: Extracts scalar float loss.
   - `return np.array(trajectory), loss_history`: Returns $(N+1, 2)$ trajectory array and list of length $N$.
5. `def plot_trajectories(trajectories, title):`:
   - `x = np.linspace(-5.0, 5.0, 300)`, `y = np.linspace(-5.0, 5.0, 300)`: Creates 300-point uniform grid across $[-5, 5]$.
   - `X, Y = np.meshgrid(x, y)`: Generates 2D coordinate matrices of shape $(300, 300)$.
   - `Z = 0.5 * (0.1 * X**2 + 2.0 * Y**2)`: Evaluates ravine loss across whole grid.
   - `plt.contour(X, Y, Z, levels=35, cmap="Blues_r", alpha=0.6)`: Renders 35 contour ellipses showing the elongated valley.
   - `for name, traj in trajectories.items(): plt.plot(...)`: Plots connected trajectory path with point markers.
   - `plt.scatter([0], [0], color="red", marker="*", s=250, label="Global Minimum (0,0)", zorder=10)`: Marks minimum at origin with red star.

---

### Cell 5: Step 1 — Plain Stochastic Gradient Descent (SGD) (Code)

```python
# Run Plain SGD
sgd_traj, sgd_loss = track_optimizer(torch.optim.SGD, {'lr': 0.7})

plot_trajectories({"Plain SGD (lr=0.18)": sgd_traj}, "Plain SGD: Bouncing across steep canyon walls")
```

#### Theory & Update Rule
$$w_{t+1} = w_t - \eta \nabla f(w_t)$$

- Coordinate updates:
  - $w_{0, t+1} = w_{0, t} - \eta (0.1 w_{0, t}) = (1 - 0.1 \eta) w_{0, t}$
  - $w_{1, t+1} = w_{1, t} - \eta (2.0 w_{1, t}) = (1 - 2.0 \eta) w_{1, t}$
- **The Dilemma:**
  - For stability along $w_1$, we require $|1 - 2.0 \eta| < 1 \implies \eta < 1.0$. At $\eta = 0.7$, $(1 - 2.0(0.7)) = -0.4$, causing sign flips (oscillations back and forth across $w_1 = 0$).
  - For fast progress along $w_0$, we need $1 - 0.1 \eta \ll 1 \implies \eta \approx 10.0$. But setting $\eta = 10.0$ causes $|1 - 2.0(10)| = |-19| = 19 \gg 1$, leading to immediate numerical explosion (divergence) along $w_1$.
- **Visual Output:**
  - Shows SGD bouncing wildly back and forth between $w_1 = +4$ and $w_1 = -2.5$ while making painfully slow forward progress along $w_0$.

---

### Cell 7: Step 2 — SGD + Momentum (Code)

```python
# Run SGD + Momentum using native PyTorch argument
momentum_traj, momentum_loss = track_optimizer(torch.optim.SGD, {'lr': 0.08, 'momentum': 0.9})

plot_trajectories({
    "Plain SGD": sgd_traj,
    "SGD + Momentum": momentum_traj
}, "Momentum: Dampening oscillations and accelerating along the valley")
```

#### Theory & Update Rule
$$\begin{aligned}
v_{t+1} &= \beta v_t + \nabla f(w_t) \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}$$
*(In PyTorch's default formulation with $\beta = 0.9$, $\eta = 0.08$)*

- **Physical Intuition:** Adds mass / inertia to the optimizer (like a heavy iron ball rolling down the surface).
  - Along $w_1$ (steep axis): Successive gradients point in opposite directions ($+g_1, -g_1, +g_1$). The momentum accumulation $\beta v_t$ averages these opposing vectors out toward $\approx 0$, dampening oscillations.
  - Along $w_0$ (gentle axis): Successive gradients point persistently in the negative direction ($-g_0, -g_0, -g_0$). The momentum accumulation builds velocity up to the terminal factor $\frac{1}{1 - \beta} = \frac{1}{1 - 0.9} = 10\times$, accelerating travel speed along the valley floor.
- **Visual Output:**
  - The trajectory turns smoothly toward the center line $w_1 = 0$ and races horizontally toward $w_0 = 0$, reaching near the global minimum within 50 steps.

---

### Cell 9: Step 3 — Nesterov Accelerated Gradient (NAG) (Code)

```python
# Run Nesterov Accelerated Gradient via PyTorch
nag_traj, nag_loss = track_optimizer(torch.optim.SGD, {'lr': 0.08, 'momentum': 0.9, 'nesterov': True})

plot_trajectories({
    "SGD + Momentum": momentum_traj,
    "NAG": nag_traj
}, "Momentum vs. NAG: The look-ahead brake in action")
```

#### Theory & Update Rule
$$\begin{aligned}
v_{t+1} &= \beta v_t + \nabla f(w_t - \eta \beta v_t) \\
w_{t+1} &= w_t - \eta v_{t+1}
\end{aligned}$$

- **Intuition (Look-Ahead Gradient):**
  - Standard momentum evaluates the gradient at current position $w_t$, then steps by combined velocity. If the ball is moving very fast toward a rising slope, standard momentum overshoots before the gradient can slow it down.
  - NAG evaluates the gradient at the *predicted future position* $(w_t - \eta \beta v_t)$. If the future position is ascending an opposing canyon wall, the look-ahead gradient immediately applies a corrective brake *before* the step is taken.
- **Visual Output:**
  - NAG displays significantly reduced overshoot at turning points compared to standard momentum, hugging the valley center tighter.

---

### Cell 11: Step 4 — Adaptive Step Sizes: AdaGrad (Code)

```python
# Run AdaGrad
adagrad_traj, adagrad_loss = track_optimizer(torch.optim.Adagrad, {'lr': 0.7})

plot_trajectories({
    "Plain SGD": sgd_traj,
    "AdaGrad": adagrad_traj
}, "AdaGrad: Straightens the path by scaling each coordinate")
```

#### Theory & Update Rule
$$\begin{aligned}
G_{t+1} &= G_t + g_t^2 \quad (\text{element-wise square}) \\
w_{t+1} &= w_t - \frac{\eta}{\sqrt{G_{t+1} + \epsilon}} \odot g_t
\end{aligned}$$

- **Intuition:**
  - Maintains coordinate-wise sum of historical squared gradients in $G_t \in \mathbb{R}^2$.
  - For $w_1$ (steep axis): Large gradients $g_1^2$ accumulate rapidly into $G_1$, causing $\frac{\eta}{\sqrt{G_1}}$ to shrink drastically, automatically taming wild oscillations.
  - For $w_0$ (gentle axis): Small gradients $g_0^2$ accumulate slowly into $G_0$, keeping the effective learning rate $\frac{\eta}{\sqrt{G_0}}$ large.
- **AdaGrad's Fatal Flaw:**
  - Because $g_t^2 \ge 0$, $G_t$ increases monotonically ($G_t \to \infty$ as $t \to \infty$).
  - The effective learning rate $\eta / \sqrt{G_t} \to 0$ prematurely, causing training to stall completely before reaching the true minimum in long optimization horizons.

---

### Cell 13: Step 5 — Fixing the Decay: RMSProp (Code)

```python
# Run RMSProp
rmsprop_traj, rmsprop_loss = track_optimizer(torch.optim.RMSprop, {'lr': 0.15, 'alpha': 0.9})

plot_trajectories({
    "AdaGrad": adagrad_traj,
    "RMSProp": rmsprop_traj
}, "AdaGrad vs. RMSProp: Leaky memory fixes premature stagnation")
```

#### Theory & Update Rule
$$\begin{aligned}
v_{t+1} &= \alpha v_t + (1 - \alpha) g_t^2 \quad (\text{Exponential Moving Average with } \alpha = 0.9) \\
w_{t+1} &= w_t - \frac{\eta}{\sqrt{v_{t+1} + \epsilon}} \odot g_t
\end{aligned}$$

- **Key Innovation (Hinton, 2012):**
  - Replaces AdaGrad's infinite sum with a leaky exponential moving average of squared gradients (EMA parameter $\alpha$, typically $0.9$ or $0.99$).
  - $v_t$ represents the recent localized scale of gradients rather than the entire training history.
  - The effective learning rate remains bounded and active indefinitely, allowing sustained progress along flat valleys without stalling.
- **Visual Output:**
  - RMSProp eliminates AdaGrad's late-stage slowdown and moves directly toward the origin.

---

### Cell 15: Step 6 — Combining Momentum & Adaptive Scales: Adam (Code)

```python
# Run Adam
adam_traj, adam_loss = track_optimizer(torch.optim.Adam, {'lr': 0.45, 'betas': (0.9, 0.999)})

plot_trajectories({
    "Plain SGD": sgd_traj,
    "SGD + Momentum": momentum_traj,
    "RMSProp": rmsprop_traj,
    "Adam": adam_traj
}, "Adam: The complete package (1st moment + 2nd moment + bias correction)")
```

#### Theory & Complete Update Rule (Kingma & Ba, 2014)
1. **1st Moment (Direction / Momentum):**
   $$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad (\beta_1 = 0.9)$$
2. **2nd Moment (Uncentered Variance / Scale):**
   $$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad (\beta_2 = 0.999)$$
3. **Bias Correction:**
   $$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$$
   *(Since $m_0 = \mathbf{0}$ and $v_0 = \mathbf{0}$, uncorrected moments are heavily biased toward zero in early steps when $\beta_1, \beta_2 \approx 1$. Dividing by $(1 - \beta^t)$ removes this initialization bias)*
4. **Parameter Update:**
   $$w_t = w_{t-1} - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \odot \hat{m}_t \quad (\epsilon = 10^{-8})$$

---

### Cell 17: The Grand Showdown (All 6 Optimizers Comparison) (Code)

```python
optimizers_config = {
    "Plain SGD": (torch.optim.SGD, {'lr': 0.18}),
    "SGD + Momentum": (torch.optim.SGD, {'lr': 0.08, 'momentum': 0.9}),
    "NAG": (torch.optim.SGD, {'lr': 0.08, 'momentum': 0.9, 'nesterov': True}),
    "AdaGrad": (torch.optim.Adagrad, {'lr': 1.5}),
    "RMSProp": (torch.optim.RMSprop, {'lr': 0.15, 'alpha': 0.9}),
    "Adam": (torch.optim.Adam, {'lr': 0.35, 'betas': (0.9, 0.999)})
}

all_trajectories = {}
all_losses = {}

for name, (opt_cls, kwargs) in optimizers_config.items():
    traj, loss = track_optimizer(opt_cls, kwargs, steps=60)
    all_trajectories[name] = traj
    all_losses[name] = loss

# 1. Trajectory Plot
plot_trajectories(all_trajectories, "The Grand Race: 6 Optimizers on an Anisotropic Ravine")

# 2. Loss Convergence Curves
plt.figure(figsize=(9, 4.5))
for name, loss_hist in all_losses.items():
    plt.plot(loss_hist, label=name, lw=2)

plt.yscale('log')
plt.title("Convergence Speed (Log Loss vs. Step)", fontsize=13, fontweight='bold')
plt.xlabel("Step", fontsize=11)
plt.ylabel("Loss (Log Scale)", fontsize=11)
plt.grid(True, which="both", linestyle="--", alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()
```

#### Line-by-Line Explanation
1. `optimizers_config`: Dictionary mapping display names to tuples of `(PyTorch Optimizer Class, Hyperparameter Dict)`.
   - `Plain SGD`: `lr = 0.18`
   - `SGD + Momentum`: `lr = 0.08`, `momentum = 0.9`
   - `NAG`: `lr = 0.08`, `momentum = 0.9`, `nesterov = True`
   - `AdaGrad`: `lr = 1.5`
   - `RMSProp`: `lr = 0.15`, `alpha = 0.9`
   - `Adam`: `lr = 0.35`, `betas = (0.9, 0.999)`
2. `all_trajectories`, `all_losses`: Storage containers for the 60-step tracking runs.
3. `for name, (opt_cls, kwargs) in optimizers_config.items():`: Iterates through each optimizer, collecting trajectories and loss curves.
4. `plt.yscale('log')`: Plots loss on a logarithmic scale ($\log_{10}(\text{Loss})$) to clearly differentiate asymptotic convergence near $10^{-6}$ vs $10^{-1}$.

---

### Cell 22: Mini-Challenge: Escaping a Saddle Point (Code)

```python
def saddle_loss(w):
    return 0.2 * w[0]**2 - 0.5 * w[1]**2

# Custom runner for saddle function
def track_saddle(optimizer_cls, opt_kwargs, start_pos=[-2.0, 0.01], steps=40):
    w = torch.tensor(start_pos, dtype=torch.float32, requires_grad=True)
    optimizer = optimizer_cls([w], **opt_kwargs)
    trajectory = [w.detach().clone().numpy()]

    for _ in range(steps):
        optimizer.zero_grad()
        loss = saddle_loss(w)
        loss.backward()
        optimizer.step()
        trajectory.append(w.detach().clone().numpy())

    return np.array(trajectory)

# Run both optimizers
sgd_saddle = track_saddle(torch.optim.SGD, {'lr': 0.1})
adam_saddle = track_saddle(torch.optim.Adam, {'lr': 0.2})

# Plot saddle trajectories
x = np.linspace(-3.0, 3.0, 200)
y = np.linspace(-3.0, 3.0, 200)
X, Y = np.meshgrid(x, y)
Z = 0.2 * X**2 - 0.5 * Y**2

plt.figure(figsize=(8, 5.5))
plt.contour(X, Y, Z, levels=30, cmap="Spectral", alpha=0.6)
plt.plot(sgd_saddle[:, 0], sgd_saddle[:, 1], 'ro-', label="SGD (lr=0.1) - Stuck in flat zone", lw=2)
plt.plot(adam_saddle[:, 0], adam_saddle[:, 1], 'bo-', label="Adam (lr=0.2) - Escapes saddle rapidly", lw=2)
plt.scatter([0], [0], color="black", marker="x", s=150, label="Saddle Point (0,0)", zorder=5)

plt.title("Mini-Challenge: Escaping a Saddle Point", fontsize=12, fontweight='bold')
plt.xlabel("$w_0$")
plt.ylabel("$w_1$")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.show()
```

#### Line-by-Line Explanation
1. `def saddle_loss(w): return 0.2 * w[0]**2 - 0.5 * w[1]**2`:
   - Saddle point at $(0, 0)$ where $\nabla f(0, 0) = [0, 0]^T$.
   - Along $w_0$, curvature is $+0.4 > 0$ (local minimum).
   - Along $w_1$, curvature is $-1.0 < 0$ (local maximum / descent cliff).
2. `start = [-2.0, 0.01]`: Initializes parameter with tiny perturbation ($+0.01$) along negative curvature axis $w_1$.
3. `sgd_saddle`: Plain SGD has $\nabla_{w_1} f = -1.0(0.01) = -0.01$. The gradient is so small that SGD moves almost purely along $w_0 \to 0$ and gets completely trapped at the saddle point $(0, 0)$ for dozens of steps.
4. `adam_saddle`: Adam's second moment accumulator $v_1$ recognizes the tiny gradient scale along $w_1$ and scales the step up by $\frac{1}{\sqrt{\hat{v}_1}}$, while momentum builds forward velocity. Adam quickly breaks symmetry and descends down the negative curvature slope to $-\infty$.

---

## 3. Comprehensive Optimizer Zoo Cheat Sheet

| Optimizer | Core Equation / State Update | Key Strength | Critical Weakness | Default Hyperparameters |
| :--- | :--- | :--- | :--- | :--- |
| **Plain SGD** | $w \leftarrow w - \eta \nabla f$ | Simplest computation, zero memory overhead | Oscillates violently in ravines; stalls at saddle points | $\text{lr} = 0.01 - 0.1$ |
| **Momentum** | $v \leftarrow \beta v + \nabla f$<br>$w \leftarrow w - \eta v$ | Dampens oscillations; accelerates through flat plateaus | Prone to overshooting sharp minima | $\text{lr} = 0.01$, $\beta = 0.9$ |
| **NAG** | $v \leftarrow \beta v + \nabla f(w - \eta \beta v)$<br>$w \leftarrow w - \eta v$ | Look-ahead evaluation acts as an adaptive brake | Requires extra gradient evaluation / internal staging | $\text{lr} = 0.01$, $\beta = 0.9$ |
| **AdaGrad** | $G \leftarrow G + g^2$<br>$w \leftarrow w - \frac{\eta}{\sqrt{G+\epsilon}} g$ | Automatic coordinate-wise scaling; excellent for sparse data | Monotonically shrinking learning rate kills learning early | $\text{lr} = 0.01$, $\epsilon = 10^{-10}$ |
| **RMSProp** | $v \leftarrow \alpha v + (1-\alpha) g^2$<br>$w \leftarrow w - \frac{\eta}{\sqrt{v+\epsilon}} g$ | Exponential moving average prevents learning rate decay | Non-momentum updates can still jitter in stochastic settings | $\text{lr} = 0.001$, $\alpha = 0.99$ |
| **Adam** | $m \leftarrow \beta_1 m + (1-\beta_1) g$<br>$v \leftarrow \beta_2 v + (1-\beta_2) g^2$<br>$w \leftarrow w - \frac{\eta}{\sqrt{\hat{v}}+\epsilon} \hat{m}$ | SOTA default; handles sparse gradients, non-stationary objectives, and noise | Can generalize slightly worse than tuned SGD on standard image benchmarks | $\text{lr} = 0.001$, $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$ |
| **AdamW** | $w \leftarrow (1 - \eta \lambda) w - \frac{\eta}{\sqrt{\hat{v}}+\epsilon} \hat{m}$ | Decouples $L_2$ weight decay $\lambda$ from adaptive gradient scaling | Requires tuning weight decay factor | $\text{lr} = 0.0001 - 0.001$, $\text{wd} = 0.01$ |

---

## 4. Key Takeaways & Summary

1. **Curvature Condition Number:** When Hessian eigenvalues differ significantly ($\kappa \gg 1$), isotropic gradient descent fails because no single learning rate $\eta$ is simultaneously stable for steep directions and fast for shallow directions.
2. **Momentum as a Low-Pass Filter:** Adding a velocity vector cancels high-frequency oscillations along orthogonal canyon walls while integrating low-frequency persistent signals along the descent valley.
3. **Adaptive Gradients as Coordinate Preconditioning:** 2nd-moment methods (RMSProp, Adam) approximate the diagonal of the inverse Hessian matrix ($H_{ii}^{-1/2} \approx 1/\sqrt{v_i}$), normalizing the step size across all parameter coordinates.
4. **Adam as General Purpose King:** By unifying first-moment directional smoothing, second-moment coordinate normalization, and initialization bias corrections, Adam functions reliably across deep vision, NLP, and reinforcement learning domains with minimal manual tuning.
