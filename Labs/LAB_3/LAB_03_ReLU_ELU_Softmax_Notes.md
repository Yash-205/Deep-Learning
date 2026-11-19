# Lab 03: Modern Activation Functions (ReLU, Leaky ReLU, ELU, Softmax) — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Conceptual Motivation

In deep neural networks, early activations (Sigmoid, Tanh) saturate at $|z| \gg 0$, causing vanishing gradients. Modern deep architectures rely on **non-saturating** activations:
1. **ReLU (Rectified Linear Unit)**: $f(z) = \max(0, z)$. Non-saturating for $z > 0$, computationally trivial, but susceptible to the **Dying ReLU** problem ($f'(z) = 0$ for $z < 0$).
2. **Leaky ReLU**: $f(z) = \max(\alpha z, z)$ (typically $\alpha = 0.01$). Maintains a small slope $\alpha$ on the negative domain, preventing permanently dead neurons.
3. **ELU (Exponential Linear Unit)**: $f(z) = z$ for $z \ge 0$ and $\alpha(e^z - 1)$ for $z < 0$. Smooth continuous derivative at $z=0$ (when $\alpha=1$), with a smooth saturation asymptote at $-\alpha = -1.0$.
4. **Softmax**: Normalizes unnormalized real logit vectors into categorical probability distributions ($\sum p_i = 1$).

---

## 2. Mathematical Formulations & Derivatives

| Activation | Forward Formulation $f(z)$ | Derivative $f'(z) = \frac{\partial f}{\partial z}$ | Range | Negative Slope / Asymptote |
| :--- | :--- | :--- | :--- | :--- |
| **ReLU** | $\max(0, z) = \begin{cases} z & z \ge 0 \\ 0 & z < 0 \end{cases}$ | $\begin{cases} 1 & z > 0 \\ 0 & z < 0 \end{cases}$ | $[0, \infty)$ | Slope $= 0$ (Dying ReLU risk) |
| **Leaky ReLU** | $\max(\alpha z, z) = \begin{cases} z & z \ge 0 \\ \alpha z & z < 0 \end{cases}$ | $\begin{cases} 1 & z > 0 \\ \alpha & z < 0 \end{cases}$ | $(-\infty, \infty)$ | Constant slope $\alpha = 0.01$ |
| **ELU** | $\begin{cases} z & z \ge 0 \\ \alpha(e^z - 1) & z < 0 \end{cases}$ | $\begin{cases} 1 & z \ge 0 \\ \alpha e^z = f(z) + \alpha & z < 0 \end{cases}$ | $(-\alpha, \infty)$ | Asymptote $\to -\alpha = -1.0$ as $z \to -\infty$ |
| **Softmax** | $\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$ | $\frac{\partial \sigma_i}{\partial z_j} = \sigma_i (\delta_{ij} - \sigma_j)$ | $(0, 1), \; \sum \sigma_i = 1$ | Multi-class categorical probability |

---

## 3. Cell-by-Cell & Line-by-Line Code & Output Breakdown

---

### Cell 1: Environment Setup
```python
import math
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt

torch.manual_seed(7)
torch.set_printoptions(precision=6, sci_mode=False)
print(f"PyTorch {torch.__version__} | device: CPU")
```
- **Line 1-5**: Loads mathematics, NumPy, PyTorch, Functional library (`F`), and Matplotlib.
- **Line 7-9**: Sets PRNG seed and 6-decimal precision formatting.
- **Output**:
  ```text
  PyTorch 2.11.0+cpu | device: CPU
  ```

---

### Cell 2: Creating the Continuous Domain Grid
```python
z = torch.linspace(-6, 6, 1201, requires_grad=True)
```
- **Line 1 (`z = torch.linspace(-6, 6, 1201, requires_grad=True)`)**:
  - Generates 1201 finely spaced points over $[-6.0, +6.0]$ ($\Delta z = 0.01$).
  - `requires_grad=True`: Instructs PyTorch to build a dynamic computation graph to track gradients w.r.t. `z`.

---

### Cell 3: Forward Pass of Modern Activations
```python
relu_y = F.relu(z)
leaky_y = F.leaky_relu(z, negative_slope=0.01)
elu_y = F.elu(z, alpha=1.0)
```
- **Line 1 (`relu_y = F.relu(z)`)**: Computes $\max(0, z)$. All negative values are clamped to `0.0`.
- **Line 2 (`leaky_y = F.leaky_relu(z, negative_slope=0.01)`)**: Computes $\max(0.01z, z)$. Negative values are scaled by $0.01$.
- **Line 3 (`elu_y = F.elu(z, alpha=1.0)`)**: Computes $z$ for $z \ge 0$, and $1.0 \cdot (e^z - 1)$ for $z < 0$.

---

### Cell 4 & 5: Autograd Backward Gradient Calculations
```python
relu_grad = torch.autograd.grad(relu_y.sum(), z, retain_graph=True)[0]
leaky_grad = torch.autograd.grad(leaky_y.sum(), z, retain_graph=True)[0]
elu_grad = torch.autograd.grad(elu_y.sum(), z)[0]
```
- **Explanation**:
  - `torch.autograd.grad(outputs, inputs)` computes vector-Jacobian products $\nabla_z (\sum y)$.
  - `retain_graph=True`: Keeps intermediate computation graph buffers alive in memory so subsequent gradient queries on the same graph (e.g. for `leaky_y` and `elu_y`) do not fail.

---

### Cell 6: Visualizing $f(z)$ and $f'(z)$ Curves
```python
names = ["ReLU", "Leaky ReLU", "ELU"]
outputs = [relu_y.detach(), leaky_y.detach(), elu_y.detach()]
grads = [relu_grad.detach(), leaky_grad.detach(), elu_grad.detach()]

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for col, (name, out, grad) in enumerate(zip(names, outputs, grads)):
    # Row 0: Activation Function f(z)
    axes[0, col].plot(z.detach(), out, color="#22577A", linewidth=2)
    axes[0, col].set_title(name)
    axes[0, col].set_ylabel("f(z)")

    # Row 1: First Derivative f'(z)
    axes[1, col].plot(z.detach(), grad, color="#D1495B", linewidth=2)
    axes[1, col].set_ylabel("f'(z)")
    axes[1, col].set_xlabel("z")

    # Add crosshairs for visual clarity
    for ax in axes[:, col]:
        ax.axvline(0, color="0.75", linewidth=1)
        ax.axhline(0, color="0.75", linewidth=1)
        ax.grid(alpha=0.2)

# ELU lower horizontal asymptote at y = -1.0
axes[0, 2].axhline(-1.0, color="0.75", linewidth=1, linestyle="--")

plt.tight_layout()
plt.show()

center = len(z) // 2 # Index corresponding to z = 0.0
print(f"At z=0: ReLU'={relu_grad[center].item():.2f}, "
      f"Leaky ReLU'={leaky_grad[center].item():.2f}, ELU'={elu_grad[center].item():.2f}")
print(f"At z={z[0].item():.1f}: ReLU={relu_y[0].item():.3f}")
```
- **Output Metrics**:
  ```text
  At z=0: ReLU'=0.50, Leaky ReLU'=0.01, ELU'=1.00
  At z=-6.0: ReLU=0.000
  ```
- **Key Visual Observations**:
  - **ReLU**: Sharp elbow at $z=0$. Derivative is a discontinuous step from $0 \to 1$.
  - **Leaky ReLU**: Non-zero negative slope ($0.01$). Derivative is $0.01$ for $z<0$ and $1.0$ for $z>0$.
  - **ELU**: Completely smooth transition at $z=0$. Asymptotically flattens to $-1.0$ as $z \to -\infty$, pushing mean activations closer to zero.

---

## 4. Softmax Function & Numerical Stability

When computing Softmax:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
If $z_i \ge 1000$, standard floating-point arithmetic overflows to `inf`.

### The Shift Invariance / Log-Sum-Exp Trick:
$$\sigma(\mathbf{z})_i = \frac{e^{z_i - C}}{\sum_{j=1}^K e^{z_j - C}}, \qquad \text{where } C = \max_j(z_j)$$
```python
def numerically_stable_softmax(logits):
    z_max = torch.max(logits, dim=-1, keepdim=True).values
    exp_shifted = torch.exp(logits - z_max)
    return exp_shifted / torch.sum(exp_shifted, dim=-1, keepdim=True)
```
- **Guarantee**: All exponent terms satisfy $z_i - C \le 0$, ensuring $e^{z_i - C} \in (0, 1]$, which eliminates numerical overflow.
