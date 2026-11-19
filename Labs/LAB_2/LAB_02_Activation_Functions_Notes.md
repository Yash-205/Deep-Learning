# Lab 02: Activation Functions — Hard Step, Sigmoid, and Tanh
**Notebook Reference:** `Lab_2 (Activation_Functions).ipynb`  
**Focus:** Mathematical definitions, derivatives, saturation regions, Autograd mechanics (`torch.autograd.grad`), analytical vs autograd chain rule verification, and one-neuron gradient descent updates.

---

## 1. Theoretical Foundations & Mathematical Formulas

| Activation Function | Formula $f(z)$ | Derivative $f'(z) = \frac{df}{dz}$ | Output Range | Key Characteristics & Failure Modes |
| :--- | :--- | :--- | :--- | :--- |
| **Hard Step** | $f(z) = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$ | $f'(z) = 0 \text{ for } z \neq 0$, undefined at $z=0$ | $\{0, 1\}$ | Gradient is $0$ almost everywhere; completely blocks backpropagation. |
| **Sigmoid ($\sigma$)** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | $\sigma'(z) = \sigma(z)(1 - \sigma(z))$ | $(0, 1)$ | Smooth, non-zero-centered (mean $\approx 0.5$), maximum slope is $0.25$ at $z=0$. Saturates for $|z| > 4$. |
| **Tanh** | $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$ | $\tanh'(z) = 1 - \tanh^2(z)$ | $(-1, 1)$ | Zero-centered, steeper slope than Sigmoid ($4\times$ steeper at $z=0$, maximum slope is $1.0$). Saturates for $|z| > 3$. |

---

## 2. Cell-by-Cell & Line-by-Line Code Breakdown

---

### Cell 2: Module Imports and Environment Configuration (Code)

```python
import math
import torch
import matplotlib.pyplot as plt

torch.manual_seed(7)
torch.set_printoptions(precision=6, sci_mode=False)
print(f"PyTorch {torch.__version__} | device: CPU")
```

#### Line-by-Line Explanation & Function Calls
1. `import math`: Imports Python's built-in mathematical library providing floating-point precision comparisons like `math.isclose()`.
2. `import torch`: Imports PyTorch root library for tensor creation and automatic differentiation (`torch.autograd`).
3. `import matplotlib.pyplot as plt`: Imports Matplotlib plotting interface for rendering activation curves and computational graphs.
4. `torch.manual_seed(7)`: Sets the pseudo-random seed to `7` for deterministic execution across CPU operations.
5. `torch.set_printoptions(precision=6, sci_mode=False)`:
   - Function called: `torch.set_printoptions()`
   - `precision=6`: Formats printed tensor floating-point numbers to 6 decimal places.
   - `sci_mode=False`: Suppresses scientific exponent notation (e.g., prints `0.000185` instead of `1.85e-4`).
6. `print(f"PyTorch {torch.__version__} | device: CPU")`: Outputs PyTorch runtime version and confirms execution on host CPU.

---

### Cell 4: Teacher Demo — Outputs and Derivatives of Step, Sigmoid, and Tanh (Code)

```python
z = torch.linspace(-6, 6, 1201, requires_grad=True)

step_y = (z >= 0).float()
sigmoid_y = torch.sigmoid(z)
tanh_y = torch.tanh(z)

# A hard comparison has no grad_fn. Its mathematical derivative is 0
# almost everywhere and undefined at 0, so we display 0 away from the jump.
step_grad = torch.zeros_like(z)
sigmoid_grad = torch.autograd.grad(sigmoid_y.sum(), z, retain_graph=True)[0]
tanh_grad = torch.autograd.grad(tanh_y.sum(), z)[0]

names = ["Hard step", "Sigmoid", "Tanh"]
outputs = [step_y.detach(), sigmoid_y.detach(), tanh_y.detach()]
grads = [step_grad.detach(), sigmoid_grad.detach(), tanh_grad.detach()]

fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)
for col, (name, out, grad) in enumerate(zip(names, outputs, grads)):
    axes[0, col].plot(z.detach(), out, color="#22577A", linewidth=2)
    axes[0, col].set_title(name)
    axes[0, col].set_ylabel("f(z)")
    axes[1, col].plot(z.detach(), grad, color="#D1495B", linewidth=2)
    axes[1, col].set_ylabel("f'(z)")
    axes[1, col].set_xlabel("z")
    for ax in axes[:, col]:
        ax.axvline(0, color="0.75", linewidth=1)
        ax.grid(alpha=0.2)
axes[1, 0].text(0.04, 0.82, "0 a.e.; undefined at z=0", transform=axes[1, 0].transAxes)
plt.tight_layout()
plt.show()

center = len(z) // 2
print(f"At z=0: sigmoid'={sigmoid_grad[center].item():.2f}, tanh'={tanh_grad[center].item():.2f}")
print("Hard-step output requires grad?", step_y.requires_grad)
```

#### Line-by-Line Explanation & Function Calls
1. `z = torch.linspace(-6, 6, 1201, requires_grad=True)`:
   - Function called: `torch.linspace(-6, 6, 1201)` creates a 1D tensor of 1,201 evenly spaced points in the domain $[-6.0, 6.0]$ with step size $\Delta z = \frac{12}{1200} = 0.01$.
   - Argument: `requires_grad=True` instructs PyTorch autograd engine to track every operation on `z` for derivative evaluation.
2. `step_y = (z >= 0).float()`:
   - Expression: `(z >= 0)` performs element-wise boolean thresholding returning a boolean tensor.
   - Method called: `.float()` casts boolean values (`True`/`False`) to floating point (`1.0`/`0.0`).
   - *Note:* Because comparison operators are non-differentiable, `step_y.requires_grad` evaluates to `False` and has no `grad_fn`.
3. `sigmoid_y = torch.sigmoid(z)`:
   - Function called: `torch.sigmoid(z)` computes element-wise $\sigma(z) = \frac{1}{1 + e^{-z}}$, attaching a `SigmoidBackward0` computational graph node.
4. `tanh_y = torch.tanh(z)`:
   - Function called: `torch.tanh(z)` computes element-wise $\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$, attaching a `TanhBackward0` graph node.
5. `step_grad = torch.zeros_like(z)`:
   - Function called: `torch.zeros_like(z)` allocates a tensor of zeros matching the shape and dtype of `z` to represent the derivative of the step function almost everywhere ($f'(z) = 0$).
6. `sigmoid_grad = torch.autograd.grad(sigmoid_y.sum(), z, retain_graph=True)[0]`:
   - Functions called:
     - `sigmoid_y.sum()`: Reduces vector tensor to a scalar sum so autograd can differentiate scalar $S = \sum_i \sigma(z_i)$.
     - `torch.autograd.grad(outputs, inputs, retain_graph=True)`: Computes $\frac{\partial S}{\partial z_i} = \sigma'(z_i)$ for all 1,201 points without populating `.grad` attributes.
     - `retain_graph=True`: Keeps intermediate autograd buffers in memory so subsequent backward calls can execute.
     - `[0]`: Extracts the gradient tensor corresponding to the first input (`z`).
7. `tanh_grad = torch.autograd.grad(tanh_y.sum(), z)[0]`:
   - Computes $\frac{\partial (\sum \tanh(z_i))}{\partial z_i} = 1 - \tanh^2(z_i)$ via `torch.autograd.grad`.
8. `names = ["Hard step", "Sigmoid", "Tanh"]`: List of string titles.
9. `outputs = [step_y.detach(), sigmoid_y.detach(), tanh_y.detach()]`:
   - Method called: `.detach()` returns a view tensor sharing storage but severed from the autograd graph for safe plotting.
10. `grads = [step_grad.detach(), sigmoid_grad.detach(), tanh_grad.detach()]`: Detached derivative tensors.
11. `fig, axes = plt.subplots(2, 3, figsize=(14, 6), sharex=True)`:
    - Function called: `plt.subplots(2, 3)` creates a $2 \times 3$ subplot matrix (Row 0: Activation $f(z)$, Row 1: Derivative $f'(z)$).
12. `for col, (name, out, grad) in enumerate(zip(names, outputs, grads)):`:
    - `axes[0, col].plot(z.detach(), out, color="#22577A", linewidth=2)`: Plots blue activation curve.
    - `axes[0, col].set_title(name)`: Sets column title.
    - `axes[0, col].set_ylabel("f(z)")`: Sets y-axis label for activation row.
    - `axes[1, col].plot(z.detach(), grad, color="#D1495B", linewidth=2)`: Plots red derivative curve.
    - `axes[1, col].set_ylabel("f'(z)")`: Sets y-axis label for derivative row.
    - `axes[1, col].set_xlabel("z")`: Sets x-axis label.
    - `for ax in axes[:, col]: ax.axvline(0, color="0.75", linewidth=1); ax.grid(alpha=0.2)`: Draws vertical reference line at $z=0$ and subtle grid.
13. `axes[1, 0].text(0.04, 0.82, "0 a.e.; undefined at z=0", transform=axes[1, 0].transAxes)`: Annotates the zero derivative of hard step.
14. `plt.tight_layout()`, `plt.show()`: Renders the $2\times 3$ visualization figure.
15. `center = len(z) // 2`: Index corresponding to $z=0$ (index $600$).
16. `print(...)`:
    - `sigmoid_grad[center].item()`: Extracts Python float `0.25`.
    - `tanh_grad[center].item()`: Extracts Python float `1.00`.
    - `step_y.requires_grad`: Prints `False`.

#### Printed Output
```text
At z=0: sigmoid'=0.25, tanh'=1.00
Hard-step output requires grad? False
```

---

### Cell 7: Student Activity 1 — Activation-Gradient Probe Function (Code & Solution)

```python
A1_READY = True  # Changed from False to True

def student_probe(name, value):
    z = torch.tensor(float(value), requires_grad=True)
    if name == "sigmoid":
        activation = torch.sigmoid(z)
    elif name == "tanh":
        activation = torch.tanh(z)
    else:
        raise ValueError("name must be 'sigmoid' or 'tanh'")
    
    gradient = torch.autograd.grad(activation, z)[0]
    return activation.item(), gradient.item()

if not A1_READY:
    print("Complete student_probe, change A1_READY to True, and run again.")
else:
    s0 = student_probe("sigmoid", 0.0)
    t0 = student_probe("tanh", 0.0)
    assert math.isclose(s0[0], 0.5, abs_tol=1e-6)
    assert math.isclose(s0[1], 0.25, abs_tol=1e-6)
    assert math.isclose(t0[0], 0.0, abs_tol=1e-6)
    assert math.isclose(t0[1], 1.0, abs_tol=1e-6)
    print("Tests passed!")
    for value in [-4.0, 0.0, 4.0]:
        print(value, student_probe("sigmoid", value), student_probe("tanh", value))
```

#### Line-by-Line Explanation & Function Calls
1. `def student_probe(name, value):`: Custom probe taking activation string identifier and scalar coordinate.
   - `z = torch.tensor(float(value), requires_grad=True)`: Creates root scalar tensor with gradient tracking enabled.
   - `if name == "sigmoid": activation = torch.sigmoid(z)`: Evaluates Sigmoid function.
   - `elif name == "tanh": activation = torch.tanh(z)`: Evaluates Tanh function.
   - `else: raise ValueError(...)`: Guards against invalid activation names.
   - `gradient = torch.autograd.grad(activation, z)[0]`: Calls PyTorch autograd engine to calculate scalar derivative $\frac{d(\text{activation})}{dz}$.
   - `return activation.item(), gradient.item()`: Converts 0D scalar tensors into native Python `float` numbers using `.item()`.
2. Test assertions with `math.isclose()`:
   - At $z=0.0$:
     - Sigmoid: $\sigma(0) = 0.5$, $\sigma'(0) = 0.5(1-0.5) = 0.25$.
     - Tanh: $\tanh(0) = 0.0$, $\tanh'(0) = 1 - 0^2 = 1.0$.
   - Prints `"Tests passed!"`.

#### Activity 1 Evidence Table Outputs
| $z$ | Sigmoid Activation | Sigmoid Gradient $\sigma'(z)$ | Tanh Activation | Tanh Gradient $\tanh'(z)$ | Which Gradient is Larger? |
|---:|:---:|:---:|:---:|:---:|:---:|
| **$-4.0$** | $0.017986$ | **$0.017663$** | $-0.999329$ | $0.001341$ | **Sigmoid** ($13.2\times$ larger) |
| **$0.0$** | $0.500000$ | $0.250000$ | $0.000000$ | **$1.000000$** | **Tanh** ($4\times$ larger) |
| **$+4.0$** | $0.982014$ | **$0.017663$** | $+0.999329$ | $0.001341$ | **Sigmoid** ($13.2\times$ larger) |

**Conclusion:** At the center ($z=0$), Tanh produces $4\times$ larger gradients than Sigmoid; however, in saturation regions ($|z| \ge 4$), Tanh decays much faster toward zero ($1 - \tanh^2(4) \approx 0.0013$) than Sigmoid ($\sigma(4)(1-\sigma(4)) \approx 0.0177$).

---

### Cell 10: Teacher Demo — Computational Graph Visualization (Code)

```python
from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(figsize=(14, 5))
ax.set_xlim(0, 14)
ax.set_ylim(-0.6, 5)
ax.axis("off")

def circle_node(xy, text, color):
    node = Circle(xy, 0.55, facecolor=color, edgecolor="#17324D", linewidth=2)
    ax.add_patch(node)
    ax.text(*xy, text, ha="center", va="center", fontsize=12, fontweight="bold")

def box_node(xy, width, text, color):
    x0, y0 = xy[0] - width / 2, xy[1] - 0.55
    node = FancyBboxPatch(
        (x0, y0), width, 1.1,
        boxstyle="round,pad=0.08,rounding_size=0.12",
        facecolor=color, edgecolor="#17324D", linewidth=2,
    )
    ax.add_patch(node)
    ax.text(*xy, text, ha="center", va="center", fontsize=11, fontweight="bold")

def arrow(start, end, label=None, label_xy=None, color="#17324D"):
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=18,
        linewidth=2, color=color, shrinkA=4, shrinkB=4,
    ))
    if label:
        lx, ly = label_xy or ((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.25)
        ax.text(lx, ly, label, ha="center", va="center", fontsize=10, color=color)

# Forward-pass nodes with the lecture values.
circle_node((1.0, 3.4), "x = 2", "#D8F3DC")
circle_node((1.0, 1.3), "b = 0", "#FFF3BF")
box_node((3.8, 2.5), 2.2, "Weighted sum\nz = wx + b = 2", "#DDEBFF")
box_node((6.5, 2.5), 1.7, "Sigmoid\nσ(z)", "#E5DBFF")
circle_node((9.0, 2.5), "ŷ =\n0.880797", "#FFD8A8")
circle_node((9.0, 0.7), "y = 0", "#F1F3F5")
box_node((12.0, 2.5), 2.3, "Loss\nL = 0.387902", "#FFC9C9")

arrow((1.55, 3.4), (2.65, 2.75), "weight w = 1", (2.1, 3.35))
arrow((1.55, 1.3), (2.65, 2.2), "bias", (2.05, 1.55))
arrow((4.9, 2.5), (5.65, 2.5))
arrow((7.35, 2.5), (8.45, 2.5))
arrow((9.55, 2.5), (10.85, 2.5), "prediction", (10.2, 2.85))
arrow((9.55, 0.7), (11.25, 1.95), "target", (10.25, 1.15))

ax.text(
    7.0, 4.35,
    "FORWARD PASS: compute the prediction and loss",
    ha="center", fontsize=14, fontweight="bold", color="#17324D",
)
ax.text(
    7.0, -0.25,
    "BACKWARD PASS: dL/dw = 0.184956     →     UPDATE: w = 1 - 0.5(0.184956) = 0.907522",
    ha="center", fontsize=12, fontweight="bold", color="#B42318",
)
plt.tight_layout()
plt.show()
```

#### Line-by-Line Explanation & Function Calls
1. `from matplotlib.patches import Circle, FancyBboxPatch, FancyArrowPatch`: Imports geometric vector primitives from Matplotlib.
2. `def circle_node(...)`, `def box_node(...)`, `def arrow(...)`: Helper routines constructing nodes and directed edges using `ax.add_patch()`.
3. Renders the full forward pass flow ($x, b \to z \to \hat{y} \to L$) and annotations for the backward pass update:
   $$\frac{\partial L}{\partial w} = 0.184956 \implies w_{\text{new}} = 1 - 0.5(0.184956) = 0.907522$$

---

### Cell 11: Teacher Demo — Manual Chain Rule vs. PyTorch Autograd Trace (Code)

```python
x = torch.tensor(2.0)
w = torch.tensor(1.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)
target = torch.tensor(0.0)
alpha = 0.5

z = w * x + b
prediction = torch.sigmoid(z)
loss = 0.5 * (prediction - target) ** 2
loss.backward()

sigmoid_slope = prediction.detach() * (1 - prediction.detach())
manual_grad = (prediction.detach() - target) * sigmoid_slope * x
with torch.no_grad():
    new_w = w - alpha * w.grad

print(f"1. weighted sum z              = {z.item():.6f}")
print(f"2. prediction sigmoid(z)      = {prediction.item():.6f}")
print(f"3. error (prediction-target) = {(prediction-target).item():.6f}")
print(f"4. sigmoid slope f'(z)        = {sigmoid_slope.item():.6f}")
print(f"5. manual dL/dw               = {manual_grad.item():.6f}")
print(f"6. autograd dL/dw             = {w.grad.item():.6f}")
print(f"7. updated weight             = {new_w.item():.6f}")

assert torch.allclose(manual_grad, w.grad, atol=1e-6)
assert math.isclose(new_w.item(), 0.907522, abs_tol=1e-6)
print("Manual chain rule and PyTorch autograd agree.")
```

#### Mathematical Derivation of Every Step
1. **Weighted sum:**
   $$z = w \cdot x + b = 1.0(2.0) + 0.0 = 2.000000$$
2. **Prediction:**
   $$\hat{y} = \sigma(z) = \frac{1}{1 + e^{-2.0}} = 0.880797$$
3. **Error:**
   $$\text{error} = \hat{y} - y = 0.880797 - 0.0 = 0.880797$$
4. **Squared Error Loss:**
   $$L = \frac{1}{2}(\hat{y} - y)^2 = \frac{1}{2}(0.880797)^2 = 0.387902$$
5. **Sigmoid Slope:**
   $$\frac{\partial \hat{y}}{\partial z} = \hat{y}(1 - \hat{y}) = 0.880797(1 - 0.880797) = 0.104994$$
6. **Chain Rule for $\frac{\partial L}{\partial w}$:**
   $$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w} = (\hat{y} - y) \cdot [\hat{y}(1 - \hat{y})] \cdot x$$
   $$\frac{\partial L}{\partial w} = (0.880797) \times (0.104994) \times (2.0) = 0.184956$$
7. **Weight Update Step ($\alpha = 0.5$):**
   $$w_{\text{new}} = w - \alpha \frac{\partial L}{\partial w} = 1.0 - 0.5(0.184956) = 0.907522$$

#### Line-by-Line Explanation & Function Calls
- `loss.backward()`: Traverses the autograd DAG in reverse, computing $\frac{\partial L}{\partial w}$ and populating `w.grad`.
- `with torch.no_grad():`: Context manager that disables gradient tracking during the parameter update $w_{\text{new}} = w - \alpha w.\text{grad}$ so the update step is not recorded in the graph.
- `assert torch.allclose(manual_grad, w.grad, atol=1e-6)`: Validates that analytical chain rule matches automatic differentiation to within $10^{-6}$.
- `assert math.isclose(new_w.item(), 0.907522, abs_tol=1e-6)`: Confirms final updated weight.

#### Printed Output
```text
1. weighted sum z              = 2.000000
2. prediction sigmoid(z)      = 0.880797
3. error (prediction-target) = 0.880797
4. sigmoid slope f'(z)        = 0.104994
5. manual dL/dw               = 0.184956
6. autograd dL/dw             = 0.184956
7. updated weight             = 0.907522
Manual chain rule and PyTorch autograd agree.
```

---

### Cell 14: Student Activity 2 — Generalized One-Neuron Update Function (Code & Solution)

```python
A2_READY = True  # Changed from False to True

def student_one_update(x_value, w_value, b_value, y_value, alpha, activation_name):
    x = torch.tensor(float(x_value))
    w = torch.tensor(float(w_value), requires_grad=True)
    b = torch.tensor(float(b_value), requires_grad=True)
    target = torch.tensor(float(y_value))
    
    z = w * x + b
    if activation_name == "sigmoid":
        prediction = torch.sigmoid(z)
    elif activation_name == "tanh":
        prediction = torch.tanh(z)
    else:
        raise ValueError("activation_name must be 'sigmoid' or 'tanh'")
        
    loss = 0.5 * (prediction - target) ** 2
    loss.backward()
    
    with torch.no_grad():
        new_w = w - alpha * w.grad
        
    return {
        "z": z.item(),
        "prediction": prediction.item(),
        "loss": loss.item(),
        "gradient": w.grad.item(),
        "new_w": new_w.item()
    }

if not A2_READY:
    print("Complete student_one_update, change A2_READY to True, and run again.")
else:
    result = student_one_update(2, 1, 0, 0, 0.5, "sigmoid")
    assert math.isclose(result["z"], 2.0, abs_tol=1e-6)
    assert math.isclose(result["gradient"], 0.184956, abs_tol=1e-6)
    assert math.isclose(result["new_w"], 0.907522, abs_tol=1e-6)
    print("Tests passed!", result)
```

#### Activity 2 Experiments & Analysis
1. **Experiment 1: Double Learning Rate ($\alpha = 1.0$)**
   - `student_one_update(2, 1, 0, 0, 1.0, "sigmoid")`
   - Result: Gradient $\frac{\partial L}{\partial w} = 0.184956$ (unchanged!). Update step doubles: $\Delta w = 1.0(0.184956) = 0.184956 \implies w_{\text{new}} = 1.0 - 0.184956 = 0.815044$.
2. **Experiment 2: Invert Target from $0$ to $1$**
   - `student_one_update(2, 1, 0, 1, 0.5, "sigmoid")`
   - Result: Error is now $\hat{y} - y = 0.880797 - 1.0 = -0.119203 < 0$.
   - Gradient: $\frac{\partial L}{\partial w} = (-0.119203)(0.104994)(2.0) = -0.025031$.
   - Updated weight: $w_{\text{new}} = 1.0 - 0.5(-0.025031) = 1.012516$ (**weight increases!**).
   - *Explanation:* Since target $y=1 > \hat{y}$, the network must increase prediction, which requires increasing $w$ because $x=2 > 0$.
3. **Experiment 3: Tanh Activation**
   - `student_one_update(1, 0, 0, 1, 1.0, "tanh")`
   - $z = 0(1) + 0 = 0 \implies \hat{y} = \tanh(0) = 0.0$.
   - Error: $\hat{y} - y = 0 - 1 = -1.0$.
   - Tanh slope: $1 - \tanh^2(0) = 1.0$.
   - Gradient: $\frac{\partial L}{\partial w} = (-1.0)(1.0)(1.0) = -1.000000$.
   - Updated weight: $w_{\text{new}} = 0.0 - 1.0(-1.0) = 1.000000$.

---

### Cell 17: Pair Challenge — Center vs. Saturation Analysis (Code)

```python
def teacher_reference_update(z_value, activation_name):
    x = torch.tensor(1.0)
    w = torch.tensor(0.0, requires_grad=True)
    b = torch.tensor(float(z_value), requires_grad=True)
    target = torch.tensor(1.0)
    z = w * x + b
    prediction = torch.sigmoid(z) if activation_name == "sigmoid" else torch.tanh(z)
    loss = 0.5 * (prediction - target) ** 2
    loss.backward()
    return prediction.item(), w.grad.item(), (w - w.grad).item()

print(f"{'z':>4} {'activation':>10} {'prediction':>12} {'dL/dw':>12} {'new w':>12}")
print("-" * 56)
for z_value in [0.0, 2.0, 4.0]:
    for activation_name in ["sigmoid", "tanh"]:
        pred, grad, new_w = teacher_reference_update(z_value, activation_name)
        print(f"{z_value:4.1f} {activation_name:>10} {pred:12.6f} {grad:12.6f} {new_w:12.6f}")

sigmoid_center = teacher_reference_update(0, "sigmoid")[2]
tanh_center = teacher_reference_update(0, "tanh")[2]
print(f"\nAt z=0, tanh step / sigmoid step = {tanh_center / sigmoid_center:.1f}x")
```

#### Line-by-Line Explanation & Mathematical Breakdown
- The experiment fixes $x=1.0, y=1.0, w=0.0, \alpha=1.0$ and varies initial $z \in \{0.0, 2.0, 4.0\}$ by setting $b = z$.
- At $z=0.0$:
  - Sigmoid: $\hat{y} = 0.5$, $\text{error} = 0.5 - 1.0 = -0.5$, $\sigma'(0) = 0.25 \implies \frac{\partial L}{\partial w} = -0.125 \implies w_{\text{new}} = +0.125$.
  - Tanh: $\hat{y} = 0.0$, $\text{error} = 0.0 - 1.0 = -1.0$ ($2\times$ error), $\tanh'(0) = 1.0$ ($4\times$ derivative) $\implies \frac{\partial L}{\partial w} = -1.0 \implies w_{\text{new}} = +1.0$.
  - Ratio: $\frac{\text{tanh step}}{\text{sigmoid step}} = \frac{1.0}{0.125} = 8.0\times$ larger!

#### Printed Output
```text
   z activation   prediction        dL/dw        new w
--------------------------------------------------------
 0.0    sigmoid     0.500000    -0.125000     0.125000
 0.0       tanh     0.000000    -1.000000     1.000000
 2.0    sigmoid     0.880797    -0.012516     0.012516
 2.0       tanh     0.964028    -0.002541     0.002541
 4.0    sigmoid     0.982014    -0.000318     0.000318
 4.0       tanh     0.999329    -0.000001     0.000001

At z=0, tanh step / sigmoid step = 8.0x
```

---

## 3. Exit Ticket Solutions & Key Conceptual Questions

### Q1: Why does a hard step fail under ordinary gradient descent?
**Answer:** The hard step function has a mathematical derivative of zero everywhere except at $z=0$ (where it is non-differentiable/undefined). By the chain rule, $\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot f'(z) \cdot x = 0$, producing zero gradients and causing weight updates to freeze permanently.

### Q2: What two factors made the Tanh update at $z=0$ eight times larger than the Sigmoid update?
**Answer:**
1. **Error factor ($2\times$):** Since target $y=1.0$, Tanh predicts $\hat{y}=0.0$ (error $= -1.0$), which is twice the Sigmoid error ($\hat{y}=0.5$, error $= -0.5$).
2. **Derivative factor ($4\times$):** Tanh's slope at $z=0$ is $\tanh'(0) = 1.0$, which is four times the Sigmoid slope ($\sigma'(0) = 0.25$).
- Total multiplier $= 2 \times 4 = 8\times$.

### Q3: If a model is confidently wrong in a saturated region, what happens to learning?
**Answer:** In saturated regions ($|z| \gg 0$), the activation derivatives $\sigma'(z)$ and $\tanh'(z)$ become vanishingly small ($\approx 0$). Even when the error $(\hat{y} - y)$ is large (confidently wrong), multiplying by the tiny derivative yields a near-zero gradient $\frac{\partial L}{\partial w} \approx 0$, causing backpropagation to stall (the **vanishing gradient problem**).
