# Lab 01: The XOR Problem & Multi-Layer Perceptrons (MLP) — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Theoretical Foundations

In 1969, Marvin Minsky and Seymour Papert proved in *Perceptrons* that a **Single-Layer Perceptron (SLP)** with linear decision boundaries cannot solve non-linearly separable functions like **Exclusive-OR (XOR)**:
$$\text{XOR}(x_1, x_2) = (x_1 \lor x_2) \land \neg(x_1 \land x_2) = \text{OR}(x_1, x_2) \land \text{NAND}(x_1, x_2)$$

This lab demonstrates:
1. **Linear Separability of AND & OR Gates** using a Single Perceptron (`nn.Linear(2, 1)` + step activation).
2. **Failure of SLP on XOR** (the linear separability limitation).
3. **Solving XOR with a 2-Layer MLP** by mapping inputs into a non-linear latent feature space using hidden units.

---

## 2. Cell-by-Cell & Line-by-Line Code & Output Breakdown

---

### Section 1: AND Gate with a Single-Layer Perceptron

#### Cell 1: Importing PyTorch Modules
```python
import torch
import torch.nn as nn
```
- **Line 1 (`import torch`)**: Imports core PyTorch tensor library.
- **Line 2 (`import torch.nn as nn`)**: Imports the neural network module containing layer definitions (e.g. `nn.Linear`).
- **Output**: None (successful import).

---

#### Cell 2: Creating AND Dataset
```python
X = torch.tensor([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]], dtype=torch.float32)

y = torch.tensor([[0], [0], [0], [1]], dtype=torch.float32)
```
- **Line 1-5 (`X = torch.tensor(...)`)**: Defines input truth table matrix of shape $(4, 2)$ containing all 2-bit binary combinations.
- **Line 7 (`y = torch.tensor(...)`)**: Defines the target AND column vector of shape $(4, 1)$ where only $(1, 1) \to 1$.
- **Output**: Tensor representations in memory.

---

#### Cell 3 & 4: Initializing Model and Setting Analytical Weights
```python
# Single perceptron: 2 inputs -> 1 output
model = nn.Linear(2, 1)

# Manually set weights and bias for AND
model.weight.data = torch.tensor([[1., 1.]])
model.bias.data = torch.tensor([-1.5])
```
- **Line 2 (`model = nn.Linear(2, 1)`)**: Instantiates affine transformation $z = \mathbf{X} \mathbf{W}^T + b$, where $\mathbf{W} \in \mathbb{R}^{1 \times 2}$ and $b \in \mathbb{R}^{1}$.
- **Line 5 (`model.weight.data = torch.tensor([[1., 1.]])`)**: Manually sets weights $w_1 = 1.0, w_2 = 1.0$.
- **Line 6 (`model.bias.data = torch.tensor([-1.5])`)**: Manually sets bias $b = -1.5$.
- **Linear equation**: $z = 1.0 x_1 + 1.0 x_2 - 1.5$.

---

#### Cell 5 & 6: Forward Pass & Step Thresholding
```python
# Forward pass
z = model(X)
print(z)

pred = (z >= 0).int()
print(pred)
```
- **Line 2 (`z = model(X)`)**: Computes affine linear logit scores:
  - $(0, 0) \implies 1(0) + 1(0) - 1.5 = -1.5$
  - $(0, 1) \implies 1(0) + 1(1) - 1.5 = -0.5$
  - $(1, 0) \implies 1(1) + 1(0) - 1.5 = -0.5$
  - $(1, 1) \implies 1(1) + 1(1) - 1.5 = +0.5$
- **Line 5 (`pred = (z >= 0).int()`)**: Implements Heaviside step activation: returns `1` if $z \ge 0$, else `0`.
- **Output**:
  ```text
  z:
  tensor([[-1.5000],
          [-0.5000],
          [-0.5000],
          [ 0.5000]], grad_fn=<AddmmBackward0>)

  pred:
  tensor([[0],
          [0],
          [0],
          [1]], dtype=torch.int32)
  ```

---

#### Cell 7: Verifying Predictions vs Targets
```python
print("Predicted:", pred.squeeze().tolist())
print("Actual:   ", y.squeeze().int().tolist())
```
- **Line 1-2**: Squeezes $(4, 1)$ tensors to 1D and converts to Python lists.
- **Output**:
  ```text
  Predicted: [0, 0, 0, 1]
  Actual:    [0, 0, 0, 1]
  ```
  *(100% Accuracy for AND gate using single linear boundary $x_1 + x_2 - 1.5 = 0$).*

---

#### Cell 8: Decision Boundary Visualization (AND Gate)
```python
from matplotlib.figure import Figure
from IPython.display import display
from matplotlib.colors import ListedColormap

my_colors = ListedColormap(['red', 'green'])
x_line = torch.tensor([-0.25, 1.7])
y_line = 1.5 - x_line # from x1 + x2 - 1.5 = 0 => x2 = 1.5 - x1

fig = Figure(figsize=(5, 4))
ax = fig.subplots()
ax.scatter(X[:, 0], X[:, 1], c=y, cmap=my_colors)
ax.plot(x_line, y_line)
ax.set(title="AND using SLP", xlabel="x1", ylabel="x2", xlim=(-0.3, 1.7), ylim=(-0.3, 1.7))
for i in range(len(X)):
    ax.text(X[i, 0] + 0.03, X[i, 1] + 0.03, f"y={int(y[i])}")
display(fig)
```
- **Output**: Generates a 2D scatter plot showing a clean dividing line $x_2 = 1.5 - x_1$ separating $(1, 1)$ (green) from the other three points (red).

---

### Section 2: OR Gate with a Single-Layer Perceptron

#### Cell 9 & 10: OR Data & Weights Configuration
```python
# OR gate data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [1]], dtype=torch.float32)

model = nn.Linear(2, 1)
model.weight.data = torch.tensor([[1., 1.]])
model.bias.data = torch.tensor([-0.5]) # Decision boundary: x1 + x2 - 0.5 = 0

z = model(X)
pred = (z >= 0).int()

print("z values: ", z.squeeze().tolist())
print("Predicted:", pred.squeeze().tolist())
print("Actual:   ", y.squeeze().int().tolist())
```
- **Affine Calculation**:
  - $(0, 0) \implies 0 + 0 - 0.5 = -0.5 \implies \text{pred} = 0$
  - $(0, 1) \implies 0 + 1 - 0.5 = +0.5 \implies \text{pred} = 1$
  - $(1, 0) \implies 1 + 0 - 0.5 = +0.5 \implies \text{pred} = 1$
  - $(1, 1) \implies 1 + 1 - 0.5 = +1.5 \implies \text{pred} = 1$
- **Output**:
  ```text
  z values:  [-0.5, 0.5, 0.5, 1.5]
  Predicted: [0, 1, 1, 1]
  Actual:    [0, 1, 1, 1]
  ```
  *(100% Accuracy for OR gate with line $x_2 = 0.5 - x_1$).*

---

### Section 3: The Failure of SLP on XOR

#### Cell 11: Attempting to Classify XOR with a Single Perceptron
```python
# XOR gate data
X = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

model = nn.Linear(2, 1)
model.weight.data = torch.tensor([[1., 1.]])
model.bias.data = torch.tensor([-0.5]) # Trying OR-like weights

z = model(X)
pred = (z >= 0).int()

print("XOR using single perceptron")
print("z values: ", z.squeeze().tolist())
print("Predicted:", pred.squeeze().tolist())
print("Actual:   ", y.squeeze().int().tolist())
```
- **Output**:
  ```text
  XOR using single perceptron
  z values:  [-0.5, 0.5, 0.5, 1.5]
  Predicted: [0, 1, 1, 1]
  Actual:    [0, 1, 1, 0]
  ```
- **Analysis**:
  - Point $(1, 1)$ yields target $y=0$, but the model outputs $\hat{y}=1$ because any single straight line placing $(0, 1)$ and $(1, 0)$ on the positive side **must** also place $(1, 1)$ on the positive side.
  - **Conclusion**: A single straight line cannot separate $\{ (0,1), (1,0) \}$ from $\{ (0,0), (1,1) \}$.

---

### Section 4: Solving XOR with Multi-Layer Perceptron (MLP)

#### Cell 12: Hidden Layer Feature Transformation & Complete Forward Pass
```python
import torch

def step(x):
    return (x >= 0).float()

def hidden_layer_xor_forward(x):
    # Hidden neuron 1: OR Gate (w1=1, w2=1, bias=-0.5)
    h_or = step(1 * x[:, 0:1] + 1 * x[:, 1:2] - 0.5)

    # Hidden neuron 2: NAND Gate (w1=-1, w2=-1, bias=1.5)
    h_nand = step(-1 * x[:, 0:1] - 1 * x[:, 1:2] + 1.5)

    # Output neuron: AND Gate of (h_or, h_nand) (w1=1, w2=1, bias=-1.5)
    output = step(1 * h_or + 1 * h_nand - 1.5)

    hidden_values = torch.cat([h_or, h_nand], dim=1)
    return output, hidden_values

# Forward pass
preds, hidden_values = hidden_layer_xor_forward(X)

print("Hidden layer values: [OR, NAND]")
print(hidden_values.numpy().astype(int))

print("\nModel predictions:")
print("predictions: ", preds.squeeze().int().tolist())
print("targets:     ", y.squeeze().int().tolist())
```
- **Step-by-Step Mathematical Flow**:
  1. Input $(0, 0)$:
     - $h_{\text{OR}} = \text{step}(0 + 0 - 0.5) = 0$
     - $h_{\text{NAND}} = \text{step}(0 + 0 + 1.5) = 1$
     - Latent coordinate: $(0, 1)$
     - Output $= \text{step}(1(0) + 1(1) - 1.5) = \text{step}(-0.5) = \mathbf{0}$
  2. Input $(0, 1)$:
     - $h_{\text{OR}} = \text{step}(0 + 1 - 0.5) = 1$
     - $h_{\text{NAND}} = \text{step}(0 - 1 + 1.5) = 1$
     - Latent coordinate: $(1, 1)$
     - Output $= \text{step}(1(1) + 1(1) - 1.5) = \text{step}(+0.5) = \mathbf{1}$
  3. Input $(1, 0)$:
     - $h_{\text{OR}} = \text{step}(1 + 0 - 0.5) = 1$
     - $h_{\text{NAND}} = \text{step}(-1 + 0 + 1.5) = 1$
     - Latent coordinate: $(1, 1)$
     - Output $= \text{step}(1(1) + 1(1) - 1.5) = \text{step}(+0.5) = \mathbf{1}$
  4. Input $(1, 1)$:
     - $h_{\text{OR}} = \text{step}(1 + 1 - 0.5) = 1$
     - $h_{\text{NAND}} = \text{step}(-1 - 1 + 1.5) = 0$
     - Latent coordinate: $(1, 0)$
     - Output $= \text{step}(1(1) + 1(0) - 1.5) = \text{step}(-0.5) = \mathbf{0}$

- **Output**:
  ```text
  Hidden layer values: [OR, NAND]
  [[0 1]
   [1 1]
   [1 1]
   [1 0]]

  Model predictions:
  predictions:  [0, 1, 1, 0]
  targets:      [0, 1, 1, 0]
  ```

---

## 3. Geometric Interpretation of the Hidden Layer Solution

In the original space $(x_1, x_2)$, XOR points form a non-separable square:
- Class 0: $(0, 0)$ and $(1, 1)$ (opposite diagonals).
- Class 1: $(0, 1)$ and $(1, 0)$ (opposite diagonals).

In the transformed hidden space $(h_1, h_2) = (\text{OR}, \text{NAND})$:
- $(0, 0) \to (0, 1)$ (Class 0)
- $(1, 1) \to (1, 0)$ (Class 0)
- $(0, 1) \to (1, 1)$ (Class 1)
- $(1, 0) \to (1, 1)$ (Class 1)

In this new representation, all Class 1 samples collapse into the single point $(1, 1)$, and a standard linear AND boundary ($h_1 + h_2 - 1.5 = 0$) separates Class 1 from Class 0.
