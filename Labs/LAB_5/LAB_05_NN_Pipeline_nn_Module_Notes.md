# Lab 05: Building a Binary Classifier with PyTorch's `nn.Module` — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Learning Objectives

This lab implements an end-to-end binary classification pipeline using PyTorch's `torch.nn.Module` subclassing on the real-world **Wisconsin Breast Cancer Dataset** (569 samples, 30 geometric features, predicting Malignant vs. Benign).

### Key Architectural Concepts:
1. **`nn.Module` Encapsulation**: Registering sub-layers (`nn.Linear`, `nn.Sigmoid`) in `__init__()` and defining data flow in `forward()`.
2. **Parameter Tracking**: Dynamic parameter discovery via `model.parameters()`.
3. **Autograd Engine**: Tracing computational graphs, calculating gradients with `loss.backward()`, and resetting accumulation buffers with `optimizer.zero_grad()`.
4. **Data Preprocessing**: Preventing train-test data leakage, feature scaling (`StandardScaler`), label encoding (`LabelEncoder`), and resolving `float32` vs `float64` tensor dtype mismatches.

---

## 2. Cell-by-Cell & Line-by-Line Code & Output Breakdown

---

### Step 1: Environment Setup & Seed Configuration
```python
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
```
- **Line 1-9**: Imports required data manipulation, machine learning, and deep learning libraries.
- **Line 11-13**: Sets deterministic seeds across NumPy and PyTorch PRNGs.

---

### Step 2: Data Ingestion, Cleaning & Preprocessing
```python
# 1. Load raw CSV
df = pd.read_csv('https://raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/refs/heads/master/data.csv')

# 2. Clean irrelevant and corrupted columns (id and Unnamed: 32)
df.drop(columns=['id', 'Unnamed: 32'], inplace=True)

# 3. Separate features and target
X = df.drop(columns=['diagnosis']) # (569, 30)
y = df['diagnosis']                # (569,)

# 4. Train/Test Split (80% Train, 20% Test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

# 5. Feature Scaling (Zero-mean, unit variance)
# IMPORTANT: fit_transform ONLY on training set to prevent data leakage!
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 6. Label Encoding ('M' -> 1, 'B' -> 0)
encoder = LabelEncoder()
y_train = encoder.fit_transform(y_train)
y_test = encoder.transform(y_test)

# 7. Convert NumPy arrays to PyTorch Float Tensors (Cast to float32 to avoid Double vs Float errors)
X_train_tensor = torch.from_numpy(X_train.astype(np.float32))
X_test_tensor  = torch.from_numpy(X_test.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
y_test_tensor  = torch.from_numpy(y_test.astype(np.float32))

print("X_train_tensor shape:", X_train_tensor.shape)
print("y_train_tensor shape:", y_train_tensor.shape)
```
- **Output**:
  ```text
  X_train_tensor: torch.Size([455, 30])
  y_train_tensor: torch.Size([455])
  ```

---

### Step 3: Custom `nn.Module` Architecture & Loss Definition
```python
class MySimpleNN(nn.Module):
    def __init__(self, num_features):
        super().__init__() # Must be called first to initialize module registry
        # Learnable affine transformation layer: (Batch_Size, 30) -> (Batch_Size, 1)
        self.linear = nn.Linear(num_features, 1)
        # Non-linear probability squashing: (0, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, features):
        out = self.linear(features)
        out = self.sigmoid(out)
        return out

# Instantiate model, loss criterion, and optimizer
model = MySimpleNN(X_train_tensor.shape[1])
learning_rate = 0.1
epochs = 25

loss_function = nn.BCELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
```
- **Line 1-13 (`MySimpleNN`)**:
  - `num_features = 30`: Matches input tensor column dimension.
  - `out_features = 1`: Outputs single probability $P(\text{Malignant})$.
  - `self.sigmoid`: Squashes $z = \mathbf{w}^T \mathbf{x} + b \in \mathbb{R}$ into $\hat{y} \in (0, 1)$.
- **Line 16-20**:
  - `model.parameters()`: Automatically passes all learnable weights ($W \in \mathbb{R}^{1 \times 30}, b \in \mathbb{R}^1$) to Stochastic Gradient Descent (`optim.SGD`).

---

### Step 4: The 5-Step PyTorch Training Loop
```python
train_losses = []

for epoch in range(epochs):
    # --- Step 1: Forward Pass ---
    y_pred = model(X_train_tensor) # Shape: (455, 1)
    
    # --- Step 2: Loss Calculation ---
    # Reshape target from (455,) -> (455, 1) to avoid tensor broadcasting bug!
    loss = loss_function(y_pred, y_train_tensor.view(-1, 1))
    
    # --- Step 3: Zero Gradients ---
    # Gradients accumulate in PyTorch; must clear previous epoch buffers
    optimizer.zero_grad()
    
    # --- Step 4: Backward Pass (Autograd) ---
    loss.backward()
    
    # --- Step 5: Parameter Update Step ---
    optimizer.step()
    
    train_losses.append(loss.item())
    print(f'Epoch: {epoch + 1:2d}, Loss: {loss.item():.6f}')
```
- **Epoch Loss Progression Output**:
  ```text
  Epoch:  1, Loss: 0.670354
  Epoch:  2, Loss: 0.536520
  Epoch:  3, Loss: 0.455513
  Epoch:  4, Loss: 0.401915
  Epoch:  5, Loss: 0.363852
  Epoch:  6, Loss: 0.335302
  Epoch:  7, Loss: 0.312960
  Epoch:  8, Loss: 0.294886
  ...
  Epoch: 24, Loss: 0.182454
  Epoch: 25, Loss: 0.179303
  ```

---

### Step 5: Model Evaluation & Testing Accuracy
```python
# 1. Switch to evaluation context (disables autograd graph building)
with torch.no_grad():
    y_pred_test = model(X_test_tensor) # Shape: (114, 1)
    # 2. Decision thresholding: probability >= 0.5 -> Class 1 (Malignant)
    y_pred_labels = (y_pred_test > 0.5).float()
    
    # 3. Compute accuracy
    accuracy = (y_pred_labels == y_test_tensor.view(-1, 1)).float().mean()
    print(f'Final Test Set Accuracy: {accuracy.item() * 100:.2f}%')
```
- **Output**:
  ```text
  Final Test Set Accuracy: 98.25%
  ```

---

### Step 6: Loss Curve Visualization
```python
plt.figure(figsize=(7, 4))
plt.plot(range(1, epochs + 1), train_losses, color='blue', marker='o')
plt.title("Training Loss Curve over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Binary Cross-Entropy Loss")
plt.grid(True, alpha=0.3)
plt.show()
```
- **Output**: Generates a smooth, monotonically decreasing loss trajectory converging from $0.670 \to 0.179$.

---

## 3. Top 3 PyTorch Pitfalls & Best Practices

1. **Missing `optimizer.zero_grad()`**:
   - PyTorch accumulates gradients in `.grad` by default. Omitting `zero_grad()` compounds gradients across epochs, leading to catastrophic divergent updates.
2. **Tensor Shape Mismatch in Loss Functions (`(N,)` vs `(N, 1)`)**:
   - If model outputs `(N, 1)` and target is `(N,)`, PyTorch silently broadcasts them into an `(N, N)` matrix without raising an error, producing garbage loss values. Always use `target.view(-1, 1)` or `target.unsqueeze(1)`.
3. **Omitting `torch.no_grad()` in Evaluation**:
   - Running inference without `with torch.no_grad():` retains intermediate activation tensors in RAM/VRAM, leading to memory leaks.
