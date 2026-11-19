# Lab 04: Loss Functions in Practice — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Conceptual Scope

Loss functions measure the objective error between model predictions $\hat{y}$ and ground-truth targets $y$. This lab benchmarks:
1. **Regression Losses (MSE vs. MAE)**: Using California Housing data to analyze outlier sensitivity and loss landscape geometry.
2. **Binary Classification Loss (BCE vs. BCEWithLogits)**: Using Breast Cancer diagnosis data to study numerical stability and the "double-sigmoid" pitfall.
3. **Multi-Class Loss (Categorical Cross-Entropy)**: Using the 10-class Digits dataset with `nn.CrossEntropyLoss`.

---

## 2. Cell-by-Cell & Line-by-Line Code & Output Breakdown

---

### Section 1: Regression Losses (MSE vs. MAE) on California Housing

#### Cell 1: Environment Setup & Seed
```python
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing, load_breast_cancer, load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

torch.manual_seed(42)
```
- **Line 1-7**: Loads scientific packages, PyTorch neural network modules, Scikit-Learn standard datasets, dataset splitters, and feature scalers.
- **Line 8 (`torch.manual_seed(42)`)**: Seeds PRNG for reproducibility.

---

#### Cell 2 to 5: Data Preparation & Outlier Injection
```python
housing = fetch_california_housing(as_frame=True)
df = housing.frame

# Subsample 400 rows for clean 1D visualization
df_sub = df.sample(n=400, random_state=42).reset_index(drop=True)

# 1D Feature: Median Income ('MedInc') -> Target: Median House Value ('MedHouseVal')
X_reg = df_sub[["MedInc"]].values.astype(np.float32)
y_reg = df_sub["MedHouseVal"].values.astype(np.float32)

# Synthetic Outlier Injection: Push 3 samples (indices 5, 120, 300) far away by +20.0
outlier_idx = [5, 120, 300]
y_reg_outliers = y_reg.copy()
y_reg_outliers[outlier_idx] += 20.0

X_reg_t = torch.tensor(X_reg)
y_clean_t = torch.tensor(y_reg)
y_outlier_t = torch.tensor(y_reg_outliers)
```
- **Outputs**:
  ```text
  X_reg shape: (400, 1)
  y_reg shape: (400,)
  ```

---

#### Cell 6 & 7: Linear Model Forward Pass
```python
torch.manual_seed(42)
model = nn.Linear(in_features=1, out_features=1)
y_pred_t = model(X_reg_t).squeeze(1)

print("First 5 predictions:", y_pred_t[:5])
```
- **Output**:
  ```text
  Weight: 0.764539, Bias: 0.830008
  First 5 predictions: tensor([2.1154, 2.7653, 3.4907, 5.2166, 3.6779], grad_fn=<SliceBackward0>)
  ```

---

#### Cell 8 to 11: MSE vs. MAE (Manual vs Built-in)
```python
# 1. Mean Squared Error (MSE / L2)
def mse_manual(y_true, y_pred):
    return torch.mean((y_true - y_pred)**2)

# 2. Mean Absolute Error (MAE / L1)
def mae_manual(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

mse_fn = nn.MSELoss()
mae_fn = nn.L1Loss()

mse_clean = mse_fn(y_pred_t, y_clean_t).item()
mse_out = mse_fn(y_pred_t, y_outlier_t).item()
mae_clean = mae_fn(y_pred_t, y_clean_t).item()
mae_out = mae_fn(y_pred_t, y_outlier_t).item()
```
- **Quantitative Comparison Table Output**:
  ```text
  Metric              Clean data     Outlier data   Change         
  -----------------------------------------------------------------
  MSE (manual)        4.4830         6.9770         +2.4940 (+55.6%)
  MSE (built-in)      4.4830         6.9770         +2.4940 (+55.6%)
  MAE (manual)        1.8151         1.9398         +0.1247 (+6.9%)
  MAE (built-in)      1.8151         1.9398         +0.1247 (+6.9%)
  ```
- **Analysis**:
  - In MSE, squaring just 3 outliers inflated the loss by **$+55.6\%$**.
  - In MAE, linear penalty limited the outlier inflation to only **$+6.9\%$**.

---

#### Cell 12: Plotly Loss Surface Sweep
```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

weights = np.linspace(-1, 1, 100)
# Evaluates loss across weight range for MSE vs MAE (clean vs with outliers)
```
- **Key Geometric Insights**:
  - **MSE Loss Landscape**: The outlier parabola (red) becomes steeper and its minimum **shifts significantly** along the horizontal axis. The model is forced to compromise fit on all 397 normal points to accommodate 3 outliers.
  - **MAE Loss Landscape**: The outlier V-shape (red) shifts up almost by a uniform vertical constant. The location of the minimum **barely moves**, preserving the optimal weight.

---

### Section 2: Binary Classification & The Double-Sigmoid Pitfall

#### Cell 13 to 17: Breast Cancer Dataset Preprocessing
```python
bc = load_breast_cancer()
X_bc = bc.data.astype(np.float32)
y_bc = bc.target.astype(np.float32)

# Stratified Split (80% Train, 20% Test)
X_train_bc, X_test_bc, y_train_bc, y_test_bc = train_test_split(
    X_bc, y_bc, test_size=0.2, random_state=42, stratify=y_bc
)

scaler_bc = StandardScaler()
X_train_bc = scaler_bc.fit_transform(X_train_bc).astype(np.float32)
X_test_bc = scaler_bc.transform(X_test_bc).astype(np.float32)

X_test_bc_t = torch.tensor(X_test_bc)
y_test_bc_t = torch.tensor(y_test_bc)

model_bc = nn.Linear(in_features=30, out_features=1)
with torch.no_grad():
    test_logits = model_bc(X_test_bc_t[:5]).squeeze(1)
    test_probs = torch.sigmoid(test_logits)
```
- **Outputs**:
  ```text
  Raw logits:     tensor([ 1.1869, -0.1040, -0.2470,  0.4076,  2.0179])
  Probabilities:  tensor([0.7662,  0.4740,  0.4386,  0.6005,  0.8827])
  ```

---

#### Cell 18 to 21: Demonstrating `BCELoss` vs `BCEWithLogitsLoss`
```python
# 1. BCELoss expects probabilities
bce_loss_fn = nn.BCELoss()
bce_result = bce_loss_fn(test_probs, y_test_bc_t[:5])

# 2. COMMON MISTAKE: Passing probabilities to BCEWithLogitsLoss
bce_logits_loss_fn = nn.BCEWithLogitsLoss()
bce_logits_buggy = bce_logits_loss_fn(test_probs, y_test_bc_t[:5])

# 3. CORRECT USE: Passing raw logits to BCEWithLogitsLoss
bce_logits_fixed = bce_logits_loss_fn(test_logits, y_test_bc_t[:5])

print(f"BCELoss(probabilities)   [Correct]: {bce_result.item():.6f}")
print(f"BCEWithLogitsLoss(probs) [BUGGY]:   {bce_logits_buggy.item():.6f}")
print(f"BCEWithLogitsLoss(logits)[Correct]: {bce_logits_fixed.item():.6f}")
print(f"Does the fixed version match BCELoss? {torch.allclose(bce_result, bce_logits_fixed, atol=1e-5)}")
```
- **Output**:
  ```text
  BCELoss(probabilities)   [Correct]: 1.085942
  BCEWithLogitsLoss(probs) [BUGGY]:   0.846862
  BCEWithLogitsLoss(logits)[Correct]: 1.085942
  Does the fixed version match BCELoss? True
  ```
- **Why does the bug occur?**
  `nn.BCEWithLogitsLoss()` internally applies Sigmoid. When fed `test_probs`, it applies Sigmoid **a second time**: $\sigma(\sigma(z))$. This squashes all probabilities toward $0.5$, artificially hiding model confidence and producing corrupted loss gradients.

---

### Section 3: Multi-Class Cross-Entropy on Digits Dataset

#### Cell 22 to 25: Loading Digits & Model Architecture
```python
digits = load_digits()
X_dig = digits.data.astype(np.float32)       # (1797, 64) pixel features
y_dig = digits.target.astype(np.int64)      # (1797,) integer class labels in [0, 9]

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(
    X_dig, y_dig, test_size=0.2, random_state=42, stratify=y_dig
)

scaler_d = StandardScaler()
X_test_d = scaler_d.fit_transform(X_test_d).astype(np.float32)
X_test_d_t = torch.tensor(X_test_d)
y_test_d_t = torch.tensor(y_test_d)

# Multi-class Linear Model: 64 inputs -> 10 class logits
model_dig = nn.Linear(in_features=64, out_features=10)

with torch.no_grad():
    test_logits_d = model_dig(X_test_d_t[:5])
    test_probs_d = torch.softmax(test_logits_d, dim=1)

# Loss calculation: nn.CrossEntropyLoss expects raw logits directly
criterion_ce = nn.CrossEntropyLoss()
ce_result = criterion_ce(test_logits_d, y_test_d_t[:5])

print("Logits shape:               ", test_logits_d.shape)
print(f"CrossEntropyLoss(raw logits): {ce_result.item():.4f}")
print("Sum of Softmax Probabilities:", test_probs_d[0].sum().item())
```
- **Output**:
  ```text
  Logits shape:                torch.Size([5, 10])
  CrossEntropyLoss(raw logits): 2.1846
  Sum of Softmax Probabilities: 1.000000
  ```

---

## 3. Loss Selection Rules Summary

| Task | Output Activation | Loss Function | Input to Loss Function |
| :--- | :--- | :--- | :--- |
| **Standard Regression** | None / Linear | `nn.MSELoss()` | Predicted continuous scalar $\hat{y}$ |
| **Outlier-Heavy Regression**| None / Linear | `nn.L1Loss()` or `nn.HuberLoss()` | Predicted continuous scalar $\hat{y}$ |
| **Binary Classification** | None / Linear | `nn.BCEWithLogitsLoss()` | Raw unnormalized logit $z \in \mathbb{R}$ |
| **Multi-Class ($K$ Classes)** | None / Linear | `nn.CrossEntropyLoss()` | Raw unnormalized logits $\mathbf{z} \in \mathbb{R}^K$ |
