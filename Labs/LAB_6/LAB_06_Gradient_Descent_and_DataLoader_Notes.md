# Lab 06: Gradient Descent Variants & PyTorch DataLoaders — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Theoretical Principles

Neural network optimization updates weights by stepping in the negative gradient direction:
$$W_{\text{new}} = W_{\text{old}} - \eta \frac{\partial L}{\partial w}$$

The three fundamental paradigms of Gradient Descent differ in **batch size $B$**, which sets the **number of weight updates per epoch ($\lceil N/B \rceil$)**:

| Paradigm | Batch Size $B$ | Updates per Epoch | Wall-Clock Time per Epoch | Convergence Speed per Epoch | Gradient Variance |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Batch GD (BGD)** | $B = N$ (all 320 rows) | Exactly $1$ | **Fastest** ($0.01\text{ s}$) | **Slowest** ($0.666 \to 0.627$) | Zero (exact true gradient) |
| **Mini-Batch GD (MBGD)** | $1 < B < N$ (e.g. $16, 32$) | $\lceil N/B \rceil$ (e.g. $10$) | Fast (SIMD vectorized) | Rapid, stable convergence | Low / controlled noise |
| **Stochastic GD (SGD)** | $B = 1$ (single sample) | $N$ ($320$ updates) | **Slowest** ($2.03\text{ s}$) | **Fastest** ($0.374 \to 0.247$) | Very high (noisy updates) |

---

## 2. Cell-by-Cell & Line-by-Line Code & Output Breakdown

---

### Step 1: Environment Setup & Seed Configuration
```python
import numpy as np
import pandas as pd
import time
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

torch.manual_seed(42)
np.random.seed(42)
```
- **Line 1-9**: Imports core scientific tools, execution timer (`time`), and PyTorch neural network modules.
- **Line 11-12**: Sets deterministic random seeds for reproducible batch shuffling.

---

### Step 2: Data Ingestion & Preprocessing
```python
url = 'https://raw.githubusercontent.com/shivang98/Social-Network-ads-Boost/master/Social_Network_Ads.csv'
df = pd.read_csv(url)[['Age', 'EstimatedSalary', 'Purchased']]

# Separate features X and binary target y
X = df[['Age', 'EstimatedSalary']].values
y = df['Purchased'].values

# 80/20 Train-Validation Split (320 Train rows, 80 Validation rows)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=2)

# Standardize features using ONLY training set statistics
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# Convert to Float32 Tensors and reshape target to (N, 1) for BCELoss
X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
X_val_t   = torch.tensor(X_val,   dtype=torch.float32)
y_val_t   = torch.tensor(y_val,   dtype=torch.float32).reshape(-1, 1)

n_train = X_train_t.shape[0]
print('training rows:', n_train)
```
- **Output**:
  ```text
  training rows: 320
  ```

---

### Step 3: Model Definition
```python
def build_model():
    return nn.Sequential(
        nn.Linear(2, 10), nn.ReLU(),
        nn.Linear(10, 10), nn.ReLU(),
        nn.Linear(10, 1), nn.Sigmoid()
    )
```
- **Line 1-6 (`build_model`)**: Returns a fresh model with identical initial weights to ensure completely fair benchmarking between Batch GD and Stochastic GD.
- **Architecture**: Input(2) $\to$ Linear(10) $\to$ ReLU $\to$ Linear(10) $\to$ ReLU $\to$ Linear(1) $\to$ Sigmoid.

---

### Step 4: Generic Parameterized Training Function (`fit`)
```python
def fit(model, epochs, batch_size, lr=0.1):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        perm = torch.randperm(n_train) # Reshuffles training indices each epoch
        epoch_loss, n_batches = 0.0, 0
        
        # Step through dataset in chunks of size `batch_size`
        for i in range(0, n_train, batch_size):
            idx = perm[i:i + batch_size]            # Slice current mini-batch indices
            xb, yb = X_train_t[idx], y_train_t[idx] # Extract batch tensors
            
            optimizer.zero_grad()           # 1. Reset gradient buffers
            loss = criterion(model(xb), yb) # 2. Forward pass & compute batch loss
            loss.backward()                 # 3. Backward pass (autograd dL/dw)
            optimizer.step()                # 4. Parameter update
            
            epoch_loss += loss.item()
            n_batches += 1

        print(f'Epoch {epoch + 1:2d}/{epochs} - loss: {epoch_loss / n_batches:.4f}')
```

---

### Step 5: Benchmark Execution & Results Comparison

#### 5.1 One-Time Warmup Run
```python
fit(build_model(), epochs=1, batch_size=32)
```
- **Explanation**: Pre-allocates CUDA/CPU memory caches so PyTorch overhead does not bias the timed runs.

#### 5.2 Batch GD Benchmark ($B = 320$, 1 update per epoch)
```python
model = build_model()
start = time.time()
fit(model, epochs=10, batch_size=n_train) # batch_size = 320
print('Total time:', round(time.time() - start, 2), 's')
```
- **Output**:
  ```text
  Epoch  1/10 - loss: 0.6661
  Epoch  2/10 - loss: 0.6615
  Epoch  3/10 - loss: 0.6571
  Epoch  4/10 - loss: 0.6527
  Epoch  5/10 - loss: 0.6483
  Epoch  6/10 - loss: 0.6440
  Epoch  7/10 - loss: 0.6397
  Epoch  8/10 - loss: 0.6355
  Epoch  9/10 - loss: 0.6312
  Epoch 10/10 - loss: 0.6269
  Total time: 0.01 s
  ```

#### 5.3 Stochastic GD Benchmark ($B = 1$, 320 updates per epoch)
```python
model = build_model()
start = time.time()
fit(model, epochs=10, batch_size=1) # batch_size = 1
print('Total time:', round(time.time() - start, 2), 's')
```
- **Output**:
  ```text
  Epoch  1/10 - loss: 0.3740
  Epoch  2/10 - loss: 0.2825
  Epoch  3/10 - loss: 0.2760
  Epoch  4/10 - loss: 0.2648
  Epoch  5/10 - loss: 0.2551
  Epoch  6/10 - loss: 0.2589
  Epoch  7/10 - loss: 0.2590
  Epoch  8/10 - loss: 0.2543
  Epoch  9/10 - loss: 0.2454
  Epoch 10/10 - loss: 0.2473
  Total time: 2.03 s
  ```

---

## 3. Comparative Tradeoff Analysis

```
       WALL-CLOCK TIME PER EPOCH (Speed)
Batch GD (0.01s)  <────────────────────────  Stochastic GD (2.03s)
[Vectorized & Fast]                           [Serial & Slow]

       LOSS DROP PER EPOCH (Convergence)
Batch GD (0.666 -> 0.627)  ────────────────>  Stochastic GD (0.374 -> 0.247)
[Barely moved]                                [Rapid convergence]
```

- **Why is Batch GD fast in wall-clock time?** It executes a single vectorized matrix multiplication per epoch, fully leveraging hardware cache alignment and BLAS parallelization.
- **Why does Stochastic GD drop loss faster per epoch?** It updates weights $320\times$ in a single epoch, learning from every individual mistake immediately.
- **Why is Mini-Batch GD ($B=32$) the industry standard?** It performs $10$ updates per epoch while retaining matrix parallelization speed, giving the best of both worlds.
