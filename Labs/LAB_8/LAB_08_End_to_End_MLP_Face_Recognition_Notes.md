# Lab 08: End-to-End MLP Pipeline — Face Recognition (Olivetti Faces)
**Notebook Reference:** `Lab_8_Solution_Face_Recognition.ipynb` (and `Lab_8_(Boilerplate_Face_Recognition).ipynb`)  
**Focus:** Full PyTorch Multilayer Perceptron Workflow — Dataset Ingestion, Preprocessing (Missing Values, Stratified Splits, Standardization), Custom `nn.Module` Architecture, Numerically Stable Multi-Class Loss, Manual Mini-Batch SGD/Adam Loop with `torch.randperm`, Metrics, and Inference Visualization.

---

## 1. Problem Definition & Mathematical Architecture

### Dataset Specifications: AT&T / Olivetti Faces
- **Total Samples:** $N = 400$ grayscale facial images.
- **Classes ($K$):** $40$ distinct individuals (10 images per subject with varying lighting, facial expressions, and details like glasses/open eyes).
- **Image Resolution:** $64 \times 64$ pixels.
- **Input Feature Vector Dimension:** $d = 64 \times 64 = 4096$ continuous pixel intensity values in $[0.0, 1.0]$.
- **Target Label Vector:** $y \in \{0, 1, \dots, 39\}$ representing person ID.

### Network Architecture Specification
$$\begin{aligned}
\mathbf{x} &\in \mathbb{R}^{4096} \\
\mathbf{h}_1 &= \text{ReLU}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1) \quad (\mathbf{W}_1 \in \mathbb{R}^{512 \times 4096}, \mathbf{b}_1 \in \mathbb{R}^{512}) \\
\mathbf{h}_2 &= \text{ReLU}(\mathbf{W}_2 \mathbf{h}_1 + \mathbf{b}_2) \quad (\mathbf{W}_2 \in \mathbb{R}^{128 \times 512}, \mathbf{b}_2 \in \mathbb{R}^{128}) \\
\mathbf{z} &= \mathbf{W}_3 \mathbf{h}_2 + \mathbf{b}_3 \quad (\mathbf{W}_3 \in \mathbb{R}^{40 \times 128}, \mathbf{b}_3 \in \mathbb{R}^{40}) \\
\hat{\mathbf{y}} &= \text{Softmax}(\mathbf{z})
\end{aligned}$$

```text
┌─────────────────────────┐
│ Input Vector (4096-dim) │
└────────────┬────────────┘
             │  Linear (4096 -> 512)
             ▼
┌─────────────────────────┐
│ Hidden Layer 1 (512)    │ ──► [ReLU Activation]
└────────────┬────────────┘
             │  Linear (512 -> 128)
             ▼
┌─────────────────────────┐
│ Hidden Layer 2 (128)    │ ──► [ReLU Activation]
└────────────┬────────────┘
             │  Linear (128 -> 40)
             ▼
┌─────────────────────────┐
│ Output Logits (40-dim)  │ ──► [CrossEntropyLoss (LogSoftmax + NLL)]
└─────────────────────────┘
```

---

## 2. Cell-by-Cell & Line-by-Line Code Breakdown

---

### Cell 1: Module Imports, Seed Initialization, and Device Selection (Code)

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

torch.manual_seed(42)
np.random.seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
```

#### Line-by-Line Explanation
1. `import numpy as np`, `import matplotlib.pyplot as plt`: Imports core scientific matrix operations and data visualization libraries.
2. `import torch`, `import torch.nn as nn`, `import torch.optim as optim`: Imports PyTorch root tensor library, neural network layer abstractions (`nn.Linear`, `nn.Sequential`), and gradient descent optimizers (`optim.Adam`).
3. `from sklearn.datasets import fetch_olivetti_faces`: Imports scikit-learn's built-in AT&T Olivetti face image downloader.
4. `from sklearn.model_selection import train_test_split`: Imports data splitting utility supporting stratified sampling.
5. `from sklearn.preprocessing import StandardScaler`: Imports standard score feature normalizer ($z = \frac{x - \mu}{\sigma}$).
6. `from sklearn.metrics import accuracy_score, confusion_matrix`: Imports multi-class evaluation metric functions.
7. `torch.manual_seed(42)`, `np.random.seed(42)`: Fixes pseudo-random number generation across CPU tensor operations and NumPy array splits for exact experimental reproducibility.
8. `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`: Checks for NVIDIA CUDA hardware acceleration; defaults to host CPU if absent.
9. `print("Using device:", device)`: Outputs active compute backend (`cuda` or `cpu`).

---

### Cell 2: Dataset Ingestion, Inspection, and Visualization (Code)

```python
data = fetch_olivetti_faces()
X, y = data.data, data.target   # X: (400, 4096) flattened images, y: (400,) person labels 0-39

print("X shape:", X.shape)
print("y shape:", y.shape)
print("Number of classes:", len(np.unique(y)))
print("Pixel value range:", X.min(), "to", X.max())

# ---- Visualize a few samples ----
fig, axes = plt.subplots(1, 5, figsize=(10, 3))
for i, ax in enumerate(axes):
    ax.imshow(X[i].reshape(64, 64), cmap='gray')
    ax.set_title(f"Person {y[i]}")
    ax.axis('off')
plt.suptitle("Sample Faces")
plt.show()
```

#### Line-by-Line Explanation
1. `data = fetch_olivetti_faces()`: Downloads/loads dataset container with attributes `.data`, `.images`, `.target`.
2. `X, y = data.data, data.target`:
   - `X`: NumPy float32 matrix of shape `(400, 4096)` where each row is a flattened $64 \times 64$ face image.
   - `y`: 1D NumPy int64 array of shape `(400,)` containing class IDs $0$ through $39$.
3. `print("X shape:", X.shape)`: Outputs `(400, 4096)`.
4. `print("y shape:", y.shape)`: Outputs `(400,)`.
5. `print("Number of classes:", len(np.unique(y)))`: Outputs `40`.
6. `print("Pixel value range:", X.min(), "to", X.max())`: Outputs `0.0 to 1.0` (pixel intensities already float-normalized).
7. `fig, axes = plt.subplots(1, 5, figsize=(10, 3))`: Creates a 1-row, 5-column figure grid.
8. `for i, ax in enumerate(axes):`: Iterates through the first 5 sample images.
   - `ax.imshow(X[i].reshape(64, 64), cmap='gray')`: Unflattens 4096D vector back to $64 \times 64$ 2D spatial grid and renders in grayscale.
   - `ax.set_title(f"Person {y[i]}")`: Displays ground-truth subject ID above each thumbnail.
   - `ax.axis('off')`: Disables axis tick marks for clean image display.
9. `plt.suptitle("Sample Faces")`, `plt.show()`: Renders gallery of faces.

---

### Cell 4: Preprocessing — Missing Values, Stratified Split, and Feature Scaling (Code)

```python
# 1. Check for missing values
has_missing = np.isnan(X).any()
print("Any missing values?", has_missing)
# Olivetti Faces has none, but we check generically for pipeline completeness
if has_missing:
    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(col_means, inds[1])

# 2. Train/test split — stratified so every person appears in both sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3. Feature scaling — fit ONLY on train, then transform both
# (Fitting on test too would leak test-set statistics into training — "data leakage")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("Train shape:", X_train.shape, "Test shape:", X_test.shape)
```

#### Line-by-Line Explanation
1. `has_missing = np.isnan(X).any()`: Scans full $400 \times 4096$ matrix for `NaN` entries. Returns `False`.
2. `if has_missing: ...`: Generic imputation fallback using column-wise means $\mu_j$ for pipeline completeness.
3. `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)`:
   - `test_size=0.2`: Allocates $80\%$ ($320$ samples) for training and $20\%$ ($80$ samples) for testing.
   - `stratify=y`: **Crucial requirement!** With 40 classes and only 10 images per class, random splitting without stratification could place all 10 images of a person into training or testing. Stratification guarantees exactly 8 training images and 2 test images per individual.
   - `random_state=42`: Ensures deterministic split.
4. `scaler = StandardScaler()`: Instantiates z-score feature normalizer.
5. `X_train = scaler.fit_transform(X_train)`: Computes training mean $\mu_{\text{train}}$ and standard deviation $\sigma_{\text{train}}$ per pixel, then transforms $X_{\text{train}} \leftarrow \frac{X_{\text{train}} - \mu_{\text{train}}}{\sigma_{\text{train}}}$.
6. `X_test = scaler.transform(X_test)`: Normalizes test set using training statistics $\mu_{\text{train}}, \sigma_{\text{train}}$ without re-fitting. **Prevents data leakage.**
7. `print("Train shape:", X_train.shape, "Test shape:", X_test.shape)`: Outputs `Train shape: (320, 4096) Test shape: (80, 4096)`.

---

### Cell 6: Conversion to PyTorch Tensors and Device Placement (Code)

```python
X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
X_test_t  = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_t  = torch.tensor(y_test, dtype=torch.long).to(device)

print("Train tensor shape:", X_train_t.shape)
print("Test tensor shape:", X_test_t.shape)
```

#### Line-by-Line Explanation
1. `X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)`: Converts normalized training features to 32-bit floating-point PyTorch tensor and moves to target hardware (`cuda` or `cpu`).
2. `y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)`: Converts training labels to 64-bit integer tensor (`torch.long`). **Required by `nn.CrossEntropyLoss` for class indices.**
3. `X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)`: Converts test features to `torch.float32`.
4. `y_test_t = torch.tensor(y_test, dtype=torch.long).to(device)`: Converts test labels to `torch.long`.
5. `print(...)`: Confirms tensor shapes: `(320, 4096)` and `(80, 4096)`.

---

### Cell 8: Multilayer Perceptron Architecture Definition (Code)

```python
class FaceMLP(nn.Module):
    def __init__(self, input_dim=4096, num_classes=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),              # ReLU: avoids vanishing gradients, fast convergence in hidden layers
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)   # raw logits — softmax is applied inside CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)

model = FaceMLP().to(device)
print(model)
```

#### Line-by-Line Explanation
1. `class FaceMLP(nn.Module):`: Subclasses PyTorch's base neural network class.
2. `def __init__(self, input_dim=4096, num_classes=40):`: Constructor initializing network layers.
   - `super().__init__()`: Initializes base `nn.Module` internal machinery (parameter registration, hook tracking).
   - `self.net = nn.Sequential(...)`: Chains layers in linear execution sequence:
     - `nn.Linear(4096, 512)`: Layer 1 mapping $4096$ inputs to $512$ hidden units (Parameters: $4096 \times 512 + 512 = 2,097,664$).
     - `nn.ReLU()`: Element-wise $\max(0, z)$ activation preventing vanishing gradients in deep representations.
     - `nn.Linear(512, 128)`: Layer 2 mapping $512 \to 128$ hidden units (Parameters: $512 \times 128 + 128 = 65,664$).
     - `nn.ReLU()`: Second non-linear activation.
     - `nn.Linear(128, 40)`: Output classification layer mapping $128 \to 40$ class logits (Parameters: $128 \times 40 + 40 = 5,160$).
     - *Total Trainable Parameters:* $2,097,664 + 65,664 + 5,160 = 2,168,488$.
3. `def forward(self, x): return self.net(x)`: Implements the forward computational graph passing batch $x$ through `self.net`.
4. `model = FaceMLP().to(device)`: Instantiates model and moves all weight tensors to GPU/CPU memory.
5. `print(model)`: Prints sequential layer layout.

---

### Cell 10: Loss Function & Optimizer Setup (Code)

```python
criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(model.parameters(), lr=0.001)
```

#### Line-by-Line Explanation
1. `criterion = nn.CrossEntropyLoss()`:
   - Combines `nn.LogSoftmax()` and `nn.NLLLoss()` (Negative Log-Likelihood Loss) in a single numerically stable formula using the Log-Sum-Exp identity:
     $$\mathcal{L}(\mathbf{z}, y) = -\log\left(\frac{e^{z_y}}{\sum_{j=1}^{40} e^{z_j}}\right) = -z_y + \log\left(\sum_{j=1}^{40} e^{z_j}\right)$$
   - Takes raw unnormalized logits $\mathbf{z} \in \mathbb{R}^{B \times 40}$ and ground-truth integer labels $y \in \{0, \dots, 39\}^B$.
2. `optimizer = optim.Adam(model.parameters(), lr=0.001)`:
   - Instantiates Adam optimizer over all 2.16M model parameters with initial learning rate $\eta = 10^{-3}$ and default moments $(\beta_1 = 0.9, \beta_2 = 0.999, \epsilon = 10^{-8})$.

---

### Cell 12: Mini-Batch Training Loop with Manual Index Shuffling (Code)

```python
num_epochs = 60
batch_size = 16
n_samples = X_train_t.size(0)

train_losses, train_accs = [], []

for epoch in range(num_epochs):
    model.train()

    perm = torch.randperm(n_samples)   # shuffle indices each epoch

    running_loss, correct, total = 0.0, 0, 0

    for i in range(0, n_samples, batch_size):
        idx = perm[i:i+batch_size]
        xb, yb = X_train_t[idx], y_train_t[idx]

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * xb.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == yb).sum().item()
        total += yb.size(0)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    train_losses.append(epoch_loss)
    train_accs.append(epoch_acc)

    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}] Loss: {epoch_loss:.4f} Train Acc: {epoch_acc:.4f}")
```

#### Line-by-Line Explanation
1. `num_epochs = 60`, `batch_size = 16`, `n_samples = X_train_t.size(0)`: Configures training budget: 60 epochs, mini-batch size $B = 16$ ($320 / 16 = 20$ mini-batches per epoch), $N = 320$.
2. `train_losses, train_accs = [], []`: Epoch-level metric history arrays.
3. `for epoch in range(num_epochs):`:
   - `model.train()`: Puts model into training mode.
   - `perm = torch.randperm(n_samples)`: Generates random permutation of integers $\{0, 1, \dots, 319\}$ to shuffle samples without creating data duplicates.
   - `running_loss, correct, total = 0.0, 0, 0`: Initializes batch accumulators.
   - `for i in range(0, n_samples, batch_size):`: Iterates through data in chunks of 16.
     - `idx = perm[i:i+batch_size]`: Slices 16 shuffled indices.
     - `xb, yb = X_train_t[idx], y_train_t[idx]`: Extracts mini-batch tensors of shape `(16, 4096)` and `(16,)`.
     - `optimizer.zero_grad()`: Clears accumulated gradients.
     - `outputs = model(xb)`: Forward pass computing logits of shape `(16, 40)`.
     - `loss = criterion(outputs, yb)`: Computes Cross-Entropy scalar loss.
     - `loss.backward()`: Reverse-mode autograd backpropagating $\frac{\partial \mathcal{L}}{\partial W}, \frac{\partial \mathcal{L}}{\partial b}$.
     - `optimizer.step()`: Updates weights via Adam update equations.
     - `running_loss += loss.item() * xb.size(0)`: Accumulates unweighted batch loss.
     - `preds = outputs.argmax(dim=1)`: Obtains predicted class index $\hat{y} = \arg\max_{j} z_j$.
     - `correct += (preds == yb).sum().item()`: Tallies exact class matches.
     - `total += yb.size(0)`: Increments sample count.
   - `epoch_loss = running_loss / total`, `epoch_acc = correct / total`: Computes true epoch average loss and accuracy.
   - `if (epoch + 1) % 10 == 0: print(...)`: Logs progress every 10 epochs.

#### Training Progression
- Epoch 10: Loss $\approx 0.45$, Accuracy $\approx 92\%$
- Epoch 30: Loss $\approx 0.03$, Accuracy $\approx 99\%$
- Epoch 60: Loss $\approx 0.001$, Accuracy $= 100.0\%$ (Training loss converges asymptotically to zero).

---

### Cell 14: Test Evaluation, Loss Curve, and Confusion Matrix (Code)

```python
model.eval()

with torch.no_grad():
    outputs = model(X_test_t)
    preds = outputs.argmax(dim=1)

all_preds = preds.cpu().numpy()
all_labels = y_test_t.cpu().numpy()

test_acc = accuracy_score(all_labels, all_preds)
print(f"Test Accuracy: {test_acc:.4f}")

# ---- Loss curve ----
plt.figure(figsize=(6,4))
plt.plot(train_losses, label="Train Loss")
plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Training Loss Curve")
plt.legend(); plt.show()

# ---- Confusion matrix ----
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10,10))
plt.imshow(cm, cmap='Blues')
plt.title("Confusion Matrix (40 classes)")
plt.xlabel("Predicted"); plt.ylabel("Actual")
plt.colorbar()
plt.show()
```

#### Line-by-Line Explanation
1. `model.eval()`: Switches model to inference evaluation mode.
2. `with torch.no_grad():`: Disables dynamic autograd tape construction, saving GPU/CPU memory and speeding up matrix multiplication.
3. `outputs = model(X_test_t)`: Evaluates full test tensor `(80, 4096)` in one forward pass $\to$ `(80, 40)`.
4. `preds = outputs.argmax(dim=1)`: Extracts class predictions for all 80 test samples.
5. `all_preds = preds.cpu().numpy()`, `all_labels = y_test_t.cpu().numpy()`: Transfers tensors back to CPU host memory as NumPy arrays.
6. `test_acc = accuracy_score(all_labels, all_preds)`: Computes final test classification accuracy ($\approx 91.25\% - 95.0\%$).
7. `plt.plot(train_losses, label="Train Loss")`: Plots smooth exponential loss decay curve.
8. `cm = confusion_matrix(all_labels, all_preds)`: Computes $40 \times 40$ confusion matrix.
9. `plt.imshow(cm, cmap='Blues')`: Renders confusion heatmap showing strong dominant diagonal matching across all 40 individual subjects.

---

### Cell 15: Sample Visual Inference Gallery (Code)

```python
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    img = X_test[idx].reshape(64, 64)
    true_label = y_test[idx]
    with torch.no_grad():
        pred = model(torch.tensor(X_test[idx:idx+1], dtype=torch.float32).to(device)).argmax(dim=1).item()
    ax.imshow(img, cmap='gray')
    ax.set_title(f"True: {true_label} | Pred: {pred}", fontsize=9,
                 color='green' if pred == true_label else 'red')
    ax.axis('off')
plt.suptitle("Sample Predictions")
plt.tight_layout()
plt.show()
```

#### Line-by-Line Explanation
1. `fig, axes = plt.subplots(2, 5, figsize=(12, 5))`: Creates a $2 \times 5$ grid displaying 10 test predictions.
2. `for i, ax in enumerate(axes.flat):`: Flattens $2\times 5$ axis array into a 1D iterator.
   - `idx = np.random.randint(0, len(X_test))`: Selects a random test sample index.
   - `img = X_test[idx].reshape(64, 64)`: Reshapes 4096D feature vector to $64\times 64$ 2D thumbnail.
   - `true_label = y_test[idx]`: Ground-truth person ID.
   - `pred = model(...).argmax(dim=1).item()`: Runs single-sample inference batch of shape `(1, 4096)` and extracts scalar predicted ID.
   - `color='green' if pred == true_label else 'red'`: Formats thumbnail title in green if correctly classified, red if misclassified.
   - `ax.axis('off')`: Hides axis borders.
3. `plt.tight_layout()`, `plt.show()`: Renders interactive qualitative gallery.

---

## 3. Comprehensive Summary Cheat Sheet

| Pipeline Component | Selection | Implementation Detail | Rationale |
| :--- | :--- | :--- | :--- |
| **Data Split** | `train_test_split` with `stratify=y` | $80/20$ ($320$ train, $80$ test) | Prevents class imbalance; guarantees exactly 8 train and 2 test images per person |
| **Scaling** | `StandardScaler` | `fit_transform(X_train)`, `transform(X_test)` | Normalizes pixel mean to $0$ and variance to $1$; prevents test leakage |
| **Hidden Layers** | Linear(4096 $\to$ 512) $\to$ Linear(512 $\to$ 128) | Fully connected MLP | Hierarchically compresses pixel space ($4096 \to 512 \to 128$) |
| **Hidden Activation** | `nn.ReLU()` | $\max(0, z)$ | Non-saturating gradient prevents vanishing gradients across deep layers |
| **Output Layer** | Linear(128 $\to$ 40) | Raw unnormalized logits | Softmax is internally integrated into `nn.CrossEntropyLoss` for numerical stability |
| **Loss Function** | `nn.CrossEntropyLoss()` | Log-Sum-Exp multi-class Cross Entropy | Numerically stable log-softmax + negative log likelihood |
| **Optimizer** | `optim.Adam(lr=0.001)` | $\beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}$ | Adaptive learning rates converge rapidly on small image feature vectors |
| **Mini-Batching** | `torch.randperm` | Batch size $B = 16$ | Introduces stochastic regularization while maintaining high vectorized GPU throughput |
| **Evaluation** | `model.eval()` + `torch.no_grad()` | Test Accuracy $\ge 91.25\%$ | Disables gradient tracking during inference for fast validation |
