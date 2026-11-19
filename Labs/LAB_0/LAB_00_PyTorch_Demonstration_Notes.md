# Lab 00: PyTorch Demonstration — Line-by-Line Code & Output Breakdown

---

## 1. Overview & Objectives

PyTorch is a dynamic tensor computation framework with hardware acceleration (CUDA/MPS) and automatic differentiation (`autograd`). This lab establishes foundational mechanics:
- Initializing tensors with various distributions and memory patterns.
- Manipulating data types, shapes, and dimensionalities.
- Tensor operations: element-wise arithmetic, reductions, matrix algebra, and nonlinearities.
- Hardware management: querying CUDA devices, tensor device migration, and CPU vs. GPU performance benchmarking.
- Interoperability: zero-copy conversion between NumPy arrays and PyTorch tensors.
- 2D pixel manipulation and visualization using Matplotlib.

---

## 2. Cell-by-Cell, Line-by-Line Code & Output Analysis

---

### Cell 1: Importing PyTorch
```python
import torch
```
- **Line 1 (`import torch`)**: Loads the primary PyTorch C++ backend engine and Python API into the current namespace.
- **Output**: None (successful import).

---

### Cell 2: Checking PyTorch Version
```python
print(torch.__version__)
```
- **Line 1 (`print(torch.__version__)`)**: Accesses the metadata string attribute `__version__` indicating the installed PyTorch build and CUDA compilation version.
- **Output**:
  ```text
  2.11.0+cu128
  ```
  *(Indicates PyTorch 2.11.0 compiled with CUDA 12.8 support).*

---

### Cell 3: Querying Hardware / GPU Availability
```python
if torch.cuda.is_available():
    print("GPU is avl")
    print(f"using gpu:{torch.cuda.get_device_name(0)}")
else:
    print("GPU not avl")
```
- **Line 1 (`if torch.cuda.is_available():`)**: Queries the CUDA driver API to check whether a compatible NVIDIA GPU and CUDA toolkit are detected.
- **Line 2 (`print("GPU is avl")`)**: Confirms CUDA acceleration is active.
- **Line 3 (`print(f"using gpu:{torch.cuda.get_device_name(0)}")`)**: Retrieves the hardware model name of GPU device index `0`.
- **Line 4-5 (`else: print("GPU not avl")`)**: Fallback branch when running strictly on CPU.
- **Output**:
  ```text
  GPU is avl
  using gpu:Tesla T4
  ```

---

### Cell 4 & 5: Uninitialized Tensors (`torch.empty`)
```python
torch.empty(2, 3) # create tensors with shape of 2*3 with memory location or without values.
a = torch.empty(2, 3)
```
- **Line 1 (`torch.empty(2, 3)`)**: Allocates a block of memory of shape $(2, 3)$ without initializing values. The entries contain leftover residual data ("garbage values") from whatever was previously in that RAM/VRAM block.
- **Line 2 (`a = torch.empty(2, 3)`)**: Assigns the $(2 \times 3)$ uninitialized tensor to variable `a`.
- **Output**:
  ```text
  tensor([[0.0000e+00, 0.0000e+00, 0.0000e+00],
          [0.0000e+00, 2.4242e-43, 1.4013e-45]])
  ```

---

### Cell 6: Type Inspection
```python
type(a)
```
- **Line 1 (`type(a)`)**: Inspects the Python class type of the variable `a`.
- **Output**:
  ```text
  torch.Tensor
  ```

---

### Cell 7: Zeros Initialization (`torch.zeros`)
```python
torch.zeros(2, 3)
```
- **Line 1 (`torch.zeros(2, 3)`)**: Allocates a tensor of shape $2 \times 3$ and initializes all elements to floating-point `0.0` (default dtype: `torch.float32`).
- **Output**:
  ```text
  tensor([[0., 0., 0.],
          [0., 0., 0.]])
  ```

---

### Cell 8: Ones Initialization (`torch.ones`)
```python
torch.ones(3, 4)
```
- **Line 1 (`torch.ones(3, 4)`)**: Allocates a tensor of shape $3 \times 4$ with all elements initialized to `1.0`.
- **Output**:
  ```text
  tensor([[1., 1., 1., 1.],
          [1., 1., 1., 1.],
          [1., 1., 1., 1.]])
  ```

---

### Cell 9: Uniform Random Numbers (`torch.rand`)
```python
torch.rand(2, 3)
```
- **Line 1 (`torch.rand(2, 3)`)**: Samples random floating-point values drawn from a uniform continuous distribution over the half-open interval $[0, 1)$.
- **Output**:
  ```text
  tensor([[0.8013, 0.7088, 0.3822],
          [0.9561, 0.8861, 0.1350]])
  ```

---

### Cell 10: Manual Seed Reproducibility (`torch.manual_seed`)
```python
torch.manual_seed(100)
torch.rand(3, 4)
```
- **Line 1 (`torch.manual_seed(100)`)**: Sets the seed of the CPU pseudo-random number generator (PRNG) to integer `100`, ensuring deterministic outputs across runs.
- **Line 2 (`torch.rand(3, 4)`)**: Generates a $3 \times 4$ uniform random tensor using the seeded PRNG state.
- **Output**:
  ```text
  tensor([[0.1117, 0.8158, 0.2626, 0.4839],
          [0.6765, 0.7539, 0.2627, 0.0428],
          [0.2080, 0.1180, 0.1217, 0.7356]])
  ```

---

### Cell 11: Explicit Tensor Creation (`torch.tensor`)
```python
torch.tensor([[1, 2, 4], [3, 4, 4]])
```
- **Line 1 (`torch.tensor(...)`)**: Constructs a 2D tensor of shape $(2, 3)$ directly from nested Python lists. Infers integer type `torch.int64`.
- **Output**:
  ```text
  tensor([[1, 2, 4],
          [3, 4, 4]])
  ```

---

### Cell 12: Range Tensors (`torch.arange`)
```python
torch.arange(2, 100, 2)
```
- **Line 1 (`torch.arange(start=2, end=100, step=2)`)**: Returns a 1D tensor of evenly spaced integer values from `2` up to (but excluding) `100` with increment step `2`. Total elements $= \frac{100 - 2}{2} = 49$.
- **Output**:
  ```text
  tensor([ 2,  4,  6,  8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36,
          38, 40, 42, 44, 46, 48, 50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70, 72,
          74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94, 96, 98])
  ```

---

### Cell 13: Linear Spacing (`torch.linspace`)
```python
torch.linspace(0, 20, 10) # evenly spaced values starting from 0 to 20
```
- **Line 1 (`torch.linspace(start=0, end=20, steps=10)`)**: Generates 10 linearly spaced points over the closed interval $[0, 20]$ with step $\Delta = \frac{20 - 0}{10 - 1} = \frac{20}{9} \approx 2.2222$.
- **Output**:
  ```text
  tensor([ 0.0000,  2.2222,  4.4444,  6.6667,  8.8889, 11.1111, 13.3333, 15.5556,
          17.7778, 20.0000])
  ```

---

### Cell 14: Identity Matrix (`torch.eye`)
```python
torch.eye(5)
```
- **Line 1 (`torch.eye(5)`)**: Constructs a 2D square Identity matrix $I_5$ of shape $(5, 5)$ where diagonal entries are `1.0` and all off-diagonal entries are `0.0`.
- **Output**:
  ```text
  tensor([[1., 0., 0., 0., 0.],
          [0., 1., 0., 0., 0.],
          [0., 0., 1., 0., 0.],
          [0., 0., 0., 1., 0.],
          [0., 0., 0., 0., 1.]])
  ```

---

### Cell 15: Constant Value Initialization (`torch.full`)
```python
torch.full((2, 3), 5)
```
- **Line 1 (`torch.full(size=(2, 3), fill_value=5)`)**: Creates a tensor of shape $(2, 3)$ filled entirely with the scalar value `5`.
- **Output**:
  ```text
  tensor([[5, 5, 5],
          [5, 5, 5]])
  ```

---

### Cell 16 & 17: Inspecting Shape & Like Constructors
```python
p = torch.tensor([[11, 11, 12], [2, 3, 4]])
p.shape
```
- **Line 1 (`p = ...`)**: Creates a $(2, 3)$ integer tensor.
- **Line 2 (`p.shape`)**: Returns a `torch.Size([2, 3])` tuple.
- **Output**:
  ```text
  torch.Size([2, 3])
  ```

```python
torch.empty_like(p)
torch.zeros_like(p)
torch.ones_like(p)
```
- **`torch.empty_like(p)`**: Creates uninitialized tensor matching shape $(2, 3)$ and dtype (`int64`) of `p`.
- **`torch.zeros_like(p)`**: Creates all-zero tensor matching shape and dtype of `p`.
  - **Output**: `tensor([[0, 0, 0], [0, 0, 0]])`
- **`torch.ones_like(p)`**: Creates all-ones tensor matching shape and dtype of `p`.
  - **Output**: `tensor([[1, 1, 1], [1, 1, 1]])`

```python
torch.rand_like(p)
```
- **Explanation**: Raises `NotImplementedError: "check_uniform_bounds" not implemented for 'Long'`. `torch.rand_like` requires a floating-point dtype (e.g. `float32`), but `p` has dtype `int64` (`Long`).

---

### Cell 18, 19, 20: Data Types & Casting (`dtype`, `.to`)
```python
p.dtype
```
- **Output**: `torch.int64`

```python
torch.tensor([1, 2, 4], dtype=torch.float64)
```
- **Line 1**: Explicitly assigns double precision 64-bit float dtype.
- **Output**: `tensor([1., 2., 4.], dtype=torch.float64)`

```python
p.to(torch.float32)
```
- **Line 1 (`p.to(...)`)**: Casts existing tensor `p` to 32-bit single precision float (`torch.float32`).
- **Output**:
  ```text
  tensor([[11., 11., 12.],
          [ 2.,  3.,  4.]])
  ```

---

### Cell 21: Scalar Arithmetic & Broadcasting
```python
t1 = torch.rand(4, 5)
t1 + 2
```
- **Line 1 (`t1 = torch.rand(4, 5)`)**: Generates a $(4 \times 5)$ random tensor.
- **Line 2 (`t1 + 2`)**: **Broadcasting**: The scalar `2` is expanded to shape $(4, 5)$ and added element-wise.
- **Output**: Adds `2.0` to each entry in `t1`.

```python
t1 * (2 ^ 3)
```
- **Explanation**: In Python, `^` is bitwise XOR, not exponentiation. $2 \oplus 3 = 1$. Thus, `t1 * 1` returns identical entries to `t1`. *(For exponentiation, use `2**3 = 8`)*.

```python
((t1 * 100) // 3) % 2
```
- **Explanation**: Multiplies `t1` by 100, performs floor division `//` by 3, then applies modulo `% 2` to yield binary integer residues `0.` or `1.`.

---

### Cell 22: Element-wise Tensor-Tensor Operations
```python
a = torch.rand(4, 5)
b = torch.rand(4, 5)
a - b # Element-wise subtraction
a + b # Element-wise addition
a * b # Hadamard (element-wise) product
```
- **Outputs**: Each entry $(i, j)$ in the resulting tensor is computed directly from $a_{ij} \text{ op } b_{ij}$.

---

### Cell 23: Reduction Operations (Sum, Min, Max, Argmin, Argmax)
```python
e = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
```
- Creates a $(2 \times 3)$ tensor with discrete uniform values in $[0, 10)$, e.g.:
  $$\begin{bmatrix} 5. & 6. & 9. \\ 2. & 3. & 1. \end{bmatrix}$$

| Command | Mathematical Operation | Code Meaning | Exact Output for Example |
| :--- | :--- | :--- | :--- |
| `torch.sum(e)` | $\sum_{i,j} e_{ij}$ | Sum of all elements | `tensor(26.)` |
| `torch.sum(e, dim=0)` | $\sum_{i} e_{ij}$ | Column-wise sum (collapses rows) | `tensor([ 7., 9., 10.])` |
| `torch.max(e)` | $\max_{i,j} e_{ij}$ | Global maximum scalar | `tensor(9.)` |
| `torch.min(e)` | $\min_{i,j} e_{ij}$ | Global minimum scalar | `tensor(1.)` |
| `torch.argmax(e)` | $\arg\max \text{vec}(e)$ | Flattened 1D index of maximum | `tensor(2)` *(Index $(0, 2)$ in flattened 1D)* |
| `torch.argmin(e)` | $\arg\min \text{vec}(e)$ | Flattened 1D index of minimum | `tensor(5)` *(Index $(1, 2)$ in flattened 1D)* |

---

### Cell 24: Matrix Multiplication & Dot Product
```python
m1 = torch.randint(size=(4, 4), low=0, high=10)
m2 = torch.randint(size=(4, 4), low=0, high=10)
torch.matmul(m1, m2) # Or m1 @ m2
```
- **Line 3 (`torch.matmul(m1, m2)`)**: Performs linear algebraic matrix multiplication $C_{ik} = \sum_j A_{ij} B_{jk}$.

```python
vector1 = torch.tensor([1, 2])
vector2 = torch.tensor([3, 4])
torch.dot(vector1, vector2)
```
- **Line 3 (`torch.dot(...)`)**: Calculates 1D Euclidean inner product $1(3) + 2(4) = 3 + 8 = 11$.
- **Output**: `tensor(11)`

---

### Cell 25: Transpose & Activation Functions
```python
f = torch.randint(size=(2, 3), low=0, high=10, dtype=torch.float32)
torch.transpose(f, 0, 1) # Swaps dim 0 and dim 1 (shape 2x3 -> 3x2)
torch.log(f)             # Natural logarithm ln(x) element-wise
torch.exp(f)             # Exponential e^x element-wise
torch.softmax(f, dim=0)  # Softmax normalized down columns
torch.sigmoid(f)         # Logistic sigmoid 1 / (1 + e^-x)
torch.relu(f)            # Rectified Linear Unit max(0, x)
```

---

### Cell 26 & 27: Device Management & GPU Benchmarking
```python
device = torch.device('cuda')
a = torch.rand(2, 3)
b = a.to(device) # Migrates tensor from Host (CPU RAM) to Device (GPU VRAM)
```

```python
import time
size = 10000 # 10,000 x 10,000 matrices (100 million floats = 400 MB each)

matrix_cpu1 = torch.rand(size, size)
matrix_cpu2 = torch.rand(size, size)

matrix_gpu1 = matrix_cpu1.to(device)
matrix_gpu2 = matrix_cpu2.to(device)

# CPU benchmark
start_time = time.time()
result_cpu = torch.matmul(matrix_cpu1, matrix_cpu2)
cpu_time = time.time() - start_time

# GPU benchmark
start_time = time.time()
result_gpu = torch.matmul(matrix_gpu1, matrix_gpu2)
torch.cuda.synchronize() # Blocks CPU until all queued CUDA GPU kernels finish
gpu_time = time.time() - start_time

print(f"CPU time: {cpu_time:.4f} seconds")
print(f"GPU time: {gpu_time:.4f} seconds")
```
- **Output**:
  ```text
  CPU time: 16.4400 seconds
  GPU time: 0.5624 seconds
  ```
- **Explanation**: GPU achieves a **$\approx 30\times$ speedup** via massive SIMD parallelization over thousands of CUDA cores.

---

### Cell 28 & 29: NumPy Interoperability
```python
import numpy as np

arr = np.array([1, 2, 3])

# Method 1: Zero-copy memory sharing
tensor_shared = torch.from_numpy(arr)

# Method 2: Memory copy
tensor_copied = torch.tensor(arr)

# Convert Tensor back to NumPy (shares memory on CPU)
arr_back = tensor_shared.numpy()
```
- **Memory Sharing**: Modifying `arr[0] = 99` will immediately reflect in `tensor_shared[0]` without memory reallocation.

---

### Cell 30: Drawing a Tensor Smiley Face
```python
import matplotlib.pyplot as plt

# 1. Initialize 10x10 zero grid (black canvas)
face = torch.zeros(10, 10)

# 2. Draw border (white = 1.0)
face[0, :] = 1     # top row
face[9, :] = 1     # bottom row
face[:, 0] = 1     # left column
face[:, 9] = 1     # right column

# 3. Draw eyes
face[2, 2] = 1     # left eye
face[2, 7] = 1     # right eye

# 4. Draw mouth
face[7, 2:8] = 1   # mouth line spanning columns 2 to 7 on row 7

# 5. Display with Matplotlib
plt.imshow(face, cmap='gray')
plt.title("My Tensor Smiley")
plt.axis('off')
plt.show()
```
- **Output**: Renders a $10 \times 10$ grayscale bitmap smiling face image.
