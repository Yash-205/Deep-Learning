# Lecture 9: Weight Initialization Dynamics

## 1. Why Does Initialization Matter?

Before a neural network takes a single gradient descent step, its weights are already set. 

If weights start in a bad state:
* Signals can **explode** to infinity (NaNs, overflow).
* Signals can **vanish** to zero (gradients become $10^{-15}$, no learning occurs).
* Neurons can get **stuck in saturation** (for Tanh/Sigmoid) or **die permanently** (for ReLU).
* Neurons can get trapped in **symmetry lock** (computing identical features forever).

A good initialization ensures that:
1. **Forward pass**: Activation magnitudes remain steady from layer 1 all the way to layer 100.
2. **Backward pass**: Gradient magnitudes remain healthy as they flow backward from the loss to the input.
3. **Symmetry breaking**: Every neuron learns distinct, diverse features.

---

## 2. Breaking the Symmetry: Why Zero and Constant Weights Fail

Suppose we initialize all weights to zero ($W = 0$) or to an identical constant ($W = c$):

```
Input x1 ───► [Neuron 1] ───► Output
         ╲   ╱
          ╲ ╱
Input x2 ───► [Neuron 2] ───► Output
```

1. **In the Forward Pass**:
   The preactivation of unit $i$ is:
   $$z_i = \sum_{j} W_{ij} x_j + b_i$$
   If every row of $W$ is identical and biases $b_i$ are equal, then:
   $$z_1 = z_2 = \dots = z_m$$
   Every neuron computes the exact same activation $a_i = \phi(z_i)$.

2. **In the Backward Pass**:
   The gradient with respect to weight $W_{ij}$ is:
   $$\frac{\partial \mathcal{L}}{\partial W_{ij}} = \delta_i \cdot x_j \quad \text{where } \delta_i = \frac{\partial \mathcal{L}}{\partial z_i}$$
   Because outgoing connections are also symmetric, every neuron receives the exact same sensitivity $\delta_1 = \delta_2$.
   Therefore, the gradient $\frac{\partial \mathcal{L}}{\partial W_{ij}}$ is identical for all neurons $i$.

3. **The Result (Symmetry Lock)**:
   Every single neuron in the layer gets the exact same update:
   $$W_{1:} = W_{2:} = \dots = W_{m:}$$
   No matter how long you train, the neurons remain completely identical! A hidden layer of 1,000 neurons acts like a single neuron.

> **Key Rule**: We **must** use independent random numbers for weights to break symmetry. Biases, however, can safely start at 0 because the randomized weights have already broken the symmetry.

---

## 3. Naive Random Initialization: The Scaling Dilemma

Random weights solve symmetry, but **what scale (variance $\sigma^2$) should we pick?**

Consider a linear neuron summing $n_{\text{in}}$ inputs:
$$z = \sum_{j=1}^{n_{\text{in}}} w_j a_j$$

Assuming weights $w_j$ and inputs $a_j$ are independent with zero mean:
$$\mathbb{E}[z] = 0$$
$$\text{Var}(z) = \sum_{j=1}^{n_{\text{in}}} \text{Var}(w_j) \cdot \mathbb{E}[a_j^2] = n_{\text{in}} \cdot \sigma_w^2 \cdot \mathbb{E}[a^2]$$

The per-layer signal scaling factor is:
$$\text{Multiplier} = n_{\text{in}} \cdot \sigma_w^2$$

Across $L$ layers:
$$\mathbb{E}[a_L^2] \approx \left(n_{\text{in}} \sigma_w^2\right)^L \mathbb{E}[a_0^2]$$

* **If $n_{\text{in}} \sigma_w^2 > 1$ (Weights too large)**:
  * For linear networks: Activations explode exponentially ($2^L \to \infty$).
  * For Tanh/Sigmoid networks: Preactivations become very large ($\pm 10$), pushing activations to $\pm 1.0$ (saturation). Since $\tanh'(z) = 1 - \tanh^2(z) \approx 0$, local derivatives drop to zero, causing **vanishing gradients**!
* **If $n_{\text{in}} \sigma_w^2 < 1$ (Weights too small)**:
  * Signals fade exponentially toward zero. By layer 10, activations and gradients are essentially zero.

---

## 4. Xavier / Glorot Initialization (2010)

Proposed by Xavier Glorot and Yoshua Bengio for networks using **Tanh, Sigmoid, and Linear** activations.

### The Logic
* **Preserving Forward Signal**: We want the variance of activations to stay constant across layers:
  $$n_{\text{in}} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{\text{in}}}$$
* **Preserving Backward Gradient**: We want the variance of backpropagated gradients to stay constant across layers:
  $$n_{\text{out}} \cdot \text{Var}(w) = 1 \implies \text{Var}(w) = \frac{1}{n_{\text{out}}}$$
* **The Compromise**: When a layer changes size ($n_{\text{in}} \neq n_{\text{out}}$), we take the harmonic mean compromise:
  $$\text{Var}(w) = \frac{2}{n_{\text{in}} + n_{\text{out}}}$$

### Sampling Formulas
1. **Xavier Normal**:
   $$w \sim \mathcal{N}\left(0, \, \sigma = \sqrt{\frac{2}{n_{\text{in}} + n_{\text{out}}}}\right)$$
2. **Xavier Uniform**:
   $$w \sim \mathcal{U}(-b, b) \quad \text{where } b = \sqrt{\frac{6}{n_{\text{in}} + n_{\text{out}}}}$$
   *(Because the variance of a uniform distribution on $[-b, b]$ is $\frac{b^2}{3}$; setting $\frac{b^2}{3} = \frac{2}{n_{\text{in}}+n_{\text{out}}}$ gives $b = \sqrt{\frac{6}{n_{\text{in}}+n_{\text{out}}}}$).*

### Why Does Xavier Fail for ReLU?
ReLU is defined as $\text{ReLU}(z) = \max(0, z)$.
* For a symmetric distribution around zero, ReLU completely discards the negative half.
* It removes **half the second moment** (energy) of the signal:
  $$\mathbb{E}[\text{ReLU}(z)^2] = \frac{1}{2} \mathbb{E}[z^2]$$
* With Xavier initialization ($n_{\text{in}} \text{Var}(w) = 1$), each ReLU layer multiplies the signal energy by $\frac{1}{2}$:
  $$\text{Signal at layer } L = \left(\frac{1}{2}\right)^L \cdot \text{Signal at layer 0}$$
* In a 10-layer network: $(1/2)^{10} \approx 0.00097$ (99.9% signal lost).
* In a 50-layer network: $(1/2)^{50} \approx 8.88 \times 10^{-16}$ (signal and gradients completely vanish!).

---

## 5. He / Kaiming Initialization (2015)

Proposed by Kaiming He et al. specifically for **ReLU and Leaky ReLU** activations.

### The Logic: Compensate for the Rectifier
Since ReLU cuts signal energy in half ($\times \frac{1}{2}$), **double the initial weight variance** ($\times 2$) to balance it perfectly:

$$\text{Var}(w) = \frac{2}{n_{\text{in}}}$$

Then:
$$\text{Per-layer multiplier} = n_{\text{in}} \cdot \text{Var}(w) \cdot \frac{1}{2} = n_{\text{in}} \cdot \left(\frac{2}{n_{\text{in}}}\right) \cdot \frac{1}{2} = 1.0$$

The signal magnitude remains completely stable across arbitrary depths (even 100+ layers)!

### Sampling Formulas
1. **He Normal**:
   $$w \sim \mathcal{N}\left(0, \, \sigma = \sqrt{\frac{2}{n_{\text{in}}}}\right)$$
2. **He Uniform**:
   $$w \sim \mathcal{U}(-b, b) \quad \text{where } b = \sqrt{\frac{6}{n_{\text{in}}}}$$

### Extension for Leaky ReLU
Leaky ReLU allows a small negative slope $\alpha$:
$$f(z) = \begin{cases} z & \text{if } z \ge 0 \\ \alpha z & \text{if } z < 0 \end{cases}$$
* The negative half retains fraction $\alpha^2$ of its squared energy.
* The total retained fraction is $\frac{1 + \alpha^2}{2}$.
* Compensating for this fraction gives the adjusted variance:
  $$\text{Var}(w) = \frac{2}{(1 + \alpha^2) \cdot n_{\text{in}}}$$
* Notice that when $\alpha = 0$ (standard ReLU), this reduces directly to $\frac{2}{n_{\text{in}}}$.

---

## 6. Understanding the Nuance: Second Moment vs Variance

* **Second Moment ($q = \mathbb{E}[a^2]$)**: Measures total signal energy.
* **Centered Variance ($\text{Var}(a) = \mathbb{E}[a^2] - (\mathbb{E}[a])^2$)**:
  * For zero-mean activations (like Tanh, Linear), $\mathbb{E}[a] = 0$, so $\text{Var}(a) = \mathbb{E}[a^2]$.
  * For ReLU, all negative values become 0, creating a **positive mean**:
    $$\mathbb{E}[a] = \sqrt{\frac{v}{2\pi}} \quad (\approx 0.564 \text{ when preactivation variance } v = 2)$$
  * Therefore:
    $$\text{Var}(a) = v \left(\frac{1}{2} - \frac{1}{2\pi}\right) \approx 0.682$$
  * He initialization preserves the **second moment (energy)** at 1.0, not the centered variance.

---

## 7. Practical Field Guide & Diagnostic Table

| Initializer | Weight Variance $\text{Var}(w)$ | Best Suited For | Common Pitfall / Danger |
| :--- | :--- | :--- | :--- |
| **Zero / Constant** | $0$ | **Biases only** | **Symmetry Lock**: All hidden neurons compute identical features forever. |
| **Naive Random** | Arbitrary $\sigma^2$ (e.g. 1.0) | Small demo experiments | Width-dependent signal explosion or decay. |
| **Xavier / Glorot** | $\frac{2}{n_{\text{in}} + n_{\text{out}}}$ | **Tanh, Sigmoid, Linear, Softmax** | With ReLU, signals decay by $50\%$ at every layer ($\to 0$). |
| **He / Kaiming** | $\frac{2}{n_{\text{in}}}$ | **ReLU, GELU** | With Linear/Tanh, signal energy doubles at every layer ($\to \infty$). |
| **He (Leaky)** | $\frac{2}{(1 + \alpha^2) n_{\text{in}}}$ | **Leaky ReLU, PReLU** (slope $\alpha$) | Must adjust for the specific slope $\alpha$. |

### How to Calculate $n_{\text{in}}$ (Fan-in) in Code
* **Linear / Fully-Connected Layer**:
  $$n_{\text{in}} = \text{number of input features}$$
* **Convolutional 2D Layer**:
  $$n_{\text{in}} = \text{input channels} \times \text{kernel height} \times \text{kernel width}$$
  *(e.g., Conv2D with 64 input channels and a $3 \times 3$ kernel has $n_{\text{in}} = 64 \times 3 \times 3 = 576$).*
