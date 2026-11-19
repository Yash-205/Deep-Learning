# Deep Learning — Worksheet 02: The Activation Function and the Update Rule

**Complete Solutions, Mathematical Derivations & Step-by-Step Analysis**

---

## Part 1: What an Activation Function Does

### 1.1 The Perceptron Computation

A single perceptron executes two sequential operations:

1. **Linear Arithmetic (Pre-activation)**:
   $$
   z = wx + b
   $$
2. **Non-linear Mapping (Activation)**:
   $$
   \hat{y} = f(z)
   $$

```
  x (Input) ───► [ (w · x + b) = z ] ───► [ Activation f(z) ] ───► ŷ (Prediction)
```

---

### 1.2 Why $f$ is Needed

- **Example Scenario**: A perceptron evaluates whether a student passes (Target $y \in \{0, 1\}$).
- **Parameters**: Weight $w = 1$, Bias $b = -40$.

|   Student   | Marks ($x$) |      Weighted Sum$z = wx + b$      | Interpretation                                |
| :---------: | :-----------: | :-----------------------------------: | :-------------------------------------------- |
| **A** |    $65$    | $z = (1)(65) + (-40) = \mathbf{25}$ | $25$ is a raw score, not a binary decision. |
| **B** |    $32$    | $z = (1)(32) + (-40) = \mathbf{-8}$ | $-8$ is a raw score, not a binary decision. |

> **Key Takeaway**: The question asks for **Pass ($1$) or Fail ($0$)**. The arithmetic produced $+25$ and $-8$. The activation function $f$ is required to transform arbitrary real numbers $z \in \mathbb{R}$ into a valid decision/probability.

---

### 1.3 The Threshold (Step / Heaviside) Function

$$
f(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{if } z < 0 \end{cases}
$$

Applying $f(z)$ to the students:

- **Student A**: $z = 25 \implies \hat{y} = f(25) = \mathbf{1}$ (**Pass**)
- **Student B**: $z = -8 \implies \hat{y} = f(-8) = \mathbf{0}$ (**Fail**)

---

### 1.4 The Role of $f$

For a single perceptron, the activation function $f$:

1. Takes $z$, which may be any real number ($-\infty < z < \infty$).
2. Returns an output in a fixed, bounded range (e.g., $\{0, 1\}$ or $[0, 1]$).
3. Enables the perceptron to output an **actionable decision or probability** rather than an unbounded measurement.

---

## Part 2: The Update Rule & Chain Rule Derivation

### 2.1 Core Definitions

- **Linear Combination**: $z = wx + b$
- **Output / Prediction**: $\hat{y} = f(z)$
- **Loss Function (Mean Squared Error for single sample)**:
  $$
  L = \frac{1}{2}(\hat{y} - y)^2
  $$

  *(where $y$ is the ground-truth target, $L$ is the loss, and $\alpha > 0$ is the learning rate)*.

---

### 2.2 Gradient Descent Formulation

We want to minimize $L$. The gradient $\frac{\partial L}{\partial w}$ points in the direction of steepest increase in loss, so we move in the opposite direction:

$$
w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}
$$

---

### 2.3 Derivation of $\frac{\partial L}{\partial w}$ via the Chain Rule

Since $w$ affects $z$, $z$ affects $\hat{y}$, and $\hat{y}$ affects $L$, we apply the Chain Rule:

$$
\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} \cdot \frac{\partial z}{\partial w}
$$

#### Evaluating Each Factor:

1. **Factor 1 ($\frac{\partial L}{\partial \hat{y}}$)**:
   $$
   \frac{\partial L}{\partial \hat{y}} = \frac{\partial}{\partial \hat{y}}\left[\frac{1}{2}(\hat{y} - y)^2\right] = \frac{1}{2} \cdot 2(\hat{y} - y)^1 \cdot 1 = \mathbf{(\hat{y} - y)}
   $$
2. **Factor 2 ($\frac{\partial \hat{y}}{\partial z}$)**:
   $$
   \frac{\partial \hat{y}}{\partial z} = \frac{\partial}{\partial z}[f(z)] = \mathbf{f'(z)}
   $$
3. **Factor 3 ($\frac{\partial z}{\partial w}$)**:
   $$
   \frac{\partial z}{\partial w} = \frac{\partial}{\partial w}[wx + b] = \mathbf{x}
   $$

#### Multiplying the Three Factors:

$$
\frac{\partial L}{\partial w} = (\hat{y} - y) f'(z) x
$$

---

### 2.4 Complete Parameter Update Rule

$$
w_{new} = w_{old} - \alpha (\hat{y} - y) f'(z) x
$$

> **The Three Governing Factors**:
>
> 1. **The Error term $(\hat{y} - y)$**: Measures how wrong the prediction is.
> 2. **The Derivative term $f'(z)$**: Measures the sensitivity / responsiveness of the activation at operating point $z$.
> 3. **The Input term $x$**: Measures the scale of the feature associated with weight $w$.

---

## Part 3: The Step Function Cannot Learn

### 3.1 Mathematical Derivative of the Step Function

- For $z > 0$: $f(z) = 1$ (constant) $\implies f'(z) = \mathbf{0}$
- For $z < 0$: $f(z) = 0$ (constant) $\implies f'(z) = \mathbf{0}$
- At $z = 0$: $f(z)$ undergoes a discontinuous jump $\implies f'(0)$ is **undefined**.
- **Conclusion**:
  $$
  f'(z) = 0 \quad \text{for all } z \neq 0
  $$

---

### 3.2 Step Function Numerical Walkthrough

**Given Setup**: $x = 2, w = 1, b = 0, y = 0, \alpha = 0.5$.

1. **Weighted Sum**:
   $$
   z = wx + b = (1)(2) + 0 = \mathbf{2}
   $$
2. **Activation**:
   $$
   \hat{y} = f(2) = \mathbf{1} \quad (\text{since } 2 \ge 0)
   $$
3. **Error**:
   $$
   \hat{y} - y = 1 - 0 = \mathbf{1} \quad (\text{Maximum possible error!})
   $$
4. **Derivative**:
   $$
   f'(2) = \mathbf{0}
   $$
5. **Gradient**:
   $$
   \frac{\partial L}{\partial w} = (\hat{y} - y) f'(z) x = (1)(0)(2) = \mathbf{0}
   $$
6. **Update**:
   $$
   w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} = 1 - (0.5)(0) = \mathbf{1}
   $$

---

### 3.3 Critical Conclusion

- **The Contradiction**: The error is $1.0$ (the maximum possible mistake), yet the weight change is exactly **zero**.
- **Fill-in Explanations**:
  - The update is zero because **the derivative $f'(z) = 0$**.
  - Increasing $\alpha$ (even to $1{,}000{,}000$) would **have no effect / not change the weight**, because **multiplying any learning rate by zero always yields zero ($\alpha \times 0 = 0$)**.
- **Fundamental Rule**: A usable activation function must be differentiable and have a **non-zero derivative** over a useful operating domain.

---

## Part 4: The Sigmoid Function

### 4.1 Definition

$$
\sigma(z) = \frac{1}{1 + e^{-z}} = (1 + e^{-z})^{-1}
$$

- Smooth, continuous, everywhere differentiable.
- Output range: $0 < \sigma(z) < 1$.
- Symmetry point: $\sigma(0) = 0.5$.

---

### 4.2 Complete Derivation of $\sigma'(z)$

$$
\sigma(z) = (1 + e^{-z})^{-1}
$$

1. **Step 1 (Chain Rule)**:
   $$
   \sigma'(z) = -(1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z})
   $$
2. **Step 2 (Inner Derivative)**:
   $$
   \frac{d}{dz}(1 + e^{-z}) = -e^{-z}
   $$
3. **Step 3 (Substitute & Simplify)**:
   $$
   \sigma'(z) = -(1 + e^{-z})^{-2}(-e^{-z}) = \frac{e^{-z}}{(1 + e^{-z})^2}
   $$
4. **Step 4 (Algebraic Identity: $e^{-z} = (1 + e^{-z}) - 1$)**:
   $$
   \sigma'(z) = \frac{(1 + e^{-z}) - 1}{(1 + e^{-z})^2}
   $$
5. **Step 5 (Split into Two Fractions)**:
   $$
   \sigma'(z) = \frac{1 + e^{-z}}{(1 + e^{-z})^2} - \frac{1}{(1 + e^{-z})^2} = \frac{1}{1 + e^{-z}} - \left(\frac{1}{1 + e^{-z}}\right)^2
   $$
6. **Step 6 (Express in Terms of $\sigma(z)$)**:
   $$
   \sigma'(z) = \sigma(z) - \sigma(z)^2 = \mathbf{\sigma(z)(1 - \sigma(z))}
   $$

---

## Part 5: Worked Example — Sigmoid Moves the Correct Way

**Given Setup** *(Identical parameters to Part 3)*: $x = 2, w = 1, b = 0, y = 0, \alpha = 0.5$.

1. **Weighted Sum**:
   $$
   z = wx + b = (1)(2) + 0 = \mathbf{2.000000}
   $$
2. **Activation**:
   $$
   \hat{y} = \sigma(2) = \frac{1}{1 + e^{-2}} = \frac{1}{1 + 0.135335} = \frac{1}{1.135335} = \mathbf{0.880797}
   $$
3. **Error**:
   $$
   \hat{y} - y = 0.880797 - 0 = \mathbf{0.880797}
   $$
4. **Derivative**:
   $$
   \sigma'(2) = \sigma(2)(1 - \sigma(2)) = (0.880797)(1 - 0.880797) = (0.880797)(0.119203) = \mathbf{0.104994}
   $$
5. **Gradient**:
   $$
   \frac{\partial L}{\partial w} = (\hat{y} - y) \sigma'(z) x = (0.880797)(0.104994)(2) = \mathbf{0.184956}
   $$
6. **Update**:
   $$
   w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} = 1.0 - (0.5)(0.184956) = 1.0 - 0.092478 = \mathbf{0.907522}
   $$

---

### Direction Check & Comparison with Part 3

| Requirement                                    | Mathematical Relation                     | Logical Consequence                   |
| :--------------------------------------------- | :---------------------------------------- | :------------------------------------ |
| **Target is $y = 0$**                  | Current$\hat{y} = 0.880797$ is too high | $\hat{y}$ must decrease             |
| **$\hat{y} = \sigma(z)$ is monotonic** | $\sigma(z)$ increases with $z$        | $z$ must decrease                   |
| **$z = wx + b$ with $x = 2 > 0$**    | $z$ increases with $w$                | $w$ must decrease                   |
| **Result**                               | $w : 1.0 \to 0.907522$                  | **Weight decreased correctly!** |

#### Direct Comparison: Threshold vs. Sigmoid

| Activation          |  $f'(2)$  | $\frac{\partial L}{\partial w}$ | $w_{new}$ |              Learning Occurred?              |
| :------------------ | :----------: | :-------------------------------: | :----------: | :-------------------------------------------: |
| **Threshold** | $0.000000$ |           $0.000000$           | $1.000000$ |          **No** (Weight stuck)          |
| **Sigmoid**   | $0.104994$ |           $0.184956$           | $0.907522$ | **Yes** (Weight updated towards target) |

---

## Part 6: Practice Numerical Activity

**Given Setup**: $x = 3, w = 0.5, b = -1, y = 1, \alpha = 2$. *(Given $e^{-0.5} = 0.606531$)*.

1. **Step 1 — Weighted Sum**:
   $$
   z = wx + b = (0.5)(3) + (-1) = 1.5 - 1.0 = \mathbf{0.500000}
   $$
2. **Step 2 — Activation**:
   $$
   \hat{y} = \sigma(0.5) = \frac{1}{1 + e^{-0.5}} = \frac{1}{1 + 0.606531} = \frac{1}{1.606531} = \mathbf{0.622459}
   $$
3. **Step 3 — Error**:
   $$
   \hat{y} - y = 0.622459 - 1.0 = \mathbf{-0.377541}
   $$
4. **Step 4 — Derivative**:
   $$
   \sigma'(0.5) = \sigma(0.5)(1 - \sigma(0.5)) = (0.622459)(1 - 0.622459) = (0.622459)(0.377541) = \mathbf{0.235004}
   $$
5. **Step 5 — Gradient**:
   $$
   \frac{\partial L}{\partial w} = (\hat{y} - y) \sigma'(z) x = (-0.377541)(0.235004)(3) = \mathbf{-0.266170}
   $$
6. **Step 6 — Update**:
   $$
   w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w} = 0.5 - 2(-0.266170) = 0.5 + 0.532341 = \mathbf{1.032341}
   $$
7. **Step 7 — Direction Check & Justification**:
   - **Direction**: $w$ **increased** from $0.5 \to 1.032341$.
   - **Justification**: The target $y = 1$ is greater than the current prediction $\hat{y} = 0.622459$, requiring $\hat{y}$ and $z$ to increase. Since input $x = 3 > 0$, increasing $w$ increases $z$, driving $\hat{y}$ closer to $1$.

---

## Part 7: The Largest Sigmoid Gradient

### 7.1 Maximum Derivative Analysis

- The peak of $\sigma'(z)$ occurs at **$z = 0$**.
- At $z = 0$, $\sigma(0) = 0.5$.
- Maximum derivative value:
  $$
  \sigma'_{max} = \sigma'(0) = (0.5)(1 - 0.5) = \mathbf{0.25}
  $$
- **Universal Bound**: $\sigma'(z) \le 0.25$ for all $z \in \mathbb{R}$.

---

### 7.2 The Inherent Shrinkage Problem

**Given Setup (Best possible case at $z = 0$)**: $x = 1, w = 0, b = 0, y = 1, \alpha = 1$.

- $z = (0)(1) + 0 = \mathbf{0}$
- $\hat{y} = \sigma(0) = \mathbf{0.5}$
- $\hat{y} - y = 0.5 - 1.0 = \mathbf{-0.5}$
- $\sigma'(0) = \mathbf{0.25}$ (the theoretical maximum possible)
- Gradient: $\frac{\partial L}{\partial w} = (-0.5)(0.25)(1) = \mathbf{-0.125}$
- Update: $w_{new} = 0 - (1)(-0.125) = \mathbf{0.125}$

> **Takeaway**: Even when the error is substantial ($-0.5$) and the sigmoid is at its most sensitive point ($z = 0$), the maximum derivative of $0.25$ **shrinks the gradient by $4\times$**. In deep networks with stacked sigmoid layers, chaining multiple $\le 0.25$ factors causes gradients to vanish exponentially.

---

## Part 8: The Tanh Function

### 8.1 Definition & Properties

$$
\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
$$

- Output range: $-1 < \tanh(z) < 1$.
- Zero-centered: passes through $(0, 0)$.

---

### 8.2 Derivation of $\tanh'(z)$

Let $u = e^z - e^{-z}$ and $v = e^z + e^{-z}$, so $\tanh(z) = \frac{u}{v}$.

1. **Step 1 (Differentiate Parts)**:

   $$
   u' = \frac{d}{dz}(e^z - e^{-z}) = e^z + e^{-z} = v
   $$

   $$
   v' = \frac{d}{dz}(e^z + e^{-z}) = e^z - e^{-z} = u
   $$
2. **Step 2 (Quotient Rule)**:

   $$
   \tanh'(z) = \frac{u'v - uv'}{v^2} = \frac{v \cdot v - u \cdot u}{v^2} = \frac{v^2 - u^2}{v^2}
   $$
3. **Step 3 (Simplify)**:

   $$
   \tanh'(z) = \frac{v^2}{v^2} - \frac{u^2}{v^2} = 1 - \left(\frac{u}{v}\right)^2 = \mathbf{1 - \tanh^2(z)}
   $$
4. **Value at $z = 0$**:

   $$
   \tanh'(0) = 1 - \tanh^2(0) = 1 - 0^2 = \mathbf{1.00}
   $$

---

### 8.3 & 8.4 Numerical with Tanh

**Given Setup** *(Identical to Part 7.2)*: $x = 1, w = 0, b = 0, y = 1, \alpha = 1$.

- $z = (0)(1) + 0 = \mathbf{0}$
- $\hat{y} = \tanh(0) = \mathbf{0}$
- $\hat{y} - y = 0 - 1 = \mathbf{-1.0}$
- $\tanh'(0) = \mathbf{1.0}$
- Gradient: $\frac{\partial L}{\partial w} = (-1.0)(1.0)(1) = \mathbf{-1.0}$
- Update: $w_{new} = 0 - (1)(-1.0) = \mathbf{1.0}$

---

### 8.5 Comparison: Sigmoid vs. Tanh

| Quantity                                             | Sigmoid ($\sigma$) | Tanh | Ratio ($\text{Tanh} / \text{Sigmoid}$) |            |                                        |
| :--------------------------------------------------- | :--------------------------------------------------------------------: | :--------: | :------------------------------------: |
| **Output $\hat{y}$**                         |                                $0.5$                                |  $0.0$  |                   —                   |
| **Error $(\hat{y} - y)$**                    |                                $-0.5$                                |  $-1.0$  | $\frac{|-1.0|}{|-0.5|} = \mathbf{2}$ |
| **Derivative $f'(0)$**                       |                                $0.25$                                |  $1.00$  |   $\frac{1.00}{0.25} = \mathbf{4}$   |
| **Gradient $\frac{\partial L}{\partial w}$** |                               $-0.125$                               | $-1.000$ |             $\mathbf{8}$             |
| **Parameter Update $w_{new}$**               |                               $0.125$                               | $1.000$ |             $\mathbf{8}$             |

$$
\text{Total Step Ratio} = \frac{\text{Tanh Step}}{\text{Sigmoid Step}} = \frac{1.0}{0.125} = \mathbf{8}
$$

$$
\text{Accounting for the factor of 8} = (\text{Error Ratio: } 2) \times (\text{Derivative Ratio: } 4) = \mathbf{8}
$$

---

## Summary of the Three Activation Functions

| Activation          |                Function Formula$f(z)$                |      Derivative$f'(z)$      | Max$f'(z)$ | Trainable with Gradient Descent?                                |
| :------------------ | :----------------------------------------------------: | :----------------------------: | :----------: | :-------------------------------------------------------------- |
| **Threshold** | $\begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$ | $0 \quad (\forall z \neq 0)$ |    $0$    | **No** (Gradient is permanently zero)                     |
| **Sigmoid**   |                $\frac{1}{1 + e^{-z}}$                |  $\sigma(z)(1 - \sigma(z))$  |   $0.25$   | **Yes**, but slowly (Gradient shrinks by $\ge 4\times$) |
| **Tanh**      |         $\frac{e^z - e^{-z}}{e^z + e^{-z}}$         |       $1 - \tanh^2(z)$       |   $1.00$   | **Yes**, faster (Full gradient preserved at origin)       |

> **Final Question**: *In one line: why can a perceptron using the threshold function never be trained by this rule?*
>
> **Answer**: Because its derivative is $f'(z) = 0$ everywhere (and undefined at $z=0$), which forces $\frac{\partial L}{\partial w} = (\hat{y} - y) f'(z) x = 0$ at all times, making parameter updates impossible regardless of error size or learning rate.
