# Deep Learning — Worksheet 03: Backpropagation (The Output Layer)
**Complete Step-by-Step Solutions, Mathematical Derivations & Computational Traces**

---

## 1. Architecture & Notation

### 1.1 Network Architecture
A 2-layer fully connected network (MLP):
- **Input Layer ($l=0$)**: $2$ inputs, $x = [x_1, x_2]^T = [1.0, 2.0]^T$.
- **Hidden Layer ($l=1$)**: $3$ neurons ($h_1, h_2, h_3$) with Sigmoid activation ($\sigma$).
- **Output Layer ($l=2$)**: $2$ neurons ($o_1, o_2$) with Sigmoid activation ($\sigma$) producing predictions $\hat{y} = [\hat{y}_1, \hat{y}_2]^T$.
- **Ground Truth Targets**: $y = [y_1, y_2]^T = [1.0, 0.0]^T$.
- **Learning Rate**: $\alpha = 0.5$.

```
  Inputs (x)          Hidden Layer a^[1]          Outputs ŷ = a^[2]       Targets (y)
  
  [ x₁ = 1 ] ───┬─────────► [ h₁ ] ─────────┬─────────► [ o₁ ] ─────────► y₁ = 1
                │           [ h₂ ] ─────────┼─────────► [ o₂ ] ─────────► y₂ = 0
  [ x₂ = 2 ] ───┴─────────► [ h₃ ] ─────────┘
```

---

### 1.2 Mathematical Notation
- $w^{[l]}_{jk}$: Weight connecting neuron $k$ in layer $l-1$ to neuron $j$ in layer $l$.
- $b^{[l]}_j$: Bias of neuron $j$ in layer $l$.
- $z^{[l]}_j$: Pre-activation (linear combination) of neuron $j$ in layer $l$:
  $$z^{[l]}_j = \sum_k w^{[l]}_{jk} a^{[l-1]}_k + b^{[l]}_j$$
- $a^{[l]}_j$: Post-activation of neuron $j$ in layer $l$:
  $$a^{[l]}_j = \sigma(z^{[l]}_j) = \frac{1}{1 + e^{-z^{[l]}_j}}$$
- $\delta^{[l]}_j$: Error term (sensitivity) of neuron $j$ in layer $l$:
  $$\delta^{[l]}_j = \frac{\partial L}{\partial z^{[l]}_j}$$

---

### 1.3 Given Parameters (Old Weights & Biases)

#### Layer 1 (Input $\to$ Hidden):
$$W^{[1]} = \begin{bmatrix} w^{[1]}_{11} & w^{[1]}_{12} \\ w^{[1]}_{21} & w^{[1]}_{22} \\ w^{[1]}_{31} & w^{[1]}_{32} \end{bmatrix} = \begin{bmatrix} 0.5 & -0.2 \\ 0.3 & 0.2 \\ -0.4 & -0.6 \end{bmatrix}, \quad b^{[1]} = \begin{bmatrix} 0.4 \\ 0.3 \\ 0.1 \end{bmatrix}$$

#### Layer 2 (Hidden $\to$ Output):
$$W^{[2]} = \begin{bmatrix} w^{[2]}_{11} & w^{[2]}_{12} & w^{[2]}_{13} \\ w^{[2]}_{21} & w^{[2]}_{22} & w^{[2]}_{23} \end{bmatrix} = \begin{bmatrix} 0.5 & -0.3 & 0.8 \\ -0.6 & 0.4 & 0.2 \end{bmatrix}, \quad b^{[2]} = \begin{bmatrix} 0.2 \\ -0.1 \end{bmatrix}$$

---

## 2. Forward Pass: Input to Hidden Layer ($l=1$)

### Hidden Neuron 1 ($h_1$):
- Pre-activation:
  $$z^{[1]}_1 = w^{[1]}_{11} x_1 + w^{[1]}_{12} x_2 + b^{[1]}_1 = (0.5)(1) + (-0.2)(2) + 0.4 = 0.5 - 0.4 + 0.4 = \mathbf{0.5000}$$
- Activation:
  $$a^{[1]}_1 = \sigma(0.5) = \frac{1}{1 + e^{-0.5}} = \mathbf{0.6225}$$

---

### Hidden Neuron 2 ($h_2$):
- **Blank B1 (Linear Combination)**:
  $$z^{[1]}_2 = w^{[1]}_{21} x_1 + w^{[1]}_{22} x_2 + b^{[1]}_2 = \mathbf{(0.3)(1) + (0.2)(2) + 0.3}$$
- **Blank B2 (Pre-activation Total)**:
  $$z^{[1]}_2 = 0.3 + 0.4 + 0.3 = \mathbf{1.0000}$$
- **Blank B3 (Activation)**:
  $$a^{[1]}_2 = \sigma(1.0) = \frac{1}{1 + e^{-1.0}} = \mathbf{0.7311}$$

---

### Hidden Neuron 3 ($h_3$):
- **Blank B4 (Linear Combination)**:
  $$z^{[1]}_3 = w^{[1]}_{31} x_1 + w^{[1]}_{32} x_2 + b^{[1]}_3 = \mathbf{(-0.4)(1) + (-0.6)(2) + 0.1}$$
- **Blank B5 (Pre-activation Total)**:
  $$z^{[1]}_3 = -0.4 - 1.2 + 0.1 = \mathbf{-1.5000}$$
- **Blank B6 (Activation)**:
  $$a^{[1]}_3 = \sigma(-1.5) = \frac{1}{1 + e^{1.5}} = \mathbf{0.1824}$$

> **Hidden Activation Vector Summary**:
> $$a^{[1]} = \begin{bmatrix} a^{[1]}_1 \\ a^{[1]}_2 \\ a^{[1]}_3 \end{bmatrix} = \begin{bmatrix} 0.6225 \\ 0.7311 \\ 0.1824 \end{bmatrix}$$

---

## 3. Forward Pass: Hidden to Output Layer ($l=2$)

### Output Neuron 1 ($o_1$):
- Linear Combination:
  $$z^{[2]}_1 = w^{[2]}_{11} a^{[1]}_1 + w^{[2]}_{12} a^{[1]}_2 + w^{[2]}_{13} a^{[1]}_3 + b^{[2]}_1$$
  $$z^{[2]}_1 = (0.5)(0.6225) + (-0.3)(0.7311) + (0.8)(0.1824) + 0.2$$
  $$z^{[2]}_1 = 0.3113 - 0.2193 + 0.1459 + 0.2$$
- **Blank B7 (Pre-activation Total)**:
  $$z^{[2]}_1 = \mathbf{0.4378}$$
- **Blank B10 (Prediction $\hat{y}_1$)**:
  $$\hat{y}_1 = a^{[2]}_1 = \sigma(0.4378) = \mathbf{0.6077}$$

---

### Output Neuron 2 ($o_2$):
- **Blank B8 (Linear Combination)**:
  $$z^{[2]}_2 = w^{[2]}_{21} a^{[1]}_1 + w^{[2]}_{22} a^{[1]}_2 + w^{[2]}_{23} a^{[1]}_3 + b^{[2]}_2$$
  $$z^{[2]}_2 = \mathbf{(-0.6)(0.6225) + (0.4)(0.7311) + (0.2)(0.1824) - 0.1}$$
- **Blank B9 (Pre-activation Total)**:
  $$z^{[2]}_2 = -0.3735 + 0.2924 + 0.0365 - 0.1 = \mathbf{-0.1446}$$
- **Blank B11 (Prediction $\hat{y}_2$)**:
  $$\hat{y}_2 = a^{[2]}_2 = \sigma(-0.1446) = \mathbf{0.4639}$$

> **Output Prediction Vector Summary**:
> $$\hat{y} = \begin{bmatrix} \hat{y}_1 \\ \hat{y}_2 \end{bmatrix} = \begin{bmatrix} 0.6077 \\ 0.4639 \end{bmatrix}$$

---

## 4. Error and Total Loss Computation

### 4.1 Error Terms ($e_j = \hat{y}_j - y_j$)
- **Blank B12 (Error for Neuron 1)**:
  $$e_1 = \hat{y}_1 - y_1 = 0.6077 - 1.0 = \mathbf{-0.3923} \quad (\text{Prediction is too low})$$
- **Blank B13 (Error for Neuron 2)**:
  $$e_2 = \hat{y}_2 - y_2 = 0.4639 - 0.0 = \mathbf{0.4639} \quad (\text{Prediction is too high})$$

---

### 4.2 Squared Errors & Total Loss
Total Mean Squared Error:
$$L = \frac{1}{2}\left(e_1^2 + e_2^2\right) = \frac{1}{2}\Big((\hat{y}_1 - y_1)^2 + (\hat{y}_2 - y_2)^2\Big)$$

- **Blank B14 ($e_1^2$)**:
  $$e_1^2 = (-0.3923)^2 = \mathbf{0.1539}$$
- **Blank B15 ($e_2^2$)**:
  $$e_2^2 = (0.4639)^2 = \mathbf{0.2152}$$
- **Blank B16 (Total Loss $L$)**:
  $$L = \frac{1}{2}(0.1539 + 0.2152) = \frac{1}{2}(0.3691) = \mathbf{0.1846}$$

---

## 5. What a Weight Touches & The Chain Rule

### ❓ Comprehension Check (Q0)
> **Question**: *If we change weight $w^{[2]}_{11}$, what downstream quantities change?*
> 
> **Answer**: **$z^{[2]}_1$, then $\hat{y}_1$, then $L$**.
> *(Hidden activation $a^{[1]}_1$ was computed before $w^{[2]}_{11}$, and Output 2 has its own separate weights).*

---

### 5.1 The Three Links of Backpropagation
By the Chain Rule:
$$\frac{\partial L}{\partial w^{[2]}_{11}} = \underbrace{\frac{\partial L}{\partial \hat{y}_1}}_{\text{Link 1 (Error)}} \cdot \underbrace{\frac{\partial \hat{y}_1}{\partial z^{[2]}_1}}_{\text{Link 2 (Activation Slope)}} \cdot \underbrace{\frac{\partial z^{[2]}_1}{\partial w^{[2]}_{11}}}_{\text{Link 3 (Source Signal)}}$$

#### Link 1: Error at the Output
$$\frac{\partial L}{\partial \hat{y}_1} = \frac{\partial}{\partial \hat{y}_1}\left[\frac{1}{2}(\hat{y}_1 - y_1)^2 + \frac{1}{2}(\hat{y}_2 - y_2)^2\right] = (\hat{y}_1 - y_1) = \mathbf{-0.3923}$$

#### Link 2: Derivative of Output Activation
- **Blank B17 (Symbolic)**: $\frac{\partial \hat{y}_1}{\partial z^{[2]}_1} = \mathbf{\sigma'(z^{[2]}_1)}$
- **Blank B18 (Numeric)**:
  $$\sigma'(z^{[2]}_1) = \hat{y}_1(1 - \hat{y}_1) = (0.6077)(1 - 0.6077) = (0.6077)(0.3923) = \mathbf{0.2384}$$

#### Link 3: Derivative of Pre-activation
- **Blank B19 (Symbolic for $w^{[2]}_{11}$)**: $\frac{\partial z^{[2]}_1}{\partial w^{[2]}_{11}} = \mathbf{a^{[1]}_1}$
- **Blank B20 (Numeric for $w^{[2]}_{11}$)**: $\frac{\partial z^{[2]}_1}{\partial w^{[2]}_{11}} = a^{[1]}_1 = \mathbf{0.6225}$
- **Blank B21 (Symbolic & Numeric for $w^{[2]}_{12}$)**: $\frac{\partial z^{[2]}_1}{\partial w^{[2]}_{12}} = a^{[1]}_2 = \mathbf{0.7311}$
- **Blank B22 (Derivative for Bias $b^{[2]}_1$)**: $\frac{\partial z^{[2]}_1}{\partial b^{[2]}_1} = \mathbf{1}$

---

## 6. Multiply, Then Name Output Error Sensitivity $\delta^{[2]}_1$

### 6.1 Multiplying the Three Links for $w^{[2]}_{11}$
- **Blank B23 (Gradient Value)**:
  $$\frac{\partial L}{\partial w^{[2]}_{11}} = (-0.3923) \cdot (0.2384) \cdot (0.6225) = \mathbf{-0.0582}$$

---

### 6.2 Sense Check on Gradient Direction
- **Blank B24 (Direction Sense Check)**:
  > **Prompt**: *The gradient $\frac{\partial L}{\partial w^{[2]}_{11}} = -0.0582$ is negative. What will the update do to this weight?*
  > 
  > **Answer**: **Increase it, because output 1 is too low ($\hat{y}_1 < y_1$) and hidden neuron 1 is switched on ($a^{[1]}_1 > 0$).**
  > *(The update rule subtracts the gradient: $w_{new} = w_{old} - \alpha(-\text{grad}) = w_{old} + \alpha|\text{grad}|$, increasing the weight).*

---

### 6.3 Defining the Error Term $\delta^{[2]}_1$
The first two links belong to the destination neuron, not the individual incoming weights:
$$\delta^{[2]}_1 = \frac{\partial L}{\partial z^{[2]}_1} = (\hat{y}_1 - y_1) \cdot \sigma'(z^{[2]}_1)$$

- **Blank B25 (Compute $\delta^{[2]}_1$)**:
  $$\delta^{[2]}_1 = (-0.3923)(0.2384) = \mathbf{-0.0935}$$

- **Blank B26 (General Gradient Rule for Output 1)**:
  $$\frac{\partial L}{\partial w^{[2]}_{1k}} = \mathbf{\delta^{[2]}_1 \cdot (\text{the activation it comes from } a^{[1]}_k)}$$

---

## 7. The Other Weights into Output 1

With $\delta^{[2]}_1 = -0.0935$ already computed, we calculate the remaining gradients in a single multiplication:

- **Blank B27 (Gradient for $w^{[2]}_{12}$)**:
  $$\frac{\partial L}{\partial w^{[2]}_{12}} = \delta^{[2]}_1 \cdot a^{[1]}_2 = (-0.0935)(0.7311) = \mathbf{-0.0684}$$
- **Blank B28 (Gradient for $w^{[2]}_{13}$)**:
  $$\frac{\partial L}{\partial w^{[2]}_{13}} = \delta^{[2]}_1 \cdot a^{[1]}_3 = (-0.0935)(0.1824) = \mathbf{-0.0171}$$

---

## 8. Output Neuron 2 Calculations (Your Turn)

### 8.1 Error and Delta for Output 2
- **Blank B29 (Error)**:
  $$\hat{y}_2 - y_2 = 0.4639 - 0.0 = \mathbf{0.4639}$$
- **Blank B30 (Slope $\sigma'(z^{[2]}_2)$)**:
  $$\sigma'(z^{[2]}_2) = \hat{y}_2(1 - \hat{y}_2) = (0.4639)(1 - 0.4639) = (0.4639)(0.5361) = \mathbf{0.2487}$$
- **Blank B31 (Compute $\delta^{[2]}_2$)**:
  $$\delta^{[2]}_2 = (\hat{y}_2 - y_2) \cdot \sigma'(z^{[2]}_2) = (0.4639)(0.2487) = \mathbf{0.1154}$$

---

### 8.2 The Three Gradients for Output 2
- **Blank B32 (Gradient for $w^{[2]}_{21}$)**:
  $$\frac{\partial L}{\partial w^{[2]}_{21}} = \delta^{[2]}_2 \cdot a^{[1]}_1 = (0.1154)(0.6225) = \mathbf{0.0718}$$
- **Blank B33 (Gradient for $w^{[2]}_{22}$)**:
  $$\frac{\partial L}{\partial w^{[2]}_{22}} = \delta^{[2]}_2 \cdot a^{[1]}_2 = (0.1154)(0.7311) = \mathbf{0.0844}$$
- **Blank B34 (Gradient for $w^{[2]}_{23}$)**:
  $$\frac{\partial L}{\partial w^{[2]}_{23}} = \delta^{[2]}_2 \cdot a^{[1]}_3 = (0.1154)(0.1824) = \mathbf{0.0210}$$

---

## 9. The Parameter Update & Verification

### 9.1 Parameter Update Rule ($\alpha = 0.5$)
$$w_{new} = w_{old} - \alpha \frac{\partial L}{\partial w}$$

- **Worked Example ($w^{[2]}_{11}$)**:
  $$w^{[2]}_{11, new} = 0.5 - 0.5(-0.0582) = 0.5 + 0.0291 = \mathbf{0.5291}$$
- **Blank B35 ($w^{[2]}_{12}$)**:
  $$w^{[2]}_{12, new} = -0.3 - 0.5(-0.0684) = -0.3 + 0.0342 = \mathbf{-0.2658}$$
- **Blank B36 ($w^{[2]}_{21}$)**:
  $$w^{[2]}_{21, new} = -0.6 - 0.5(0.0718) = -0.6 - 0.0359 = \mathbf{-0.6359}$$
- **Blank B37 (Sense Check)**:
  > **Prompt**: *Output 2 was too high. Its weights all went {{slot}}.*
  > 
  > **Answer**: **down** *(all gradients for output 2 are positive, so subtracting $\alpha \cdot \text{grad}$ decreases all three weights).*

---

### 9.2 Complete Table of Output Layer Updates

| Weight | Old Value $w_{old}$ | Gradient $\frac{\partial L}{\partial w}$ | Step $-\alpha \frac{\partial L}{\partial w}$ | New Value $w_{new}$ | Movement Direction |
| :---: | :---: | :---: | :---: | :---: | :---: |
| $w^{[2]}_{11}$ | $0.5000$ | $-0.0582$ | $+0.0291$ | $\mathbf{0.5291}$ | **UP** $\uparrow$ |
| $w^{[2]}_{12}$ | $-0.3000$ | $-0.0684$ | $+0.0342$ | $\mathbf{-0.2658}$ | **UP** $\uparrow$ |
| $w^{[2]}_{13}$ | $0.8000$ | $-0.0171$ | $+0.0086$ | $\mathbf{0.8086}$ | **UP** $\uparrow$ |
| $w^{[2]}_{21}$ | $-0.6000$ | $+0.0718$ | $-0.0359$ | $\mathbf{-0.6359}$ | **DOWN** $\downarrow$ |
| $w^{[2]}_{22}$ | $0.4000$ | $+0.0844$ | $-0.0422$ | $\mathbf{0.3578}$ | **DOWN** $\downarrow$ |
| $w^{[2]}_{23}$ | $0.2000$ | $+0.0210$ | $-0.0105$ | $\mathbf{0.1895}$ | **DOWN** $\downarrow$ |

---

### 9.3 Verification of Loss Reduction
Re-evaluating the network after this single update:
- **New Predictions**: $\hat{y} = [0.6183, 0.4502]^T$
  - $\hat{y}_1$ increased from $0.6077 \to 0.6183$ (closer to target $1.0$).
  - $\hat{y}_2$ decreased from $0.4639 \to 0.4502$ (closer to target $0.0$).
- **New Total Loss**:
  $$L_{new} = \frac{1}{2}\Big((0.6183 - 1)^2 + (0.4502 - 0)^2\Big) = \frac{1}{2}(0.1457 + 0.2027) = \mathbf{0.1742}$$
  $$\text{Loss decreased from } \mathbf{0.1846 \to 0.1742} \quad \checkmark$$

---

## 10. General Backpropagation Formula & Checkpoint

$$\boxed{\frac{\partial L}{\partial w^{[l]}_{jk}} = \delta^{[l]}_j \cdot a^{[l-1]}_k}$$

> **Core Principle**:
> **A weight's gradient is the error sensitivity at its destination ($\delta^{[l]}_j$) times the activation signal at its source ($a^{[l-1]}_k$).**

### General Formulation for Any Layer $l$:
- **Output Layer Error ($l = L$)**:
  $$\delta^{[L]}_j = (\hat{y}_j - y_j) \cdot f'(z^{[L]}_j)$$
- **Hidden Layer Error ($l < L$)**:
  $$\delta^{[l]}_j = \left(\sum_m w^{[l+1]}_{mj} \delta^{[l+1]}_m\right) \cdot f'(z^{[l]}_j)$$
