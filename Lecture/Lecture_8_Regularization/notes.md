# Lecture 8: Regularization in Deep Learning

## 1. What is Regularization?

> **Official Definition (Goodfellow et al., Deep Learning Book):**
> *"Regularization is any modification we make to a learning algorithm that is intended to reduce its generalization error but not its training error."*

In simple terms: **Regularization prevents overfitting.** It stops a neural network from memorizing training data and forces it to learn real, robust patterns that work on new, unseen data.

---

## 2. The Core Philosophy: Why Do Smaller Weights Mean Simpler Models?

Imagine a neural network predicting house prices based on size $x$:

* **Model A (Large weights)**: $y = 1000 \cdot x_1 - 850 \cdot x_2 + 920 \cdot x_3$
  * Even a tiny change in an input feature causes the output to swing violently. The decision boundary is jagged and chaotic, fitting every single noisy outlier.
* **Model B (Small weights)**: $y = 0.5 \cdot x_1 + 0.3 \cdot x_2 + 0.2 \cdot x_3$
  * Small changes in inputs produce smooth, gradual changes in output. The decision boundary is smooth and stable.

Regularization methods encourage smaller weights, producing a simpler, smoother function that generalizes better.

---

## 3. Modifying the Loss Function: L1 and L2 Regularization

We modify our loss function by adding a **penalty term** $\Omega(w)$ that punishes large weights:

$$
\mathcal{L}_{\text{regularized}}(w) = \mathcal{L}_{\text{data}}(w) + \lambda \Omega(w)
$$

* $\mathcal{L}_{\text{data}}(w)$: Loss FUnction
* $\Omega(w)$: Penalty on the size of weights.
* $\lambda$ (lambda): The regularization strength hyperparameter.
  * If $\lambda = 0$: No regularization (standard training).
  * If $\lambda$ is too large: Weights are crushed toward 0, causing underfitting.

### Fun Fact: What does "L" stand for?

The "L" stands for **$L_p$ norm** in mathematics: $\|w\|_p = \left(\sum_i |w_i|^p\right)^{1/p}$.

---

### A. L2 Regularization (Ridge / Weight Decay)

* **What it adds**: The sum of **squared magnitudes** of the weights:

  $$
  \Omega(w) = \frac{1}{2} \|w\|_2^2 = \frac{1}{2} \sum_{j} w_j^2
  $$
* **Derivative**: $\frac{\partial \Omega}{\partial w_j} = w_j$
* **Weight Update Rule (Gradient Descent)**:

  $$
  w \leftarrow w - \eta \left( \frac{\partial \mathcal{L}_{\text{data}}}{\partial w} + \lambda w \right)
  $$

  Rearranging terms:
  $$
  w \leftarrow (1 - \eta \lambda) w - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial w}
  $$
* **Why it is called "Weight Decay"**:
  Notice the factor $(1 - \eta \lambda)$. Since $\eta \lambda > 0$, this factor is strictly less than 1 (e.g., $0.99$). At every single update, the weight is first shrunk (decayed) by a percentage before taking the gradient step!

---

### B. L1 Regularization (Lasso)

* **What it adds**: The sum of **absolute values** of the weights:
  $$
  \Omega(w) = \|w\|_1 = \sum_{j} |w_j|
  $$
* **Derivative**: $\text{sign}(w_j) = \begin{cases} +1 & \text{if } w_j > 0 \\ -1 & \text{if } w_j < 0 \end{cases}$
* **Weight Update Rule**:
  $$
  w \leftarrow w - \eta \lambda \cdot \text{sign}(w) - \eta \frac{\partial \mathcal{L}_{\text{data}}}{\partial w}
  $$
* **Constant Pull**: Unlike L2, the penalty force does not shrink as the weight gets smaller. It continuously pulls the weight toward zero at a constant speed $(\eta \lambda)$, driving unimportant weights strictly to **exactly 0.0**.

---

### C. L1 vs L2: Sparsity and Robustness

| Feature                          | L1 Regularization (Lasso)                                                                                 | L2 Regularization (Ridge / Weight Decay)                                                                             |
| :------------------------------- | :-------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------- |
| **Penalty Term**           | $\lambda \sum \|w_j\|$                                                                                  | $\frac{\lambda}{2} \sum w_j^2$                                                                                     |
| **Gradient Force**         | Constant ($\pm 1$) regardless of weight size | Proportional to weight size ($w$)                      |                                                                                                                      |
| **Sparsity**               | **High Sparsity**: Drives many weights to **exactly 0**. Acts as automatic feature selection. | **Low Sparsity**: Drives weights to very small numbers (e.g., 0.01, 0.002), but rarely exactly 0.              |
| **Robustness to Outliers** | **More Robust**: Error penalty grows linearly. Outliers don't distort all weights.                  | **Less Robust**: Error penalty grows quadratically ($w^2$). Network expends massive effort fitting outliers. |
| **Geometric Intuition**    | Diamond-shaped constraint contour. Contours touch at sharp axis corners where weights are 0.              | Circular/Spherical constraint contour. Contours touch smoothly along curves away from axes.                          |
| **Typical Weight Output**  | e.g.,`[0.8, 0.0, 1.2, 0.0]`                                                                             | e.g.,`[0.31, 0.08, 0.42, 0.11]`                                                                                    |

---

## 4. Heuristic Regularization Techniques

### A. Early Stopping

* **The Concept**: During training, training loss keeps decreasing continuously. However, validation loss decreases at first, hits a minimum (the sweet spot), and then begins to rise as the network starts memorizing noise.
* **The Rule**: Stop training as soon as the validation error begins to climb.
* **Saving Best Weights**: Save a checkpoint of the model parameters whenever validation loss hits a new all-time low. When training finishes or triggers early stopping, restore that saved checkpoint.
* **Pros**: Simple, free, zero mathematical overhead, saves GPU time.
* **Cons**: Needs a validation set; may stop prematurely if validation loss temporarily plateaus (fix: use a "patience" parameter, e.g., wait 5–10 epochs before stopping).

---

### B. Model Ensembling & Bagging

* **The Concept**: Train multiple models independently and average their predictions:
  $$
  \hat{y}_{\text{ensemble}} = \frac{1}{M} \sum_{m=1}^{M} \hat{y}_m
  $$
* **Why it works**: If individual models make random, uncorrelated errors, averaging them cancels out the random errors (reduces variance by up to $1/M$), while preserving the correct signals.
* **The Problem for Deep Learning**: Training 10 or 20 massive deep neural networks from scratch is computationally impossible or too expensive in practice.

---

### C. Dropout: The Efficient Neural Network Ensemble

Introduced by Nitish Srivastava, Geoffrey Hinton, et al. (2014).

```
   Normal Network                    Network with Dropout
     [O]   [O]                           [X]   [O]
    /   \ /   \                         /       \
  [O]   [O]   [O]                     [O]   [X]   [O]
    \   / \   /                         \       /
     [O]   [O]                           [O]   [X]
```

#### How it works:

1. **During Training**:
   * For every mini-batch, randomly turn off (drop) each hidden neuron with probability $p$ (typically $p = 0.5$ for hidden layers, $p = 0.1 - 0.2$ for input layers).
   * Dropped neurons output 0 and receive no gradient updates during backward propagation.
   * Every batch effectively trains a different subnetwork!
2. **Why It Reduces Overfitting**:
   * **Breaks Co-adaptation**: Neurons cannot rely on a neighboring neuron to fix their mistakes. Each neuron must learn genuinely useful, independent features.
   * **Implicit Ensemble**: A network with $N$ neurons contains $2^N$ possible subnetworks. Dropout trains an ensemble of $2^N$ subnetworks with shared weights at the cost of training just one network!
3. **During Inference (Testing)**:
   * No neurons are dropped. All neurons are active to use the full predictive power.
   * **Inverted Dropout** (Standard implementation in PyTorch/TensorFlow):
     During training, scale active neurons by $\frac{1}{1 - p}$:
     $$
     \tilde{a} = \frac{a \cdot \text{mask}}{1 - p}
     $$

     Because activations are scaled up during training, **no changes or scaling are needed during testing/inference**!

---

### D. Batch Normalization (BN) as a Regularizer

* **Primary Purpose**: Accelerates training and prevents vanishing/exploding gradients by normalizing layer inputs across the mini-batch to zero mean and unit variance.
* **Regularization Side-Effect**:
  * Because mini-batch mean $\mu_B$ and variance $\sigma_B^2$ are calculated over a random sample of training examples, they fluctuate randomly from batch to batch.
  * This injects slight stochastic noise into the activations of each layer.
  * Just like Dropout, this noise prevents neurons from memorizing exact values. In many modern architectures, Batch Normalization is so effective at regularizing that Dropout can be reduced or eliminated!

---

## 5. Increasing Data: Data Augmentation

When you cannot easily collect more real data, you create synthetic variations of your existing training data:

* **Computer Vision**: Random horizontal flipping, random rotation ($\pm 15^\circ$), random cropping, color jittering, random scaling, Gaussian noise.
* **Why it works**:
  * An image of a cat flipped horizontally is still a cat.
  * An image of a car shifted 5 pixels to the right is still a car.
  * It forces the network to learn invariant features (shapes, textures) rather than memorizing the exact pixel grid coordinates.

---

## 6. Regularization Techniques Summary

| Approach                       | Method                         | How It Fights Overfitting                                                           |
| :----------------------------- | :----------------------------- | :---------------------------------------------------------------------------------- |
| **Loss Modification**    | **L2 (Weight Decay)**    | Shrinks all weights toward 0; keeps decision boundaries smooth.                     |
| **Loss Modification**    | **L1 (Lasso)**           | Forces unimportant weights strictly to 0; creates sparse models.                    |
| **Training Heuristic**   | **Early Stopping**       | Halts training before validation loss begins to degrade.                            |
| **Architecture / Noise** | **Dropout**              | Randomly disables neurons; prevents co-adaptation and acts as an implicit ensemble. |
| **Architecture / Noise** | **Batch Normalization**  | Normalizes layer inputs; mini-batch statistics inject stabilizing noise.            |
| **Data-Centric**         | **Data Augmentation**    | Artificially expands dataset with realistic transformations; prevents memorization. |
| **Multi-Model**          | **Ensembling / Bagging** | Averages predictions across multiple trained models to cancel out variance.         |
