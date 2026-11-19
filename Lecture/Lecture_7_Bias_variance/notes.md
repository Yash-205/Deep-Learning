# Lecture 7: Bias and Variance in Deep Learning

## 1. The Big Picture: Why Do Models Fail?

When we train a machine learning or deep learning model, the real-world data we feed it is messy, noisy, and full of natural variations.

Our ultimate goal is **generalization**—we want the model to perform well not just on the data we used for training, but on brand new, unseen data.

Two primary roadblocks stand in the way of good generalization:

1. **Underfitting (High Bias)**
2. **Overfitting (High Variance)**

---

## 2. Issues in Model Fitting

When we train a neural network, its performance on the **Training Set** versus the **Evaluation / Validation Set** tells us exactly what kind of problem we have:

| Characteristic     | Training Set Error   | Evaluation Set Error | Model Status             | Diagnosis                                                                   |
| :----------------- | :------------------- | :------------------- | :----------------------- | :-------------------------------------------------------------------------- |
| **Underfit** | High (Poor)          | High (Poor)          | High Bias, Low Variance  | Model is too simple; failed to learn the underlying pattern.                |
| **Overfit**  | Very Low (Very Good) | High (Very Poor)     | Low Bias, High Variance  | Model is too complex; memorized training noise instead of general patterns. |
| **Good Fit** | Low (Good)           | Low (Good)           | Balanced Bias & Variance | Sweet spot! Model captures true patterns and generalizes well.              |

---

## 3. What is Bias? What is Variance?

### Bias (Error from overly simplistic assumptions)

* **Simple Definition**: Bias measures how far off the model's average predictions are from the true real-world pattern.
* **Analogy**: Imagine a student who assumes every math problem can be solved using simple addition. No matter what problem you give them, they always apply addition. Their assumptions are too rigid and fundamentally flawed.
* **In Neural Networks**: High bias happens when the network's capacity is too small (e.g., too few layers or neurons, or trying to fit non-linear curves with a straight line).

### Variance (Error from sensitivity to training data fluctuations)

* **Simple Definition**: Variance measures how much the model's predictions would change if we trained it on a different sample of the training data.
* **Analogy**: Imagine a student who memorizes the exact numbers of every practice question from a textbook. If the exam changes the numbers even slightly, the student fails completely.
* **In Neural Networks**: High variance happens when the network has too many parameters relative to the dataset. It fits every single outlier and random noise point in the training set.

### Prediction Error Decomposition

The total expected prediction error can be mathematically broken down into three components:

$$
\text{Total Expected Error} = \text{Bias}^2 + \text{Variance} + \sigma^2
$$

Where:

* **$\text{Bias}^2$**: Systematic error due to wrong assumptions.
* **$\text{Variance}$**: Sensitivity to specific data samples.
* **$\sigma^2$ (Irreducible Error)**: Inherent noise in the real world (measurement errors, unobserved variables) that no model can ever predict.

```
Total Error
    ▲
    │         \             /   <--- Validation Error (U-Curve)
    │          \   Sweet   /
    │           \   Spot  /
    │  High Bias \   │   / High Variance
    │ (Underfit)  \_ ▼ _/   (Overfit)
    │────────────────────────────
    │              \
    │               \_____      <--- Training Error (keeps dropping)
    └─────────────────────────────► Model Capacity (Width / Depth)
```

---

## 4. How to Counter Underfitting (High Bias)

### The Cause

The model's capacity is simply too small to capture the true underlying relationships in the data.

### The Solutions

To fix high bias, we must **increase model capacity**:

1. **Make the network deeper**: Add more hidden layers.
2. **Make the network wider**: Increase the number of neurons per hidden layer.
3. **Use better / non-linear features**: Ensure non-linear activation functions (like ReLU) are being used.
4. **Train longer**: Increase the number of training epochs or adjust learning rates if training stopped prematurely.
5. **Reduce regularization**: If penalty weights (L1/L2) are too strong, loosen them.

> **Key Rule**: A good strategy in deep learning is to first build a model that can easily overfit the training set (proving it has enough capacity), and then apply regularization to bring the validation error down.

---

## 5. How to Counter Overfitting (High Variance)

### The Cause

The model has so much capacity that it adapts too well to the specific training examples and learns statistical noise that does not exist in the general population.

### The Solutions

To fix high variance, we constrain the model so it learns simpler, smoother functions:

1. **Regularization**: Add a penalty to the loss function (L1 or L2 weight decay) to keep weights small.
2. **Dropout**: Randomly turn off neurons during training to prevent co-adaptation.
3. **Early Stopping**: Monitor validation loss and stop training before the model starts learning noise.
4. **Batch Normalization**: Adds slight stochastic noise per batch and stabilizes training.
5. **Data Augmentation**: Artificially create more training examples (rotations, crops, flips) so the model cannot memorize specific samples.
6. **Ensembling / Bagging**: Average the predictions of multiple models to cancel out individual variances.

---

## 6. Quick Summary & Exam Checklist

* **Underfitting** = High Bias, Low Variance = Poor Train, Poor Test = Fix by **increasing capacity** (more layers/neurons).
* **Overfitting** = Low Bias, High Variance = Great Train, Poor Test = Fix by **regularization & more data**.
* **Good Fit** = Balanced Bias & Variance = Low Train Error, Low Test Error.
* **The Goal**: Keep model capacity just high enough to capture true patterns, combined with regularization to ensure generalizability.
