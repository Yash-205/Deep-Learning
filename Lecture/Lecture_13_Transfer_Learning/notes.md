# Lecture 13: Transfer Learning — Borrow, Adapt, See Further

> **Worksheet File**: [Open Lecture 13 Interactive Worksheet](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_13_Transfer_Learning/Lecture_13_Transfer_Learning.html)  
> **Core Theme**: Reusing features from a pretrained backbone network instead of training from scratch on small datasets.

---

## 1. The Basics: The 120-Image Small-Data Dilemma

### The Scenario
Imagine your team needs to classify **3 rare leaf species** ($C = 3$). You only have **120 labelled images** in total (40 images per species). A pretrained **ResNet-18** model exists, which contains **11.18 million parameters** ($11,176,512$ weights).

### Why Training from Scratch Fails
* If you initialize ResNet-18 randomly and train it on 120 images, you have over **93,000 parameters for every single training image**:
  $$\frac{11,180,000}{120} \approx 93,166 : 1$$
* When a model has vastly more capacity than the data can constrain, it simply memorizes pixel noise, achieving 100% training accuracy while completely failing on new test images (**catastrophic overfitting**).

### The Solution: Transfer Learning
A model trained on a massive dataset (like **ImageNet** with 1.28 million images across 1,000 classes) has already spent days learning a rich "visual vocabulary" (edges, textures, shapes). Instead of throwing that away, we **reuse the learned feature extractor (the backbone)** and replace only the final classification layer (**the task head**).

---

## 2. Pretrained Models: Splitting Backbone and Head

A network is split into two parts:
$$\hat{y} = g_\phi(f_{\theta^*}(x))$$

1. **Backbone ($f_{\theta^*}$):** The pretrained feature extractor with frozen weights $\theta^*$. It takes an image $x$ and outputs a compact feature vector of dimension $d$ (for ResNet-18, $d = 512$).
2. **Task Head ($g_\phi$):** A new, small linear classifier with parameters $\phi = (\mathbf{W}, \mathbf{b})$:
   $$\mathbf{W} \in \mathbb{R}^{C \times d}, \quad \mathbf{b} \in \mathbb{R}^C$$

### Counting the Head Parameters
* Feature dimension: $d = 512$
* Number of target classes: $C = 3$
* **Weights:** $3 \times 512 = 1,536$
* **Biases:** $3$
* **Total trainable parameters:** $1,536 + 3 = \mathbf{1,539}$
* *Key Takeaway:* We reduce our optimization problem from finding 11.18 million weights down to finding just **1,539 weights**!

### What Transfers vs. What Changes
* **Early layers:** Detect generic visual primitives (edges, color gradients, simple textures). These transfer universally to almost any visual task.
* **Late layers:** Detect task-specific semantics (dog breeds, car grilles). These may need adaptation if the new task is very different from ImageNet.
* **Input Preprocessing:** Pretrained models require input images to match the exact statistical distribution they were trained on:
  * Resize to $256 \times 256$, center crop to $224 \times 224$.
  * Normalize RGB channels using ImageNet mean $\mu = [0.485, 0.456, 0.406]$ and std $\sigma = [0.229, 0.224, 0.225]$.

---

## 3. Feature Extraction & Global Average Pooling (GAP)

### How Feature Extraction Works
Before reaching the classifier, the convolutional layers output a 3D feature map of shape $C_{\text{channels}} \times H \times W$. To turn this 2D spatial map into a single vector, modern networks use **Global Average Pooling (GAP)**:
$$z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W A_{i, j}^c$$
where $A^c$ is the activation grid for channel $c$. Once pooled into vector $\mathbf{z}$, the predicted class probabilities are computed via Softmax:
$$\mathbf{p} = \text{softmax}(\mathbf{W}\mathbf{z} + \mathbf{b})$$

### Exact Numerical Walkthrough (From the Worksheet)
Suppose we have a 2-channel feature map of size $2 \times 2$:
* **Channel 1 Grid:** $\begin{bmatrix} 1 & 3 \\ 2 & 2 \end{bmatrix} \implies z_1 = \frac{1 + 3 + 2 + 2}{4} = \frac{8}{4} = \mathbf{2}$
* **Channel 2 Grid:** $\begin{bmatrix} 0 & 2 \\ 4 & 2 \end{bmatrix} \implies z_2 = \frac{0 + 2 + 4 + 2}{4} = \frac{8}{4} = \mathbf{2}$
* **Resulting Feature Vector:** $\mathbf{z} = [2, 2]^T$

Now, pass $\mathbf{z}$ into a 2-class linear classifier where $\mathbf{W} = \begin{bmatrix} 1 & -0.5 \\ -0.5 & 1 \end{bmatrix}$ and $\mathbf{b} = \begin{bmatrix} 1 \\ 1 \end{bmatrix}$:
* Logit 1: $1(2) - 0.5(2) + 1 = 2 - 1 + 1 = \mathbf{2}$
* Logit 2: $-0.5(2) + 1(2) + 1 = -1 + 2 + 1 = \mathbf{2}$
* Probabilities: $\text{softmax}([2, 2]) = [0.5, 0.5]$
* Cross-Entropy Loss for class 0: $\mathcal{L} = -\ln(0.5) \approx \mathbf{0.693}$

### Feature Caching (Speedup)
Since the backbone is frozen, passing an image through it always yields the exact same vector $\mathbf{z}$.
* We can pass our 120 images through the backbone **just once**, save the 120 vectors to memory, and train the linear head directly on them. This trains in seconds and saves 99% of GPU compute!

---

## 4. Layer Freezing & The BatchNorm Safety Rule

### Masked Gradient Updates
To keep weights fixed during training, we apply a binary update mask $m_\ell \in \{0, 1\}$:
$$\theta_\ell \leftarrow \theta_\ell - \eta_\ell \cdot m_\ell \cdot \nabla_{\theta_\ell} \mathcal{L}$$
* If layer $\ell$ is **frozen** ($m_\ell = 0$): $\Delta \theta_\ell = 0$. The weight never changes.
  * *Example:* If a frozen weight starts at $2.000$, its gradient is $0.4$, and learning rate is $0.1$, its value after the step is still **$2.000$**.
* If layer $\ell$ is **trainable** ($m_\ell = 1$): It updates normally via gradient descent.

### The BatchNorm Trap (Crucial Exam Concept)
Batch Normalization layers maintain moving averages of dataset mean and variance ($\hat{\mu}, \hat{\sigma}^2$).
* If you set `requires_grad = False` on parameters but leave the model in **training mode**, BatchNorm will continue updating its running statistics using noisy statistics from your tiny 120-image batches. This corrupts the robust ImageNet statistics!
* **The Rule:** During fixed feature extraction, always keep the frozen backbone in **evaluation mode (`model.eval()`)** so running statistics remain fixed.

---

## 5. Data Augmentation: Transformations & Semantic Auditing

### The Mathematics of Augmentation
Augmentation artificially expands training data by sampling label-preserving transformations $T$ from a distribution $\mathcal{A}$:
$$\min_{\theta, \phi} \frac{1}{N} \sum_{i=1}^N \mathbb{E}_{T \sim \mathcal{A}} \left[ \mathcal{L}(g_\phi(f_\theta(T(x_i))), y_i) \right]$$

### Common Transformations
* **Horizontal Flip:** $x'[r, c] = x[r, W - 1 - c]$ (reverses column indices).
* **Brightness Adjustment:** $x' = \min(9, \max(0, x + \delta))$ (adds offset $\delta$ and clips to valid pixel range).
* **Right Shift:** Inserts zeros on the left and discards the rightmost column.
* **$90^\circ$ Clockwise Rotation:** Maps output $[r, c]$ from input $[H - 1 - c, r]$.

### Semantic Auditing: Is an Augmentation Safe?
An augmentation is only valid if it **preserves the true label** ($y(T(x)) = y(x)$):
* **Leaf Species:** A flipped leaf is still the same leaf species $\implies$ **Safe**.
* **Arrow Direction (Left vs. Right):** A horizontal flip turns a left arrow into a right arrow $\implies$ **Unsafe** (corrupts the label unless the label is inverted too).
* **Digit Recognition:** Flipping a "6" turns it into a "9" $\implies$ **Unsafe**.

### Preventing Data Leakage
* **Always split into Train, Validation, and Test sets FIRST**, before applying augmentation.
* If you augment before splitting, augmented twins of the same original photograph end up in both training and validation sets, giving artificially high, fake validation accuracy.

---

## 6. Context & The Bridge to Vision Transformers (ViT)

### Receptive Field in CNNs
The receptive field $r_\ell$ is the spatial window of input pixels that can influence a neuron at layer $\ell$:
$$r_\ell = r_{\ell-1} + (k_\ell - 1) \cdot j_{\ell-1}, \quad j_\ell = j_{\ell-1} \cdot s_\ell$$
where $k$ is kernel size, $s$ is stride, and $j$ is the cumulative stride jump.
* For $L$ layers of $3 \times 3$ convolutions ($k=3$) with stride 1:
  $$r = 1 + 2L$$
  * 1 layer: $r = 1 + 2(1) = 3 \times 3$
  * 2 layers: $r = 1 + 2(2) = 5 \times 5$
  * 3 layers: $r = 1 + 2(3) = 7 \times 7$

### Why Convolutions Struggle with Distant Clues
* A CNN's receptive field grows only linearly with depth.
* If a visual decision depends on comparing two distant marks (e.g. top-left corner vs. bottom-right corner), an early CNN layer cannot compare them directly.
* Furthermore, the **effective** receptive field is a bell curve (Gaussian) heavily concentrated at the center, leaving outer borders weak.

### The Vision Transformer (ViT) Solution
Instead of sliding local kernels, ViT cuts the image into non-overlapping patches:
* For a $224 \times 224$ image split into $16 \times 16$ patches ($P=16$):
  $$N = \left(\frac{H}{P}\right) \times \left(\frac{W}{P}\right) = \left(\frac{224}{16}\right) \times \left(\frac{224}{16}\right) = 14 \times 14 = \mathbf{196 \text{ patch tokens}}$$
* Each patch directly attends to every other patch via **Self-Attention**:
  $$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$
* **Quadratic Attention Cost:** Every patch compares itself to all 196 patches:
  $$\text{Attention scores per head} = N \times N = 196 \times 196 = \mathbf{38,416 \text{ scores}}$$
* *Benefit:* Every patch can see the entire image immediately at Layer 1.

---

## 7. Review Questions & Answers (From the Worksheet)

1. **A 512-feature backbone feeds a 3-class linear head with bias. How many head parameters are learned?**  
   &rarr; **1539** ($3 \times 512 + 3$).
2. **Fixed feature extraction means:**  
   &rarr; **Keep the backbone fixed and train the new head**.
3. **Two $2 \times 2$ feature maps are $[1, 3; 2, 2]$ and $[0, 2; 4, 2]$. After global average pooling, what is the vector?**  
   &rarr; **[2, 2]** (mean of map 1 is 2; mean of map 2 is 2).
4. **A frozen weight starts at 2.000. Its gradient is 0.4 and learning rate is 0.1. What is its value after the optimizer step?**  
   &rarr; **2.000** (frozen weights receive zero update, $m = 0$).
5. **For a frozen BatchNorm backbone during fixed feature extraction, the safest default is:**  
   &rarr; **Evaluation mode with fixed running statistics (`model.eval()`)**.
6. **To prevent augmented siblings from leaking across splits:**  
   &rarr; **Split original examples first, then augment training examples only**.
7. **A horizontal flip for a left-versus-right arrow task is:**  
   &rarr; **Unsafe unless the label is changed** (flipping changes left into right).
8. **A $224 \times 224$ image split into $16 \times 16$ patches creates 196 tokens. How many patch-to-patch attention scores are formed per head?**  
   &rarr; **38,416** ($196 \times 196$).
