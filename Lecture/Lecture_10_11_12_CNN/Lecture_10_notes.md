# Lecture 10: Convolutional Neural Networks (CNN) Fundamentals & LeNet-5

## 1. Biological Motivation: How the Brain Sees

In the late 1950s and 1960s, neurophysiologists **David Hubel and Torsten Wiesel** conducted groundbreaking experiments on the cat visual cortex (winning the 1981 Nobel Prize).

They discovered that the brain does **not** process an entire image pixel by pixel:

1. **Simple Cells**: Fire only when they detect an edge or line at a **specific orientation** (e.g., vertical, horizontal, $45^\circ$) in a **specific fixed spot**.
2. **Complex Cells**: Fire when they see the same orientation anywhere in a local area (**translation invariance**).
3. **Hierarchical Processing**:
   * Low levels detect edges and corners.
   * Mid levels combine edges into textures and geometric shapes.
   * High levels assemble shapes into complex semantic objects (faces, cars, birds).

CNNs are directly inspired by this hierarchical, local receptive field design.

---

## 2. Why Fully Connected Networks (ANNs / MLPs) Fail for Images

Suppose you feed a standard $1000 \times 1000$ RGB color image into a standard feed-forward Multi-Layer Perceptron (MLP):

```
Image (Flattened) ───────────────────► Hidden Layer (1,000 Neurons)
[3,000,000 pixels]                   [1,000 units]
```

### Three Major Fatal Flaws:

1. **Loss of 2D Spatial Structure**:
   * Images have high local correlation: neighboring pixels belong to the same object (e.g., eye pixels are adjacent to nose pixels).
   * Flattening an image into a 1D vector of numbers treats pixels 2 inches apart the exact same way as pixels right next to each other.
2. **Many Parameter**:
   * A $1000 \times 1000 \times 3$ image has $3,000,000$ input features.
   * Connecting this to a modest hidden layer of 1,000 neurons requires:
     $$
     3,000,000 \times 1,000 = 3,000,000,000 \text{ weights (3 Billion parameters!)}
     $$
   * This guarantees massive overfitting and exceeds GPU memory.
3. **No Translation Invariance**:
   * If an MLP is trained on images of a cat centered in the frame, and a new photo has the cat shifted 10 pixels to the left, all input numbers shift into entirely different weights. The MLP fails to recognize it.

---

## 3. Invariance vs Equivariance

* **Translation Invariance**: $f(T(x)) = f(x)$
  * Moving the object in the input does **not** change the final prediction.
  * **Use case**: Image Classification (whether the dog is on the left or right, the label is still "Dog").
* **Translation Equivariance**: $f(T(x)) = T(f(x))$
  * Moving the object in the input moves the output prediction by the exact same amount.
  * **Use case**: Semantic Segmentation and Object Detection (if a car moves right by 20 pixels, its segmentation mask/bounding box must also move right by 20 pixels).

---

## 4. The Convolution Operation: Step-by-Step

### The Flashlight Analogy

Imagine a huge painting hanging in a dark room. You cannot see everything at once. You hold a small flashlight that illuminates a $3 \times 3$ patch. You shine it on the top-left, examine the details, slide it slightly to the right, examine the next patch, and continue across the whole canvas.

```
Input Image (6 x 6)            Kernel / Filter (3 x 3)          Feature Map (4 x 4)
[ 1  0  0  0  0  1 ]                 [  1 -1 -1 ]                    [  3  0  1  3 ]
[ 0  1  0  0  1  0 ]        *        [ -1  1 -1 ]         =          [ -1  1  3  0 ]
[ 0  0  1  1  0  0 ]                 [ -1 -1  1 ]                    [ ...         ]
```

* **Element-wise Multiplication & Sum**:
  For each patch, multiply each image pixel by the corresponding filter weight, sum them all together, and add a learnable bias:
  $$
  z = \left(\sum_{i} \sum_{j} X_{i,j} \cdot K_{i,j}\right) + b
  $$
* **Filters are Feature Detectors**:
  * A filter with positive weights along a vertical line will output a large positive number ("light up") when placed over a vertical edge.
  * These filter weights are **not hand-crafted**—they are learned automatically through backpropagation!

### Working with Color / Multi-Channel Images (3D Filters)

* A grayscale image has 1 channel $\implies$ Filter is $2\text{D}$ ($K \times K$).
* An RGB image has 3 channels $\implies$ Filter is a **3D volume** ($C_{\text{in}} \times K \times K$).
  * The filter depth **must match** the input channel depth!
  * For an RGB image, a $3 \times 3$ filter has shape $3 \times 3 \times 3 = 27$ weights.
  * The convolutions across all 3 channels are summed together into a single 2D feature map.
* To detect $C_{\text{out}}$ different patterns (e.g., edges, corners, textures), we use $C_{\text{out}}$ separate 3D filters, producing an output volume of depth $C_{\text{out}}$.

---

## 5. The Four Master Knobs (Hyperparameters)

### 1. Kernel / Filter Size ($K$ or $F$)

The spatial width and height of the sliding window (commonly $3 \times 3$, $5 \times 5$, or $7 \times 7$).

### 2. Stride ($S$)

The number of pixels the filter shifts per step:

* $S = 1$: Shifts 1 pixel at a time (standard fine-grained extraction).
* $S = 2$: Shifts 2 pixels at a time, skipping every other pixel (halves spatial resolution).

### 3. Padding ($P$)

Adding rows and columns of zeros around the border of the input:

* **Valid Padding ($P = 0$)**: No padding. The output feature map shrinks after every layer.
* **Same Padding ($P = \frac{K - 1}{2}$ for stride 1)**: Output spatial dimensions remain identical to input dimensions.
* **Why Padding is Essential**:
  1. **Prevents over-shrinkage**: Without padding, repeated convolutions quickly reduce a $32 \times 32$ image down to $1 \times 1$.
  2. **Preserves border & corner information**: Without padding, corner pixels are only sampled once by the filter, while central pixels are sampled dozens of times. Zero-padding ensures edge pixels receive fair attention.

### 4. Dilation ($D$)

Spacing between kernel elements (Atrous Convolution):

* $D = 1$: Standard convolution (adjacent weights).
* $D = 2$: Inserts one hole (zero) between weights, expanding the filter's field of view without adding any new parameters!

---

## 6. The Universal Output Dimension Formula

For an input of spatial size $W \times H$, kernel size $K$, padding $P$, and stride $S$:

$$
W_{\text{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

$$
H_{\text{out}} = \left\lfloor \frac{H - K + 2P}{S} \right\rfloor + 1
$$

### Example Walkthrough:

* Input: $224 \times 224$
* Filter: $K = 7 \times 7$, Stride $S = 2$, Padding $P = 3$
* Calculation:
  $$
  W_{\text{out}} = \frac{224 - 7 + 2(3)}{2} + 1 = \frac{223}{2} + 1 = 111.5 + 1 \implies 112
  $$
* Output size is $112 \times 112$!

---

## 7. Two Core Principles that Save Parameters

1. **Local Connectivity (Sparse Interactions)**:
   * Instead of every hidden neuron connecting to every input pixel, each neuron connects only to a tiny local patch (e.g., $3 \times 3 = 9$ pixels).
2. **Weight Sharing (Parameter Sharing)**:
   * The exact same $3 \times 3$ filter slides across the entire image.
   * If a feature (like a cat ear) is useful in the top-left, the same filter can detect it in the bottom-right.
   * A single $3 \times 3$ filter has only **10 parameters** (9 weights + 1 bias), regardless of whether the image is $32 \times 32$ or $4000 \times 4000$!

---

## 8. Pooling Layers (Downsampling)

Pooling reduces the spatial resolution (width and height) of feature maps while keeping the number of channels intact.

```
Max Pooling (2x2 filter, stride 2)
[ 1  3 | 2  4 ]          [ 3 | 4 ]
[ 2  0 | 1  1 ]   ───►   [ ─ ┼ ─ ]
[──────┼──────]          [ 8 | 5 ]
[ 5  8 | 0  3 ]
[ 2  6 | 5  1 ]
```

* **Max Pooling**: Takes the maximum value in each window. Preserves the most dominant detected features.
* **Average Pooling**: Computes the mean value. Produces a smoother aggregation.
* **Why Pool?**
  1. **Reduces parameter count and computation** in downstream layers.
  2. **Provides local translation invariance**: If an edge shifts slightly inside the $2 \times 2$ window, the maximum value remains the same!
  3. **Increases receptive field** of later layers.

---

## 9. Receptive Field (RF)

> **Definition**: The **Receptive Field** of a neuron is the region in the original input image that directly influences that neuron's activation.

* In Layer 1 with a $3 \times 3$ kernel: RF is $3 \times 3$.
* In Layer 2 with a $3 \times 3$ kernel: Each unit looks at $3 \times 3$ units of Layer 1. The effective RF on the input expands to **$5 \times 5$**!
* In Layer 3 with a $3 \times 3$ kernel: The effective RF expands to **$7 \times 7$**!

### The Power of Stacking Small Kernels:

Stacking two $3 \times 3$ convolutions gives the exact same receptive field as one $5 \times 5$ convolution, but:

* Parameters for one $5 \times 5$: $5^2 = 25$
* Parameters for two $3 \times 3$: $2 \times (3^2) = 18$ (**28% parameter reduction!**)
* Plus, you get **two non-linear activation functions** (ReLUs) instead of one, making the network much more expressive!

---

## 10. Complete CNN Architecture & LeNet-5 (1998)

### Standard Full Pipeline

$$
\text{Input Image} \to [\text{Conv} \to \text{ReLU} \to \text{Pool}] \times N \to \text{Flatten} \to [\text{Dense} \to \text{ReLU}] \to \text{Dense} \to \text{Softmax}
$$

### LeNet-5 Architecture (Yann LeCun, 1998)

Designed for automated handwritten digit recognition on checks (MNIST dataset):

```
Input (32x32x1)
  │
  ▼
[C1: Conv 5x5, s=1, pad=0] ──► 6 feature maps of size 28x28
  │
  ▼
[S2: Avg Pool 2x2, s=2]   ──► 6 feature maps of size 14x14
  │
  ▼
[C3: Conv 5x5, s=1, pad=0] ──► 16 feature maps of size 10x10
  │
  ▼
[S4: Avg Pool 2x2, s=2]   ──► 16 feature maps of size 5x5
  │
  ▼
[C5: Conv 5x5 (Dense)]    ──► 120 feature maps of size 1x1
  │
  ▼
[F6: Fully Connected]     ──► 84 neurons
  │
  ▼
[Output: RBF / Softmax]   ──► 10 class probabilities (digits 0 to 9)
```

LeNet-5 established the foundational template for all modern computer vision architectures.
