# Lecture 11: Modern CNN Architectures (AlexNet, VGG, ResNet)

## 1. The Quest for Deeper Networks

Following LeNet-5 (1998), neural networks entered a period of stagnation until 2012.
Between 2012 and 2016, computer vision was revolutionized through the rapid evolution of deep convolutional architectures:

* **AlexNet (2012)**: 8 layers $\implies$ proved deep learning works on large-scale data.
* **VGG-16 (2014)**: 16–19 layers $\implies$ standardized $3 \times 3$ convolutions and depth.
* **ResNet (2015)**: 152 layers $\implies$ broke the depth barrier with residual skip connections.

---

## 2. Upsampling Techniques: Recovering Resolution

In tasks like **Semantic Segmentation** (predicting a class for every pixel) and **Image Generation**, networks first compress the image down (Encoder) and must then **upsample** it back to full resolution (Decoder).

```
   Downsampling (Encoder)                  Upsampling (Decoder)
Image [224x224] ──► Feature Map [7x7] ──► Reconstructed Image [224x224]
```

### The 4 Main Upsampling Methods:

1. **Nearest Neighbor**:
   * Duplicates each pixel into a $2 \times 2$ block.
   * Simple, fast, but creates sharp, blocky artifacts. Non-learnable.
2. **Bilinear Interpolation**:
   * Calculates weighted averages between neighboring sample points.
   * Produces smooth transitions. Non-learnable.
3. **Max Unpooling**:
   * During the forward pass of a downsampling Max Pooling layer, we record the exact coordinates (the "argmax switches") of where the maximum values came from.
   * In the unpooling layer, we place each value directly back into its recorded original position and fill all other slots with zeros.
4. **Transposed Convolution ("Deconvolution" / Fractionally Strided Conv)**:
   * A **learnable** upsampling layer with trainable weights!
   * In standard convolution, multiple input pixels produce 1 output pixel.
   * In transposed convolution, 1 input pixel is multiplied by a $K \times K$ kernel and broadcasted across multiple output pixels.
   * Mathematically, its operation is represented by the **transpose of the convolution weight matrix** ($C^T$).

---

## 3. AlexNet (2012): The Spark of Modern AI

Created by Alex Krizhevsky, Ilya Sutskever, and Geoffrey Hinton. Won the ImageNet 2012 competition, crushing classical computer vision by dropping the top-5 error rate from **~26% down to 16.4%**.

```
Input (224x224x3)
  ├── Conv1: 96 filters (11x11, s=4) + ReLU + LRN + MaxPool(3x3, s=2)
  ├── Conv2: 256 filters (5x5, pad=2) + ReLU + LRN + MaxPool(3x3, s=2)
  ├── Conv3: 384 filters (3x3, pad=1) + ReLU
  ├── Conv4: 384 filters (3x3, pad=1) + ReLU
  ├── Conv5: 256 filters (3x3, pad=1) + ReLU + MaxPool(3x3, s=2)
  ├── FC6: 4096 neurons + ReLU + Dropout(0.5)
  ├── FC7: 4096 neurons + ReLU + Dropout(0.5)
  └── FC8: 1000 neurons + Softmax
```

### Breakthrough Innovations of AlexNet:

1. **ReLU Activations**: Replaced Tanh/Sigmoid. Avoided saturation in positive values and enabled 6x faster training.
2. **Multi-GPU Parallelism**: Split the network across two 3GB NVIDIA GTX 580 GPUs because one GPU could not fit the model.
3. **Dropout (0.5)**: Used in FC6 and FC7 to prevent the massive 60M parameters from overfitting.
4. **Data Augmentation**: Random $224 \times 224$ crops from $256 \times 256$ images, horizontal flips, and PCA color jittering.
5. **Overlapping Max Pooling**: Used window size $3 \times 3$ with stride $S = 2$ instead of non-overlapping $2 \times 2$ (reduced error by 0.5%).
6. **Local Response Normalization (LRN)**: Simulated biological lateral inhibition (later superseded by Batch Normalization).

---

## 4. VGG Network (2014): Simplicity and Extreme Depth

Created by Karen Simonyan and Andrew Zisserman (Visual Geometry Group, Oxford). Achieved a top-5 error rate of **6.8%** on ImageNet.

### The Core Philosophy: Only $3 \times 3$ Filters Everywhere

AlexNet used giant $11 \times 11$ and $5 \times 5$ filters. VGG showed that you **never need filters larger than $3 \times 3$**!

#### Why Stacking $3 \times 3$ Filters is Mathematically Superior:

* **Two stacked $3 \times 3$ conv layers** have the exact same effective receptive field as **one $5 \times 5$ conv layer**:
  * Parameters for $5 \times 5$: $1 \times (5^2 \times C^2) = 25 C^2$
  * Parameters for two $3 \times 3$: $2 \times (3^2 \times C^2) = 18 C^2$ (**28% parameter savings!**)
* **Three stacked $3 \times 3$ conv layers** have the exact same effective receptive field as **one $7 \times 7$ conv layer**:
  * Parameters for $7 \times 7$: $49 C^2$
  * Parameters for three $3 \times 3$: $3 \times 9 C^2 = 27 C^2$ (**45% parameter savings!**)
* **Extra Non-Linearities**: Stacking three $3 \times 3$ layers introduces three consecutive ReLU activation functions instead of one, making the network vastly more capable of learning complex decision boundaries!

### VGG-16 Architecture Structure:

VGG-16 consists of 5 modular convolutional blocks followed by 3 FC layers:

* **Block 1**: 2 $\times$ [Conv $3 \times 3$, 64 filters] $\to$ MaxPool ($2 \times 2$, $s=2$)
* **Block 2**: 2 $\times$ [Conv $3 \times 3$, 128 filters] $\to$ MaxPool
* **Block 3**: 3 $\times$ [Conv $3 \times 3$, 256 filters] $\to$ MaxPool
* **Block 4**: 3 $\times$ [Conv $3 \times 3$, 512 filters] $\to$ MaxPool
* **Block 5**: 3 $\times$ [Conv $3 \times 3$, 512 filters] $\to$ MaxPool
* **Dense Layers**: FC 4096 $\to$ FC 4096 $\to$ FC 1000 $\to$ Softmax
* **Total Parameters**: ~138 Million (over 100M reside in the first FC layer!).

---

## 5. The Degradation Problem & The Shattered Gradient

If deeper networks learn better representations, can we simply keep adding layers (e.g., 50, 100 layers)?

### The Experiment (He et al., 2015):

Researchers compared a **20-layer plain network** with a **56-layer plain network**:

* Result: The **56-layer network had higher training error AND higher test error** than the 20-layer network!

```
Error Rate
  ▲
  │        /‾‾‾‾‾‾‾‾‾‾‾‾‾‾ 56-layer plain network (Worse!)
  │       /
  │      /________________ 20-layer plain network
  │
  └────────────────────────► Training Epochs
```

* **Why this is NOT Overfitting**: Overfitting means low training error and high test error. Here, **training error was also worse**.
* **The Culprit: The Shattered Gradient Problem**:
  * In deep plain sequential networks, gradients must backpropagate through dozens of repeated weight multiplications.
  * The correlation between the input and the gradient drops to zero.
  * The loss surface looks like a chaotic landscape of jagged white noise mountains instead of a smooth bowl. Gradient descent cannot navigate it.

---

## 6. ResNet (Residual Networks, 2015)

Introduced by Kaiming He, Xiangyu Zhang, Shaoqing Ren, and Jian Sun (Microsoft Research). Won ImageNet 2015 with a **152-layer network** and a top-5 error rate of **3.57%** (beating human performance of ~5.1%).

### The Core Breakthrough: Residual Connections

Instead of forcing a stack of layers to directly fit the underlying target mapping $H(x)$, we let the layers fit a **residual mapping** $\mathcal{F}(x)$:

$$
\mathcal{F}(x) = H(x) - x \implies H(x) = \mathcal{F}(x) + x
$$

```
              x (Input)
              │ ──┐ (Skip / Shortcut Connection)
              ▼   │
        ┌───────────┐
        │ Weight L1 │
        └─────┬─────┘
              ▼ ReLU
        ┌───────────┐
        │ Weight L2 │
        └─────┬─────┘
              ▼
           F(x)
              │   │
              ▼   ▼
              [ + ]  ◄── Add Input directly (Element-wise)
                │
                ▼ ReLU
             Output
```

### Why Do Skip Connections Work So Well?

1. **Identity Shortcut**: If the optimal transformation for a layer is to do nothing (identity mapping), it is notoriously difficult for non-linear layers to learn $H(x) = x$. But with a residual connection, it is trivially easy: gradient descent just pushes all weights in $\mathcal{F}(x)$ toward **zero** ($\mathcal{F}(x) \to 0$), leaving $H(x) = x$!
2. **Unimpeded Gradient Highway**: During backpropagation, the gradient of the loss $\mathcal{L}$ with respect to input $x$ is:
   $$
   \frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial H} \cdot \left( \frac{\partial \mathcal{F}}{\partial x} + 1 \right) = \frac{\partial \mathcal{L}}{\partial H} \cdot \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{L}}{\partial H}
   $$

   The "$+ 1$" term means gradients can flow **directly backward** to the earliest layers without being multiplied by weight matrices, completely eliminating vanishing gradients!
3. **Implicit Ensemble of Paths**: Expanding an $N$-block ResNet computational graph reveals it behaves like an ensemble of $2^N$ sub-paths of varying depths.

---

## 7. Residual Block Types

### 1. BasicBlock (Used in ResNet-18 and ResNet-34)

* Consists of two consecutive $3 \times 3$ convolutions:
  $$
  x \to [\text{Conv } 3 \times 3] \to [\text{BN}] \to [\text{ReLU}] \to [\text{Conv } 3 \times 3] \to [\text{BN}] + x \to [\text{ReLU}]
  $$

### 2. Bottleneck Block (Used in ResNet-50, 101, 152)

When networks become very deep, two $3 \times 3$ convs become computationally expensive. The Bottleneck block sandwiches a $3 \times 3$ conv between two $1 \times 1$ convs:

1. **$1 \times 1$ Conv**: Slashes channel dimensions (e.g., from 256 down to 64).
2. **$3 \times 3$ Conv**: Performs spatial convolution on the reduced 64 channels.
3. **$1 \times 1$ Conv**: Restores (expands) channel dimensions back to 256.

* Provides a **4x reduction in computation**, making 152-layer networks practical!

### Matching Dimensions: Projection Shortcut

When the spatial size is halved (stride 2) or channel count increases:

* We cannot do a direct element-wise addition $x + \mathcal{F}(x)$ because shapes mismatch.
* Solution: Apply a **$1 \times 1$ convolution with stride 2** ($W_s x$) on the shortcut path to match dimensions.

---

## 8. The Exploding Variance Problem & Batch Normalization

* Even when using He initialization within residual blocks, each branch adds variability.
* When recombining the residual branch with the shortcut branch:
  $$
  \text{Var}(x + \mathcal{F}(x)) \approx \text{Var}(x) + \text{Var}(\mathcal{F}(x)) = 1 + 1 = 2
  $$
* Across $K$ residual blocks, the variance doubles at each stage:
  $$
  \text{Variance at block } K = 2^K
  $$
* In a 50-block network, $2^{50}$ causes catastrophic numerical overflow.
* **The Fix**: **Batch Normalization (BN)** placed inside each residual block standardizes the input distribution. When combined with He initialization, it stabilizes variance so it grows linearly rather than exponentially, making ultra-deep ResNets stable.
