# Lectures 10, 11 & 12: The Complete Guide to Convolutional Neural Networks & Computer Vision

> **Overview**: This master document covers the complete progression of deep learning in computer vision across Lectures 10, 11, and 12—from fundamental convolutional mechanics to modern state-of-the-art vision systems.
>
> **Individual Deep-Dive Notes**:
> * [Lecture 10 Detailed Notes](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_10_11_12_CNN/Lecture_10_notes.md)
> * [Lecture 11 Detailed Notes](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_10_11_12_CNN/Lecture_11_notes.md)
> * [Lecture 12 Detailed Notes](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_10_11_12_CNN/Lecture_12_notes.md)
> * [CNN Formulas & Architectures Reference Guide](file:///Users/yashagarwal/Downloads/Newton%20College/Deep%20Learning/Lecture/Lecture_10_11_12_CNN/CNN_Formulas_and_Architectures.md)

---

# Part 1: Lecture 10 — CNN Fundamentals & LeNet-5

## 1. Why Standard Neural Networks Fail for Images
When an image is flattened into a 1D vector and passed to a standard feed-forward Multi-Layer Perceptron (MLP):
1. **Destruction of 2D Spatial Structure**: Pixel adjacency (e.g., eye next to nose) is completely erased.
2. **Parameter Explosion**: A $1000 \times 1000 \times 3$ image has 3M inputs. Connecting to a 1,000-neuron layer requires $3 \times 10^9$ weights (3 Billion parameters!).
3. **No Translation Invariance**: If an object shifts by 5 pixels, all input values shift to different weights, causing recognition to fail.

## 2. Invariance vs Equivariance
* **Translation Invariance**: $f(T(x)) = f(x)$. Shifting the input does **not** change the output label (used in Image Classification).
* **Translation Equivariance**: $f(T(x)) = T(f(x))$. Shifting the input shifts the output by the exact same amount (used in Object Detection and Segmentation).

## 3. The Convolution Operation & The 4 Master Knobs
A learnable filter slides over the image. At each step, it computes an element-wise multiplication, sums the results, and adds a bias:
$$z = \left(\sum X_{i,j} \cdot K_{i,j}\right) + b$$

For multi-channel images (e.g. RGB with 3 channels), the filter is a **3D volume** ($3 \times K \times K$). The filter channel depth must match the input channel depth. $C_{\text{out}}$ filters produce an output volume of depth $C_{\text{out}}$.

### The 4 Knobs:
1. **Kernel Size ($K$)**: Dimensions of the sliding filter (commonly $3 \times 3$ or $5 \times 5$).
2. **Stride ($S$)**: Step size the filter moves (e.g., $S=1$ moves 1 pixel; $S=2$ moves 2 pixels, halving spatial size).
3. **Padding ($P$)**:
   * **Valid**: No padding ($P=0$). Output shrinks.
   * **Same**: $P = \frac{K-1}{2}$ (for $S=1$). Output dimensions stay identical to input.
   * *Why pad?* Prevents rapid shrinkage and stops corner/edge pixels from being ignored.
4. **Dilation ($D$)**: Spaces between kernel weights ($D=2$ skips 1 pixel), expanding receptive field without adding parameters.

### Output Dimension Formula:
$$W_{\text{out}} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$$

## 4. Why CNNs are Parameter-Efficient
1. **Local Connectivity**: Each neuron connects only to a small local patch ($K \times K$).
2. **Weight Sharing**: The exact same filter weights slide across the entire image.

## 5. Pooling & Receptive Field
* **Max Pooling**: Selects the maximum value in a window (e.g. $2 \times 2$, stride 2). Reduces compute, downsamples resolution, and gives local translation invariance.
* **Receptive Field (RF)**: The input region influencing a neuron.
  * Stacking two $3 \times 3$ conv layers produces an effective $5 \times 5$ RF.
  * Parameters: $2 \times 3^2 = 18$ vs $5^2 = 25$ (**28% parameter savings** + two ReLUs).

## 6. LeNet-5 (1998)
Pioneering architecture by Yann LeCun for handwritten digit recognition (MNIST):
$$\text{Input } (32\times32) \to \text{C1 } (5\times5, 6) \to \text{S2 } (2\times2) \to \text{C3 } (5\times5, 16) \to \text{S4 } (2\times2) \to \text{C5 } (120) \to \text{F6 } (84) \to \text{Output } (10)$$

---

# Part 2: Lecture 11 — Modern CNNs: AlexNet, VGG & ResNet

## 1. Upsampling Methods (Recovering Resolution)
1. **Nearest Neighbor**: Simple $2 \times 2$ pixel copying (blocky, non-learnable).
2. **Bilinear Interpolation**: Weighted average of neighboring samples (smooth, non-learnable).
3. **Max Unpooling**: Reuses saved coordinates from forward Max Pooling; places values back into recorded positions.
4. **Transposed Convolution (Deconvolution)**: Trainable upsampling layer using the transpose of the convolution weight matrix ($C^T$).

## 2. AlexNet (2012)
Won ImageNet 2012 (top-5 error: 16.4%, 8 layers):
* **ReLU**: 6x faster training, no positive saturation.
* **Multi-GPU**: Model split across two 3GB GTX 580 GPUs.
* **Dropout (0.5)** in dense FC layers to prevent overfitting.
* **Data Augmentation**: Random cropping, flips, color jitter.
* **Overlapping Pooling**: $3 \times 3$ window with stride 2.

## 3. VGG-16 (2014)
Showed that **$3 \times 3$ filters everywhere** are mathematically superior:
* Stacking two $3 \times 3$ convs equals one $5 \times 5$ conv (28% fewer params).
* Stacking three $3 \times 3$ convs equals one $7 \times 7$ conv (45% fewer params).
* 13 Conv layers + 3 FC layers = 16 layers (~138M parameters).

## 4. The Degradation & Shattered Gradient Problem
* When training plain 56-layer networks vs 20-layer networks, **both training and test error get worse**!
* Not overfitting, but an optimization failure: in deep sequential networks, gradient correlation with inputs drops to zero ("shattered gradients"), turning the loss surface into chaotic white noise.

## 5. ResNet (2015) & Skip Connections
Introduced residual skip connections: $y = \mathcal{F}(x) + x$.
* **Why it works**:
  * Pushing residual weights $\mathcal{F}(x) \to 0$ makes learning identity mappings trivial.
  * Gradient highway: $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{L}}{\partial y}$. The direct $+1$ term passes gradients back unimpeded.
* **Blocks**:
  * **BasicBlock**: Two $3 \times 3$ convs (ResNet-18/34).
  * **Bottleneck Block**: $1 \times 1 \to 3 \times 3 \to 1 \times 1$ convs (ResNet-50/101/152) to save 4x compute.
  * **Projection Shortcut**: $1 \times 1$ conv with stride 2 used when spatial sizes or channel counts mismatch.
* **Batch Normalization (BN)**: Eliminates the exponential $2^K$ variance explosion caused by adding $x + \mathcal{F}(x)$, stabilizing training.

---

# Part 3: Lecture 12 — Advanced Architectures & Computer Vision Tasks

## 1. U-Net (2015): Biomedical Semantic Segmentation
* **Symmetrical U-Shape**:
  * **Contracting Path (Encoder)**: Convolutions + Max Pooling to extract high-level context ("What").
  * **Expanding Path (Decoder)**: Transposed convolutions to restore spatial resolution ("Where").
  * **Skip Concatenations**: Direct horizontal connections that copy fine-grained spatial feature maps across from the encoder to the decoder, restoring sharp boundary definitions.

## 2. GoogLeNet / Inception (2014)
* **Multi-Scale Processing**: Applies $1 \times 1$, $3 \times 3$, $5 \times 5$ convolutions and $3 \times 3$ Max Pooling **in parallel** on the same input, concatenating all channels.
* **$1 \times 1$ Bottlenecks**: Compresses channel depth before expensive $3 \times 3$ and $5 \times 5$ convolutions, slashing computation by $>70\%$.
* **Global Average Pooling (GAP)**: Replaced massive FC layers, dropping parameters to only 7M.
* **Auxiliary Classifiers**: Injected gradients directly into intermediate layers during training.

## 3. Evolution of Object Detection
Object detection performs **Classification** + **Bounding Box Localization** $[x, y, w, h]$.

| Detection Architecture | Region Proposal Mechanism | Feature Extraction | Inference Speed | Key Limitation / Advantage |
| :--- | :--- | :--- | :--- | :--- |
| **R-CNN (2014)** | Selective Search (~2,000 proposals) | Crops each proposal; runs CNN 2,000 times | ~47 seconds / image | Extremely slow; proposals run on CPU. |
| **Fast R-CNN (2015)** | Selective Search | **Single CNN pass**; RoI Pooling extracts regions from shared feature map | ~2 seconds / image | 25x faster; Selective Search proposal is the bottleneck. |
| **Faster R-CNN (2015)**| **Region Proposal Network (RPN)** with Anchor Boxes | Single CNN pass; shared convolutional backbone | ~0.2 sec (5–7 FPS) | End-to-end GPU training; near-real-time. |
| **YOLO (2016)** | **Single-Stage**: Unified grid regression | Single forward pass predicts boxes & classes directly | **45+ FPS** (True Real-Time) | Ultra-fast; global image reasoning. |

## 4. Modern Frontiers
* **DETR (Detection Transformer)**: Treats detection as a direct set-prediction problem using Transformer encoder-decoders and bipartite matching loss (no anchor boxes or NMS needed).
* **Segment Anything Model (SAM)**: Foundational zero-shot segmentation model capable of interactive prompting with points, boxes, and text masks.

---

# Master Architectures Comparison Sheet

| Architecture | Year | Depth | Parameters | Top-5 Error | Core Breakthrough |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LeNet-5** | 1998 | 7 | ~60K | N/A (MNIST) | First modern CNN template (Conv + Pool + FC). |
| **AlexNet** | 2012 | 8 | ~60M | 16.4% | ReLU, Dropout, Multi-GPU; sparked deep learning era. |
| **VGG-16** | 2014 | 16 | ~138M | 6.8% | Replaced large filters with stacks of small $3 \times 3$ convolutions. |
| **GoogLeNet** | 2014 | 22 | ~7M | 6.7% | Parallel multi-scale Inception modules + $1 \times 1$ bottlenecks + GAP. |
| **ResNet-152** | 2015 | 152 | ~60M | 3.57% | Residual skip connections; broke depth barrier; beat human performance. |
| **U-Net** | 2015 | ~23 | ~30M | N/A (Biomed) | Encoder-Decoder with skip concatenations for pixel-level segmentation. |
| **Faster R-CNN**| 2015 | Backbone+ | ~40M+ | High mAP | RPN + Anchor Boxes for unified near-real-time detection. |
| **YOLO** | 2016 | 24 | ~45M | High mAP | Single-stage grid regression for 45+ FPS real-time detection. |
