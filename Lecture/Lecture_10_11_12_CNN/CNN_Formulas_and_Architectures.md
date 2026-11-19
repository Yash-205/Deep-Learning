# CNN Formulas & Landmark Architectures: Complete Reference Guide

This reference document compiles all the mathematical formulas, equations, variable definitions, and landmark architectures for Convolutional Neural Networks (CNNs) and Computer Vision across Lectures 10, 11, and 12.

---

# Part 1: All CNN Formulas

## Master Table of CNN Formulas

| # | Concept / Operation | Mathematical Formula | Purpose / Explanation |
| :---: | :--- | :---: | :--- |
| **1** | **Output Size (Conv / Pool)** | $O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1$  | Calculates output spatial dimension ($H_{\text{out}}$ or $W_{\text{out}}$). |
| **2** | **Output Size (Dilated Conv)** | $O = \left\lfloor \frac{W - (K - 1)D - 1 + 2P}{S} \right\rfloor + 1$  | Calculates output spatial dimension when dilation ($D$) is applied. |
| **3** | **Effective Kernel Size** | $K_{\text{eff}} = (K - 1)D + 1$ | Effective receptive span of a dilated filter (e.g., $K=3, D=2 \implies K_{\text{eff}}=5$). |
| **4** | **Same Padding (Stride 1)** | $P = \frac{K - 1}{2}$ | Preserves spatial resolution ($O = W$) for odd-sized kernels at stride $S = 1$. |
| **5** | **Convolution Parameters** | $\text{Params} = C_{\text{out}} \times (C_{\text{in}} \cdot K_h \cdot K_w + 1)$ | Trainable weights and biases. $+1$ accounts for one bias per filter. |
| **6** | **Linear / FC Parameters** | $\text{Params} = N_{\text{in}} \cdot N_{\text{out}} + N_{\text{out}}$ | Trainable weights and biases in a dense layer. |
| **7** | **Receptive Field Growth** | $RF_{\ell} = RF_{\ell-1} + (K_{\ell} - 1) \cdot D_{\ell} \cdot J_{\ell-1}$ | Theoretical receptive field at layer $\ell$. $J_{\ell-1}$ is the cumulative jump. |
| **8** | **Cumulative Stride (Jump)** | $J_{\ell} = J_{\ell-1} \cdot S_{\ell}, \quad J_0 = 1$ | Distance in original input pixels between adjacent receptive field centers. |
| **9** | **RF of $N$ Stacked $3 \times 3$ Convs** | $RF = 2N + 1$ | Receptive field for $N$ consecutive stride-1, dilation-1 $3 \times 3$ convs. |
| **10** | **Parameter Ratio (Stacked vs Large)** | $\text{Ratio} = \frac{9N}{(2N + 1)^2}$ | Weight ratio: $N$ stacked $3 \times 3$ convs vs. one $(2N+1)\times(2N+1)$ conv. |
| **11** | **Transposed Conv Output Size** | $O = (W_{\text{in}} - 1)S - 2P + D(K - 1) + P_{\text{out}} + 1$ | Restored spatial dimension after learnable upsampling (deconvolution). |
| **12** | **Residual Block Forward Pass** | $y = \mathcal{F}(x) + x$ | Adds learned residual mapping $\mathcal{F}(x)$ directly to identity shortcut $x$. |
| **13** | **Residual Gradient Highway** | $\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \left(\frac{\partial \mathcal{F}}{\partial x} + I\right)$ | Unobstructed gradient backpropagation via $+I$, mitigating vanishing gradients. |
| **14** | **Unnormalized Residual Variance** | $\text{Var}(y_K) \approx 2^K \cdot \text{Var}(x_0)$ | Variance doubles at each residual block without Batch Normalization ($2^K$). |
| **15** | **Conv Computational Cost (MACs)** | $\text{MACs} = H_{\text{out}} W_{\text{out}} C_{\text{out}} C_{\text{in}} K_h K_w$ | Multiply-Accumulate operations (FLOPs $\approx 2 \times \text{MACs}$). |

## Input Channels vs. Output Channels

| Term | Meaning | Concrete Example |
| :--- | :--- | :--- |
| **Input Channels ($C_{\text{in}}$)** | Number of 2D feature maps entering the convolutional layer. | An RGB image has $C_{\text{in}} = 3$ channels (Red, Green, Blue). |
| **Kernel / Filter Volume** | A learnable 3D weight tensor spanning all input channels. | For $C_{\text{in}} = 3$, one $3 \times 3$ filter has $3 \times 3 \times 3 = 27$ weights. |
| **Output Channels ($C_{\text{out}}$)** | Total number of filters applied; each filter produces one 2D map. | Applying 16 filters produces $C_{\text{out}} = 16$ output feature maps. |
| **Spatial Dimensions** | Height ($H$) and width ($W$) of feature maps (distinct from channels). | Tensor of shape $32 \times 32 \times 3$ has $H=32, W=32$, and $C=3$. |
| **Layer 1 Example** | Input: $32 \times 32 \times 3$; 16 filters of size $3 \times 3$, stride 1, padding 1. | Output: $32 \times 32 \times 16$. Spatial size kept; channels grow $3 \to 16$. |
| **Layer 2 Example** | Next layer receives all 16 channels from Layer 1. | With 32 filters ($3 \times 3$), each filter has shape $3 \times 3 \times 16$; output is $32 \times 32 \times 32$. |
| **Parameter Formula** | Total weights $= C_{\text{out}} \times (C_{\text{in}} \cdot K_h \cdot K_w + 1)$. | For Layer 1: $16 \times (3 \times 3 \times 3 + 1) = 448$ learnable parameters. |

## Notation Reference

| Symbol | Mathematical Meaning |
| :---: | :--- |
| $W, H$ | Input spatial width and height (in pixels) |
| $W_{\text{in}}$ | Input spatial dimension for transposed convolution |
| $O, W_{\text{out}}, H_{\text{out}}$ | Output spatial dimension, width, and height |
| $K, K_h, K_w$ | Kernel size, kernel height, and kernel width |
| $P$ | Padding (number of zero pixels added to each border) |
| $S$ | Stride (filter step size) |
| $D$ | Dilation rate ($D=1$ is standard; $D=2$ skips 1 pixel) |
| $P_{\text{out}}$ | Output padding in transposed convolution |
| $C_{\text{in}}$ | Number of input channels |
| $C_{\text{out}}$ | Number of output channels (number of distinct filters) |
| $N_{\text{in}}, N_{\text{out}}$ | Number of input features and output neurons in a linear / dense layer |
| $RF_{\ell}$ | Theoretical receptive field at layer $\ell$ |
| $J_{\ell}$ | Cumulative jump (stride) between receptive field centers at layer $\ell$ |
| $\mathcal{F}(x)$ | Residual mapping function learned inside a residual block |
| $I$ | Identity matrix |
| $\text{Var}(x)$ | Statistical variance of tensor activations |
| $N$ | Number of stacked convolutional layers |
| $\lfloor \cdot \rfloor$ | Floor operation (rounds down to nearest integer) |

---

# Part 2: All Landmark CNN Architectures

### 📊 Master Table of CNN Architectures

| Architecture | Year | Creators / Team | Depth | Parameters | Input Resolution | Key Defining Innovations | Benchmark Performance |
| :--- | :---: | :--- | :---: | :---: | :---: | :--- | :---: |
| **LeNet-5** | 1998 | Yann LeCun et al. (AT&T Labs) | **7** | **~60 K** | $32 \times 32 \times 1$ | • First modern CNN pipeline (Conv $\to$ AvgPool $\to$ FC)<br>• Weight sharing & local receptive fields for digit recognition | 0.95% error (MNIST) |
| **AlexNet** | 2012 | A. Krizhevsky, I. Sutskever, G. Hinton (Toronto) | **8** | **~60 M** | $224 \times 224 \times 3$ | • **ReLU** activation (6x faster convergence)<br>• **Dropout (0.5)** in FC layers<br>• Parallel training across **2 GPUs** (GTX 580)<br>• Overlapping Max Pooling ($3\times3$, stride 2) | **16.4%** Top-5 (ImageNet 2012 Winner) |
| **VGG-16 / 19** | 2014 | K. Simonyan & A. Zisserman (Oxford VGG) | **16 / 19** | **~138 M** (VGG-16)<br>**~144 M** (VGG-19) | $224 \times 224 \times 3$ | • Standardized **$3\times3$ filters everywhere**<br>• Showed stacked $3\times3$ beats large $5\times5$ and $7\times7$ filters<br>• Modular blocks doubling channels after each pool | **6.8%** Top-5 (ImageNet 2014) |
| **GoogLeNet** | 2014 | Christian Szegedy et al. (Google) | **22** | **~7 M** | $224 \times 224 \times 3$ | • **Inception Module**: parallel multi-scale convs ($1\times1, 3\times3, 5\times5$)<br>• **$1\times1$ Bottlenecks** slash compute by $>70\%$<br>• **Global Average Pooling (GAP)** eliminates heavy FC layers<br>• Auxiliary classification heads for gradient injection | **6.67%** Top-5 (ImageNet 2014 Winner) |
| **ResNet** | 2015 | K. He, X. Zhang, S. Ren, J. Sun (Microsoft) | **50 / 101 / 152** | **~25.6 M** (ResNet-50)<br>**~60.2 M** (ResNet-152) | $224 \times 224 \times 3$ | • **Residual Skip Connections** ($y = \mathcal{F}(x) + x$)<br>• Solved degradation & shattered gradient problem<br>• **Bottleneck blocks** ($1\times1 \to 3\times3 \to 1\times1$)<br>• Broke the 100+ layer barrier; beat human performance | **3.57%** Top-5 (ImageNet 2015 Winner) |
| **U-Net** | 2015 | O. Ronneberger, P. Fischer, T. Brox (Freiburg) | **23** | **~30 M** | Variable (e.g. $572 \times 572$) | • Symmetrical **U-shape Encoder-Decoder**<br>• **Skip Concatenations**: transfers fine-grained encoder maps directly to decoder<br>• Dense pixel-level biomedical semantic segmentation | ISBI Cell Tracking Winner |
| **R-CNN** | 2014 | Ross Girshick et al. (UC Berkeley) | Backbone (AlexNet) | ~60 M | Crops ($224 \times 224$) | • **Two-stage Object Detection**<br>• **Selective Search** proposes ~2,000 regions<br>• Runs CNN forward pass 2,000 times per image<br>• Linear SVM classifier + Bounding box regressor | 53.7% mAP (Slow: ~47s / image) |
| **Fast R-CNN** | 2015 | Ross Girshick (Microsoft Research) | Backbone (VGG-16) | ~138 M | Full image ($W \times H \times 3$) | • **Single CNN pass** over the entire image<br>• **RoI Pooling**: pools fixed-size features directly from shared feature map<br>• Joint multi-task training (classification + regression) | ~2.0s / image (25x faster) |
| **Faster R-CNN**| 2015 | S. Ren, K. He, R. Girshick, J. Sun | Backbone (VGG/ResNet) | ~138 M+ | Full image ($W \times H \times 3$) | • Replaced CPU Selective Search with **Region Proposal Network (RPN)**<br>• **Anchor Boxes** (3 scales $\times$ 3 aspect ratios = 9 anchors)<br>• Fully differentiable GPU end-to-end detection | **5–7 FPS** (~0.2s / image) |
| **YOLO** | 2016 | J. Redmon, S. Divvala, R. Girshick, A. Farhadi | **24 Conv + 2 FC** | **~45 M** | $448 \times 448 \times 3$ | • **Single-Stage Detection**: frames detection as unified grid regression<br>• Divides image into $S \times S$ grid ($7 \times 7$); predicts boxes & classes in **one forward pass**<br>• Global context reasoning with high throughput | **45 FPS** (Real-Time) / 155 FPS (Fast YOLO) |

---

### 🔍 Explanation of Terms & Architectural Modules

* **Top-5 Error**: The percentage of test images where the true category is not among the model's top 5 highest-probability predictions (standard metric on ImageNet with 1,000 classes).
* **mAP (Mean Average Precision)**: The primary evaluation metric for object detection, measuring precision across multiple Recall thresholds and Intersection-over-Union (IoU) levels.
* **FPS (Frames Per Second)**: Inference throughput. $\ge 30\text{ FPS}$ is considered true real-time video speed.
* **$1\times1$ Convolution (Bottleneck / Cross-Channel Pooling)**:
  * Computes linear combinations across input channels without changing spatial height or width.
  * Used to compress channel depth (e.g. from 256 down to 64) before expensive $3\times3$ or $5\times5$ filters, slashing computation.
* **Global Average Pooling (GAP)**:
  * Takes an entire $H \times W$ feature map and computes its average into a single scalar value.
  * Replaces parameter-heavy fully connected layers (e.g., reducing 100M+ weights in VGG down to 0 weights in GoogLeNet/ResNet), preventing overfitting.
* **Skip / Residual Connection**:
  * An alternate computational pathway that bypasses one or more layers and adds the input directly to the layer output ($y = \mathcal{F}(x) + x$).
  * Creates an unobstructed gradient highway that eliminates vanishing gradients in networks with 100+ layers.
* **Skip Concatenation (in U-Net)**:
  * Rather than adding ($x + \mathcal{F}(x)$), U-Net **stacks** the encoder's feature map alongside the decoder's feature map along the channel axis.
  * Restores high-resolution spatial details (edges, boundaries) lost during downsampling.
* **Inception Module**:
  * A modular block that runs $1\times1$, $3\times3$, $5\times5$ convolutions and $3\times3$ max-pooling in parallel on the same input, capturing features at multiple spatial scales simultaneously.
* **RoI Pooling (Region of Interest Pooling)**:
  * Projects arbitrary-sized bounding box proposals onto a shared convolutional feature map and crops/downsamples them into a standard fixed size (e.g., $7 \times 7$) so they can feed into dense layers.
* **RPN (Region Proposal Network)**:
  * A lightweight convolutional network that slides over the shared feature map to score regions for "objectness" and propose initial bounding box coordinates.
* **Anchor Boxes**:
  * Pre-defined reference bounding boxes of diverse scales (e.g., $128^2, 256^2, 512^2$) and aspect ratios ($1:1, 1:2, 2:1$) placed at every position across a feature map to detect objects of various shapes.
