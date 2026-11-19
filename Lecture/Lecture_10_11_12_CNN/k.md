
Ran command: `python3 -c "
import re

def find_math(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    # Look for math expressions, formulas, variable names
    # e.g., patterns like W_out, stride, padding, receptive field, parameters, FLOPS, etc.
    matches = re.findall(r'(\([^\)]*[\+\-\*\/][^\)]*\)|[HhWw]_(?:out|in)|(?:kernel|stride|padding|receptive|parameters|channels)[\w\s\=\+\-\*\/\(\)]+)', content)
    print(f'=== {filename} ===')
    for m in matches[:30]:
        clean = ' '.join(m.split())
        if len(clean) > 5:
            print('-', clean)

find_math('Lecture/Lecture_10_11_12_CNN/L10_CNN_Worksheet.html')
find_math('Lecture/Lecture_10_11_12_CNN/L11-12_CNN_Worksheet.html')
"`Ran command:`python3 -c "
from bs4 import BeautifulSoup
import re

for fn in ['Lecture/Lecture_10_11_12_CNN/L10_CNN_Worksheet.html', 'Lecture/Lecture_10_11_12_CNN/L11-12_CNN_Worksheet.html']:
    with open(fn, 'r', encoding='utf-8', errors='ignore') as f:
        soup = BeautifulSoup(f.read(), 'html.parser')
    print(f'=== {fn} ===')
    for code in soup.find_all(['code', 'pre', 'span', 'div']):
        t = code.get_text().strip()
        if any(sym in t for sym in ['W_out', 'H_out', 'floor', 'stride', 'padding', 'receptive', 'Var(', 'O =', 'params', 'RF']):
            if len(t) < 200 and '\n' not in t:
                print('  >', t)
"`

Here is the complete compilation of **all CNN formulas** and **all landmark architectures** covered across the lectures, structured into comparative tables with detailed variable explanations below each table.

---

# Part 1: All CNN Formulas

### 📊 Master Table of CNN Formulas

|      #      | Concept / Operation                                              | Mathematical Formula                                                                                                                                                         | Purpose / When to Use                                                                                         |
| :----------: | :--------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------ |
| **1** | **Output Spatial Dimension** *(Standard Conv / Pooling)* | 
$$
O = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

                                                                                                                   | Calculates the output height or width of a feature map after convolution or pooling.                          |
| **2** | **Output Spatial Dimension** *(Dilated Conv)*            | 
$$
O = \left\lfloor \frac{W - (K - 1)D - 1 + 2P}{S} \right\rfloor + 1
$$

                                                                                                        | Calculates output size when using dilated (atrous) convolution.                                               |
| **3** | **Effective Kernel Size** *(Dilation)*                   | 
$$
K_{\text{eff}} = K + (K - 1)(D - 1) = (K - 1)D + 1
$$

                                                                                                                        | The effective physical span of a kernel when spaces are inserted between its weights.                         |
| **4** | **Same Padding** *(Preserve Resolution)*                 | 
$$
P_{\text{same}} = \frac{K - 1}{2}
$$

                                                                                                                                         | Determines how many zeros to pad on each side to keep$O = W$ (for odd $K$ and $S = 1$).                 |
| **5** | **Total Parameters** *(Conv Layer)*                      | 
$$
\text{Params} = C_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w + 1)
$$

                                                                                           | Calculates the number of learnable weights and biases in a 2D convolutional layer.                            |
| **6** | **Total Parameters** *(Linear / FC Layer)*               | 
$$
\text{Params} = N_{\text{in}} \times N_{\text{out}} + N_{\text{out}}
$$

                                                                                                      | Counts the total weights and biases in a standard fully connected dense layer.                                |
| **7** | **Receptive Field Growth** *(Recursive)*                 | 
$$
RF_{\ell} = RF_{\ell-1} + (K_{\ell} - 1) \cdot J_{\ell-1}
$$

                                                                                                                 | Computes how many pixels of the original input image influence a neuron at layer$\ell$.                     |
| **8** | **Cumulative Stride / Jump**                               | 
$$
J_{\ell} = J_{\ell-1} \cdot S_{\ell} \quad (\text{with } J_0 = 1)
$$

                                                                                                         | Tracks the cumulative stride distance between adjacent receptive field centers at layer$\ell$.              |
| **9** | **Effective RF of $N$ Stacked $3\times3$ Convs**       | 
$$
RF = 2N + 1
$$

                                                                                                                                                               | Fast rule for effective field of view when stacking$N$ consecutive stride-1 $3\times3$ convs.             |
| **10** | **Parameter Savings of Stacking $3\times3$**             | 
$$
\text{Ratio} = \frac{N \times 3^2}{(2N + 1)^2} = \frac{9N}{(2N + 1)^2}
$$

                                                                                                    | Ratio of parameters in$N$ stacked $3\times3$ layers vs. a single $(2N+1)\times(2N+1)$ conv layer.       |
| **11** | **Transposed Conv Output Size** *(Upsampling)*           | 
$$
O = (W_{\text{in}} - 1) \cdot S - 2P + K + P_{\text{out}}
$$

                                                                                                                 | Determines the restored spatial dimension after learnable upsampling (deconvolution).                         |
| **12** | **Residual Block Forward Pass**                            | 
$$
y = \mathcal{F}(x, \{W_i\}) + x
$$

                                                                                                                                           | The core ResNet skip connection formula, learning a residual mapping$\mathcal{F}(x)$.                       |
| **13** | **Residual Gradient Highway**                              | 
$$
\frac{\partial \mathcal{L}}{\partial x} = \frac{\partial \mathcal{L}}{\partial y} \cdot \frac{\partial \mathcal{F}}{\partial x} + \frac{\partial \mathcal{L}}{\partial y}
$$

 | Proves gradients flow directly back via the additive$+1$ identity shortcut, preventing vanishing gradients. |
| **14** | **Variance Explosion in Plain ResNet**                     | 
$$
\text{Var}(y_K) \approx 2^K \cdot \text{Var}(x_0)
$$

 | Demonstrates that without Batch Normalization, variance doubles at every residual block ($2^K$).                    |                                                                                                               |
| **15** | **Computational Cost (FLOPs / MACs)**                      | 
$$
\text{MACs} = H_{\text{out}} \times W_{\text{out}} \times C_{\text{out}} \times (C_{\text{in}} \times K_h \times K_w)
$$

                                                     | Measures the Multiply-Accumulate Operations (FLOPs$\approx 2 \times \text{MACs}$) for a Conv layer.         |

---

### 🔍 Explanation of Variables Used in the Formula Table

* $W$ / $W_{\text{in}}$: Spatial **width** of the input feature map (measured in pixels).
* $H$ / $H_{\text{in}}$: Spatial **height** of the input feature map (measured in pixels).
* $O$ / $W_{\text{out}}$ / $H_{\text{out}}$: Output spatial width or height produced by the layer.
* $K$ / $K_h$ / $K_w$: **Kernel (filter) size**. $K$ is the dimension for square kernels ($K \times K$); $K_h$ and $K_w$ are kernel height and width.
* $P$: **Padding**. The number of rows and columns of zeros added around the border of the input image.
* $P_{\text{same}}$: Padding value specifically chosen to keep the output resolution equal to the input resolution ($O = W$).
* $S$: **Stride**. The number of pixels the sliding filter shifts horizontally or vertically at each step.
* $D$: **Dilation rate**. The spacing between kernel elements (e.g., $D=1$ is standard convolution; $D=2$ skips 1 pixel between weights).
* $K_{\text{eff}}$: **Effective kernel size** when dilation is applied.
* $C_{\text{in}}$: Number of **input channels** (e.g., $1$ for grayscale, $3$ for RGB, or the number of feature maps from the previous layer).
* $C_{\text{out}}$: Number of **output channels** (which equals the number of distinct filters applied by that layer).
* $N_{\text{in}}$ / $N_{\text{out}}$: Number of input and output nodes (neurons) in a Fully Connected (Dense) layer.
* $RF_{\ell}$: **Receptive Field** at layer $\ell$ (the width/height of the input image region that affects a neuron at layer $\ell$).
* $J_{\ell}$: **Jump (cumulative stride)** at layer $\ell$. Tracks the spatial distance between the centers of two adjacent receptive fields.
* $N$: Number of consecutive convolutional layers stacked together.
* $y$: The final output of a residual block.
* $x$: The input tensor fed into the residual block.
* $\mathcal{F}(x, \{W_i\})$: The **residual function** learned by the weight layers inside the block (e.g., two $3 \times 3$ convs).
* $\mathcal{L}$: The scalar loss function of the neural network.
* $\text{Var}(\cdot)$: Statistical variance of the activations.
* $K$ (in $2^K$): The total number of stacked residual blocks.
* $\text{MACs}$: **Multiply-Accumulate Operations** (one multiplication followed by one addition: $a \times b + c$).
* $\lfloor \cdot \rfloor$: The **floor operation** (rounds down to the nearest integer).

---

# Part 2: All Landmark CNN Architectures

### 📊 Master Table of CNN Architectures

| Architecture                             | Year | Primary Creators / Team                                             |       Depth (Layers)       |                    Learnable Parameters                    |                                                                                                                                                   Input Image Resolution                                                                                                                                                   | Key Defining Innovations                                                                                                                                                                                                                                        |          Top-5 Error / Primary Benchmark          |
| :--------------------------------------- | :--: | :------------------------------------------------------------------ | :------------------------: | :---------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------: |
| **LeNet-5**                        | 1998 | Yann LeCun et al. (AT&T Labs)                                       |        **7**        |                  **~60,000** (60 K)                  |                                                                                                                                            $32 \times 32 \times 1$ (Grayscale)                                                                                                                                            | • First modern CNN pipeline (Conv$\to$ AvgPool $\to$ FC)• Weight sharing & local receptive fields for digits                                                                                                                                              |                0.95% error on MNIST                |
| **AlexNet**                        | 2012 | Alex Krizhevsky, Ilya Sutskever, Geoffrey Hinton (Univ. of Toronto) | **8** (5 Conv, 3 FC) |                **~60 Million** (60 M)                |                                        $224 \times 224 \times 3$ (RGB) | • Popularized **ReLU** (6x faster convergence)• **Dropout (0.5)** in FC layers• Split training across **2 GPUs** (NVIDIA GTX 580)• Overlapping Max Pooling ($3\times3$, stride 2)                                        | **16.4%** Top-5 (ImageNet 2012 Winner)                                                                                                                                                                                                                    |                                                    |
| **VGG-16 / VGG-19**                | 2014 | Karen Simonyan & Andrew Zisserman (Oxford VGG)                      |     **16 / 19**     |     **~138 M** (VGG-16)**~144 M** (VGG-19)     |                                                           $224 \times 224 \times 3$ | • Proved **$3\times3$ filters everywhere** are superior to larger filters ($5\times5, 7\times7$)• Modular block structure: doubles channels after every pool                                                           | **6.8%** Top-5 (ImageNet 2014 2nd Place)                                                                                                                                                                                                                  |                                                    |
| **GoogLeNet (Inception-v1)**       | 2014 | Christian Szegedy et al. (Google)                                   |        **22**        |                 **~7 Million** (7 M)                 | $224 \times 224 \times 3$ | • **Inception Module**: multi-scale parallel convs ($1\times1, 3\times3, 5\times5$)• **$1\times1$ Conv Bottlenecks** slash compute by $>70\%$• **Global Average Pooling (GAP)** eliminates heavy FC layers• Auxiliary classification heads for gradient injection | **6.67%** Top-5 (ImageNet 2014 Winner)                                                                                                                                                                                                                    |                                                    |
| **ResNet (ResNet-50 / 101 / 152)** | 2015 | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun (Microsoft)       |   **50, 101, 152**   | **~25.6 M** (ResNet-50)**~60.2 M** (ResNet-152) |                      $224 \times 224 \times 3$ | • **Residual Skip Connections** ($y = \mathcal{F}(x) + x$)• Solved degradation & shattered gradient problem• **Bottleneck blocks** ($1\times1 \to 3\times3 \to 1\times1$)• Broke the 100+ layer barrier; beat human performance                      | **3.57%** Top-5 (ImageNet 2015 Winner)                                                                                                                                                                                                                    |                                                    |
| **U-Net**                          | 2015 | Olaf Ronneberger, Philipp Fischer, Thomas Brox (Univ. of Freiburg)  |        **23**        |                **~30 Million** (30 M)                |                                                                                                                                              Variable (e.g.$572 \times 572$)                                                                                                                                              | • Symmetrical**U-shape Encoder-Decoder**• **Skip Concatenations**: copies fine-grained spatial maps from encoder directly to decoder• Precision pixel-level biomedical semantic segmentation                                                     |       Winner of ISBI Cell Tracking Challenge       |
| **R-CNN**                          | 2014 | Ross Girshick et al. (UC Berkeley)                                  |     Backbone (AlexNet)     |                            ~60 M                            |                                                                                                                                            Cropped regions ($224 \times 224$)                                                                                                                                            | •**Two-stage Object Detection**• Uses **Selective Search** to propose ~2,000 regions• Warps each region and runs CNN separately 2,000 times• Linear SVM for classification + Bounding box regressor                                             |    53.7% mAP on PASCAL VOC (Slow: 47s / image)    |
| **Fast R-CNN**                     | 2015 | Ross Girshick (Microsoft Research)                                  |     Backbone (VGG-16)     |                           ~138 M                           |                                                                                                                                            Full image ($W \times H \times 3$)                                                                                                                                            | •**Single CNN pass** over the whole image• **RoI Pooling**: extracts fixed-size features directly from the shared feature map for each proposed region• Multi-task loss (jointly trains classification & regression)                             |       ~2.0s / image (25x faster than R-CNN)       |
| **Faster R-CNN**                   | 2015 | Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun                   |   Backbone (VGG/ResNet)   |                           ~138 M+                           |                                                                                                                                            Full image ($W \times H \times 3$)                                                                                                                                            | • Replaced CPU Selective Search with**Region Proposal Network (RPN)**• **Anchor Boxes** (3 scales $\times$ 3 aspect ratios = 9 anchors)• Fully differentiable end-to-end detection on GPU                                                      | **5–7 FPS** (~0.2s / image, near real-time) |
| **YOLO (You Only Look Once)**      | 2016 | Joseph Redmon, Santosh Divvala, Ross Girshick, Ali Farhadi          |  **24 Conv + 2 FC**  |                **~45 Million** (45 M)                |                                                                                                                                                 $448 \times 448 \times 3$                                                                                                                                                 | •**Single-Stage Detection**: frames detection as a unified grid regression problem• Divides image into $S \times S$ grid ($7 \times 7$); predicts bounding boxes and classes in **one forward pass**• Full-image global contextual reasoning | **45 FPS** (Real-Time) / 155 FPS (Fast YOLO) |

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
