# Lecture 12: Advanced CNN Architectures & Computer Vision Applications

## 1. Beyond Standard Image Classification

While networks like VGG and ResNet mastered whole-image classification ("Is this a dog?"), real-world computer vision requires answering far more demanding questions:
1. **Semantic Segmentation**: What object is at *every single pixel*? (Autonomous driving, medical tumors).
2. **Multi-Scale Feature Extraction**: How do we recognize objects whether they are massive or microscopic?
3. **Object Detection**: *Where* are all the objects, and *what* are they? (Bounding boxes + labels in real time).

---

## 2. U-Net: Biomedical Image Segmentation (2015)

Introduced by Olaf Ronneberger, Philipp Fischer, and Thomas Brox. Designed for segmenting biological cells and organs in microscopy and medical scans.

### The Problem with Traditional CNNs
Standard classification CNNs downsample images with pooling layers. By the final layer, you know **what** object is present, but you have completely lost the fine-grained spatial coordinates of **where** its boundaries are.

### The U-Net Architecture (The "U" Shape)

```
Input Image (572x572)                                   Output Segmentation Mask (388x388)
     │                                                              ▲
 [2x Conv 3x3] ───────────── Skip Connection ────────────► [2x Conv 3x3]
     │                                                              ▲
 [MaxPool 2x2]                                                 [Up-Conv 2x2]
     │                                                              ▲
 [2x Conv 3x3] ───────────── Skip Connection ────────────► [2x Conv 3x3]
     │                                                              ▲
 [MaxPool 2x2]                                                 [Up-Conv 2x2]
     │                                                              ▲
 [2x Conv 3x3] ───────────── Skip Connection ────────────► [2x Conv 3x3]
     │                                                              ▲
 [MaxPool 2x2]                                                 [Up-Conv 2x2]
     │                                                              ▲
     └────────────────► [Bottleneck: 2x Conv 3x3] ──────────────────┘
```

### The Three Pillars of U-Net:
1. **Contracting Path (Encoder)**:
   * Repeated stacks of: `[Conv 3x3] -> [ReLU] -> [Conv 3x3] -> [ReLU] -> [MaxPool 2x2]`.
   * Progressively cuts spatial resolution while doubling feature channels. Captures rich, high-level contextual semantics ("What").
2. **Expanding Path (Decoder)**:
   * Uses **Transposed Convolutions (Up-convolutions)** to progressively restore spatial dimensions back to full resolution ("Where").
3. **Skip Connections (Concatenation)**:
   * **The Secret Weapon**: Direct horizontal shortcuts copy high-resolution feature maps from the encoder directly across to the decoder, **concatenating** them along the channel axis.
   * Restores crisp, pixel-perfect boundary details (edges, borders) that were destroyed by max-pooling in the encoder!

---

## 3. GoogLeNet / Inception Network (2014)

Created by Christian Szegedy et al. at Google. Won ImageNet 2014 with a top-5 error rate of **6.67%**.

### The Core Problem: The Scale Dilemma
In photos, visual patterns appear at wildly different sizes:
* A dog sitting close to the camera needs a large $5 \times 5$ or $7 \times 7$ filter.
* A dog far away in the background needs a tiny $3 \times 3$ filter.
* Choosing a single filter size forces an undesirable compromise.

### The Inception Solution: Multi-Scale Processing in Parallel
Instead of picking one filter size, the **Inception Module** applies multiple filter sizes **in parallel on the exact same input** and stacks their outputs together!

```
                       Input Feature Map
                       ┌───────┬───────┬───────┐
                       │       │       │       │
                       ▼       ▼       ▼       ▼
                     [1x1]   [1x1]   [1x1]   [MaxPool]
                     Conv    Conv    Conv     3x3, s=1
                       │       │       │       │
                       │       ▼       ▼       ▼
                       │     [3x3]   [5x5]   [1x1]
                       │     Conv    Conv    Conv
                       │       │       │       │
                       └───────┼───────┴───────┘
                               ▼
                    [Filter Concatenation]
```

### The $1 \times 1$ Convolution as a Dimensionality Bottleneck
Running $5 \times 5$ convolutions directly on deep feature maps (e.g., 192 channels) requires astronomical amounts of computation.

* **How $1 \times 1$ Conv Works**:
  * A $1 \times 1$ convolution computes a weighted sum across all channels for each individual pixel position.
  * It acts as **cross-channel pooling**: if you pass a $28 \times 28 \times 192$ feature map through thirty-two $1 \times 1$ filters, the output is $28 \times 28 \times 32$.
  * The spatial dimensions are untouched, but the channel depth is compressed by **$83\%$**!
* By inserting a $1 \times 1$ conv before the expensive $3 \times 3$ and $5 \times 5$ convs, total computation was slashed by **over $70\%$**, allowing GoogLeNet to be 22 layers deep while using only **7 Million parameters** (compared to VGG's 138 Million!).

### Two Additional GoogLeNet Innovations:
1. **Global Average Pooling (GAP)**:
   * Replaced the massive 100M+ parameter Fully Connected layers at the end with a single Global Average Pooling layer.
   * Averages each feature map over its entire spatial dimensions into a single value, eliminating overfitting and drastically reducing parameters.
2. **Auxiliary Classifiers**:
   * Attached two temporary softmax output branches to intermediate layers during training.
   * Injected direct gradient signals into earlier layers to combat vanishing gradients (discarded during inference).

---

## 4. Why Inception Fails at Real-Time Object Detection

Although Inception was a breakthrough for image classification, it struggled with object detection:
1. **Sliding Window Bottleneck**: Testing thousands of image crops across multiple scales wastes massive amounts of GPU compute.
2. **Context Blindness**: Passing isolated cropped patches through an Inception network loses the global context of where the object sits in the overall scene.
3. **Complex Branching Latency**: Inception's branching pathways introduce memory-access and scheduling overhead that hurts real-time frame rates (FPS).

---

## 5. The Evolution of Object Detection

Object detection requires two simultaneous tasks:
1. **Classification**: "What is this object?"
2. **Localization**: "Where is it?" (Predicting bounding box coordinates $[x, y, w, h]$).

```
   R-CNN (2014)           Fast R-CNN (2015)          Faster R-CNN (2015)            YOLO (2016)
~47 sec / image          ~2.0 sec / image             ~0.2 sec (5-7 FPS)          45+ FPS (Real-time)
Selective Search +       Selective Search +          RPN inside CNN +            Single-stage
Crop 2,000 times         Single CNN + RoI Pool       RoI Pool + Anchors          Grid regression
```

---

### A. R-CNN (Region-based CNN, 2014 - Ross Girshick et al.)
1. **Selective Search**: An unsupervised algorithm groups pixels into ~2,000 region proposals (candidate boxes).
2. **Warp & Crop**: Each candidate box is cropped from the image and distorted to a fixed $224 \times 224$ size.
3. **Forward Pass**: Each of the 2,000 crops is passed through a CNN independently.
4. **Classification & Bounding Box**: An SVM classifies the category, and a linear regressor refines the coordinates.
* **The Fatal Flaw**: Running 2,000 CNN forward passes takes **47 seconds per image**!

---

### B. Fast R-CNN (2015 - Ross Girshick)
* **The Breakthrough**: Pass the **entire original image through the CNN once** to produce a global feature map!
* **RoI Pooling (Region of Interest Pooling)**:
  * Projects the 2,000 region proposals directly onto the shared global feature map.
  * Divides each proposed region into a fixed grid (e.g., $7 \times 7$) and performs max-pooling.
  * Outputs a standardized $7 \times 7 \times C$ vector for every proposal without re-running the CNN!
* **Speedup**: Slashed time from 47 seconds to **~2 seconds per image** (25x faster).
* **The Remaining Bottleneck**: Selective Search proposal generation still ran slowly on the CPU (~2 seconds).

---

### C. Faster R-CNN (2015 - Shaoqing Ren, Kaiming He, Ross Girshick, Jian Sun)
* **The Breakthrough**: Eliminate Selective Search entirely and let a neural network generate the proposals!
* **Region Proposal Network (RPN)**:
  * A small, fast convolutional network that slides directly over the shared convolutional feature map.
  * Predicts whether a region contains an object ("objectness score") and estimates initial coordinates.
* **Anchor Boxes**:
  * At every sliding position, the RPN evaluates $k$ predefined reference boxes of varying scales ($128^2, 256^2, 512^2$) and aspect ratios ($1:1, 1:2, 2:1$). Typically $k = 9$ anchor boxes per position.
* **Speed**: Runs at **5–7 FPS** (~200ms per image), achieving true near-real-time end-to-end detection on GPU!

---

### D. Single-Stage Detectors: YOLO (You Only Look Once, 2016 - Joseph Redmon et al.)
Two-stage detectors (Faster R-CNN) first propose regions and then classify them. YOLO treats detection as a **single, unified regression problem**:

```
Input Image ──► Single CNN Pass ──► S x S Grid Output [x, y, w, h, Confidence, Class_Probs]
```

1. **How it works**:
   * Divides the input image into an $S \times S$ grid (e.g., $7 \times 7$).
   * If an object's center falls into a grid cell, that cell is responsible for detecting it.
   * Each cell simultaneously predicts $B$ bounding boxes, confidence scores, and conditional class probabilities in **one single forward pass**.
2. **Speed & Real-time Performance**:
   * Standard YOLO processes images at **45 FPS** (Fast YOLO reaches **155 FPS**).
   * Because it views the entire image at once during inference, it encodes global contextual information, resulting in significantly fewer false positive background detections.

---

## 6. SOTA Frontiers in Computer Vision

* **DETR (DEtection TRansformer, Carion et al., 2020)**:
  * Replaces anchor boxes and non-maximum suppression (NMS) with a Transformer encoder-decoder architecture using set-based global bipartite matching loss.
* **Segment Anything Model (SAM, Meta AI, 2023)**:
  * A foundational promptable segmentation model capable of zero-shot segmentation across diverse visual domains with interactive points, bounding boxes, or text prompts.

---

## 7. Master Comparison Table of Landmark CNN Architectures

| Model | Year | Layers | Parameters | Top-5 Error | Key Defining Innovations |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **LeNet-5** | 1998 | 7 | ~60 K | N/A (MNIST) | First modern CNN; Conv + AvgPool + FC. |
| **AlexNet** | 2012 | 8 | ~60 M | 16.4% | ReLU, Multi-GPU, Dropout, Data Augmentation. |
| **VGG-16** | 2014 | 16 | ~138 M | 6.8% | Replaced large filters with stacks of small $3 \times 3$ filters. |
| **GoogLeNet** | 2014 | 22 | ~7 M | 6.7% | Inception multi-scale modules, $1 \times 1$ bottlenecks, Global Avg Pooling. |
| **ResNet-152** | 2015 | 152 | ~60 M | 3.57% | Residual skip connections; shattered gradient elimination; beat human performance. |
| **U-Net** | 2015 | ~23 | ~30 M | N/A (Biomed) | Symmetrical Encoder-Decoder with skip concatenations for pixel-level segmentation. |
| **Faster R-CNN**| 2015 | Backbone+ | ~40 M+ | mAP Metric | Region Proposal Network (RPN) + Anchor Boxes for near-real-time detection. |
| **YOLO** | 2016 | 24 | ~45 M | mAP Metric | Single-stage grid regression for 45+ FPS real-time object detection. |
