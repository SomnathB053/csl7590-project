# Lightweight Semantic Segmentation Model

## Overview
This project implements a **computationally lightweight deep learning model for semantic segmentation** using **PyTorch**. The goal is to achieve a balance between **segmentation accuracy** and **computational efficiency** for real-world deployment.

The model is evaluated on the **PASCAL VOC 2012** dataset, which contains pixel-level annotations for 21 classes (20 object classes + background).

---

## Model Architecture

The implemented model follows an **Encoder–Decoder architecture** with **Depthwise Separable Convolutions** to reduce computational cost.

### Encoder
The encoder progressively reduces spatial resolution while increasing feature depth.

Layers:
- Conv(3 → 32, stride=2)
- DSConv(32 → 64, stride=2)
- DSConv(64 → 128, stride=2)
- DSConv(128 → 256, stride=2)

### Bottleneck
- DSConv(256 → 256)

### Decoder
The decoder upsamples feature maps to restore spatial resolution.

Layers:
- Transposed Conv(256 → 128)
- DSConv(128 → 128)
- Transposed Conv(128 → 64)
- DSConv(64 → 64)
- Transposed Conv(64 → 32)
- DSConv(32 → 32)
- Transposed Conv(32 → 32)

### Output Layer
- 1×1 convolution to produce **21 class logits per pixel**

Final predictions are resized to match the original input resolution.

---

## Depthwise Separable Convolution

To reduce FLOPs and parameter count, the model uses **Depthwise Separable Convolutions**, which split a standard convolution into:

1. **Depthwise convolution** (per-channel filtering)
2. **Pointwise convolution (1×1)** for channel mixing

This significantly reduces computational cost compared to standard convolutions.

---

## Dataset

**PASCAL VOC 2012**

- 21 semantic classes
- Pixel-level segmentation masks
- RGB images

Typical input resolution used for the model: **304 × 304**.

---

project/
│
├── models/
│   ├── __init__.py
│   └── segmentation_model.py
│
├── train.py
├── dataset.py
├── utils.py
└── README.md

## Installation

Clone the repository:

```bash
git clone <repository_url>
cd csl7590-project