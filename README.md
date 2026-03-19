# ISBI2026 FreqDINO: Frequency-Guided Adaptation for Generalized Boundary-Aware Ultrasound Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2512.11335-b31b1b.svg)](https://arxiv.org/abs/2512.11335)
[![GitHub](https://img.shields.io/github/stars/MingLang-FD/FreqDINO?style=social)](https://github.com/MingLang-FD/FreqDINO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **FreqDINO** bridges the gap between natural-image vision transformers and medical image segmentation by injecting wavelet-based frequency cues into a frozen DINOv3 backbone — no full fine-tuning required.

---

## 👥 Authors

<div align="center">



</div>

---

## 📰 News

- **[2026.03]** 🎉 Code for FreqDINO is now publicly available!

---

## 🔍 Overview

FreqDINO enhances DINOv3-based medical image segmentation through three complementary frequency-domain modules:

| Module | Description |
|--------|-------------|
| **MFEA** | Multi-Scale Frequency Extraction and Alignment |
| **FBAA** | Frequency-Guided Boundary Refinement |
| **FGBP** | Multi-Task Boundary-Guided Decoder |

---

## 🛠 Setup

```bash
git clone https://github.com/MingLang-FD/FreqDINO.git
cd FreqDINO
```

**Key requirements**: CUDA 12.2+, PyTorch 2.0+, Python 3.10+

```bash
pip install timm pytorch-wavelets albumentations monai pytorch-lightning \
            scikit-image opencv-python tqdm seaborn pandas matplotlib
```

### 📂 Data Structure

```
FreqDINO
├── datasets
│   ├── image
│   │   ├── case_001.png
│   │   └── ...
│   ├── mask
│   │   ├── case_001.png
│   │   └── ...
│   └── data_split.json
```

The `data_split.json` should follow this format:

```json
{
  "train": ["case_001.png", "case_004.png"],
  "valid": ["case_002.png"],
  "test":  ["case_003.png"]
}
```

---

## 🚀 Usage

**Training:**

```bash
python train.py \
    --dataset BUSI \
    --batch 16 \
    --lr 0.0001 \
    --epoch 300 \
    --use_msfe \
    --use_fbaa \
    --use_fgbp \
    --use_cross_attn \
    --decoder_type boundary_guided \
    --wavelet haar
```

**Evaluation:**

```bash
python eval.py \
    --dataset BUSI \
    --model outputs/wts/<checkpoint>.pth \
    --use_msfe \
    --use_fbaa \
    --use_fgbp \
    --use_cross_attn \
    --decoder_type boundary_guided
```

---

## 📜 Citation

If you find this work helpful, please consider citing:

---

## 🙏 Acknowledgements

- [DINOv3 (timm)](https://github.com/huggingface/pytorch-image-models)
- [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets)
