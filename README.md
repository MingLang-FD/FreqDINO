<div align="center">
# ISBI 2026 FreqDINO: Frequency-Guided Adaptation for Generalized Boundary-Aware Ultrasound Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2512.11335-b31b1b.svg)](https://arxiv.org/abs/2512.11335)
[![GitHub](https://img.shields.io/github/stars/MingLang-FD/FreqDINO?style=social)](https://github.com/MingLang-FD/FreqDINO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **FreqDINO** is a frequency-guided segmentation framework that adapts a frozen DINOv3 backbone for ultrasound image segmentation. It addresses DINOv3's limited sensitivity to ultrasound-specific boundary degradation by introducing three complementary modules: **MFEA** for multi-scale frequency extraction and alignment via Haar wavelet transform, **FGBR** for boundary prototype distillation and feature refinement, and **MBGD** for joint boundary-semantic prediction. FreqDINO achieves **86.52% Dice** on BUSI and strong zero-shot generalization on TN3K.

---

## 👥 Authors

<div align="center">

Yixuan Zhang\*, Qing Xu\*, Yue Li\*, Xiangjian He†, Qian Zhang†, Mainul Haque, Rong Qu, Wenting Duan, Zhen Chen

\* Equal contribution &nbsp;&nbsp; † Corresponding author

**University of Nottingham Ningbo China · University of Nottingham · University of Lincoln · Yale University**

</div>

---

## 📰 News

- **[2026.03]** 🎉 Code for FreqDINO is now publicly available!
- **[2025.12]** 🎉 FreqDINO accepted at **ISBI 2026**!

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
```bibtex
@article{zhang2025freqdino,
  title={FreqDINO: Frequency-Guided Adaptation for Generalized Boundary-Aware Ultrasound Image Segmentation},
  author={Zhang, Yixuan and Xu, Qing and Li, Yue and He, Xiangjian and Zhang, Qian and Haque, Mainul and Qu, Rong and Duan, Wenting and Chen, Zhen},
  journal={arXiv preprint arXiv:2512.11335},
  year={2025}
}
```

---

## 🙏 Acknowledgements

- [DINOv3 (timm)](https://github.com/huggingface/pytorch-image-models)
- [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets)
