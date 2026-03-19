<div align="center">
# ISBI 2026 FreqDINO: Frequency-Guided Adaptation for Generalized Boundary-Aware Ultrasound Image Segmentation

[![arXiv](https://img.shields.io/badge/arXiv-2512.11335-b31b1b.svg)](https://arxiv.org/abs/2512.11335)
[![GitHub](https://img.shields.io/github/stars/MingLang-FD/FreqDINO?style=social)](https://github.com/MingLang-FD/FreqDINO)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

</div>

---

> **FreqDINO** adapts a frozen DINOv3 backbone for ultrasound segmentation via explicit frequency-domain decomposition. Three synergistic modules — **MFEA** (multi-scale wavelet decomposition), **FGBR** (boundary prototype refinement), and **MBGD** (joint boundary-semantic decoding) — enable precise boundary perception without full fine-tuning. FreqDINO achieves **86.52% Dice** on BUSI and strong zero-shot generalization on TN3K.

---

## 👥 Authors

<div align="center">

[Yixuan Zhang](https://scholar.google.com/citations?hl=en&user=ezcvFNEAAAAJ)<sup>1\*</sup> [Qing Xu](https://scholar.google.com/citations?user=IzA-Ij8AAAAJ&hl=en&authuser=1)<sup>1,2\*</sup> [Yue Li](https://scholar.google.com/citations?hl=en&user=TvAELsAAAAAJ)<sup>1,2\*</sup> [Xiangjian He](https://scholar.google.com/citations?user=BiBXGfIAAAAJ&hl=en&oi=ao)<sup>1†</sup> [Qian Zhang](https://scholar.google.com/citations?user=nJlSf_YAAAAJ&hl=en&oi=ao)<sup>1†</sup> [Mainul Haque](https://scholar.google.com/citations?hl=en&user=Me2aCpgAAAAJ)<sup>1</sup> [Rong Qu](https://scholar.google.com/citations?user=ErszCRMAAAAJ&hl=en&oi=ao)<sup>2</sup> [Wenting Duan](https://scholar.google.com/citations?user=H9C0tX0AAAAJ&hl=en&authuser=1)<sup>3</sup> [Zhen Chen](https://franciszchen.github.io/)<sup>4</sup>

<sup>1</sup>University of Nottingham Ningbo China &emsp; <sup>2</sup>University of Nottingham &emsp; <sup>3</sup>University of Lincoln &emsp; <sup>4</sup>The Hong Kong Polytechnic University

<sup>\*</sup> Equal Contribution. &emsp; <sup>†</sup> Corresponding Author.

</div>>

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
