# FreqDINO

**FreqDINO** is a medical image segmentation framework that integrates a frozen **DINOv3** (ViT-Large/16) backbone with a set of plug-in frequency-domain modules to improve boundary-level accuracy. The core idea is to enrich the pure spatial features produced by the vision transformer with multi-scale wavelet frequency information, then decode the result into a binary segmentation mask (and optionally an auxiliary boundary map).

---

## Architecture Overview

```
Input Image (512 × 512)
        │
        ▼
┌───────────────────────────┐
│  DINOv3 Encoder           │  ViT-Large/16, backbone frozen
│  (with Adapter layers)    │  Adapter = lightweight prompt-learning MLP per block
└───────────────────────────┘
        │  spatial tokens → 2-D feature map  [B, 1024, 32, 32]
        ▼
┌───────────────────────────┐   (optional, --use_msfe)
│  MSFE                     │  Multi-Scale Frequency Extraction
│  DWT (Haar / other)       │  Decomposes features at 32×32 and 16×16 scales
│  → high_fine, high_coarse,│  High-freq: H / V / D wavelet sub-bands
│    low_global             │  Low-freq:  approximation coefficients
└───────────────────────────┘
        │
        ├──► FBAA  (optional, --use_fbaa)
        │    Frequency-Boundary Alignment Attention
        │    Generates boundary + structure attention maps and
        │    adds a weighted enhancement to the spatial feature map
        │
        └──► FGBP + CrossAttn  (optional, --use_fgbp --use_cross_attn)
             Frequency-Guided Boundary Prototype Generator
             Extracts boundary prototypes from high-freq features,
             then refines the spatial map via multi-head cross-attention
        │
        ▼
┌───────────────────────────┐
│  Decoder                  │
│  SimpleDecoder            │  U-Net-style 4× UpBlock chain → 1-ch mask logit
│  BoundaryGuidedDecoder    │  Same chain + dual head: mask logit + boundary logit
└───────────────────────────┘
        │
        ▼
  Binary segmentation mask  (+ boundary map when using boundary_guided decoder)
```

### Modules at a Glance

| Module | Flag | Description |
|---|---|---|
| `Adapter` | always on | Prompt-learning MLP injected into every ViT block |
| `MSFE` | `--use_msfe` | Wavelet-based multi-scale frequency extraction |
| `FBAA` | `--use_msfe --use_fbaa` | Frequency-boundary alignment attention |
| `FGBP + CrossAttn` | `--use_msfe --use_fgbp --use_cross_attn` | Boundary prototype + cross-modal attention (**must be enabled together**) |
| `BoundaryGuidedDecoder` | `--decoder_type boundary_guided` | Dual-head decoder (mask + boundary) |

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/MingLang-FD/FreqDINO.git
cd FreqDINO

# 2. Create a conda / virtual environment (Python 3.9+ recommended; 3.10 shown here)
conda create -n freqdino python=3.10 -y
conda activate freqdino

# 3. Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install remaining dependencies
pip install timm pytorch-wavelets albumentations monai pytorch-lightning \
            scikit-image opencv-python tqdm seaborn pandas matplotlib
```

---

## Dataset Preparation

FreqDINO expects each dataset to be organized as follows:

```
/home/datasets/<DATASET_NAME>/
├── image/
│   ├── case_001.png
│   ├── case_002.png
│   └── ...
├── mask/
│   ├── case_001.png
│   ├── case_002.png
│   └── ...
└── data_split.json
```

`data_split.json` must contain three keys: `"train"`, `"valid"`, and `"test"`, each holding a list of file names (with extension):

```json
{
  "train": ["case_001.png", "case_002.png"],
  "valid": ["case_003.png"],
  "test":  ["case_004.png"]
}
```

Masks are binary PNG files (pixel value 0 = background, > 0 = foreground). Images must be RGB PNGs.

Tested datasets include **BUSI** (Breast Ultrasound Images).

---

## Training

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

### Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--dataset` | `BUSI` | Dataset name (must match directory under `/home/datasets/`) |
| `--batch` | `16` | Batch size |
| `--lr` | `0.0001` | Initial learning rate (Adam) |
| `--epoch` | `300` | Number of training epochs |
| `--use_msfe` | off | Enable Multi-Scale Frequency Extraction |
| `--wavelet` | `haar` | Wavelet type passed to `pytorch_wavelets.DWTForward` |
| `--use_fbaa` | off | Enable Frequency-Boundary Alignment Attention (requires `--use_msfe`) |
| `--use_fgbp` | off | Enable Frequency-Guided Boundary Prototype (requires `--use_msfe`; must be paired with `--use_cross_attn`) |
| `--use_cross_attn` | off | Enable Frequency-Spatial Cross-Attention (requires `--use_msfe`; must be paired with `--use_fgbp`) |
| `--decoder_type` | `simple` | `simple` or `boundary_guided` |

Checkpoints are saved to `outputs/wts/` whenever the validation loss improves. Training curves (loss and IoU) are written to `outputs/train_data/` and `outputs/valid_data/`, and the corresponding plots to `outputs/train_image/` and `outputs/valid_image/`.

---

## Evaluation

```bash
python eval.py \
    --dataset BUSI \
    --model outputs/wts/Dinov3_baseline_adapter_msfe_fbaa_fgbp_crossattn_boundary_guided_BUSI_78.pth \
    --use_msfe \
    --use_fbaa \
    --use_fgbp \
    --use_cross_attn \
    --decoder_type boundary_guided \
    --size 512
```

The script reports:

- **IoU** (Intersection over Union)
- **Dice / DSC** (Dice Similarity Coefficient)
- **HD** (Hausdorff Distance, via MONAI)
- **Accuracy**, **Precision**, **Recall**, **F1**
- **FPS** (inference speed)

Per-image predictions are saved to `visual/<DATASET>/<model_path>/` and a summary CSV (`results.csv`) is written to the same directory.

---

## Loss Function

`BinaryMaskLoss` combines **Dice loss** and **Focal loss**:

```
L = 0.8 × DiceLoss + 0.2 × FocalLoss
```

When `--decoder_type boundary_guided` is used, an auxiliary boundary loss is added with weight `λ = 0.3`:

```
L_total = L_mask + 0.3 × L_boundary
```

The pseudo boundary ground truth is generated online by morphological dilation minus erosion of the mask.

---

## Project Structure

```
FreqDINO/
├── model.py        # All model components (Adapter, MSFE, FBAA, FGBP, CrossAttn, decoders, BaselineModel)
├── loss.py         # BinaryMaskLoss (Dice + Focal) and BinaryIoU metric
├── dataloader.py   # BinaryLoader Dataset (image + mask loading with albumentations)
├── train.py        # Training loop, argument parsing, checkpoint saving
├── eval.py         # Evaluation loop, metric computation, result saving
└── README.md
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` / `torchvision` | Deep learning framework |
| `timm` | DINOv3 pretrained ViT-Large backbone |
| `pytorch_wavelets` | Discrete Wavelet Transform for MSFE |
| `albumentations` | Image augmentation / preprocessing pipeline |
| `monai` | Hausdorff Distance metric |
| `pytorch_lightning` | Classification metrics (Accuracy, Precision, Recall, F1) used in eval |
| `scikit-image` | Image I/O in dataloader |
| `opencv-python` | Morphological operations for boundary GT generation |
| `tqdm` | Progress bars |
| `seaborn` / `matplotlib` | Training curve plots |

---

## License

This project is released under the [MIT License](LICENSE).
