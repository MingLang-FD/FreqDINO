# ISBI 2026 | FreqDINO: Frequency-Enhanced DINOv3 Adaptation for Medical Image Segmentation

### [[ArXiv Paper]()]

-------------------------------------------

## 📰 News

- **[2026.03]** We have released the code for FreqDINO!

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

The data structure is as follows.
```
FreqDINO
├── datasets
│   ├── image
│     ├── case_001.png
│     ├── ...
│   ├── mask
│     ├── case_001.png
│     ├── ...
│   ├── data_split.json
```

The json structure is as follows.

    {
      "train": ["case_001.png", "case_004.png"],
      "valid": ["case_002.png"],
      "test":  ["case_003.png"]
    }

For training, run:

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

For evaluation, run:

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

## 📜 Citation
If you find this work helpful for your project, please consider citing the following paper:

```bibtex
@inproceedings{freqdino2026,
  title     = {FreqDINO: Frequency-Enhanced DINOv3 Adaptation for Medical Image Segmentation},
  booktitle = {IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2026}
}
```

## Acknowledgements

* [DINOv3 (timm)](https://github.com/huggingface/pytorch-image-models)
* [pytorch-wavelets](https://github.com/fbcotter/pytorch_wavelets)
