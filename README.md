# UniField: A Unified Field-Aware MRI Enhancement Framework

> **MICCAI 2026** | Official PyTorch Implementation

UniField is a unified deep learning framework for MRI enhancement that is *field-aware* — it explicitly conditions on the scanner's static magnetic field strength (B₀) to adapt its enhancement strategy across different field strengths (0.35 T, 1.5 T, 3 T, 7 T) in a single model.

---

## Highlights

- **Field-Aware Conditioning**: Learnable B₀ embeddings modulate every decoder block via Feature-wise Linear Modulation (FiLM), letting one model serve all clinical field strengths.
- **Unified Architecture**: A shared encoder–decoder backbone handles both MRI super-resolution and denoising under a joint training objective.
- **Plug-and-play**: Drop-in dataset class for NIfTI/DICOM volumes with on-the-fly patch sampling and field-strength labelling.

---

## Architecture Overview

```
Input (low-quality MRI)
        │
  ┌─────▼──────┐
  │  Encoder   │  ← shared CNN feature extractor (residual blocks)
  └─────┬──────┘
        │  skip connections
  ┌─────▼───────────────────────────┐
  │  Field-Aware Decoder            │
  │  ┌──────────────────────────┐   │
  │  │ FiLM(features, B₀ embed) │×N │
  │  └──────────────────────────┘   │
  └─────────────────────────────────┘
        │
  ┌─────▼──────┐
  │    Head    │  ← task-specific output projection
  └─────┬──────┘
        │
  Enhanced MRI
```

---

## Installation

```bash
git clone https://github.com/linyiyang98/UniField.git
cd UniField
pip install -r requirements.txt
```

---

## Quick Start

### Training

```bash
python train.py --config configs/experiments/unifield_sr.yaml
```

### Inference / Testing

```bash
python test.py --config configs/experiments/unifield_sr.yaml --checkpoint path/to/best.pth
```

---

## Configuration

All hyper-parameters live in `configs/`. Copy and edit a YAML file to run your own experiments:

```yaml
# configs/experiments/unifield_sr.yaml
model:
  name: UniField
  in_channels: 1
  base_channels: 64
  num_res_blocks: 4
  num_decoder_stages: 4
  field_embed_dim: 32

data:
  train_root: /path/to/train
  val_root: /path/to/val
  patch_size: [64, 64, 32]
  scale_factor: 4
  task: sr            # 'sr' | 'denoise' | 'sr+denoise'

training:
  epochs: 200
  batch_size: 4
  lr: 1e-4
  scheduler: cosine
```

---

## Supported Field Strengths

| Label | B₀ (Tesla) |
|-------|-----------|
| 0     | 0.35 T    |
| 1     | 1.5 T     |
| 2     | 3.0 T     |
| 3     | 7.0 T     |

---

## Repository Structure

```
UniField/
├── configs/                   # YAML experiment configs
├── data/                      # Dataset and transform utilities
│   ├── dataset.py
│   └── transforms.py
├── losses/                    # Loss functions
│   └── losses.py
├── models/                    # Model definitions
│   ├── unifield.py            # Top-level UniField model
│   ├── encoder.py             # Residual encoder
│   ├── decoder.py             # FiLM-conditioned decoder
│   └── field_embed.py         # B₀ field embedding module
├── utils/                     # Metrics and visualisation helpers
│   ├── metrics.py
│   └── visualization.py
├── train.py
├── test.py
└── requirements.txt
```

---

## Citation

If you find UniField useful, please consider citing:

```bibtex
@inproceedings{unifield2026,
  title     = {UniField: A Unified Field-Aware MRI Enhancement Framework},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```

---

## License

This project is released under the MIT License.
