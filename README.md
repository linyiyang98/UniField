# UniField

Implementation of **UniField: A Unified Field-Aware MRI Enhancement Framework** ([paper](https://arxiv.org/abs/2603.09223)): FlashVSR-based MRI enhancement with LoRA and field-aware spatial-frequency loss.

Includes training, resume, evaluation, LF-only inference and two-GPU DDP. Defaults use **GPUs 4 and 5**. Third-party pretrained weights are public; **our trained LoRA and checkpoints remain private**.

## Dataset

Dataset link: https://pan.baidu.com/s/1xn_YHmi7u3Vv4aBli3XEkg 

Code: 37wu

## Install

Python 3.11; tested with PyTorch 2.10.0 + CUDA 12.8.

```bash
git clone https://github.com/linyiyang98/UniField.git
cd UniField
```

```bash
pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-tested.txt
pip install -e . --no-deps
```

## Data and weights

Use registered LF/HF NIfTI pairs on the same grid. CSV columns:

```csv
subject,center,modality,source_field,target_field,lq,hq
case001,ULF-EnC,T1,64mT,3T,case001/T1/64mT.nii.gz,case001/T1/3T.nii.gz
```

Supported modalities: T1, T2, FLAIR. Tasks: 64mT→3T and 3T→7T. Image paths are relative to `data.root`. All modalities of each subject must remain in one split.

Third-party assets are in `pretrained/FlashVSR/`; [sources, hashes and licenses](pretrained/README.md) are retained. If missing:

```bash
python -m unifield.download_weights --output pretrained/FlashVSR
```

Copy `configs/paper.yaml` to `configs/local.yaml` and set dataset paths. Prepare real frozen UMT5 text embeddings:

```bash
GPUS=4 bash scripts/prepare_text.sh --config configs/local.yaml
python -m unifield.preflight --config configs/local.yaml
```

## Train and test

```bash
bash scripts/train.sh --config configs/local.yaml --output runs/train
bash scripts/train.sh --config configs/local.yaml --output runs/train \
  --resume runs/train/latest.pt
bash scripts/test.sh --config configs/local.yaml --checkpoint runs/train/latest.pt \
  --output runs/test --lpips --native-grid
```

`--max-steps 2` provides a short training check. `GPUS=4 NPROC=1` selects a single GPU; the default is two GPUs with global batch size 2. The paper specifies single-GPU batch size 1.

LF-only inference needs no HF image:

```bash
GPUS=4 bash scripts/infer.sh --config configs/local.yaml --checkpoint runs/train/latest.pt \
  --input data/example/64mT.nii.gz --output runs/prediction.nii.gz \
  --modality T1 --source-field 64mT --target-field 3T --steps 2
```

Inference writes normalized `[0,1]` NIfTI with the transformed affine. Evaluation reports PSNR, SSIM, NRMSE and optional AlexNet LPIPS, grouped by task/modality.

## Files and validation

| Directory | Purpose |
|---|---|
| `unifield/` | Model, losses, dataset, training and evaluation |
| `scripts/` | Train/test/infer launchers and asset verification |
| `configs/` | Paper defaults and ignored local configuration |
| `pretrained/` | Public third-party weights, manifest and licenses |
| `tests/` | Nine functional tests |
| `docs/NOTES.md` | Paper comparison, implementation choices and validation scope |
| `cache/`, `runs/` | Private local embeddings, manifests, checkpoint and outputs |

GPU 4/5 training and resume passed at 256×256×160 preprocessing, 40-slice crops and LoRA rank 128. Both field transitions passed full 160-slice inference. Nine tests and isolated wheel installation passed. The checkpoint has **three training steps**; paper performance has not been reproduced. See [notes](docs/NOTES.md).

```bash
python -m unittest discover -s tests -v
```

Optional helpers: `python -m unifield.split`, `python -m unifield.ablation`, and `python -m unifield.register` (requires `pip install -e '.[registration]'`). Each supports `--help`.

Apache-2.0, with the taehv-derived decoder retaining its [MIT notice](third_party/taehv-MIT.txt); upstream attribution is in [NOTICE](NOTICE). Citation metadata is in [CITATION.cff](CITATION.cff). Private LoRA, medical data and run outputs are excluded from publication; large public pretrained assets are supplied separately from code packages.
