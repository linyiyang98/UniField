"""Inference / evaluation entry point for UniField.

Usage
-----
::

    # Evaluate on a test set (computes PSNR / SSIM):
    python test.py --config configs/experiments/unifield_sr.yaml \
                   --checkpoint checkpoints/best.pth

    # Run inference on a single NIfTI file:
    python test.py --config configs/experiments/unifield_sr.yaml \
                   --checkpoint checkpoints/best.pth \
                   --input path/to/lq.nii.gz \
                   --field 2 \
                   --output enhanced.nii.gz
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import MRIEnhancementDataset, build_val_transforms
from models import UniField
from utils import compute_metrics
from utils.visualization import save_nifti

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_model(cfg, checkpoint_path: str, device: torch.device) -> UniField:
    model = UniField.from_config(cfg).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info(f"Loaded checkpoint from {checkpoint_path} (epoch {ckpt.get('epoch', '?')})")
    return model


@torch.no_grad()
def evaluate(
    model: UniField,
    loader: DataLoader,
    device: torch.device,
    output_dir: str | None,
) -> None:
    psnr_sum, ssim_sum, n = 0.0, 0.0, 0

    for batch in loader:
        lq = batch["lq"].to(device)
        hq = batch["hq"].to(device)
        field = batch["field"].to(device)
        paths = batch["path"]

        pred = model(lq, field)
        m = compute_metrics(pred, hq)
        psnr_sum += m["psnr"]
        ssim_sum += m["ssim"]
        n += 1

        if output_dir is not None:
            for i, src_path in enumerate(paths):
                fname = Path(src_path).name.replace(".nii.gz", "_enhanced.nii.gz")
                out_path = str(Path(output_dir) / fname)
                save_nifti(pred[i], out_path)

    logger.info(
        f"Results over {n} volumes — "
        f"PSNR: {psnr_sum / max(n, 1):.2f} dB | "
        f"SSIM: {ssim_sum / max(n, 1):.4f}"
    )


@torch.no_grad()
def infer_single(
    model: UniField,
    input_path: str,
    field_label: int,
    output_path: str,
    device: torch.device,
) -> None:
    """Enhance a single NIfTI volume and save the result."""
    import nibabel as nib
    import numpy as np

    img = nib.load(input_path)
    volume = np.asarray(img.dataobj, dtype=np.float32)
    vmin, vmax = volume.min(), volume.max()
    if vmax - vmin > 1e-8:
        volume = (volume - vmin) / (vmax - vmin)

    lq = torch.from_numpy(volume).unsqueeze(0).unsqueeze(0).to(device)  # (1,1,H,W,D)
    field = torch.tensor([field_label], dtype=torch.long, device=device)

    pred = model(lq, field)  # (1,1,H,W,D)
    # Rescale back to original intensity range.
    pred_np = pred[0, 0].cpu().float().numpy() * (vmax - vmin) + vmin

    out_img = nib.Nifti1Image(pred_np, img.affine, img.header)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, output_path)
    logger.info(f"Enhanced volume saved to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="UniField inference / evaluation")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--input", type=str, default=None, help="Single NIfTI input for inference")
    parser.add_argument("--field", type=int, default=None, help="B₀ field label (0–3)")
    parser.add_argument("--output", type=str, default=None, help="Output path for single inference")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    model = load_model(cfg, args.checkpoint, device)

    if args.input is not None:
        # Single-volume inference mode.
        if args.field is None:
            parser.error("--field is required when using --input")
        output_path = args.output or str(
            Path(args.input).with_stem(Path(args.input).stem + "_enhanced")
        )
        infer_single(model, args.input, args.field, output_path, device)
    else:
        # Dataset evaluation mode.
        test_ds = MRIEnhancementDataset(
            root=cfg.data.test_root,
            patch_size=cfg.data.patch_size,
            scale_factor=cfg.data.scale_factor,
            task=cfg.data.task,
            noise_sigma=cfg.data.noise_sigma,
            transform=build_val_transforms(),
            is_train=False,
        )
        test_loader = DataLoader(
            test_ds,
            batch_size=1,
            shuffle=False,
            num_workers=cfg.data.num_workers,
            pin_memory=cfg.data.pin_memory,
        )
        out_dir = cfg.inference.output_dir if cfg.inference.output_dir else None
        evaluate(model, test_loader, device, out_dir)


if __name__ == "__main__":
    main()
