"""Training entry point for UniField.

Usage
-----
::

    python train.py --config configs/experiments/unifield_sr.yaml
    python train.py --config configs/experiments/unifield_sr.yaml training.lr=5e-5
"""

from __future__ import annotations

import argparse
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from data import MRIEnhancementDataset, build_train_transforms, build_val_transforms
from losses import UniFieldLoss
from models import UniField
from utils import compute_metrics
from utils.visualization import log_images_to_tb

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None  # type: ignore[misc,assignment]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg,
    steps_per_epoch: int,
) -> torch.optim.lr_scheduler._LRScheduler:
    name = cfg.training.scheduler
    total_steps = cfg.training.epochs * steps_per_epoch
    warmup_steps = cfg.training.warmup_epochs * steps_per_epoch

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=cfg.training.lr * 1e-3,
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=10, factor=0.5
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")
    return scheduler


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict,
    save_path: str,
) -> None:
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        },
        save_path,
    )


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: UniFieldLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg,
    writer,
    global_step: int,
) -> tuple[float, int]:
    model.train()
    total_loss = 0.0

    for batch in loader:
        lq = batch["lq"].to(device)
        hq = batch["hq"].to(device)
        field = batch["field"].to(device)

        pred = model(lq, field)
        loss_dict = criterion(pred, hq)
        loss = loss_dict["total"]

        optimizer.zero_grad()
        loss.backward()
        if cfg.training.clip_grad_norm > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.training.clip_grad_norm)
        optimizer.step()

        total_loss += loss.item()
        global_step += 1

        if writer is not None and global_step % 50 == 0:
            for k, v in loss_dict.items():
                writer.add_scalar(f"train/{k}_loss", v.item(), global_step)

    avg_loss = total_loss / max(len(loader), 1)
    logger.info(f"Epoch {epoch:04d} | train loss: {avg_loss:.6f}")
    return avg_loss, global_step


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: UniFieldLoss,
    device: torch.device,
    epoch: int,
    cfg,
    writer,
    global_step: int,
) -> dict:
    model.eval()
    total_loss = 0.0
    psnr_sum, ssim_sum, n_batches = 0.0, 0.0, 0
    logged = False

    for batch in loader:
        lq = batch["lq"].to(device)
        hq = batch["hq"].to(device)
        field = batch["field"].to(device)

        pred = model(lq, field)
        loss_dict = criterion(pred, hq)
        total_loss += loss_dict["total"].item()

        m = compute_metrics(pred, hq)
        psnr_sum += m["psnr"]
        ssim_sum += m["ssim"]
        n_batches += 1

        if writer is not None and not logged:
            log_images_to_tb(writer, "val/images", lq, pred, hq, global_step)
            logged = True

    avg = {
        "loss": total_loss / max(n_batches, 1),
        "psnr": psnr_sum / max(n_batches, 1),
        "ssim": ssim_sum / max(n_batches, 1),
    }
    logger.info(
        f"Epoch {epoch:04d} | val loss: {avg['loss']:.6f} | "
        f"PSNR: {avg['psnr']:.2f} dB | SSIM: {avg['ssim']:.4f}"
    )
    if writer is not None:
        for k, v in avg.items():
            writer.add_scalar(f"val/{k}", v, global_step)
    return avg


def main() -> None:
    parser = argparse.ArgumentParser(description="Train UniField")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    args, overrides = parser.parse_known_args()

    cfg = OmegaConf.merge(
        OmegaConf.load("configs/default.yaml"),
        OmegaConf.load(args.config),
        OmegaConf.from_dotlist(overrides),
    )

    set_seed(cfg.training.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ── Datasets ──────────────────────────────────────────────────────────
    train_ds = MRIEnhancementDataset(
        root=cfg.data.train_root,
        patch_size=cfg.data.patch_size,
        scale_factor=cfg.data.scale_factor,
        task=cfg.data.task,
        noise_sigma=cfg.data.noise_sigma,
        transform=build_train_transforms(),
        is_train=True,
    )
    val_ds = MRIEnhancementDataset(
        root=cfg.data.val_root,
        patch_size=cfg.data.patch_size,
        scale_factor=cfg.data.scale_factor,
        task=cfg.data.task,
        noise_sigma=cfg.data.noise_sigma,
        transform=build_val_transforms(),
        is_train=False,
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=cfg.data.num_workers,
        pin_memory=cfg.data.pin_memory,
    )

    # ── Model ─────────────────────────────────────────────────────────────
    model = UniField.from_config(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"UniField parameters: {n_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
    )
    scheduler = build_scheduler(optimizer, cfg, steps_per_epoch=len(train_loader))

    # ── Loss ──────────────────────────────────────────────────────────────
    criterion = UniFieldLoss(
        l1_weight=cfg.loss.l1_weight,
        ssim_weight=cfg.loss.ssim_weight,
        perceptual_weight=cfg.loss.perceptual_weight,
    ).to(device)

    # ── Logging ───────────────────────────────────────────────────────────
    writer = None
    if SummaryWriter is not None:
        writer = SummaryWriter(log_dir=cfg.training.log_dir)

    save_dir = Path(cfg.training.save_dir)
    best_psnr = 0.0
    global_step = 0

    for epoch in range(1, cfg.training.epochs + 1):
        _, global_step = train_one_epoch(
            model, train_loader, criterion, optimizer, device,
            epoch, cfg, writer, global_step,
        )

        if cfg.training.scheduler == "cosine":
            scheduler.step()

        if epoch % cfg.training.val_every == 0:
            val_metrics = validate(
                model, val_loader, criterion, device,
                epoch, cfg, writer, global_step,
            )
            if cfg.training.scheduler == "plateau":
                scheduler.step(val_metrics["loss"])

            if val_metrics["psnr"] > best_psnr:
                best_psnr = val_metrics["psnr"]
                save_checkpoint(
                    model, optimizer, epoch, val_metrics,
                    str(save_dir / "best.pth"),
                )
                logger.info(f"  ✓ New best PSNR: {best_psnr:.2f} dB — checkpoint saved.")

        if epoch % cfg.training.save_every == 0:
            save_checkpoint(
                model, optimizer, epoch, {},
                str(save_dir / f"epoch_{epoch:04d}.pth"),
            )

    if writer is not None:
        writer.close()
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
