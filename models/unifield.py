"""UniField: top-level model assembling encoder, decoder, and field embedding.

    lq_mri ──► Encoder ──► bottleneck ──┐
                   │                    │
                skips               Decoder ──► enhanced_mri
                   │      B₀ embed ──►  │
                   └────────────────────┘
"""

from __future__ import annotations

from omegaconf import DictConfig
import torch
import torch.nn as nn

from .encoder import Encoder
from .decoder import Decoder
from .field_embed import FieldEmbedding


class UniField(nn.Module):
    """Unified Field-Aware MRI Enhancement Network.

    Parameters
    ----------
    in_channels:
        Input image channels (default 1 for single-contrast MRI).
    out_channels:
        Output image channels (default 1).
    base_channels:
        Base feature channels for the encoder.
    num_res_blocks:
        Residual blocks per encoder stage.
    num_decoder_stages:
        Number of encoder/decoder stages (= depth of U-Net).
    field_embed_dim:
        Dimensionality of the B₀ field embedding.
    num_field_strengths:
        Number of supported discrete field strengths.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 64,
        num_res_blocks: int = 4,
        num_decoder_stages: int = 4,
        field_embed_dim: int = 32,
        num_field_strengths: int = 4,
    ) -> None:
        super().__init__()

        self.encoder = Encoder(
            in_channels=in_channels,
            base_channels=base_channels,
            num_stages=num_decoder_stages,
            num_res_blocks=num_res_blocks,
        )

        self.field_embed = FieldEmbedding(
            num_field_strengths=num_field_strengths,
            embed_dim=field_embed_dim,
        )

        # Skip channel sizes from finest to coarsest produced by the encoder:
        # stem → base_channels, then each stage doubles.
        skip_channels = [base_channels * (2 ** i) for i in range(num_decoder_stages)]
        # Decoder receives skips in coarsest→finest order.
        skip_channels_reversed = list(reversed(skip_channels))

        self.decoder = Decoder(
            bottleneck_channels=self.encoder.out_channels,
            skip_channels_list=skip_channels_reversed,
            out_channels=out_channels,
            cond_dim=field_embed_dim,
        )

        # Final pixel-shuffle-like upsampling to match original input size
        # (compensates for stem not downsampling).
        self.out_act = nn.Sigmoid()

    @classmethod
    def from_config(cls, cfg: DictConfig) -> "UniField":
        """Construct a UniField model from an OmegaConf DictConfig."""
        m = cfg.model
        return cls(
            in_channels=m.get("in_channels", 1),
            out_channels=m.get("out_channels", 1),
            base_channels=m.get("base_channels", 64),
            num_res_blocks=m.get("num_res_blocks", 4),
            num_decoder_stages=m.get("num_decoder_stages", 4),
            field_embed_dim=m.get("field_embed_dim", 32),
            num_field_strengths=m.get("num_field_strengths", 4),
        )

    def forward(
        self,
        lq: torch.Tensor,
        field: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        lq:
            Low-quality input MRI, shape (B, C, H, W, D).
        field:
            B₀ field-strength label, shape (B,), integer in [0, num_field_strengths).

        Returns
        -------
        torch.Tensor
            Enhanced MRI of the same spatial size as *lq*, shape (B, C, H, W, D).
        """
        field_emb = self.field_embed(field)                  # (B, embed_dim)
        bottleneck, skips = self.encoder(lq)                 # bottleneck + skip list
        # Decoder expects skips coarsest→finest.
        skips_reversed = list(reversed(skips))
        out = self.decoder(bottleneck, skips_reversed, field_emb)
        out = self.out_act(out)
        return out
