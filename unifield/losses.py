"""Field-aware spatial-frequency loss, UniField Eq. (2)."""
import torch
from torch import nn

FIELD_WEIGHTS = {("64mT", "3T"): (0.2, 0.5, 0.3), ("3T", "7T"): (0.1, 0.3, 0.6)}

class FASFL(nn.Module):
    def __init__(self, lambda_freq=0.1, alpha=1.0, band_edges=(1/3, 2/3)):
        super().__init__()
        if lambda_freq < 0 or alpha < 0 or not 0 < band_edges[0] < band_edges[1] < 1:
            raise ValueError("Invalid FASFL parameters")
        self.lambda_freq, self.alpha, self.band_edges = lambda_freq, alpha, band_edges

    def forward(self, prediction, target, source_fields, target_fields):
        if prediction.shape != target.shape or prediction.ndim != 5:
            raise ValueError("Velocity tensors must have matching B,C,D,H,W shapes")
        weights = []
        for source, destination in zip(source_fields, target_fields):
            if (source, destination) not in FIELD_WEIGHTS:
                raise ValueError(f"No paper spectral weights for {source} -> {destination}")
            weights.append(FIELD_WEIGHTS[(source, destination)])
        if len(weights) != prediction.shape[0]:
            raise ValueError("One field pair per sample is required")
        delta = prediction.float() - target.float()
        spatial = delta.abs().mean()
        if self.lambda_freq == 0:
            return spatial, {"spatial": spatial.detach(), "spectral": spatial.detach()*0}
        # Float32 FFT outside autocast: BF16 FFT is unsupported and power terms
        # are numerically sensitive. Orthogonal FFT keeps scale independent of size.
        spectrum = torch.fft.fftn(delta, dim=(-3, -2, -1), norm="ortho").abs()
        focal_error = spectrum.pow(self.alpha + 2)
        axes = [torch.fft.fftfreq(n, device=delta.device) / 0.5 for n in delta.shape[-3:]]
        grids = torch.meshgrid(*axes, indexing="ij")
        radius = torch.sqrt(sum(g*g for g in grids)) / (3**0.5)
        lo, hi = self.band_edges
        masks = [radius < lo, (radius >= lo) & (radius < hi), radius >= hi]
        bands = torch.stack([focal_error[..., m].mean(dim=(1, 2)) if m.any()
                             else focal_error.sum(dim=(1,2,3,4))*0 for m in masks], dim=1)
        spectral = (bands * delta.new_tensor(weights)).sum(dim=1).mean()
        total = spatial + self.lambda_freq * spectral
        return total, {"spatial": spatial.detach(), "spectral": spectral.detach()}
