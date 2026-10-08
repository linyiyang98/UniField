# Public third-party pretrained assets

These are the publicly released upstream pretrained files used by UniField. They are unchanged copies, verified against official upstream Git/LFS hashes before inclusion. `manifest.json` provides exact source revisions, pinned download URLs, byte sizes and SHA-256 checksums.

| Asset | Upstream |
|---|---|
| DiT `diffusion_pytorch_model_streaming_dmd.safetensors` | JunhaoZhuang/FlashVSR v1 |
| `Wan2.1_VAE.pth` | JunhaoZhuang/FlashVSR v1 |
| `LQ_proj_in.ckpt` | JunhaoZhuang/FlashVSR v1 |
| `TCDecoder.ckpt` | JunhaoZhuang/FlashVSR v1 |
| `models_t5_umt5-xxl-enc-bf16.pth` | Wan-AI/Wan2.1-T2V-1.3B |
| `google/umt5-xxl/` tokenizer files | Wan-AI/Wan2.1-T2V-1.3B |

Sources: [FlashVSR model card](https://huggingface.co/JunhaoZhuang/FlashVSR), [Wan2.1 model card](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B). Both declare Apache-2.0; original licenses are retained in `licenses/`. FlashVSR authors: Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan and Tianfan Xue. Wan2.1 assets are distributed by the Wan-AI team; the text encoder/tokenizer uses UMT5-XXL.

The public model-asset directory is `FlashVSR/`, approximately 18.3 GB. Model binaries are supplied separately from the small code archive/wheel and ordinary Git objects. The official pinned upstream links are already public and can also be used to fetch exact copies. No new external hosting destination is configured by this local preparation step.

**Our trained UniField LoRA weights are private.** No `lora_weights.pth`, trained adapter, `latest.pt`, optimizer state, training checkpoint, MRI dataset or cached MRI-specific embedding belongs in this public directory.

Verify every prepared file locally:

```bash
python scripts/verify_pretrained.py --directory pretrained
```

If only the code archive is available, retrieve the upstream assets:

```bash
python -m unifield.download_weights --output pretrained/FlashVSR
```

The original training/evaluation evidence used the same upstream pretrained assets. New users must train their own UniField LoRA because the authors' fine-tuned LoRA is not part of this release.
