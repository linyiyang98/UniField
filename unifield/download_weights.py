"""Fetch public FlashVSR v1 weights and Wan UMT5 assets from official sources."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download,hf_hub_download

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',default='pretrained/FlashVSR');a=p.parse_args()
    out=Path(a.output)
    snapshot_download('JunhaoZhuang/FlashVSR',local_dir=out,allow_patterns=[
        'diffusion_pytorch_model_streaming_dmd.safetensors','Wan2.1_VAE.pth','LQ_proj_in.ckpt','TCDecoder.ckpt','README.md'])
    hf_hub_download('Wan-AI/Wan2.1-T2V-1.3B','models_t5_umt5-xxl-enc-bf16.pth',local_dir=out)
    snapshot_download('Wan-AI/Wan2.1-T2V-1.3B',local_dir=out,allow_patterns=['google/umt5-xxl/*'])
    print('Weights prepared:',out)
if __name__=='__main__':main()
