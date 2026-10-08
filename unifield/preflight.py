"""Check assets/manifests/config before starting an expensive GPU run."""
import argparse
from pathlib import Path
import torch
from .config import load_config
from .data import read_manifest,check_splits,prompt_for
from .model import context_name

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);a=p.parse_args()
    c=load_config(a.config);m=c['model'];base=Path(m['weights_dir'])
    for name in ['diffusion_pytorch_model_streaming_dmd.safetensors','Wan2.1_VAE.pth','LQ_proj_in.ckpt','TCDecoder.ckpt']:
        path=base/name
        if not path.is_file() or path.stat().st_size<1000000:raise FileNotFoundError(f'Missing or Git LFS pointer: {path}')
    splits={}
    for split in ['train','val','test']:
        path=c['data'].get(split+'_manifest')
        if not path:continue
        rows=read_manifest(path,c['data'].get('root'));splits[split]=rows
        print(split,len(rows),'pairs',len({(r['center'],r['subject']) for r in rows}),'subjects')
        for prompt in {prompt_for(r) for r in rows}:
            path=Path(m['context_dir'])/context_name(prompt)
            if not path.is_file():raise FileNotFoundError(f'Run prepare_text first: {path}')
            obj=torch.load(path,map_location='cpu',weights_only=True)
            if obj['prompt']!=prompt or obj['context'].shape!=(1,512,4096):raise ValueError('Invalid context')
    names=list(splits)
    for i,k in enumerate(names):
        for j in names[i+1:]:check_splits(splits[k],splits[j])
    print('Preflight passed; torch',torch.__version__,'CUDA',torch.cuda.is_available())
if __name__=='__main__':main()
