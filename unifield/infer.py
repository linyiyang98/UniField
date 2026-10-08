"""Enhance a single NIfTI using only a low-field input."""
import argparse
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
from .config import load_config
from .data import MRIPairs
from .model import UniField

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint',required=True)
    p.add_argument('--input',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modality',choices=['T1','T2','FLAIR'],required=True)
    p.add_argument('--source-field',choices=['64mT','3T'],required=True)
    p.add_argument('--target-field',choices=['3T','7T'],required=True)
    p.add_argument('--steps',type=int);p.add_argument('--native-grid',action='store_true')
    a=p.parse_args();cfg=load_config(a.config)
    if (a.source_field,a.target_field) not in [('64mT','3T'),('3T','7T')]:raise ValueError('Unsupported transition')
    torch.cuda.set_device(0)
    # Reuse the training transform with two references to the LF file.
    # There is no HF file, target embedding, or target normalization here.
    row=dict(subject=Path(a.input).stem,center='input',modality=a.modality,
             source_field=a.source_field,target_field=a.target_field,lq=a.input,hq=a.input)
    batch=MRIPairs([row],tuple(cfg['data']['spatial_size']),augment=False)[0]
    model=UniField(cfg,torch.device('cuda',0))
    state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
    for k in ['lora_rank','lora_alpha','attention_backend']:
        if state['config']['model'].get(k)!=cfg['model'].get(k):raise ValueError(f'Checkpoint {k} differs')
    if state['config']['inference']['local_range']!=cfg['inference']['local_range']:raise ValueError('Checkpoint local_range differs')
    model.load_adapters(state['adapters'])
    prediction=model.enhance(batch['lq'][None].to('cuda',dtype=torch.bfloat16),batch['prompt'],
               steps=a.steps or cfg['inference']['steps'],seed=cfg['inference'].get('seed',0))
    volume=((prediction[0].mean(0)+1)/2).cpu().numpy().transpose(1,2,0)
    affine=batch['affine'].numpy()
    image=nib.Nifti1Image(volume.astype(np.float32),affine)
    image.set_qform(affine,1);image.set_sform(affine,1);image.header.set_xyzt_units('mm')
    if a.native_grid:
        from nibabel.processing import resample_from_to
        native=nib.load(a.input);image=resample_from_to(image,(native.shape,native.affine),order=1)
    out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);nib.save(image,out)
    print('Saved',out,'shape',image.shape)
if __name__=='__main__':main()
