"""Target-free ODE inference with per-volume and per-task MRI evaluation."""
import argparse
import json
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.distributed as dist
from .config import load_config,save_json
from .data import read_manifest,MRIPairs
from .distributed import setup,finish
from .model import UniField
from .metrics import volume_metrics

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--checkpoint')
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['val','test'],default='test')
    p.add_argument('--max-cases',type=int);p.add_argument('--steps',type=int)
    p.add_argument('--native-grid',action='store_true',help='Also resample normalized prediction to source NIfTI grid')
    p.add_argument('--lpips',action='store_true',help='Compute AlexNet LPIPS; requires pretrained weights')
    a=p.parse_args();cfg=load_config(a.config);rank,local,world,device=setup()
    try:
        rows=read_manifest(cfg['data'][a.split+'_manifest'],cfg['data'].get('root'))
        if a.max_cases:rows=rows[:a.max_cases]
        dataset=MRIPairs(rows,tuple(cfg['data']['spatial_size']),frames=None,augment=False)
        model=UniField(cfg,device)
        if a.checkpoint:
            state=torch.load(a.checkpoint,map_location='cpu',weights_only=True)
            # Reject checkpoint inference with the wrong adapter or sparse geometry.
            for k in ['lora_rank','lora_alpha','attention_backend']:
                if state['config']['model'].get(k)!=cfg['model'].get(k):raise ValueError(f'Checkpoint {k} differs')
            if state['config']['inference']['local_range']!=cfg['inference']['local_range']:raise ValueError('Checkpoint local_range differs')
            model.load_adapters(state['adapters'])
        net=None
        if a.lpips:
            import lpips
            net=lpips.LPIPS(net='alex').eval().to(device)
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        records=[]
        for index in range(rank,len(dataset),world):
            batch=dataset[index]
            lq=batch['lq'].unsqueeze(0).to(device,dtype=torch.bfloat16)
            # HQ is deliberately absent from enhance(): no target leakage.
            prediction=model.enhance(lq,batch['prompt'],steps=a.steps or cfg['inference']['steps'],seed=cfg['inference'].get('seed',0)+index)
            pred=((prediction[0].mean(0)+1)/2).cpu().numpy().transpose(1,2,0)
            target=((batch['hq'].mean(0)+1)/2).numpy().transpose(1,2,0)
            source=((batch['lq'].mean(0)+1)/2).numpy().transpose(1,2,0)
            scores=volume_metrics(pred,target);baseline=volume_metrics(source,target)
            if net is not None:
                with torch.inference_mode():
                    gray = prediction[0].mean(0,keepdim=True).repeat(3,1,1,1).permute(1,0,2,3)
                    gt = batch['hq'].to(device).permute(1,0,2,3)
                    values = [net(gray[j:j+16],gt[j:j+16]).flatten() for j in range(0,len(gray),16)]
                    scores['lpips']=float(torch.cat(values).mean())
            identity=f'{index:04d}_{batch["center"]}_{batch["subject"]}_{batch["modality"]}'
            # Sanitize path components from metadata.
            import re
            identity=re.sub(r'[^A-Za-z0-9_.-]','_',identity)
            case_dir=out/identity;case_dir.mkdir(exist_ok=True)
            affine=batch['affine'].numpy()
            image=nib.Nifti1Image(pred.astype(np.float32),affine)
            image.set_qform(affine,code=1);image.set_sform(affine,code=1)
            image.header.set_xyzt_units('mm')
            nib.save(image,case_dir/'prediction.nii.gz')
            nib.save(nib.Nifti1Image(target,affine),case_dir/'target_normalized.nii.gz')
            if a.native_grid:
                from nibabel.processing import resample_from_to
                native=nib.load(batch['lq_path'])
                restored=resample_from_to(image,(native.shape,native.affine),order=1)
                nib.save(restored,case_dir/'prediction_native.nii.gz')
            record={k:batch[k] for k in ['subject','center','modality','source_field','target_field']}
            record.update(index=index,rank=rank,peak_cuda_gib=torch.cuda.max_memory_allocated(device)/2**30,metrics=scores,input_metrics=baseline,shape=list(pred.shape),output=str(case_dir/'prediction.nii.gz'))
            save_json(case_dir/'metrics.json',record);records.append(record)
            print(json.dumps(record),flush=True)
        save_json(out/f'metrics_rank{rank}.json',records)
        if world>1:dist.barrier()
        if rank==0:
            all_records=[]
            for r in range(world):all_records.extend(json.loads((out/f'metrics_rank{r}.json').read_text()))
            groups={}
            for r in all_records:
                key=f'{r["source_field"]}_to_{r["target_field"]}/{r["modality"]}'
                groups.setdefault(key,[]).append(r)
            summary={k:{'count':len(v),**{m:float(np.mean([r['metrics'][m] for r in v])) for m in v[0]['metrics']}} for k,v in groups.items()}
            save_json(out/'summary.json',{'cases':len(all_records),'metrics_by_task_modality':summary,
                'overall':{m:float(np.mean([r['metrics'][m] for r in all_records])) for m in all_records[0]['metrics']},
                'checkpoint':a.checkpoint,'ode_steps':a.steps or cfg['inference']['steps']})
            print(f'Evaluation complete: {out / "summary.json"}',flush=True)
    finally:finish()

if __name__=='__main__':main()
