"""Paper-configured LoRA training, single GPU or torchrun DDP."""
import argparse
import hashlib
import json
import os
import random
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from .config import load_config, save_json
from .data import read_manifest, check_splits, MRIPairs
from .distributed import setup, finish
from .losses import FASFL
from .model import UniField

def rng_state():
    # JSON-compatible NumPy/random metadata is carried in a weights_only-safe checkpoint.
    return {'torch':torch.get_rng_state(),'cuda':torch.cuda.get_rng_state(),
            'python':random.getstate(),'numpy':(np.random.get_state()[0],np.random.get_state()[1].tolist(),*np.random.get_state()[2:])}

def restore_rng(s):
    torch.set_rng_state(s['torch']); torch.cuda.set_rng_state(s['cuda'])
    random.setstate(s['python'])
    n=s['numpy']; np.random.set_state((n[0],np.array(n[1],dtype=np.uint32),*n[2:]))

def adapter_digest(model):
    h=hashlib.sha256()
    for k,v in model.adapters().items(): h.update(k.encode());h.update(v.numpy().tobytes())
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--resume');p.add_argument('--max-steps',type=int,help='Stop at this absolute optimizer step; retain paper LR schedule')
    a=p.parse_args(); cfg=load_config(a.config)
    rank,local,world,device=setup()
    try:
        seed=cfg['training'].get('seed',2026)
        random.seed(seed+rank);np.random.seed(seed+rank);torch.manual_seed(seed+rank)
        torch.cuda.manual_seed(seed+rank)
        out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
        train=read_manifest(cfg['data']['train_manifest'],cfg['data'].get('root'))
        for key in ['val_manifest','test_manifest']:
            if cfg['data'].get(key): check_splits(train,read_manifest(cfg['data'][key],cfg['data'].get('root')))
        dataset=MRIPairs(train,tuple(cfg['data']['spatial_size']),cfg['data']['train_frames'],
                         cfg['data'].get('augment',False),seed+rank)
        sampler=DistributedSampler(dataset,num_replicas=world,rank=rank,seed=seed,shuffle=True)
        # num_workers=0 preserves sampling/augmentation RNG for exact resume.
        loader=DataLoader(dataset,batch_size=1,sampler=sampler,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed+rank))
        print(f'Rank {rank}/{world}: CUDA logical {local}, visible={os.environ.get("CUDA_VISIBLE_DEVICES")}',flush=True)
        model=UniField(cfg,device)
        optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=cfg['training']['lr'],betas=(0.5,0.999))
        total=cfg['training']['steps'];decay=cfg['training']['decay_start']
        if not 0 <= decay < total: raise ValueError('decay_start must be smaller than steps')
        scheduler=torch.optim.lr_scheduler.LambdaLR(optimizer,lambda s:1. if s<=decay else max(0.,(total-s)/(total-decay)))
        loss_fn=FASFL(**cfg['loss'])
        start=0
        if a.resume:
            state=torch.load(a.resume,map_location='cpu',weights_only=True)
            # Only path/output changes are permitted in a resumed run.
            for key in ['model','training','loss','inference']:
                old=dict(state['config'][key]);new=dict(cfg[key])
                if key=='training':
                    old.pop('save_every',None);new.pop('save_every',None)
                if key=='model':
                    for path_key in ['weights_dir','context_dir','tokenizer_dir']:
                        old.pop(path_key,None);new.pop(path_key,None)
                if old != new: raise ValueError(f'Resume configuration differs: {key}')
            if state['config']['data'] != cfg['data']: raise ValueError('Resume data configuration differs')
            if state['world_size'] != world: raise ValueError('Resume with the same number of GPUs for exact RNG/data order')
            model.load_adapters(state['adapters']);optimizer.load_state_dict(state['optimizer']);scheduler.load_state_dict(state['scheduler'])
            start=state['step'];restore_rng(state['rng'][rank]);dataset.restore_augmentation_state(state.get('augmentation_rng',[[]]*world)[rank])
            print(f'Rank {rank}: resumed at step {start}',flush=True)
        runner=DDP(model,device_ids=[local],broadcast_buffers=False) if world>1 else model
        initial_digest=adapter_digest(model)
        stop=min(total,a.max_steps or total)
        if stop <= start: raise ValueError('Requested stop step is not after the resume step')
        if rank==0:
            save_json(out/'config.json',cfg)
            save_json(out/'environment.json',{'torch':torch.__version__,'world_size':world,'visible_gpus':os.environ.get('CUDA_VISIBLE_DEVICES'),
                'gpu':torch.cuda.get_device_name(device),'train_pairs':len(train),'train_subjects':len({(r['center'],r['subject']) for r in train}),
                'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad)})
        sampler.set_epoch(start//len(loader));iterator=iter(loader)
        # Dataset randomness must not advance while fast-forwarding the sampler.
        # Advance only its index iterator; create a loader on the remaining indices.
        offset=start%len(loader)
        if offset:
            indices=list(iter(sampler))[offset:]
            iterator=iter(DataLoader(dataset,batch_size=1,sampler=indices,num_workers=0,pin_memory=True,generator=torch.Generator().manual_seed(seed+rank)))
        for step in range(start,stop):
            if step>start and step%len(loader)==0:
                sampler.set_epoch(step//len(loader));iterator=iter(loader)
            batch=next(iterator)
            runner.train();model.vae.eval();model.decoder.eval();model.lq_projection.eval()
            lq=batch['lq'].to(device,dtype=torch.bfloat16);hq=batch['hq'].to(device,dtype=torch.bfloat16)
            optimizer.zero_grad(set_to_none=True)
            prediction,target=runner(lq,hq,batch['prompt'][0])
            loss,parts=loss_fn(prediction,target,batch['source_field'],batch['target_field'])
            loss.backward()
            trainable=[(k,p) for k,p in model.named_parameters() if p.requires_grad]
            valid=bool(torch.isfinite(loss)) and all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for k,p in trainable)
            flag=torch.tensor(int(valid),device=device)
            if world>1:dist.all_reduce(flag,op=dist.ReduceOp.MIN)
            if not flag.item(): raise FloatingPointError('Nonfinite loss/gradient or disconnected LoRA parameter')
            grad_norm=torch.nn.utils.clip_grad_norm_([p for k,p in trainable],cfg['training'].get('max_grad_norm',1.))
            cross_grad=sum(float(p.grad.float().norm()) for k,p in trainable if '.cross_attn.k.' in k or '.cross_attn.v.' in k)
            lr=optimizer.param_groups[0]['lr'];optimizer.step();scheduler.step()
            record={'step':step+1,'rank':rank,'loss':float(loss.detach()),'spatial':float(parts['spatial']),
                    'spectral':float(parts['spectral']),'lr':lr,'grad_norm':float(grad_norm),'cross_kv_grad_norm':cross_grad,
                    'peak_cuda_gib':torch.cuda.max_memory_allocated(device)/2**30}
            with (out/f'train_rank{rank}.jsonl').open('a') as f:f.write(json.dumps(record,allow_nan=False)+'\n')
            print(json.dumps(record),flush=True)
            if (step+1)%cfg['training'].get('save_every',100)==0 or step+1==stop:
                digest=adapter_digest(model)
                entry={'rng':rng_state(),'augmentation_rng':dataset.augmentation_state(),'digest':digest}
                gathered=[None]*world if rank==0 else None
                if world>1:dist.gather_object(entry,gathered,dst=0)
                else:gathered=[entry]
                if rank==0:
                    if len({v['digest'] for v in gathered})!=1: raise RuntimeError('DDP adapters diverged')
                    checkpoint={'step':step+1,'world_size':world,'config':cfg,'adapters':model.adapters(),
                                'optimizer':optimizer.state_dict(),'scheduler':scheduler.state_dict(),'rng':[v['rng'] for v in gathered],'augmentation_rng':[v['augmentation_rng'] for v in gathered]}
                    tmp=out/'latest.tmp';torch.save(checkpoint,tmp);tmp.replace(out/'latest.pt')
                    save_json(out/'verification.json',{'step':step+1,'world_size':world,'synchronized_adapters':True,
                        'adapter_sha256':digest,'parameters_changed':digest!=initial_digest,'finite_gradients':True,
                        'cross_kv_grad_norm':cross_grad,'visible_gpus':os.environ.get('CUDA_VISIBLE_DEVICES')})
                if world>1:dist.barrier()
        if rank==0: print(f'Training complete: {out / "latest.pt"}',flush=True)
    finally: finish()

if __name__=='__main__':main()
