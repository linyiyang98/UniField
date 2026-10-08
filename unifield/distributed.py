import os
import torch
import torch.distributed as dist

def setup():
    rank=int(os.environ.get('RANK','0'))
    local=int(os.environ.get('LOCAL_RANK','0'))
    world=int(os.environ.get('WORLD_SIZE','1'))
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required')
    torch.cuda.set_device(local)
    if world>1: dist.init_process_group('nccl',device_id=torch.device('cuda',local))
    return rank,local,world,torch.device('cuda',local)

def finish():
    if dist.is_initialized(): dist.destroy_process_group()
