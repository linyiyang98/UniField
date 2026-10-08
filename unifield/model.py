"""FlashVSR initialization with trainable LoRA and detached frozen encoders."""
import gc
import hashlib
from pathlib import Path
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model
from .backbones import dit as dit_module
from .backbones.dit import WanModel, sinusoidal_embedding_1d
from .backbones.vae import BidirectionalWanVideoVAE
from .backbones.lq_projection import Bidirectional_LQ4x_Proj
from .backbones.tcdecoder import build_tcdecoder

TARGETS = ['self_attn.q','self_attn.k','self_attn.v','self_attn.o',
           'cross_attn.q','cross_attn.k','cross_attn.v','cross_attn.o','ffn.0','ffn.2']

def context_name(prompt):
    return hashlib.sha256(prompt.encode()).hexdigest()+'.pt'

def load_state(path):
    path = Path(path)
    if not path.is_file(): raise FileNotFoundError(path)
    return load_file(str(path),device='cpu') if path.suffix=='.safetensors' else torch.load(path,map_location='cpu',weights_only=True,mmap=True)

def materialize(constructor, state, dtype):
    with torch.device('meta'):
        model = constructor()
    model.load_state_dict(state, strict=True, assign=True)
    return model.to(dtype=dtype)

class UniField(nn.Module):
    def __init__(self, cfg, device):
        super().__init__()
        self.config, self.device, self.dtype = cfg, device, torch.bfloat16
        model_cfg = cfg['model']
        backend = model_cfg.get('attention_backend','sdpa')
        if backend not in ['sdpa','block_sparse']:
            raise ValueError('attention_backend must be sdpa or block_sparse')
        if backend == 'block_sparse' and dit_module.block_sparse_attn_func is None:
            raise RuntimeError('block_sparse_attn unavailable or incompatible; build for your torch/CUDA or select sdpa')
        dit_module.ATTENTION_BACKEND = backend
        base = Path(model_cfg['weights_dir'])
        state = load_state(base/'diffusion_pytorch_model_streaming_dmd.safetensors')
        dim = state['patch_embedding.weight'].shape[0]
        layers = max(int(k.split('.')[1]) for k in state if k.startswith('blocks.'))+1
        if dim != 1536 or layers != 30: raise ValueError('Expected FlashVSR v1 1.3B backbone')
        self.dit = materialize(lambda:WanModel(dim=1536,in_dim=16,ffn_dim=8960,out_dim=16,text_dim=4096,
                    freq_dim=256,eps=1e-6,patch_size=(1,2,2),num_heads=12,num_layers=30), state, self.dtype)
        # RoPE tables are attributes, and meta initialization leaves them on meta.
        self.dit.freqs = dit_module.precompute_freqs_cis_3d(128)
        del state
        state = load_state(base/'Wan2.1_VAE.pth')
        if not any(k.startswith('model.') for k in state): state = {'model.'+k:v for k,v in state.items()}
        self.vae = materialize(BidirectionalWanVideoVAE,state,self.dtype)
        self.vae.mean = torch.tensor([-0.7571,-0.7089,-0.9113,0.1075,-0.1745,0.9653,-0.1517,1.5508,0.4134,-0.0715,0.5517,-0.3632,-0.1922,-0.9497,0.2503,-0.2921])
        self.vae.std = torch.tensor([2.8184,1.4541,2.3275,2.6558,1.2196,1.7708,2.6052,2.0743,3.2687,2.1526,2.8652,1.5579,1.6382,1.1253,2.8251,1.9160])
        self.vae.scale = [self.vae.mean,1/self.vae.std]
        self.lq_projection = materialize(lambda:Bidirectional_LQ4x_Proj(in_dim=3,out_dim=1536,layer_num=1),load_state(base/'LQ_proj_in.ckpt'),self.dtype)
        self.decoder = build_tcdecoder(new_channels=[512,256,128,128],new_latent_channels=784,device='cpu',dtype=self.dtype)
        self.decoder.load_state_dict(load_state(base/'TCDecoder.ckpt'),strict=True)
        self.requires_grad_(False)
        self.eval()
        rank = model_cfg.get('lora_rank',128)
        self.dit = inject_adapter_in_model(LoraConfig(r=rank,lora_alpha=model_cfg.get('lora_alpha',rank),
                           init_lora_weights='gaussian',target_modules=TARGETS),self.dit)
        for p in self.dit.parameters():
            if p.requires_grad: p.data = p.data.float()
        self.to(device)
        self.contexts = {}
        gc.collect()
        self.checkpointing = model_cfg.get('gradient_checkpointing',True)

    def context(self, prompt):
        if prompt not in self.contexts:
            path = Path(self.config['model']['context_dir'])/context_name(prompt)
            payload = torch.load(path,map_location='cpu',weights_only=True)
            if payload['prompt'] != prompt or payload['context'].shape != (1,512,4096):
                raise ValueError(f'Invalid prompt embedding: {path}')
            self.contexts[prompt] = payload['context'].to(self.device,dtype=self.dtype)
        return self.contexts[prompt]

    @torch.no_grad()
    def encode_hq(self, hq):
        z = self.vae.encode(hq,device=self.device,tiled=False).to(self.device,dtype=self.dtype)
        expected = (hq.shape[0],16,hq.shape[2]//4,hq.shape[3]//8,hq.shape[4]//8)
        if z.shape != expected:
            raise ValueError(f'VAE returned {z.shape}, expected {expected}')
        return z

    @torch.no_grad()
    def condition(self, lq):
        return self.lq_projection.stream_forward(lq)

    def velocity(self, z, t, conditioning, context):
        x,(f,h,w) = self.dit.patchify(z)
        temb = self.dit.time_embedding(sinusoidal_embedding_1d(self.dit.freq_dim,t*1000))
        tmod = self.dit.time_projection(temb).unflatten(1,(6,self.dit.dim))
        ctx = self.dit.text_embedding(context)
        freqs = torch.cat([
            self.dit.freqs[0][:f].view(f,1,1,-1).expand(f,h,w,-1),
            self.dit.freqs[1][:h].view(1,h,1,-1).expand(f,h,w,-1),
            self.dit.freqs[2][:w].view(1,1,w,-1).expand(f,h,w,-1)],dim=-1).reshape(f*h*w,1,-1).to(z.device)
        topk = max(1,int((2*h*w//128)**2*self.config['inference']['topk_ratio'])-1)
        for i,block in enumerate(self.dit.blocks):
            if i < len(conditioning): x = x+conditioning[i]
            def call(value, tc, tm, block=block, block_index=i):
                return block(value,tc,tm,freqs,f,h,w,local_num=f//2,topk=topk,
                             block_id=block_index,kv_len=3,is_stream=False,
                             local_range=self.config['inference']['local_range'])
            x = checkpoint(call,x,ctx,tmod,use_reentrant=False) if self.training and self.checkpointing else call(x,ctx,tmod)
        return self.dit.unpatchify(self.dit.head(x,temb),(f,h,w))

    def forward(self,lq,hq,prompt):
        with torch.no_grad():
            clean = self.encode_hq(hq)
            noise = torch.randn_like(clean)
            t = torch.rand((lq.shape[0],),device=self.device,dtype=torch.float32)
            scale = t.view(-1,1,1,1,1)
            z = ((1-scale)*clean + scale*noise).to(self.dtype)
            cond = self.condition(lq)
        with torch.autocast('cuda',dtype=self.dtype):
            predicted = self.velocity(z,t.to(self.dtype),cond,self.context(prompt))
        return predicted, noise-clean

    @torch.no_grad()
    def enhance(self,lq,prompt,steps=1,seed=0):
        if steps < 1: raise ValueError('ODE steps must be positive')
        self.eval()
        generator = torch.Generator(device=self.device).manual_seed(seed)
        b,c,d,h,w = lq.shape
        z = torch.randn((b,16,d//4,h//8,w//8),device=self.device,dtype=self.dtype,generator=generator)
        cond = self.condition(lq)
        times = torch.linspace(1,0,steps+1,device=self.device)
        with torch.autocast('cuda',dtype=self.dtype):
            for current,following in zip(times[:-1],times[1:]):
                v = self.velocity(z,current.expand(b).to(self.dtype),cond,self.context(prompt))
                z = z + (following-current)*v
            self.decoder.clean_mem()
            output = self.decoder.decode_video(z.transpose(1,2),parallel=False,cond=lq,
                         show_progress_bar=False,bidirectional=True).transpose(1,2)*2-1
            self.decoder.clean_mem()
        if output.shape != lq.shape: raise ValueError(f'Decoder shape mismatch: {output.shape}, {lq.shape}')
        return output.clamp(-1,1).float()

    def adapters(self):
        return {k:p.detach().cpu() for k,p in self.dit.named_parameters() if p.requires_grad}

    def load_adapters(self,state):
        expected = {k for k,p in self.dit.named_parameters() if p.requires_grad}
        if set(state) != expected: raise ValueError('LoRA keys do not match configuration')
        self.dit.load_state_dict(state,strict=False)
