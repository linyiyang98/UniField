import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
from peft import LoraConfig,inject_adapter_in_model
from unifield.losses import FASFL
from unifield.backbones import dit
from unifield.data import MRIPairs,read_manifest,COLUMNS,check_splits,prompt_for
from unifield.split import split_rows
from unifield.metrics import volume_metrics

class LossTests(unittest.TestCase):
    def test_equation_and_field_weights(self):
        torch.manual_seed(4)
        p=torch.randn(2,3,8,16,16,requires_grad=True)
        t=torch.randn_like(p)
        value,parts=FASFL()(p,t,['64mT','3T'],['3T','7T'])
        error=torch.fft.fftn(p-t,dim=(-3,-2,-1),norm='ortho').abs()**3
        grids=torch.meshgrid(*[torch.fft.fftfreq(n)/.5 for n in p.shape[-3:]],indexing='ij')
        radius=torch.sqrt(sum(g*g for g in grids))/(3**.5)
        masks=[radius<1/3,(radius>=1/3)&(radius<2/3),radius>=2/3]
        means=torch.stack([error[...,m].mean((1,2)) for m in masks],dim=1)
        expected=(p-t).abs().mean()+.1*(means*torch.tensor([[.2,.5,.3],[.1,.3,.6]])).sum(1).mean()
        torch.testing.assert_close(value,expected)
        value.backward();self.assertTrue(torch.isfinite(p.grad).all());self.assertGreater(float(p.grad.abs().sum()),0)
    def test_zero_and_unknown_fields(self):
        p=torch.zeros(1,1,8,8,8,requires_grad=True)
        v,_=FASFL()(p,p.detach(),['64mT'],['3T']);v.backward()
        self.assertEqual(float(v.detach()),0);self.assertTrue(torch.isfinite(p.grad).all())
        with self.assertRaises(ValueError):FASFL()(p,p,['1.5T'],['3T'])

class AttentionTests(unittest.TestCase):
    def test_sparse_reference_matches_masked_attention_and_gradient(self):
        torch.manual_seed(1);dit.ATTENTION_BACKEND='sdpa'
        q=torch.randn(1,256,16,requires_grad=True);k=torch.randn_like(q);v=torch.randn_like(q)
        mask=torch.tensor([[[[True,False],[True,True]],[[True,False],[False,True]]]])
        actual=dit.flash_attention(q,k,v,2,attention_mask=mask)
        qh=q.reshape(1,256,2,8).transpose(1,2)
        kh=k.reshape(1,256,2,8).transpose(1,2);vh=v.reshape(1,256,2,8).transpose(1,2)
        full=mask.repeat_interleave(128,-2).repeat_interleave(128,-1)
        expected=F.scaled_dot_product_attention(qh,kh,vh,attn_mask=full).transpose(1,2).reshape(1,256,16)
        torch.testing.assert_close(actual,expected,rtol=1e-5,atol=1e-6)
        actual.square().mean().backward();self.assertTrue(torch.isfinite(q.grad).all())
    def test_cross_kv_lora_receives_gradient(self):
        layer=dit.CrossAttention(16,2)
        layer.requires_grad_(False)
        layer=inject_adapter_in_model(LoraConfig(r=2,lora_alpha=2,target_modules=['k','v']),layer)
        layer(torch.randn(1,8,16),torch.randn(1,4,16)).square().mean().backward()
        for module in ['k','v']:
            grad=getattr(layer,module).lora_B['default'].weight.grad
            self.assertIsNotNone(grad);self.assertGreater(float(grad.abs().sum()),0)

class VolumeTests(unittest.TestCase):
    def test_bidirectional_decoder_chunk_neighbors_match_full(self):
        from unifield.backbones.tcdecoder import MemBlock,TGrow,apply_model_with_memblocks
        torch.manual_seed(3)
        model=torch.nn.Sequential(MemBlock(4,4),TGrow(4,2),MemBlock(4,4),torch.nn.Conv2d(4,3,1)).eval()
        value=torch.randn(1,8,4,4,4)
        with torch.no_grad():
            full,_=apply_model_with_memblocks(model,value.clone(),True,False,mem=[],bidirectional=True)
            chunked,_=apply_model_with_memblocks(model,value.clone(),False,False,mem=[],bidirectional=True)
        torch.testing.assert_close(chunked,full,rtol=1e-5,atol=1e-6)

    def test_vae_uses_final_slices(self):
        from types import SimpleNamespace
        from unifield.backbones.vae import BidirectionalVideoVAE_
        encoder=torch.nn.AvgPool3d((4,1,1))
        stub=SimpleNamespace(encoder=encoder,conv1=lambda x:torch.cat([x,x],dim=1),z_dim=1)
        volume=torch.zeros(1,1,8,2,2);volume[:,:,5:]=1.
        latent=BidirectionalVideoVAE_.encode(stub,volume,[torch.zeros(1),torch.ones(1)])
        self.assertEqual(tuple(latent.shape),(1,1,2,2,2))
        self.assertGreater(float(latent[:,:,-1].mean()),.5)

class DataTests(unittest.TestCase):
    def test_subject_split_no_modality_leakage(self):
        rows=[dict(subject=f's{i}',center=c,modality=m,lq=f'{c}/{i}/{m}/low',hq=f'{c}/{i}/{m}/high')
              for c in ['A','B'] for i in range(10) for m in ['T1','T2']]
        split=split_rows(rows)
        self.assertEqual(split,split_rows(rows))
        self.assertEqual(len(split['test']),8)
        check_splits(split['train'],split['test'])
        with self.assertRaises(ValueError):check_splits(split['train'],split['train'][:1])
    def test_manifest_preprocessing_preserves_float_and_affine(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            a=np.linspace(0,1,32*32*8,dtype=np.float32).reshape(32,32,8)
            affine=np.diag([2.,2.,1.,1.]);affine[:3,3]=[10,20,30]
            for name in ['lq','hq']:nib.save(nib.Nifti1Image(a,affine),root/f'{name}.nii.gz')
            row=dict(subject='s1',center='A',modality='T1',source_field='64mT',target_field='3T',lq='lq.nii.gz',hq='hq.nii.gz')
            with (root/'manifest.csv').open('w') as f:
                writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader();writer.writerow(row)
            records=read_manifest(root/'manifest.csv')
            item=MRIPairs(records,(128,128,8),augment=False)[0]
            torch.testing.assert_close(item['lq'],item['hq'])
            self.assertEqual(tuple(item['lq'].shape),(3,8,128,128))
            # Image physical center must be retained across MONAI resize.
            center=item['affine'].numpy()@np.array([63.5,63.5,3.5,1.])
            expected=affine@np.array([15.5,15.5,3.5,1.])
            np.testing.assert_allclose(center,expected,atol=1e-5)
            self.assertGreater(float((((item['lq']+1)/2*255)%1).abs().max()),.1)
            self.assertEqual(prompt_for(row),'MRI T1 sequence enhancement from 64mT to 3T magnetic field')
            shifted=affine.copy();shifted[0,3]+=1.
            nib.save(nib.Nifti1Image(a,shifted),root/'hq.nii.gz')
            with self.assertRaises(ValueError):MRIPairs(records,(128,128,8))[0]
    def test_metrics_known_error(self):
        p=np.ones((8,8,8),dtype=np.float32)*.1;t=np.zeros_like(p)
        m=volume_metrics(p,t)
        self.assertAlmostEqual(m['psnr_db'],20,places=4);self.assertAlmostEqual(m['nrmse_percent'],10,places=4)

if __name__=='__main__':unittest.main()
