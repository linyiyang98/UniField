"""Cache real frozen UMT5-XXL embeddings once, then free its GPU memory."""
import argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer
import ftfy
import html
import re
from .config import load_config
from .data import read_manifest, prompt_for
from .model import load_state, materialize, context_name
from .backbones.text_encoder import WanTextEncoder

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    args=parser.parse_args()
    cfg=load_config(args.config)
    model_cfg=cfg['model']
    prompts=set()
    for key in ['train_manifest','val_manifest','test_manifest']:
        if cfg['data'].get(key):
            prompts.update(prompt_for(r) for r in read_manifest(cfg['data'][key],cfg['data'].get('root')))
    out=Path(model_cfg['context_dir']); out.mkdir(parents=True,exist_ok=True)
    pending=[]
    for p in sorted(prompts):
        dest=out/context_name(p)
        if dest.exists():
            obj=torch.load(dest,map_location='cpu',weights_only=True)
            if obj['prompt']==p and obj['context'].shape==(1,512,4096): continue
        pending.append(p)
    if not pending:
        print(f'All {len(prompts)} prompts already cached',flush=True); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required for UMT5-XXL encoding')
    tokenizer=AutoTokenizer.from_pretrained(model_cfg['tokenizer_dir'],local_files_only=True)
    print('Loading frozen UMT5-XXL...',flush=True)
    encoder=materialize(WanTextEncoder,load_state(Path(model_cfg['weights_dir'])/'models_t5_umt5-xxl-enc-bf16.pth'),torch.bfloat16).eval().requires_grad_(False).cuda()
    with torch.inference_mode():
        for prompt in pending:
            clean=re.sub(r'\s+',' ',html.unescape(html.unescape(ftfy.fix_text(prompt)))).strip()
            inputs=tokenizer(clean,return_tensors='pt',padding='max_length',truncation=True,max_length=512,add_special_tokens=True)
            ids,mask=inputs.input_ids.cuda(),inputs.attention_mask.cuda()
            context=encoder(ids,mask)
            context[:,int(mask.sum()):]=0
            torch.save({'prompt':prompt,'context':context.cpu(),'encoder':'UMT5-XXL/FlashVSR'},out/context_name(prompt))
            print(f'Cached: {prompt}',flush=True)
    print(f'Prepared {len(pending)} real prompt embeddings',flush=True)

if __name__=='__main__': main()
