"""Create manifest/config variants for Table 4 without changing subject splits."""
import argparse
import csv
from pathlib import Path
import yaml
from .config import load_config
from .data import read_manifest,COLUMNS

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True);p.add_argument('--output',required=True)
    p.add_argument('--modality',choices=['T1','T2','FLAIR'])
    p.add_argument('--source-field',choices=['64mT','3T']);p.add_argument('--without-fasrm',action='store_true')
    a=p.parse_args();cfg=load_config(a.config);out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    for split in ['train','val','test']:
        path=cfg['data'].get(split+'_manifest')
        if not path:continue
        rows=read_manifest(path,cfg['data'].get('root'))
        rows=[r for r in rows if (a.modality is None or r['modality']==a.modality) and
                               (a.source_field is None or r['source_field']==a.source_field)]
        if not rows:raise ValueError(f'Empty {split} ablation split')
        dest=out/f'{split}.csv'
        with dest.open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=COLUMNS);writer.writeheader();writer.writerows({k:r[k] for k in COLUMNS} for r in rows)
        cfg['data'][split+'_manifest']=str(dest)
    if a.without_fasrm:cfg['loss']['lambda_freq']=0.
    (out/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False))
    print('Ablation config:',out/'config.yaml')
if __name__=='__main__':main()
