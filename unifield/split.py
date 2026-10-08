"""Seeded subject-level 80/20 split per center, keeping modalities together."""
import argparse
import csv
import random
from collections import defaultdict
from pathlib import Path
from .data import read_manifest,COLUMNS,check_splits

def split_rows(rows,test_fraction=0.2,val_fraction=0.,seed=42):
    if not 0<test_fraction<1 or not 0<=val_fraction<1-test_fraction:
        raise ValueError('Invalid split fractions')
    rng=random.Random(seed);centers=defaultdict(set)
    for r in rows:centers[r['center']].add(r['subject'])
    assignments={}
    for center,subjects in sorted(centers.items()):
        ids=sorted(subjects);rng.shuffle(ids)
        if len(ids)<2:raise ValueError(f'{center} needs at least two subjects')
        train_count=int(len(ids)*(1-test_fraction-val_fraction))
        val_count=int(len(ids)*val_fraction)
        if not train_count:raise ValueError(f'{center} has no training subjects')
        for i,subject in enumerate(ids):
            assignments[center,subject]='train' if i<train_count else 'val' if i<train_count+val_count else 'test'
    out={k:[] for k in ['train','val','test']}
    for r in rows:out[assignments[r['center'],r['subject']]].append(r)
    check_splits(out['train'],out['test']);check_splits(out['train'],out['val']);check_splits(out['test'],out['val'])
    return out

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',required=True);p.add_argument('--data-root')
    p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=42)
    p.add_argument('--test-fraction',type=float,default=.2);p.add_argument('--val-fraction',type=float,default=0.)
    a=p.parse_args();rows=read_manifest(a.manifest,a.data_root)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    for split,records in split_rows(rows,a.test_fraction,a.val_fraction,a.seed).items():
        if not records:continue
        with (out/f'{split}.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=COLUMNS);w.writeheader();w.writerows({k:r[k] for k in COLUMNS} for r in records)
        print(split,len(records),'pairs',len({(r['center'],r['subject']) for r in records}),'subjects')
if __name__=='__main__':main()
