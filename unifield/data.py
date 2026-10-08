"""Explicit paired manifests and shared MONAI preprocessing."""
import csv
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from monai import transforms as T

COLUMNS = ["subject", "center", "modality", "source_field", "target_field", "lq", "hq"]

def read_manifest(path, data_root=None):
    path = Path(path).resolve()
    base = Path(data_root).resolve() if data_root else path.parent
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or not set(COLUMNS).issubset(reader.fieldnames):
            raise ValueError(f"Manifest must include {COLUMNS}: {path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty manifest: {path}")
    seen = set()
    for row in rows:
        if any(not row.get(k, "").strip() for k in COLUMNS):
            raise ValueError(f"Incomplete record: {row}")
        if (row['source_field'], row['target_field']) not in [("64mT", "3T"), ("3T", "7T")]:
            raise ValueError("Only the two paper field transitions are supported")
        if row['modality'] not in ["T1", "T2", "FLAIR"]:
            raise ValueError(f"Unsupported modality: {row['modality']}")
        for key in ['lq', 'hq']:
            p = Path(row[key])
            p = p if p.is_absolute() else base / p
            if not p.is_file():
                raise FileNotFoundError(p)
            row[key] = str(p.resolve())
        pair = (row['lq'], row['hq'])
        if pair in seen:
            raise ValueError(f"Duplicate pair: {pair}")
        seen.add(pair)
    return rows

def check_splits(train, evaluation):
    train_ids = {(r['center'], r['subject']) for r in train}
    eval_ids = {(r['center'], r['subject']) for r in evaluation}
    shared = train_ids & eval_ids
    train_paths = {r[k] for r in train for k in ['lq', 'hq']}
    eval_paths = {r[k] for r in evaluation for k in ['lq', 'hq']}
    if shared or train_paths & eval_paths:
        raise ValueError(f"Subject/file leakage between splits: {sorted(shared)}")

def prompt_for(row):
    return (f"MRI {row['modality']} sequence enhancement from {row['source_field']} "
            f"to {row['target_field']} magnetic field")

class MRIPairs(Dataset):
    def __init__(self, rows, spatial_size=(256,256,160), frames=None, augment=False, seed=0):
        self.rows, self.frames = rows, frames
        if any(n <= 0 for n in spatial_size) or spatial_size[0] % 128 or spatial_size[1] % 128:
            raise ValueError("H,W must be multiples of 128 for FlashVSR (2,8,8) windows")
        if spatial_size[2] % 8 or frames is not None and (frames % 8 or frames > spatial_size[2]):
            raise ValueError("Volume depth and sampled frames must be multiples of 8; frames <= depth")
        # Paper says 1mm Z resampling. Negative X/Y spacing keeps native values.
        self.preprocess = T.Compose([
            T.LoadImaged(keys=['lq','hq'], image_only=True),
            T.EnsureChannelFirstd(keys=['lq','hq']),
            T.Spacingd(keys=['lq','hq'], pixdim=(-1.,-1.,1.), mode=('bilinear','bilinear')),
            T.ScaleIntensityRangePercentilesd(keys=['lq','hq'], lower=0.5, upper=99.5,
                                             b_min=-1., b_max=1., clip=True),
            T.Resized(keys=['lq','hq'], spatial_size=spatial_size, mode=('trilinear','trilinear'), align_corners=False),
            T.EnsureTyped(keys=['lq','hq'], dtype=torch.float32),
        ])
        self.augment = T.Compose([
            T.RandFlipd(keys=['lq','hq'], prob=0.1, spatial_axis=0),
            T.RandFlipd(keys=['lq','hq'], prob=0.1, spatial_axis=1),
        ]) if augment else None
        self.preprocess.set_random_state(seed=seed)
        if self.augment: self.augment.set_random_state(seed=seed)

    def augmentation_state(self):
        if self.augment is None: return []
        transforms = [self.augment] + list(self.augment.transforms)
        states = []
        for t in transforms:
            n = t.R.get_state()
            states.append((n[0], n[1].tolist(), *n[2:]))
        return states

    def restore_augmentation_state(self, states):
        if not states: return
        transforms = [self.augment] + list(self.augment.transforms)
        for t,n in zip(transforms, states):
            t.R.set_state((n[0], np.array(n[1], dtype=np.uint32), *n[2:]))

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        import nibabel as nib
        a,b = nib.load(row['lq']),nib.load(row['hq'])
        if a.shape != b.shape or not np.allclose(a.affine,b.affine,atol=1e-3):
            raise ValueError("MRI pair is not registered on a common grid; register before training")
        data = self.preprocess({'lq':row['lq'],'hq':row['hq']})
        affine = data['lq'].affine.clone().double()
        if self.augment: data = self.augment(data)
        depth = data['lq'].shape[-1]
        start = int(torch.randint(depth-self.frames+1,(1,)).item()) if self.frames and self.frames < depth else 0
        end = start+self.frames if self.frames else depth
        output = {}
        for key in ['lq','hq']:
            # Preserve float MRI intensities; no PIL/uint8 round trip.
            output[key] = data[key].as_tensor()[:, :, :, start:end].permute(0,3,1,2).repeat(3,1,1,1).contiguous()
        output.update({k:row[k] for k in ['subject','center','modality','source_field','target_field']})
        output.update(prompt=prompt_for(row), affine=affine, lq_path=row['lq'])
        return output
