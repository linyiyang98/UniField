import json
from pathlib import Path
import yaml

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    required = ['model','data','training','loss','inference']
    if not isinstance(cfg,dict) or any(k not in cfg for k in required):
        raise ValueError(f"Config requires {required}")
    if cfg['training']['batch_size'] != 1:
        raise ValueError("FlashVSR block selection currently requires batch_size=1 per GPU")
    return cfg

def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
