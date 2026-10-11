"""Experiment config: validated before anything runs, hashed for exact dedup."""
import hashlib, json
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .db import ROOT


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    paper: Literal["nanda", "power", "custom"] = "nanda"
    p: int = Field(113, ge=5, le=401)
    train_frac: float = Field(0.3, gt=0.05, lt=0.95)
    weight_decay: float = Field(1.0, ge=0, le=10)
    lr: float = Field(1e-3, gt=0, le=1)
    epochs: int = Field(10000, ge=10, le=200000)
    seed: int = Field(42, ge=0, le=2**31 - 1)
    d_model: int = Field(128, ge=8, le=1024)
    n_layers: int = Field(1, ge=1, le=6)
    n_heads: int = Field(4, ge=1, le=16)
    d_mlp: int = Field(512, ge=8, le=4096)
    batch_size: int = Field(0, ge=0)  # 0 = full batch
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    eval_every: int = Field(50, ge=1, le=5000)
    label: str = ""

    def hash_dict(self):
        d = self.model_dump()
        d.pop("label"), d.pop("eval_every")  # presentation-only, must not change identity
        return d


def config_hash(cfg: dict) -> str:
    d = ExperimentConfig(**cfg).hash_dict()
    return hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()[:16]


def validate(cfg: dict):
    """Returns (ExperimentConfig|None, error|None). Rejections are logged for the audit trail."""
    try:
        return ExperimentConfig(**cfg), None
    except ValidationError as e:
        msg = "; ".join(f"{'.'.join(map(str, x['loc']))}: {x['msg']}" for x in e.errors())
        p = ROOT / "data" / "rejected_experiments.jsonl"
        p.parent.mkdir(exist_ok=True)
        with p.open("a") as f:
            f.write(json.dumps({"config": cfg, "error": msg}) + "\n")
        return None, msg


PAPERS = json.loads((Path(__file__).parent / "papers.json").read_text())
