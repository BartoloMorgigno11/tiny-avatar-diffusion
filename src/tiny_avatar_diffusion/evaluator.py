"""Frozen DR04 real-image attribute evaluator components."""
from __future__ import annotations

import csv, hashlib, json, random, platform, os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from tiny_avatar_diffusion.semantic_mapping import CANONICAL_ATTRIBUTE_ORDER, SEMANTIC_MAPPING

ROOT = Path(__file__).resolve().parents[2]
SPLIT_ROOT = ROOT / "data" / "processed" / "dr04_evaluator_v1"
CACHE_ROOT = ROOT / "data" / "cache" / "dr04_evaluator_32x32_uint8"
CLASS_COUNTS = {name: len(SEMANTIC_MAPPING[name]) for name in CANONICAL_ATTRIBUTE_ORDER}

ARCHITECTURE_ID = "dr04_shared_cnn_32_64_128_gap_linear_heads_v1"
PREPROCESSING_ID = "rgb_bilinear_antialias_32_float32_minus1_to1_v1"

class AttributeEvaluator(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        def block(a:int,b:int)->list[nn.Module]: return [nn.Conv2d(a,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.Conv2d(b,b,3,padding=1),nn.BatchNorm2d(b),nn.ReLU(),nn.MaxPool2d(2,2)]
        self.backbone=nn.Sequential(*block(3,32),*block(32,64),*block(64,128),nn.AdaptiveAvgPool2d((1,1)))
        self.heads=nn.ModuleDict({name:nn.Linear(128,count) for name,count in CLASS_COUNTS.items()})
    def forward(self,x:torch.Tensor)->dict[str,torch.Tensor]:
        if x.ndim!=4 or x.shape[1:]!=(3,32,32): raise ValueError("Expected [B,3,32,32]")
        rep=self.backbone(x).flatten(1)
        return {name:head(rep) for name,head in self.heads.items()}
    def representation(self,x:torch.Tensor)->torch.Tensor: return self.backbone(x).flatten(1)

def parameter_count() -> int: return sum(p.numel() for p in AttributeEvaluator().parameters())

def preprocess(image:Image.Image)->torch.Tensor:
    rgb=image.convert("RGB")
    resized=rgb.resize((32,32),resample=Image.Resampling.BILINEAR)
    pixels=torch.frombuffer(bytearray(resized.tobytes()),dtype=torch.uint8).reshape(32,32,3).permute(2,0,1)
    return pixels.to(torch.float32).div(255).mul(2).sub(1)

def preprocess_cached(image:Image.Image)->torch.Tensor:
    if image.mode != "RGB" or image.size != (32,32): raise ValueError("Cached evaluator image must be RGB 32x32")
    pixels=torch.frombuffer(bytearray(image.tobytes()),dtype=torch.uint8).reshape(32,32,3).permute(2,0,1)
    return pixels.to(torch.float32).div(255).mul(2).sub(1)

def load_manifest(path:Path=SPLIT_ROOT/"split_manifest.csv")->list[dict[str,str]]:
    with path.open(encoding="utf-8",newline="") as handle: return list(csv.DictReader(handle))

class EvaluatorDataset(Dataset[tuple[torch.Tensor,dict[str,torch.Tensor]]]):
    """Canonical split dataset reading only the D28 uint8 cache."""
    def __init__(self, split:str, manifest_path:Path=SPLIT_ROOT/"split_manifest.csv",cache_root:Path=CACHE_ROOT,include_image_id:bool=False):
        self.rows=[row for row in load_manifest(manifest_path) if row["split"]==split]
        if not self.rows: raise ValueError(f"Empty split {split}")
        self.cache_root=cache_root
        self.include_image_id=include_image_id
    def __len__(self)->int:return len(self.rows)
    def __getitem__(self,index:int)->tuple[torch.Tensor,dict[str,torch.Tensor]]:
        row=self.rows[index]; path=self.cache_root/f"{row['image_id']}.png"
        if not path.is_file(): raise FileNotFoundError(f"Missing evaluator cache item: {path.name}")
        with Image.open(path) as image: image.load(); x=preprocess_cached(image)
        labels={name:torch.tensor(int(row[name]),dtype=torch.long) for name in CANONICAL_ATTRIBUTE_ORDER}
        return (x,labels,row["image_id"]) if self.include_image_id else (x,labels)

def class_weight_artifact(manifest_path:Path=SPLIT_ROOT/"split_manifest.csv")->dict[str,Any]:
    raw=manifest_path.read_bytes(); manifest_hash=hashlib.sha256(raw).hexdigest(); rows=[r for r in load_manifest(manifest_path) if r["split"]=="EVAL_TRAIN"]
    attributes={}
    for name,k in CLASS_COUNTS.items():
        counts=[sum(int(row[name])==class_id for row in rows) for class_id in range(k)]
        if not all(counts): raise ValueError(f"Zero EVAL_TRAIN support for {name}")
        weights=[len(rows)/(k*count) for count in counts]
        attributes[name]={"num_classes":k,"class_ids":list(range(k)),"train_counts":counts,"class_weights":weights}
    return {"schema":"tiny_avatar_diffusion.dr04_evaluator_class_weights","version":"dr04_evaluator_v1","source_split":"EVAL_TRAIN","split_manifest_sha256":manifest_hash,"total_train_examples":len(rows),"formula":"N / (K * n_c)","attributes":attributes}

def write_class_weights(manifest_path:Path=SPLIT_ROOT/"split_manifest.csv",output_path:Path=SPLIT_ROOT/"evaluator_class_weights.json")->dict[str,Any]:
    artifact=class_weight_artifact(manifest_path); content=json.dumps(artifact,indent=2)+"\n"; artifact["canonical_payload_sha256"]=hashlib.sha256(content.encode()).hexdigest(); output_path.write_text(json.dumps(artifact,indent=2)+"\n",encoding="utf-8"); return artifact

def write_class_distribution_audit(manifest_path:Path=SPLIT_ROOT/"split_manifest.csv",output_path:Path=SPLIT_ROOT/"evaluator_class_distribution_audit.json")->dict[str,Any]:
    weights=class_weight_artifact(manifest_path); rows=load_manifest(manifest_path)
    audit={"schema":"tiny_avatar_diffusion.dr04_evaluator_class_distribution_audit","source_split":"EVAL_TRAIN","attributes":{},"ood_matches":{}}
    for name,part in weights["attributes"].items():
        counts=part["train_counts"]; values=part["class_weights"]
        audit["attributes"][name]={"counts":counts,"percentages":[100*c/weights["total_train_examples"] for c in counts],"min_support":min(counts),"max_support":max(counts),"max_min_support_ratio":max(counts)/min(counts),"min_class_weight":min(values),"max_class_weight":max(values)}
    definitions=json.loads((ROOT/"data"/"processed"/"dr01_v1"/"heldout_combinations.json").read_text(encoding="utf-8"))["final_ood_definitions"]
    for split in ("EVAL_TRAIN","EVAL_VAL"):
        subset=[row for row in rows if row["split"]==split]; per={}
        for definition in definitions:
            matches=[row for row in subset if row[definition["attribute_a"]]==str(definition["value_a"]) and row[definition["attribute_b"]]==str(definition["value_b"])]
            exclusive=sum(sum(row[d["attribute_a"]]==str(d["value_a"]) and row[d["attribute_b"]]==str(d["value_b"]) for d in definitions)==1 for row in matches)
            per[definition["name"]]={"total_matches":len(matches),"exclusive_matches":exclusive,"multi_matches":len(matches)-exclusive}
        audit["ood_matches"][split]=per
    output_path.write_text(json.dumps(audit,indent=2)+"\n",encoding="utf-8"); return audit

def seed_everything(seed:int=2026)->None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed); torch.backends.cudnn.benchmark=False; torch.backends.cudnn.deterministic=True; torch.use_deterministic_algorithms(True)

def configure_main_determinism(seed:int=2026)->None:
    """Configure frozen MAIN determinism before any CUDA work is launched."""
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Strict CUDA determinism requires CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA initialization")
    seed_everything(seed)

def make_optimizer(model:nn.Module)->torch.optim.AdamW:
    decay=[]; no_decay=[]
    for module in model.modules():
        for name,param in module.named_parameters(recurse=False):
            (decay if isinstance(module,(nn.Conv2d,nn.Linear)) and name=="weight" else no_decay).append(param)
    return torch.optim.AdamW([{"params":decay,"weight_decay":1e-4},{"params":no_decay,"weight_decay":0.0}],lr=1e-3,betas=(0.9,0.999),eps=1e-8)

def weighted_loss(logits:dict[str,torch.Tensor],targets:dict[str,torch.Tensor],weights:dict[str,torch.Tensor])->torch.Tensor:
    return sum(F.cross_entropy(logits[name],targets[name],weight=weights[name]) for name in CANONICAL_ATTRIBUTE_ORDER)/4

def checkpoint_payload(model:nn.Module,optimizer:torch.optim.Optimizer,completed_epoch:int,global_step:int,best_epoch:int|None,best_metrics:dict[str,float]|None,identity:dict[str,Any])->dict[str,Any]:
    return {"model_state_dict":model.state_dict(),"optimizer_state_dict":optimizer.state_dict(),"completed_epoch":completed_epoch,"global_optimizer_step":global_step,"best_epoch":best_epoch,"best_selection_metrics":best_metrics,"python_rng_state":random.getstate(),"numpy_rng_state":np.random.get_state(),"torch_cpu_rng_state":torch.get_rng_state(),"torch_cuda_rng_states":torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,"evaluator_training_seed":identity.get("training_seed",2026),"architecture_config_identity":identity.get("architecture"),"split_manifest_sha256":identity.get("split_manifest_sha256",identity.get("manifest")),"class_weights_sha256":identity.get("class_weights_sha256",identity.get("weights")),"training_config_identity":identity,"environment":{"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available(),"device":str(next(model.parameters()).device)},"identity":identity}

def validate_checkpoint(payload:dict[str,Any],identity:dict[str,Any])->None:
    if payload.get("identity")!=identity: raise ValueError("Checkpoint identity mismatch")

def better_checkpoint(current:dict[str,float],best:dict[str,float]|None)->bool:
    if best is None:return True
    return (current["val_macro_balanced_accuracy"],current["val_mean_macro_f1"],-current["val_loss"],-current["epoch"])>(best["val_macro_balanced_accuracy"],best["val_mean_macro_f1"],-best["val_loss"],-best["epoch"])
