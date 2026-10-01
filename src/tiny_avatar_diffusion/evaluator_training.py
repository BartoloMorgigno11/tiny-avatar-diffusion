"""DR04 evaluator epoch utilities; no training is launched at import time."""
from __future__ import annotations
import csv, platform, random
import numpy as np
from pathlib import Path
from typing import Any
import torch
from torch.utils.data import DataLoader
from tiny_avatar_diffusion.evaluator import CANONICAL_ATTRIBUTE_ORDER, weighted_loss, validate_checkpoint

HISTORY_FIELDS=["epoch","global_step","train_loss","val_loss",*[f"{a}_{m}" for a in CANONICAL_ATTRIBUTE_ORDER for m in ("accuracy","macro_f1","balanced_accuracy")],"val_macro_balanced_accuracy","val_mean_macro_f1","is_best"]
MAIN_LOADER_CONFIG={"num_workers":4,"pin_memory":True,"persistent_workers":True,"prefetch_factor":2}
def _seed_worker(worker_id:int)->None:
    seed=torch.initial_seed()%2**32; random.seed(seed); np.random.seed(seed)
def loader(dataset,shuffle:bool,seed:int=2026,epoch:int=0,*,num_workers:int=MAIN_LOADER_CONFIG["num_workers"],pin_memory:bool=MAIN_LOADER_CONFIG["pin_memory"],persistent_workers:bool=MAIN_LOADER_CONFIG["persistent_workers"],prefetch_factor:int|None=MAIN_LOADER_CONFIG["prefetch_factor"])->DataLoader:
    g=torch.Generator().manual_seed(seed+epoch)
    kwargs={"num_workers":num_workers,"pin_memory":pin_memory,"generator":g,"worker_init_fn":_seed_worker}
    if num_workers:
        kwargs.update({"persistent_workers":persistent_workers,"prefetch_factor":prefetch_factor})
    return DataLoader(dataset,batch_size=128,shuffle=shuffle,drop_last=False,**kwargs)
def _metrics(pred:torch.Tensor,target:torch.Tensor,k:int)->tuple[float,float,float]:
    accuracy=float((pred==target).float().mean()); f1=[]; recalls=[]
    for c in range(k):
        tp=((pred==c)&(target==c)).sum().item(); fp=((pred==c)&(target!=c)).sum().item(); fn=((pred!=c)&(target==c)).sum().item()
        f1.append(0.0 if 2*tp+fp+fn==0 else 2*tp/(2*tp+fp+fn)); recalls.append(0.0 if tp+fn==0 else tp/(tp+fn))
    return accuracy,sum(f1)/k,sum(recalls)/k
@torch.no_grad()
def evaluate(model, data_loader, weights, device)->dict[str,float]:
    model.eval(); losses=[]; predictions={a:[] for a in CANONICAL_ATTRIBUTE_ORDER}; targets={a:[] for a in CANONICAL_ATTRIBUTE_ORDER}
    for x,y in data_loader:
        x=x.to(device); y={a:v.to(device) for a,v in y.items()}; logits=model(x); losses.append(float(weighted_loss(logits,y,weights)))
        for a in CANONICAL_ATTRIBUTE_ORDER: predictions[a].append(logits[a].argmax(1).cpu()); targets[a].append(y[a].cpu())
    out={"val_loss":sum(losses)/len(losses)}; f1=[]; balanced=[]
    for a in CANONICAL_ATTRIBUTE_ORDER:
        acc,macro,bacc=_metrics(torch.cat(predictions[a]),torch.cat(targets[a]),weights[a].numel()); out[f"{a}_accuracy"]=acc;out[f"{a}_macro_f1"]=macro;out[f"{a}_balanced_accuracy"]=bacc;f1.append(macro);balanced.append(bacc)
    out["val_mean_macro_f1"]=sum(f1)/4;out["val_macro_balanced_accuracy"]=sum(balanced)/4;return out
def train_epoch(model,data_loader,weights,optimizer,device)->float:
    model.train(); losses=[]
    for x,y in data_loader:
        x=x.to(device); y={a:v.to(device) for a,v in y.items()}; optimizer.zero_grad(set_to_none=True); loss=weighted_loss(model(x),y,weights)
        if not torch.isfinite(loss): raise FloatingPointError("Non-finite evaluator loss")
        loss.backward(); optimizer.step(); losses.append(float(loss))
    return sum(losses)/len(losses)
def save_checkpoint(path:Path,payload:dict[str,Any])->None:path.parent.mkdir(parents=True,exist_ok=True);torch.save(payload,path)
def restore_checkpoint(path:Path,model,optimizer,identity:dict[str,Any],device:torch.device)->dict[str,Any]:
    payload=torch.load(path,map_location=device,weights_only=False); validate_checkpoint(payload,identity); model.load_state_dict(payload["model_state_dict"]); optimizer.load_state_dict(payload["optimizer_state_dict"])
    random.setstate(payload["python_rng_state"]); np.random.set_state(payload["numpy_rng_state"]); torch.set_rng_state(payload["torch_cpu_rng_state"])
    if torch.cuda.is_available() and payload["torch_cuda_rng_states"] is not None: torch.cuda.set_rng_state_all(payload["torch_cuda_rng_states"])
    return payload
def environment_metadata()->dict[str,Any]:return {"python":platform.python_version(),"torch":torch.__version__,"cuda":torch.version.cuda,"cuda_available":torch.cuda.is_available()}
def append_history(path:Path,row:dict[str,Any])->None:
    exists=path.exists()
    with path.open("a",newline="",encoding="utf-8") as f:
        writer=csv.DictWriter(f,fieldnames=HISTORY_FIELDS)
        if not exists:writer.writeheader()
        writer.writerow(row)
