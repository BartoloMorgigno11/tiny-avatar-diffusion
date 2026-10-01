"""D29 fixed-config deterministic cached DataLoader benchmark."""
from __future__ import annotations
import json, os, statistics, time
from dataclasses import dataclass
from tiny_avatar_diffusion.evaluator import EvaluatorDataset, configure_main_determinism
from tiny_avatar_diffusion.evaluator_training import loader
CONFIGS=((0,False,False,None),(2,True,True,2),(4,True,True,2))
def run_one(config:tuple[int,bool,bool,int|None],batches:int=20)->dict:
    workers,pin,persistent,prefetch=config; dataset=EvaluatorDataset("EVAL_TRAIN",include_image_id=True); data=loader(dataset,True,2026,7,num_workers=workers,pin_memory=pin,persistent_workers=persistent,prefetch_factor=prefetch); iterator=iter(data)
    start=time.perf_counter(); first=next(iterator); cold=time.perf_counter()-start; observed=[list(first[2])]; tensors=[first[0].clone()]; labels=[{k:v.clone() for k,v in first[1].items()}]; waits=[]
    for index in range(batches):
        start=time.perf_counter(); batch=next(iterator); waits.append(time.perf_counter()-start)
        if index<4: observed.append(list(batch[2])); tensors.append(batch[0].clone()); labels.append({k:v.clone() for k,v in batch[1].items()})
    mean=statistics.mean(waits); return {"workers":workers,"pin_memory":pin,"persistent_workers":persistent,"prefetch_factor":prefetch,"cold_first_batch_seconds":cold,"mean_batch_seconds":mean,"median_batch_seconds":statistics.median(waits),"min_batch_seconds":min(waits),"max_batch_seconds":max(waits),"batches_per_second":1/mean,"images_per_second":128/mean,"image_ids":observed,"tensors":tensors,"labels":labels,"batch_shape":tuple(first[0].shape),"dtype":str(first[0].dtype),"range":[float(first[0].min()),float(first[0].max())]}
def main()->None:
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG")!=":4096:8":raise RuntimeError("Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before startup")
    configure_main_determinism(); results=[run_one(config) for config in CONFIGS]; reference=results[0]
    for item in results[1:]:
        item["ids_equal_reference"]=item["image_ids"]==reference["image_ids"]
        item["tensors_equal_reference"]=all((a==b).all().item() for a,b in zip(item["tensors"],reference["tensors"]))
        item["labels_equal_reference"]=all(all((a[k]==b[k]).all().item() for k in a) for a,b in zip(item["labels"],reference["labels"]))
    for item in results:
        item.pop("tensors");item.pop("labels")
    print(json.dumps(results,indent=2))
if __name__=="__main__":main()
