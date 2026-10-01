"""Canonical D18/D27 launcher for DR04 evaluator MAIN training."""
from __future__ import annotations
import argparse,csv,hashlib,json,subprocess,time
from datetime import UTC,datetime
from pathlib import Path
from typing import Any
import torch
from tiny_avatar_diffusion.evaluator import (ARCHITECTURE_ID,CACHE_ROOT,CLASS_COUNTS,PREPROCESSING_ID,ROOT,SPLIT_ROOT,AttributeEvaluator,EvaluatorDataset,better_checkpoint,checkpoint_payload,class_weight_artifact,configure_main_determinism,make_optimizer,parameter_count,validate_checkpoint)
from tiny_avatar_diffusion.evaluator_training import MAIN_LOADER_CONFIG,append_history,evaluate,loader,restore_checkpoint,save_checkpoint,train_epoch

RUN_ID="dr04_evaluator_main_100k_seed2026_r1"; EPOCHS=50; BATCH=128; RUN_ROOT=ROOT/"results"/"runs"
EXPECTED_MANIFEST="09fa2cc89caf3f9b2e72d97b3a4cda16a2c7d728ef895c9111fc2c1caa558ee9"; EXPECTED_WEIGHTS="148edacf6ca3bf40f110105ad817d07183ba2e8738481e1f3449fca3b9d5a3a9"
def sha(path:Path)->str:
 h=hashlib.sha256();
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
 return h.hexdigest()
def git(args:list[str])->str:
 r=subprocess.run(["git",*args],cwd=ROOT,capture_output=True,text=True,check=False);return r.stdout.strip()
def canonical_config()->dict[str,Any]:
 a=class_weight_artifact();return {"run_id":RUN_ID,"training_seed":2026,"epochs":EPOCHS,"architecture":ARCHITECTURE_ID,"preprocessing":PREPROCESSING_ID,"split_manifest_sha256":sha(SPLIT_ROOT/"split_manifest.csv"),"class_weights_sha256":EXPECTED_WEIGHTS,"parameter_count":parameter_count(),"optimizer":{"name":"AdamW","lr":1e-3,"betas":[.9,.999],"eps":1e-8,"weight_decay":1e-4},"loss":"mean_four_weighted_cross_entropy","batch":BATCH,"effective_batch":BATCH,"gradient_accumulation":1,"loader":{**MAIN_LOADER_CONFIG,"train_shuffle":True,"val_shuffle":False,"drop_last":False},"precision":"fp32","scheduler":"none","amp":"none","ema":"none","determinism":{"cublas_workspace_config":":4096:8","strict":True},"selection":"macro_ba_desc,mean_macro_f1_desc,val_loss_asc,epoch_asc","nominal_batches_per_epoch":736,"expected_optimizer_steps":36800}
def config_hash(config:dict[str,Any])->str:return hashlib.sha256(json.dumps(config,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def identity(config:dict[str,Any])->dict[str,Any]:return {"architecture":ARCHITECTURE_ID,"preprocessing":PREPROCESSING_ID,"training_seed":2026,"split_manifest_sha256":config["split_manifest_sha256"],"class_weights_sha256":config["class_weights_sha256"],"config_sha256":config_hash(config)}
def cache_check()->None:
 rows=list(csv.DictReader((SPLIT_ROOT/"split_manifest.csv").open(encoding="utf-8"))); ids={r["image_id"] for r in rows}; files={p.stem for p in CACHE_ROOT.glob("*.png")}
 if len(ids)!=100000 or len(files)!=100000 or ids!=files:raise RuntimeError("Evaluator cache integrity failure")
def weights()->dict[str,torch.Tensor]:
 disk=json.loads((SPLIT_ROOT/"evaluator_class_weights.json").read_text(encoding="utf-8"));a=class_weight_artifact()
 if disk.get("canonical_payload_sha256")!=EXPECTED_WEIGHTS:raise RuntimeError("Frozen class-weight hash mismatch")
 if a["split_manifest_sha256"]!=EXPECTED_MANIFEST:raise RuntimeError("Frozen manifest mismatch")
 if sha(SPLIT_ROOT/"split_manifest.csv")!=EXPECTED_MANIFEST:raise RuntimeError("Frozen manifest hash mismatch")
 return {k:torch.tensor(v["class_weights"],dtype=torch.float32) for k,v in a["attributes"].items()}
def preflight(real:bool=True)->dict[str,Any]:
 configure_main_determinism(); c=canonical_config();
 if c["split_manifest_sha256"]!=EXPECTED_MANIFEST or c["class_weights_sha256"]!=EXPECTED_WEIGHTS or c["parameter_count"]!=293064:raise RuntimeError("Frozen identity mismatch")
 weights()
 cache_check()
 result={"config":c,"config_sha256":config_hash(c),"head":git(["rev-parse","HEAD"]),"clean":not bool(git(["status","--short"])),"timestamp_utc":datetime.now(UTC).isoformat()}
 if real:
  tr=EvaluatorDataset("EVAL_TRAIN");va=EvaluatorDataset("EVAL_VAL");next(iter(loader(tr,True,2026,0)));next(iter(loader(va,False,2026,0)));result["batch_loads"]={"train":True,"val":True}
 return result
def _history(path:Path)->list[dict[str,str]]:
 return list(csv.DictReader(path.open(encoding="utf-8"))) if path.exists() else []
def run(run_dir:Path,*,resume:bool=False,epochs:int=EPOCHS,train_data=None,val_data=None)->dict[str,Any]:
 p=preflight(real=train_data is None);c=p["config"]; ident=identity(c)
 if not resume and run_dir.exists():raise RuntimeError("NEW run refuses existing run directory")
 if not p["clean"]:raise RuntimeError("Reportable MAIN run requires clean Git tree")
 run_dir.mkdir(parents=True,exist_ok=resume); config_path=run_dir/"config.json"; history_path=run_dir/"training_history.csv"; device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); model=AttributeEvaluator().to(device);opt=make_optimizer(model); w={k:v.to(device) for k,v in weights().items()};completed=0;steps=0;best=None;best_epoch=None
 if resume:
  payload=restore_checkpoint(run_dir/"latest.pt",model,opt,ident,device);completed=payload["completed_epoch"];steps=payload["global_optimizer_step"];best=payload["best_selection_metrics"];best_epoch=payload["best_epoch"]
  if len(_history(history_path))!=completed:raise RuntimeError("History/checkpoint epoch mismatch")
 else: config_path.write_text(json.dumps(c,indent=2)+"\n",encoding="utf-8")
 train_data=train_data or EvaluatorDataset("EVAL_TRAIN");val_data=val_data or EvaluatorDataset("EVAL_VAL")
 for epoch in range(completed+1,epochs+1):
  train_loss=train_epoch(model,loader(train_data,True,2026,epoch),w,opt,device);metrics=evaluate(model,loader(val_data,False,2026,epoch),w,device);steps+=len(loader(train_data,True,2026,epoch));current={"epoch":epoch,**metrics};is_best=better_checkpoint(current,best)
  row={"epoch":epoch,"global_step":steps,"train_loss":train_loss,**metrics,"is_best":is_best};append_history(history_path,row)
  if is_best:best=current;best_epoch=epoch;save_checkpoint(run_dir/"best.pt",checkpoint_payload(model,opt,epoch,steps,best_epoch,best,ident))
  save_checkpoint(run_dir/"latest.pt",checkpoint_payload(model,opt,epoch,steps,best_epoch,best,ident))
 summary={"run_id":RUN_ID,"status":"COMPLETE","config_sha256":ident["config_sha256"],"epochs_completed":epochs,"global_optimizer_steps":steps,"best_epoch":best_epoch,"best_metrics":best,"epoch_50_metrics":current,"boundary_best_warning":best_epoch==50,"checkpoint_hashes":{name:sha(run_dir/name) for name in ("best.pt","latest.pt")},"history_sha256":sha(history_path),"config_sha256_file":sha(config_path)}
 (run_dir/"run_summary.json").write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8");return summary
def main()->None:
 ap=argparse.ArgumentParser();ap.add_argument("--resume",action="store_true");ap.add_argument("--preflight-only",action="store_true");args=ap.parse_args()
 if args.preflight_only:print(json.dumps(preflight(),indent=2));return
 print(json.dumps(run(RUN_ROOT/RUN_ID,resume=args.resume),indent=2))
if __name__=="__main__":main()
