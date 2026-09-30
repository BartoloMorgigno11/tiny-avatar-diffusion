"""Validation-only DR03 EMA checkpoint selector; never accesses test/OOD splits."""
from __future__ import annotations

import argparse, hashlib, json, time, os
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.conditional_lr_pilot import _batch_from_indices
from tiny_avatar_diffusion.debug_tiny_overfit import preload_images
from tiny_avatar_diffusion.diffusion import LinearDDPMSchedule
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

ROOT = Path(__file__).resolve().parents[2]
ALLOWED_SPLITS = ("ID_VAL", "COMP_VAL")
STEPS = (5000, 10000, 15000, 20000)
LOSS = {5000:(.010080884028231466,.011311929900735557),10000:(.006581584719117619,.008058486252846803),15000:(.00721289130068985,.008754865304958186),20000:(.004818801780115381,.005957545730959499)}

def sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda:stream.read(1024*1024),b""): digest.update(block)
    return digest.hexdigest()

def execution_identity(*, run_id:str, manifest_sha256:str, checkpoints:dict[int,Path], sampling_batch_size:int) -> dict[str,Any]:
    body={"run_id":run_id,"manifest_sha256":manifest_sha256,"training_commit":"2d38da843f792e9f91c687a7d051984134e87a0a","checkpoints":{str(k):{"path":str(v),"sha256":sha256_file(v)} for k,v in checkpoints.items()},"steps":list(STEPS),"weight_set":"EMA","timesteps":1000,"clip_denoised":True,"cfg":False,"native_resolution":32,"sampling_batch_size":sampling_batch_size,"clean_fid_version":"0.1.35","clean_fid_mode":"clean","kid_batch_size":sampling_batch_size,"seed_mapping":"manifest_generation_seed_v1"}
    body["config_sha256"]=hashlib.sha256(json.dumps(body,sort_keys=True,separators=(",",":")).encode()).hexdigest(); return body

def write_atomic(path:Path,payload:dict[str,Any]) -> None:
    path.parent.mkdir(parents=True,exist_ok=True); temporary=path.with_suffix(path.suffix+".tmp"); temporary.write_text(json.dumps(payload,indent=2)+"\n"); os.replace(temporary,path)

def load_or_initialize_progress(path:Path, identity:dict[str,Any]) -> dict[str,Any]:
    if path.exists():
        progress=json.loads(path.read_text())
        if progress.get("config_sha256") != identity["config_sha256"]: raise ValueError("incompatible checkpoint-selection progress identity")
        return progress
    return {"status":"NOT_STARTED","config_sha256":identity["config_sha256"],"manifest_sha256":identity["manifest_sha256"],"run_id":identity["run_id"],"checkpoint_hashes":identity["checkpoints"],"completed_generation":[],"completed_kid":[]}

def write_selection_markdown(path:Path, decision:dict[str,Any], *, identity:dict[str,Any], results:dict[str,Any]) -> None:
    lines=["# DR03 Validation Checkpoint Selection","",f"Status: `{decision['status']}`.","",f"Manifest SHA256: `{identity['manifest_sha256']}`.",f"Config SHA256: `{identity['config_sha256']}`.","", "ID_TEST NOT USED.","FINAL_OOD NOT USED."]
    if decision["status"]=="PASS": lines.insert(3,f"Selected EMA milestone: `{decision['selected_step']}`.")
    elif decision["status"]=="CONFLICT": lines.insert(3,"No checkpoint frozen: 20k/15k COMP_VAL KID conflict.")
    path.write_text("\n".join(lines)+"\n",encoding="utf-8")

def _manifest_path(split: str) -> Path:
    if split not in ALLOWED_SPLITS: raise ValueError("checkpoint selection permits only ID_VAL and COMP_VAL")
    return ROOT / "data" / "processed" / "dr01_v1" / "text" / "10k" / f"{split.lower()}.jsonl"

def build_manifest(output: Path, *, count: int = 256, seed: int = 2026) -> dict[str, Any]:
    entries=[]
    for split_index, split in enumerate(ALLOWED_SPLITS):
        for index, item in enumerate(_load_text_records(_manifest_path(split), split)[:count]):
            entries.append({"split":split,"image_id":item["image_id"],"caption":item["caption"],"token_ids":item["token_ids"],"attention_mask":item["attention_mask"],"source_image_path":item["source_image_path"],"generation_seed":seed + split_index * count + index})
    payload={"version":1,"splits":list(ALLOWED_SPLITS),"entries":entries}
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":")).encode(); payload["sha256"]=hashlib.sha256(canonical).hexdigest()
    output.parent.mkdir(parents=True,exist_ok=True); output.write_text(json.dumps(payload,indent=2)+"\n",encoding="utf-8")
    return payload

def load_ema_checkpoint(path: Path, *, expected_step: int, device: torch.device) -> tuple[TextConditioningEncoder, ConditionalTinyUNet]:
    checkpoint=torch.load(path,map_location=device,weights_only=False)
    if checkpoint.get("progress",{}).get("step") != expected_step: raise ValueError("milestone stored step differs from expected step")
    state=checkpoint.get("ema_state_dict")
    if not isinstance(state,dict): raise ValueError("checkpoint lacks EMA state")
    text,unet=TextConditioningEncoder().to(device),ConditionalTinyUNet(dropout=0.0).to(device)
    text.load_state_dict(state["text_encoder"]); unet.load_state_dict(state["conditional_unet"]); text.eval(); unet.eval()
    return text,unet

def tensor_to_png(tensor: torch.Tensor, path: Path) -> None:
    image=((tensor.detach().cpu().clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,0).numpy()
    path.parent.mkdir(parents=True,exist_ok=True); Image.fromarray(image,"RGB").save(path,format="PNG")

def image_name(entry: dict[str,Any]) -> str: return f"{entry['image_id']}.png"

def sample_per_seed(schedule: LinearDDPMSchedule, text: TextConditioningEncoder, unet: ConditionalTinyUNet, entries: list[dict[str,Any]], device: torch.device) -> torch.Tensor:
    """Batched reverse inference with independent persistent RNG streams per sample."""
    generators=[torch.Generator(device=device).manual_seed(e["generation_seed"]) for e in entries]
    ids=torch.tensor([e["token_ids"] for e in entries],device=device); masks=torch.tensor([e["attention_mask"] for e in entries],device=device)
    x=torch.cat([torch.randn((1,3,32,32),device=device,generator=g) for g in generators])
    text.eval(); unet.eval()
    with torch.no_grad():
        context=text(ids,masks)
        for value in range(schedule.betas.shape[0]-1,-1,-1):
            t=torch.full((len(entries),),value,device=device,dtype=torch.long)
            eps=unet(x,t,context,masks)
            noise=torch.cat([torch.randn((1,3,32,32),device=device,generator=g) for g in generators])
            x=schedule.p_sample_from_noise(x,t,eps,noise=noise,clip_denoised=True)["sample"]
    return x

def export_real(entries: list[dict[str,Any]], directory: Path) -> None:
    for entry in entries:
        path=directory/image_name(entry)
        if path.exists(): continue
        record={"source_image_path":entry["source_image_path"]}
        tensor=preload_images([record])[0]; tensor_to_png(tensor,path)

def expected_complete(entries: list[dict[str,Any]], directory: Path) -> bool:
    expected={image_name(e) for e in entries}; actual={p.name for p in directory.glob("*.png")} if directory.exists() else set()
    if actual - expected: raise ValueError("unexpected PNG in checkpoint-selection artifact directory")
    return actual == expected

def generate_split(*, entries:list[dict[str,Any]], text:TextConditioningEncoder, unet:ConditionalTinyUNet, schedule:LinearDDPMSchedule, directory:Path, batch_size:int, device:torch.device) -> None:
    for start in range(0,len(entries),batch_size):
        pending=[e for e in entries[start:start+batch_size] if not (directory/image_name(e)).exists()]
        if pending:
            samples=sample_per_seed(schedule,text,unet,pending,device)
            for entry,image in zip(pending,samples): tensor_to_png(image,directory/image_name(entry))
    if not expected_complete(entries,directory): raise RuntimeError("incomplete generated split")

def compute_kid(real_dir: Path, generated_dir: Path, *, batch_size: int, device: torch.device) -> dict[str,Any]:
    from cleanfid import fid
    started=time.perf_counter(); value=fid.compute_kid(str(real_dir),str(generated_dir),mode="clean",batch_size=batch_size,device=device)
    return {"clean_fid_version":"0.1.35","mode":"clean","batch_size":batch_size,"device":str(device),"kid":float(value),"runtime_seconds":time.perf_counter()-started}

def select(results: dict[int,dict[str,Any]]) -> dict[str,Any]:
    if not results: return {"status":"NOT_RUN","selected_step":None,"weight_set":None}
    if results[20000]["comp_val"]["kid"] <= results[15000]["comp_val"]["kid"]:
        return {"status":"PASS","selected_step":20000,"weight_set":"EMA"}
    return {"status":"CONFLICT","selected_step":None,"weight_set":None}

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output-root",type=Path,default=ROOT/"reports"/"dr03_checkpoint_selection"); parser.add_argument("--run-id",default="dr03_conditional_main_10k_seed2026_r2"); parser.add_argument("--run-root",type=Path,default=ROOT/"reports"/"dr03_main_training"); parser.add_argument("--sampling-batch-size",type=int,default=8); parser.add_argument("--execute",action="store_true")
    args=parser.parse_args(); manifest=args.output_root/"validation_quality_manifest.json"
    if not manifest.exists() or "source_image_path" not in json.loads(manifest.read_text())["entries"][0]: build_manifest(manifest)
    if not args.execute: print(manifest); return
    data=json.loads(manifest.read_text()); checkpoints={s:args.run_root/args.run_id/f"milestone_{s:05d}.pt" for s in STEPS}; identity=execution_identity(run_id=args.run_id,manifest_sha256=data["sha256"],checkpoints=checkpoints,sampling_batch_size=args.sampling_batch_size); progress_path=args.output_root/"progress.json"; progress=load_or_initialize_progress(progress_path,identity); progress["status"]="RUNNING"; write_atomic(progress_path,progress); device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); schedule=LinearDDPMSchedule().to(device); results={}
    try:
      for step in STEPS:
        checkpoint=checkpoints[step]; text,unet=load_ema_checkpoint(checkpoint,expected_step=step,device=device); item={"checkpoint_path":str(checkpoint),"checkpoint_sha256":identity["checkpoints"][str(step)]["sha256"],"weight_set":"EMA"}
        for split in ALLOWED_SPLITS:
            entries=[e for e in data["entries"] if e["split"]==split]; real=args.output_root/"real"/split.lower(); generated=args.output_root/"generated"/f"step_{step:05d}"/split.lower(); export_real(entries,real); generate_split(entries=entries,text=text,unet=unet,schedule=schedule,directory=generated,batch_size=args.sampling_batch_size,device=device); item[split.lower()] = compute_kid(real,generated,batch_size=args.sampling_batch_size,device=device); progress["completed_generation"].append(f"{step}:{split}"); progress["completed_kid"].append(f"{step}:{split}"); write_atomic(progress_path,progress)
        results[step]=item; print(f"milestone {step} complete",flush=True)
    except Exception as error:
      progress["status"]="FAILED"; progress["last_error"]=str(error); write_atomic(progress_path,progress); raise
    payload={"identity":identity,"results":results}; write_atomic(args.output_root/"validation_quality_results.json",payload); decision=select(results); write_atomic(args.output_root/"selection.json",decision); write_selection_markdown(args.output_root/"selection.md",decision,identity=identity,results=results); progress["status"]="COMPLETE"; write_atomic(progress_path,progress); print(json.dumps(decision),flush=True)
if __name__=="__main__": main()
