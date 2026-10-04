"""Development-only LPIPS sanity audit; only ID_VAL is permitted."""
from __future__ import annotations
import csv, hashlib, json, statistics
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
import torch
from PIL import Image, ImageDraw
import lpips
from tiny_avatar_diffusion.conditional_lr_pilot import _load_text_records
from tiny_avatar_diffusion.debug_tiny_overfit import preload_images

ROOT=Path(__file__).resolve().parents[1]; OUTPUT=ROOT/"results"/"audits"/"dr05_diversity_metric_sanity_v2"; MANIFEST=ROOT/"data"/"processed"/"dr01_v1"/"text"/"10k"/"id_val.jsonl"; SEED=2026
TYPES=("identical","mild_color","small_shift","structural","distinct_real")
def guard(split:str)->None:
 if split!="ID_VAL":raise ValueError("Only ID_VAL is permitted for this audit")
def select(records:list[dict],seed:int=SEED)->list[dict]:
 return sorted(records,key=lambda r:hashlib.sha256(f"{seed}|{r['image_id']}".encode()).hexdigest())[:32]
def mild(x:torch.Tensor)->torch.Tensor:
 y=x.clone();y[:,0,:,:]=(y[:,0,:,:]+16/255).clamp(-1,1);return y
def shift(x:torch.Tensor)->torch.Tensor:
 y=torch.empty_like(x);y[:,:,:,1:]=x[:,:,:,:-1];y[:,:,:,0]=x[:,:,:,0];return y
def structural(x:torch.Tensor)->torch.Tensor:
 y=x.clone();y[:,:,12:20,12:20]=x.mean(dim=(2,3),keepdim=True);return y
def variants(images:torch.Tensor)->dict[str,torch.Tensor]:
 out={"identical":images.clone(),"mild_color":mild(images),"small_shift":shift(images),"structural":structural(images),"distinct_real":torch.roll(images,-1,0)}
 assert all(value.shape==images.shape and value.min()>=-1 and value.max()<=1 for value in out.values())
 assert torch.equal(out["mild_color"][:,1:],images[:,1:]) and torch.equal(out["small_shift"][:,:,:,0],images[:,:,:,0]) and torch.equal(out["small_shift"][:,:,:,1:],images[:,:,:,:-1])
 assert torch.equal(out["structural"][:,:,:12],images[:,:,:12]) and torch.equal(out["structural"][:,:,20:],images[:,:,20:])
 return out
def describe(values:list[float])->dict[str,float|int]:
 ordered=sorted(values);return {"n":len(values),"mean":statistics.mean(values),"median":statistics.median(values),"std":statistics.stdev(values) if len(values)>1 else 0.0,"min":min(values),"max":max(values),"q25":ordered[8] if len(values)==32 else ordered[len(values)//4],"q75":ordered[24] if len(values)==32 else ordered[(3*len(values))//4]}
def rank_summary(rows:list[dict],metric:str)->dict:
 by={r["source_id"]:{x["comparison_type"]:x[metric] for x in rows if x["source_id"]==r["source_id"]} for r in rows}; finite=lambda a,b:[v[a]/v[b] for v in by.values() if v[b]!=0]
 return {"identical_smallest_fraction":sum(v["identical"]==min(v.values()) for v in by.values())/32,"structural_gt_mild_color_fraction":sum(v["structural"]>v["mild_color"] for v in by.values())/32,"structural_gt_small_shift_fraction":sum(v["structural"]>v["small_shift"] for v in by.values())/32,"distinct_real_gt_mild_color_fraction":sum(v["distinct_real"]>v["mild_color"] for v in by.values())/32,"distinct_real_gt_structural_fraction":sum(v["distinct_real"]>v["structural"] for v in by.values())/32,"median_ratios":{"structural_over_mild_color":statistics.median(finite("structural","mild_color")) if finite("structural","mild_color") else None,"small_shift_over_mild_color":statistics.median(finite("small_shift","mild_color")) if finite("small_shift","mild_color") else None,"distinct_real_over_mild_color":statistics.median(finite("distinct_real","mild_color")) if finite("distinct_real","mild_color") else None}}
def grid(images:torch.Tensor,v:dict[str,torch.Tensor],records:list[dict],rows:list[dict],path:Path)->None:
 cell=96;canvas=Image.new("RGB",(cell*4,cell*8),"white");draw=ImageDraw.Draw(canvas)
 for i in range(8):
  for j,key in enumerate(("identical","mild_color","small_shift","structural")):
   arr=((v[key][i].clamp(-1,1)+1)*127.5).round().to(torch.uint8).permute(1,2,0).numpy();im=Image.fromarray(arr).resize((64,64));canvas.paste(im,(j*cell+16,i*cell));score=next(r["lpips"] for r in rows if r["source_id"]==records[i]["image_id"] and r["comparison_type"]==key);draw.text((j*cell+2,i*cell+68),f"{key} {score:.4f}",fill="black")
 canvas.save(path)
def main()->None:
 guard("ID_VAL")
 if OUTPUT.exists():raise RuntimeError(f"Refusing to overwrite {OUTPUT}")
 records=select(_load_text_records(MANIFEST,"ID_VAL"));images=preload_images(records);v=variants(images);validation={}
 for kind,target in v.items():
  distances=(images-target).abs().mean(dim=(1,2,3));validation[kind]={"nonzero_l1_count":int((distances>0).sum()),"mean_l1":float(distances.mean()),"min_l1":float(distances.min()),"max_l1":float(distances.max())}
 if validation["structural"]["nonzero_l1_count"]==0 or validation["mild_color"]["nonzero_l1_count"]<=1 or validation["small_shift"]["nonzero_l1_count"]==0:raise RuntimeError("Controlled transform sanity failure")
 device=torch.device("cuda" if torch.cuda.is_available() else "cpu");metric=lpips.LPIPS(net="alex",version="0.1").to(device).eval()
 def measure()->list[dict]:
  out=[]
  with torch.no_grad():
   for kind,target in v.items():
    vals=metric(images.to(device),target.to(device)).flatten().cpu().tolist();l1=(images-target).abs().mean(dim=(1,2,3)).tolist()
    for i,(p,l) in enumerate(zip(vals,l1,strict=True)):out.append({"source_id":records[i]["image_id"],"paired_source_id":records[(i+1)%32]["image_id"] if kind=="distinct_real" else "","comparison_type":kind,"lpips":float(p),"l1":float(l),"ssim":None,"one_minus_ssim":None})
  return sorted(out,key=lambda r:(r["source_id"],TYPES.index(r["comparison_type"])))
 rows=measure();repeat=measure();same=all(abs(a["lpips"]-b["lpips"])<=1e-12 and abs(a["l1"]-b["l1"])<=1e-12 for a,b in zip(rows,repeat,strict=True))
 identical=[r["lpips"] for r in rows if r["comparison_type"]=="identical"];summary={kind:{"lpips":describe([r["lpips"] for r in rows if r["comparison_type"]==kind]),"l1":describe([r["l1"] for r in rows if r["comparison_type"]==kind]),"one_minus_ssim":None} for kind in TYPES};summary["lpips_rank_summary"]=rank_summary(rows,"lpips");summary["ssim_available"]=False;summary["reproducible_measurements"]=same;summary["identical_lpips_finite_and_approximately_zero"]=bool(torch.isfinite(torch.tensor(identical)).all()) and all(abs(value)<1e-7 for value in identical)
 previous=ROOT/"results"/"audits"/"dr05_diversity_metric_sanity_v1"/"selection_manifest.json";previous_ids=[item["image_id"] for item in json.loads(previous.read_text())["entries"]]
 OUTPUT.mkdir(parents=True);payload={"seed":SEED,"source_split":"ID_VAL","entries":[{"image_id":r["image_id"],"source_image_path":r["source_image_path"]} for r in records]};payload["sha256"]=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest();payload["matches_invalid_v1_selection"]=[r["image_id"] for r in records]==previous_ids;(OUTPUT/"selection_manifest.json").write_text(json.dumps(payload,indent=2,sort_keys=True)+"\n");(OUTPUT/"transform_validation.json").write_text(json.dumps(validation,indent=2,sort_keys=True)+"\n")
 with (OUTPUT/"measurements.csv").open("w",newline="",encoding="utf-8") as f: w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 (OUTPUT/"summary.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n");grid(images,v,records,rows,OUTPUT/"sanity_grid.png")
 prov={"timestamp_utc":datetime.now(UTC).isoformat(),"lpips_package":"lpips","lpips_package_version":version("lpips"),"backbone":"AlexNet","lpips_version":"0.1","torchvision_version":version("torchvision"),"input_shape":[32,3,32,32],"input_range":"[-1,1]","ssim_available":False,"ssim_reason":"Neither scikit-image nor TorchMetrics was installed before this audit; no SSIM dependency was added.","selection_seed":SEED,"central_structural_region":{"rows":[12,20],"columns":[12,20]},"ID_TEST_accessed":False,"FINAL_OOD_accessed":False,"previous_run":"dr05_diversity_metric_sanity_v1","previous_run_status":"INVALID","invalid_reason":["mild_color indexed batch instead of red channel","small_shift shifted wrong spatial dimension","structural perturbation used empty channel slice"],"scientific_protocol_changed":False,"sealed_test_accessed":False};(OUTPUT/"provenance.json").write_text(json.dumps(prov,indent=2,sort_keys=True)+"\n")
 print(json.dumps({"reproducible":same,"summary":summary},indent=2))
if __name__=="__main__":main()
