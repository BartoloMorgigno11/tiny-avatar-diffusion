"""D28 deterministic sequential builder for the evaluator uint8 PNG cache."""
from __future__ import annotations
import argparse, hashlib, json, os, tarfile
from pathlib import Path
from typing import Any
from PIL import Image
from tiny_avatar_diffusion.evaluator import CACHE_ROOT, ROOT, SPLIT_ROOT, load_manifest, preprocess, preprocess_cached

ARCHIVE=ROOT/"data"/"raw"/"cartoonset"/"archives"/"cartoonset100k.tgz"
METADATA_NAME="cache_metadata.json"
def _rank(image_id:str)->str:return hashlib.sha256(f"seed=2026|namespace=dr04-cache-equivalence|image_id={image_id}".encode()).hexdigest()
def _source_tensor(image:Image.Image):return preprocess(image)
def build(cache_root:Path=CACHE_ROOT,audit_size:int=32)->dict[str,Any]:
    manifest=SPLIT_ROOT/"split_manifest.csv"; manifest_hash=hashlib.sha256(manifest.read_bytes()).hexdigest(); ids={row["image_id"] for row in load_manifest(manifest)}
    if len(ids)!=100000:raise ValueError("Frozen evaluator manifest must contain 100000 IDs")
    cache_root.mkdir(parents=True,exist_ok=True); audit=set(sorted(ids,key=_rank)[:audit_size]); seen=set(); equal=0
    with tarfile.open(ARCHIVE,"r|gz") as archive:
        for member in archive:
            if not member.isfile() or Path(member.name).suffix.lower()!=".png":continue
            image_id=Path(member.name).stem
            if image_id not in ids:continue
            if image_id in seen:raise ValueError(f"Duplicate source image ID {image_id}")
            stream=archive.extractfile(member)
            if stream is None:raise RuntimeError(member.name)
            with Image.open(stream) as image:
                image.load(); rgb=image.convert("RGB").resize((32,32),resample=Image.Resampling.BILINEAR)
                if rgb.mode!="RGB" or rgb.size!=(32,32):raise ValueError(image_id)
                target=cache_root/f"{image_id}.png"; temporary=cache_root/f"{image_id}.tmp"
                rgb.save(temporary,format="PNG",compress_level=9); os.replace(temporary,target)
                if image_id in audit:
                    with Image.open(target) as cached: cached.load(); exact=preprocess_cached(cached)
                    if not (exact==_source_tensor(image)).all():raise ValueError(f"D25 inequivalence {image_id}")
                    equal+=1
            seen.add(image_id)
    missing=sorted(ids-seen); extra=sorted({p.stem for p in cache_root.glob("*.png")}-ids)
    if missing or extra or len(seen)!=100000:raise ValueError(f"Cache integrity missing={len(missing)} extra={len(extra)}")
    metadata={"schema_version":"dr04_evaluator_cache_v1","cache_type":"rgb_uint8_32x32","source_archive":"data/raw/cartoonset/archives/cartoonset100k.tgz","source_archive_sha256":"not_computed_unreasonable_extra_sequential_cost","image_count":len(seen),"image_size":[32,32],"channels":3,"dtype":"uint8","resize_mode":"bilinear","antialias":True,"split_manifest_sha256":manifest_hash,"equivalence_audit":{"seed":2026,"namespace":"dr04-cache-equivalence","checked":equal,"exact_equal":True}}
    (cache_root/METADATA_NAME).write_text(json.dumps(metadata,indent=2)+"\n",encoding="utf-8");return metadata
def main()->None:
    parser=argparse.ArgumentParser();parser.add_argument("--cache-root",type=Path,default=CACHE_ROOT);args=parser.parse_args();print(json.dumps(build(args.cache_root),indent=2))
if __name__=="__main__":main()
