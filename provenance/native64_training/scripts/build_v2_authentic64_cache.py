"""Build and independently verify the canonical TRAIN-only native64 cache."""
from __future__ import annotations
import argparse, hashlib, json, os, tarfile, uuid
from pathlib import Path
import numpy as np
from PIL import Image

def sha(p:Path)->str:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def pixels(stream)->np.ndarray:
 with Image.open(stream) as image:
  rgba=image.convert('RGBA'); white=Image.new('RGBA',rgba.size,(255,255,255,255)); rgb=Image.alpha_composite(white,rgba).convert('RGB').resize((64,64),Image.Resampling.BILINEAR)
  return np.frombuffer(rgb.tobytes(),dtype=np.uint8).reshape(64,64,3).transpose(2,0,1).copy()
def main():
 p=argparse.ArgumentParser();p.add_argument('--data-root',type=Path,required=True);p.add_argument('--cache-dir',type=Path,required=True);a=p.parse_args()
 manifest=a.data_root/'data/processed/dr01_v1/text/10k/train.jsonl'; rows=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x]
 if len(rows)!=8388 or any(r.get('split_role')!='TRAIN' for r in rows): raise ValueError('expected exactly canonical 8388-record TRAIN split')
 refs=[str(r['source_image_path']) for r in rows]; archives={x.split('::',1)[0] for x in refs}
 if len(archives)!=1 or any('::' not in x for x in refs): raise ValueError('original source archive references required')
 ref=next(iter(archives)); archive=Path(ref) if Path(ref).is_absolute() else a.data_root/ref; members={x.split('::',1)[1]:i for i,x in enumerate(refs)}
 a.cache_dir.mkdir(parents=True,exist_ok=True); target=a.cache_dir/'train64_rgb_uint8.npy'; meta=a.cache_dir/'train64_rgb_uint8.json'; report=a.cache_dir/'integrity_report.json'; identities=a.cache_dir/'source_identities.jsonl'
 if any(x.exists() for x in (target,meta,report,identities)): raise FileExistsError('append-only cache destination already populated')
 tmp=a.cache_dir/f'.train64.{uuid.uuid4().hex}.tmp'; out=np.lib.format.open_memmap(tmp,mode='w+',dtype=np.uint8,shape=(len(rows),3,64,64));seen=0
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   i=members.get(member.name)
   if i is None: continue
   stream=tar.extractfile(member)
   if stream is None: raise FileNotFoundError(member.name)
   out[i]=pixels(stream);seen+=1
 del out
 if seen!=len(rows): tmp.unlink(missing_ok=True);raise RuntimeError(f'cache incomplete: {seen}/{len(rows)}')
 os.replace(tmp,target); manifest_sha=sha(manifest); archive_sha=sha(archive); cache_sha=sha(target)
 identity_lines=''.join(json.dumps({'index':i,'image_id':r['image_id'],'source_image_path':r['source_image_path']},sort_keys=True)+'\n' for i,r in enumerate(rows)); identities.write_text(identity_lines,encoding='utf-8')
 # Independent decode: reopen original archive and reproduce sixteen fixed entries.
 probe_indices=np.random.default_rng(20261017).choice(len(rows),size=16,replace=False).tolist(); probe_members={refs[i].split('::',1)[1]:i for i in probe_indices}; cached=np.load(target,mmap_mode='r'); probes=[]
 with tarfile.open(archive,'r|gz') as tar:
  for member in tar:
   i=probe_members.get(member.name)
   if i is None: continue
   stream=tar.extractfile(member); direct=pixels(stream)
   if not np.array_equal(direct,cached[i]): raise RuntimeError(f'pixel mismatch at TRAIN index {i}')
   probes.append({'index':i,'image_id':rows[i]['image_id'],'source_image_path':refs[i],'source_member_sha256':hashlib.sha256(member.name.encode()).hexdigest(),'cache_tensor_sha256':hashlib.sha256(cached[i].tobytes()).hexdigest()})
 if len(probes)!=16: raise RuntimeError('independent source verification incomplete')
 provenance={'schema':'tiny_avatar_diffusion.v2_authentic_native64_cache_v1','train_manifest_sha256':manifest_sha,'source_archive_sha256':archive_sha,'cache_sha256':cache_sha,'source_identities_sha256':hashlib.sha256(identity_lines.encode()).hexdigest(),'shape':[8388,3,64,64],'dtype':'uint8','preprocessing':{'source':'original CartoonSet RGBA archive members','alpha_composite_background_rgba':[255,255,255,255],'resize':'Pillow BILINEAR directly 500x500 source to 64x64 RGB','intermediate_32x32_forbidden':True},'pixel_exact_independent_verification':{'seed':20261017,'passed':True,'samples':probes}}
 meta.write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n',encoding='utf-8');report.write_text(json.dumps({**provenance,'all_train_cache_entries_verified_shape':True,'protected_splits_opened':False},indent=2,sort_keys=True)+'\n',encoding='utf-8');print(json.dumps(provenance,sort_keys=True))
if __name__=='__main__':main()
