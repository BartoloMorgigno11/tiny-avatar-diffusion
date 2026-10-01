"""Exact metadata-only DR04-A16 maximum balanced-control cardinality audit."""
from __future__ import annotations
import hashlib, json, subprocess
from pathlib import Path
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results'/'audits'/'dr04_max_balanced_control_cardinality_v1'
ID=ROOT/'data'/'processed'/'dr01_v1'/'text'/'10k'/'id_test.jsonl'
OOD=ROOT/'data'/'processed'/'dr01_v1'/'text'/'10k'/'final_ood.jsonl'
ATTR=('face_shape','glasses','hair_color','face_color')
PAIRS={'OOD1':('face_shape',2,'glasses',3),'OOD2':('face_shape',4,'hair_color',6),'OOD3':('glasses',10,'face_color',10),'OOD4':('hair_color',0,'face_color',0)}
TYPES=('A_FIXED','B_FIXED')
def h(v): return hashlib.sha256(v.encode()).hexdigest()
def fh(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p,split):
 r=[json.loads(x) for x in p.read_text().splitlines() if x]
 if any(x['split_role']!=split for x in r): raise ValueError('split mismatch')
 return r
def belongs(x,p):
 a,u,b,v=PAIRS[p]; return x[a]==u and x[b]==v
def ordinary(x): return not any(belongs(x,p) for p in PAIRS)
def canonical(rows,split):
 d={}
 for x in rows:
  key=h(f"dr04_prompt_source_v1|{split}|{x['caption']}|{x['image_id']}")
  if x['caption'] not in d or key<d[x['caption']]['representative_source_hash']: d[x['caption']]={**{a:x[a] for a in ATTR},'caption':x['caption'],'representative_source_sample_id':x['image_id'],'representative_source_hash':key}
 return list(d.values())
def edge_cost(q,c,typ):
 a,u,b,v=PAIRS[q['pair_id']]
 if not ordinary(c): return None
 if not ((c[a]==u and c[b]!=v) if typ=='A_FIXED' else (c[a]!=u and c[b]==v)): return None
 other=[z for z in ATTR if z not in (a,b)]
 return int(c[other[0]]!=q[other[0]])+int(c[other[1]]!=q[other[1]])
def selected_ood(pool):
 out=[]
 for p in PAIRS:
  candidates=[x for x in pool if belongs(x,p) and sum(belongs(x,z) for z in PAIRS)==1]
  if len(candidates)<25: raise ValueError('not enough exclusive OOD candidates')
  for i,x in enumerate(sorted(candidates,key=lambda x:h(f"dr04_final_ood_prompt_v1|{p}|{x['caption']}"))[:25]): out.append({**x,'pair_id':p,'prompt_id':f'OOD_{p}_{i:02d}','selection_hash':h(f"dr04_final_ood_prompt_v1|{p}|{x['caption']}")})
 return out
def build(qs,cs,fix_k=None,fix_both=None,fix_cost=None,objective='k'):
 edges=[]
 for qi,q in enumerate(qs):
  for typ in TYPES:
   for ci,c in enumerate(cs):
    cost=edge_cost(q,c,typ)
    if cost is not None: edges.append((qi,typ,ci,cost,h(f"dr04_matched_control_v1|{q['pair_id']}|{typ}|{q['prompt_id']}|{c['caption']}")))
 ny=len(edges); nk=0 if fix_k is not None else 1; nb=len(qs); base_y=0; base_b=ny; base_k=ny+nb
 n=ny+nb+nk; rr=[];cc=[];vv=[];lo=[];hi=[];row=0
 # exactly K per stratum
 for p in PAIRS:
  for typ in TYPES:
   for yi,(qi,t,_,_,_) in enumerate(edges):
    if qs[qi]['pair_id']==p and t==typ: rr.append(row);cc.append(yi);vv.append(1)
   if fix_k is None: rr.append(row);cc.append(base_k);vv.append(-1);lo.append(0);hi.append(0)
   else: lo.append(fix_k);hi.append(fix_k)
   row+=1
 # one control per q/type
 for qi in range(len(qs)):
  for typ in TYPES:
   for yi,(edge_qi,t,_,_,_) in enumerate(edges):
    if edge_qi==qi and t==typ: rr.append(row);cc.append(yi);vv.append(1)
   lo.append(0);hi.append(1);row+=1
 # unique control globally
 for ci in range(len(cs)):
  for yi,(_,_,edge_ci,_,_) in enumerate(edges):
   if edge_ci==ci:rr.append(row);cc.append(yi);vv.append(1)
  lo.append(0);hi.append(1);row+=1
 # b_q represents both types selected
 for qi in range(len(qs)):
  for typ in TYPES:
   for yi,(edge_qi,t,_,_,_) in enumerate(edges):
    if edge_qi==qi and t==typ:rr.append(row);cc.append(yi);vv.append(1)
   rr.append(row);cc.append(base_b+qi);vv.append(-1);lo.append(0);hi.append(float('inf'));row+=1
  rr.append(row);cc.append(base_b+qi);vv.append(1)
  for yi,(edge_qi,_,_,_,_) in enumerate(edges):
   if edge_qi==qi:rr.append(row);cc.append(yi);vv.append(-1)
  lo.append(-float('inf'));hi.append(0);row+=1
 if fix_both is not None:
  for qi in range(len(qs)):rr.append(row);cc.append(base_b+qi);vv.append(1)
  lo.append(fix_both);hi.append(fix_both);row+=1
 if fix_cost is not None:
  for yi,(_,_,_,cost,_) in enumerate(edges):rr.append(row);cc.append(yi);vv.append(cost)
  lo.append(fix_cost);hi.append(fix_cost);row+=1
 mat=coo_matrix((vv,(rr,cc)),shape=(row,n)).tocsr(); obj=[0.]*n
 if objective=='k':obj[base_k]=-1
 elif objective=='both':
  for qi in range(len(qs)):obj[base_b+qi]=-1
 elif objective=='cost':
  for yi,(_,_,_,cost,_) in enumerate(edges):obj[yi]=cost
 elif objective=='sha':
  rank={key:i for i,key in enumerate(sorted(e[4] for e in edges))}
  for yi,(_,_,_,_,key) in enumerate(edges):obj[yi]=rank[key]
 return obj,mat,lo,hi,edges,{'y_variables':ny,'both_variables':nb,'k_variable':nk,'total_binary_variables':n,'constraints':row}
def solve(*args,**kwargs):
 obj,mat,lo,hi,edges,counts=build(*args,**kwargs)
 upper=[1]*len(obj)
 if kwargs.get('fix_k') is None: upper[-1]=25
 return milp(c=obj,integrality=[1]*len(obj),bounds=Bounds([0]*len(obj),upper),constraints=LinearConstraint(mat,lo,hi),options={'disp':False}),edges,counts
def records(qs,cs,edges,result):
 out=[]
 for yi,(qi,typ,ci,cost,key) in enumerate(edges):
  if result.x[yi]>.5:
   q=qs[qi];c=cs[ci];out.append({'prompt_id':f"AUDIT_ID_{q['pair_id']}_{typ}_{q['prompt_id'].rsplit('_',1)[1]}",'source_pair_id':q['pair_id'],'control_type':typ,'matched_ood_prompt_id':q['prompt_id'],'caption':c['caption'],**{a:c[a] for a in ATTR},'representative_source_sample_id':c['representative_source_sample_id'],'matching_cost':cost,'selection_hash':key,'ordinary_against_all_final_ood':True})
 return sorted(out,key=lambda x:x['prompt_id'])
def main():
 if OUT.exists():raise RuntimeError('audit output exists')
 qs=selected_ood(canonical(read(OOD,'FINAL_OOD'),'FINAL_OOD'));cs=canonical(read(ID,'ID_TEST'),'ID_TEST')
 r0,e0,counts=solve(qs,cs)
 if not r0.success:raise RuntimeError('maximum-K model unexpectedly infeasible')
 k=round(-r0.fun)
 r1,e1,_=solve(qs,cs,fix_k=k,objective='both');both=round(-r1.fun)
 r2,e2,_=solve(qs,cs,fix_k=k,fix_both=both,objective='cost');cost=round(r2.fun)
 r3,e3,_=solve(qs,cs,fix_k=k,fix_both=both,fix_cost=cost,objective='sha')
 witness=records(qs,cs,e3,r3)
 r4,e4,_=solve(qs,cs,fix_k=k,fix_both=both,fix_cost=cost,objective='sha');repeat=records(qs,cs,e4,r4)
 reproducible=json.dumps(witness,sort_keys=True,separators=(',',':'))==json.dumps(repeat,sort_keys=True,separators=(',',':'))
 if not reproducible:raise RuntimeError('nonreproducible SHA witness')
 by={f'{p}|{t}':sum(x['source_pair_id']==p and x['control_type']==t for x in witness) for p in PAIRS for t in TYPES}
 prompt_types={q['prompt_id']:{x['control_type'] for x in witness if x['matched_ood_prompt_id']==q['prompt_id']} for q in qs}
 diag={'cost_histogram':{str(c):sum(x['matching_cost']==c for x in witness) for c in range(3)},'mean_nuisance_cost':sum(x['matching_cost'] for x in witness)/len(witness),'both_controls':sum(v==set(TYPES) for v in prompt_types.values()),'only_a_fixed':sum(v=={'A_FIXED'} for v in prompt_types.values()),'only_b_fixed':sum(v=={'B_FIXED'} for v in prompt_types.values()),'neither':sum(not v for v in prompt_types.values())}
 summary={'status':'AUDIT_ONLY_NOT_FROZEN','head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'solver':'scipy.optimize.milp / HiGHS','scipy_version':scipy.__version__,'metadata_sha256':{'ID_TEST':fh(ID),'FINAL_OOD':fh(OOD)},'image_content_accessed':False,'candidate_ood_count':len(qs),'ordinary_id_caption_count':len(cs),'K_max':k,'control_count':8*k,'total_prompt_count_if_adopted':100+8*k,'expected_generation_count_at_5_seeds':5*(100+8*k),'solver_status':{'K':int(r0.status),'both':int(r1.status),'cost':int(r2.status),'sha':int(r3.status)},'objective_values':{'K':k,'both':both,'total_nuisance_cost':cost},'per_stratum_selected_count':by,'witness_assignment':f'{len(witness)}/{8*k}','diagnostics':diag,'reproducible':reproducible,'variable_counts':counts}
 prov={'script_sha256':fh(Path(__file__)),'exact_formulation':'binary edge assignments; per-stratum equality to K; per-OOD/type at-most-one; global control uniqueness; binary both-control indicators; sequential exact objectives K, both, cost, SHA rank.','metadata_files_read':[str(ID.relative_to(ROOT)),str(OOD.relative_to(ROOT))],'permitted_fields_read':['image_id','split_role',*ATTR,'caption'],'generator_executed':False,'evaluator_executed':False}
 OUT.mkdir(parents=True)
 for name,obj in [('summary.json',summary),('solver_provenance.json',prov),('audit_witness_controls.json',{'status':'AUDIT_WITNESS_NOT_FROZEN','prompt_records':witness})]:(OUT/name).write_text(json.dumps(obj,sort_keys=True,indent=2)+'\n')
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
