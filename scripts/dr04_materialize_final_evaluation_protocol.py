"""Deterministically materialize DR04 D1-D42a protocol metadata only."""
from __future__ import annotations
import argparse, hashlib, importlib.util, json, platform, subprocess, sys
from pathlib import Path
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/'data'/'processed'/'dr04_final_eval_v1'
TRAIN=ROOT/'data'/'processed'/'dr01_v1'/'text'/'10k'/'train.jsonl'
GEN='0bad000254c59bef3b9cf3a2e33ccc10d263f6cea231841e71950a41b7bd3884'; EVAL='3f97b8763dc8caa1148ff9434224cac619507f6a377b32bc742cd5fae204c50f'
def load_a16():
 s=importlib.util.spec_from_file_location('a16',ROOT/'scripts'/'dr04_max_balanced_control_audit.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,obj):return json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=False)+'\n'
def edit_pairs(a,train):
 pool=a.canonical(a.read(TRAIN,'TRAIN'),'TRAIN'); candidates=[]
 for target in a.ATTR:
  groups={}
  for x in pool:
   if a.ordinary(x):groups.setdefault(tuple(x[z] for z in a.ATTR if z!=target),[]).append(x)
  for group in groups.values():
   for i,x in enumerate(group):
    for y in group[i+1:]:
     if x[target]==y[target]:continue
     c1,c2=sorted((x['caption'],y['caption'])); candidates.append((target,x,y,a.h(f'dr04_minimal_edit_pair_v1|{target}|{c1}|{c2}')))
 # Deterministic SHA-ordered constrained construction explicitly tests the
 # required 80-endpoint feasibility while never consulting image content.
 ends=sorted({x['caption'] for _,x,y,_ in candidates for x in (x,y)})
 chosen=[]; used=set()
 for target in a.ATTR:
  for candidate in sorted((z for z in candidates if z[0]==target),key=lambda z:z[3]):
   if len([z for z in chosen if z[0]==target])==10:break
   if candidate[1]['caption'] in used or candidate[2]['caption'] in used:continue
   chosen.append(candidate);used.update((candidate[1]['caption'],candidate[2]['caption']))
 if len(chosen)!=40 or len(used)!=80:raise ValueError('80 unique minimal-edit endpoints infeasible')
 out=[]
 for target in a.ATTR:
  for j,(t,x,y,key) in enumerate(sorted((z for z in chosen if z[0]==target),key=lambda z:z[3])):
   if x['caption']>y['caption']:x,y=y,x
   out.append({'pair_id':f'ME_{target}_{j:02d}','target_attribute':target,'endpoint_a_prompt_id':f'ME_{target}_{j:02d}_A','endpoint_b_prompt_id':f'ME_{target}_{j:02d}_B','caption_a':x['caption'],'caption_b':y['caption'],'attributes_a':{z:x[z] for z in a.ATTR},'attributes_b':{z:y[z] for z in a.ATTR},'changed_from':x[target],'changed_to':y[target],'unchanged_attribute_values':{z:x[z] for z in a.ATTR if z!=target},'selection_hash':key,'train_known_source_sample_ids':[x['representative_source_sample_id'],y['representative_source_sample_id']]})
 return out,{'candidate_pairs':len(candidates),'endpoints_considered':len(ends),'selection':'SHA-ordered globally endpoint-unique feasibility construction'}
def construct(closure_status):
 a=load_a16();qs=a.selected_ood(a.canonical(a.read(a.OOD,'FINAL_OOD'),'FINAL_OOD'));cs=a.canonical(a.read(a.ID,'ID_TEST'),'ID_TEST')
 r0,_,_=a.solve(qs,cs);k=round(-r0.fun)
 if k!=19:raise ValueError(f'K invariant failed: {k}')
 r1,_,_=a.solve(qs,cs,fix_k=k,objective='both');both=round(-r1.fun)
 r2,_,_=a.solve(qs,cs,fix_k=k,fix_both=both,objective='cost');cost=round(r2.fun)
 r3,e3,_=a.solve(qs,cs,fix_k=k,fix_both=both,fix_cost=cost,objective='sha'); controls=a.records(qs,cs,e3,r3)
 if both!=76 or cost!=92:raise ValueError('A16 optimum invariant failed')
 selected={x['matched_ood_prompt_id'] for x in controls}
 ood=[{'prompt_id':q['prompt_id'],'domain':'OOD','pair_id':q['pair_id'],'caption':q['caption'],**{z:q[z] for z in a.ATTR},'representative_source_sample_id':q['representative_source_sample_id'],'exclusive':True,'all_final_ood_memberships':[q['pair_id']],'selection_hash':q['selection_hash'],'selected_for_matched_controls':q['prompt_id'] in selected} for q in qs]
 ids=[{**x,'domain':'ID','nuisance_cost':x.pop('matching_cost')} for x in controls]
 panel=sorted(ood+ids,key=lambda x:x['prompt_id'])
 if len(panel)!=252 or len({x['caption'] for x in panel})!=252 or len(selected)!=76:raise ValueError('main panel invariant failed')
 hist={str(i):sum(x['nuisance_cost']==i for x in ids) for i in range(3)}
 if hist!={'0':60,'1':92,'2':0}:raise ValueError('cost histogram invariant failed')
 pairs,edit_diag=edit_pairs(a,TRAIN)
 if len(pairs)!=40 or len({c for p in pairs for c in (p['caption_a'],p['caption_b'])})!=80:raise ValueError('minimal-edit invariant failed')
 seeds={'generation_seeds':[2026,2027,2028,2029,2030],'evaluator_training_seed':2026,'prompt_panel_selection_seed':2026,'bootstrap_seed':2026,'expected_main_generation_count':1260,'expected_minimal_edit_generation_count':400}
 protocol={'protocol_id':'dr04_final_eval_v1','status':'FROZEN','main_conditioning':{'ood_full_panel_prompt_count':100,'matched_comparisons_ood_prompt_count':76,'matched_id_control_count':152,'applicability':'OOD metrics use all 100 OOD prompts; matched-control comparisons use only the 76 OOD prompts with both controls and 152 ID controls.'},'metrics':{'primary_conditioning':'MAA','primary_compositional':'MacroPSR','secondary':['EAM','per-attribute Macro-F1','per-attribute Balanced Accuracy','per-combination PSR','p11/p10/p01/p00','confusion matrices','matched-control diagnostics'],'evaluator_ceiling_correction':False},'aggregation_hierarchy':['seed','prompt','held-out pair / domain','macro aggregate'],'bootstrap':{'confidence_level':0.95,'method':'percentile','repetitions':10000,'seed':2026,'cluster_unit':'prompt','all_seeds_clustered':True,'id_ood_gap':'D36 independent ID/OOD resampling'},'invalid_generation_handling':'D39: finite [3,32,32] tensors valid; exact prompt/seed retries only for technical external failures; no new seed replacement; persistent intrinsic failure remains denominator with zero failure score where required.','minimal_edit':{'metric':'target edit success plus non-target retention; secondary same-seed sensitivity diagnostic, not causal proof','qualitative_grid_rule':'fixed, no cherry-picking'},'human_audit_reference':'human_audit_protocol.json'}
 human={'status':'FROZEN_PROTOCOL_ONLY','image_count':80,'breakdown':{'ID':40,'OOD':40},'sampling':'stratified; up to 16 error-enriched by frozen rule; remainder preregistered non-error-enriched strata; no aesthetic selection','raters':{'independent':2,'third_adjudicator_for_disagreement':True},'blinding':['model identity','ID/OOD split','automatic evaluator prediction','favored hypothesis'],'questions':'four atomic attribute questions','report':['rater agreement','human-vs-evaluator agreement','optional Cohen/Fleiss-style kappa'],'evaluator_recalibration':False}
 main={'protocol_id':'dr04_final_eval_v1','status':'FROZEN','prompt_count':252,'ood_prompt_count':100,'id_prompt_count':152,'generation_seed_count':5,'expected_main_generation_count':1260,'controlled_attributes':list(a.ATTR),'caption_grammar_identity':'DR01 canonical full four-attribute caption','frozen_final_ood_definitions':a.PAIRS,'source_split_manifest_hashes':{'ID_TEST':sha(a.ID),'FINAL_OOD':sha(a.OOD)},'prompt_records':panel}
 controls_manifest={'status':'FROZEN','control_count':152,'K_per_stratum':19,'prompt_records':ids}
 edit={'status':'FROZEN','pair_count':40,'generation_seeds':seeds['generation_seeds'],'expected_generations_per_pair':10,'expected_total_generation_count':400,'endpoint_global_uniqueness':True,'pair_records':pairs,'diagnostics':edit_diag}
 freeze={'phase':'DR04','status':closure_status,'closure_audit_status':'PASS' if closure_status=='CLOSED' else None,'decision_range':'D1-D42a','superseded_decisions':['D40/D41-v1'],'superseded_reason':'structurally infeasible under exact MILP before final evaluation','evidence':{'A15_summary_sha256':'cf2ec1f4868a97a050fd3e5ea706e402632d9cb14f7ab6dcce1742df03b347fb','A15_solver_provenance_sha256':'17f8ef2b637b138d60b68bbe7ec38108ea150f08d21136ce509110aca32bca76','A16_summary_sha256':'3630d9dd8c416126447a3e712eb8691f1911a6daf1f6158d5e828caa72ddbf79','A16_solver_provenance_sha256':'67fc472ad85fe1e543b28829bf4c61195210297d8903a6947577a5882e5ad15b','A16_witness_controls_sha256':'19fcbd04e404b709bdfe336df3c634a5c53bad75c8caf53aa2d559fb63cd674b'},'active_control_protocol':'D40/D41-v2','K':19,'main_panel_prompts':252,'main_generation_slots':1260,'minimal_edit_pairs':40,'minimal_edit_generation_slots':400,'total_DR04_conditional_generation_slots_if_all_executed':1660,'generation_has_occurred':False,'final_generation_executed':False}
 log={'ID_TEST_metadata_accessed':True,'FINAL_OOD_metadata_accessed':True,'TRAIN_metadata_accessed_for_minimal_edit_construction':True,'ID_TEST_image_content_accessed':False,'FINAL_OOD_image_content_accessed':False,'generator_executed':False,'evaluator_executed':False,'final_metric_executed':False,'metadata_files_read':[str(a.ID.relative_to(ROOT)),str(a.OOD.relative_to(ROOT)),str(TRAIN.relative_to(ROOT))],'permitted_fields_read':['image_id','split_role',*a.ATTR,'caption'],'scripts_executed':['scripts/dr04_materialize_final_evaluation_protocol.py']}
 provenance={'head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'python_version':sys.version,'scipy_version':scipy.__version__,'solver':'scipy.optimize.milp / HiGHS','script_hashes':{'materializer':sha(Path(__file__)),'a16_audit':sha(ROOT/'scripts'/'dr04_max_balanced_control_audit.py')},'source_metadata_hashes':{'ID_TEST':sha(a.ID),'FINAL_OOD':sha(a.OOD),'TRAIN':sha(TRAIN)},'evaluator_checkpoint_sha256':EVAL,'generator_checkpoint_sha256':GEN,'d31_freeze_sha256':'374e23ab2cb42f9b21b813042e93cfbfcab97187fdd30621b154378fd7c2f436','selection_namespaces':['dr04_prompt_source_v1','dr04_final_ood_prompt_v1','dr04_matched_control_v1','dr04_minimal_edit_pair_v1'],'generation_seeds':seeds['generation_seeds'],'bootstrap_seed':2026,'protocol_version':'dr04_final_eval_v1'}
 diag={'matched_controls':{'both':76,'neither':24,'one_only':0,'cost_histogram':hist,'mean_cost':sum(x['nuisance_cost'] for x in ids)/152},'main_caption_unique_count':len({x['caption'] for x in panel}),'minimal_edit_endpoint_unique_count':80}
 return {'final_evaluation_protocol.json':protocol,'final_prompt_panel_manifest.json':main,'matched_controls_manifest.json':controls_manifest,'seed_manifest.json':seeds,'minimal_edit_pair_manifest.json':edit,'human_audit_protocol.json':human,'controlled_unseal_access_log.json':log,'final_eval_provenance.json':provenance,'dr04_freeze_manifest.json':freeze,'panel_diagnostics.json':diag}
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--closure-status',choices=('pending','closed'),default='pending');args=parser.parse_args()
 status='CLOSED' if args.closure_status=='closed' else 'FROZEN_PENDING_CLOSURE_AUDIT'
 one=construct(status);two=construct(status)
 if {k:write(k,v) for k,v in one.items()}!={k:write(k,v) for k,v in two.items()}:raise ValueError('nonreproducible canonical construction')
 if OUT.exists():
  protected=('final_prompt_panel_manifest.json','matched_controls_manifest.json','seed_manifest.json','minimal_edit_pair_manifest.json','final_evaluation_protocol.json','human_audit_protocol.json')
  for name in protected:
   if write(name,json.loads((OUT/name).read_text()))!=write(name,one[name]):raise ValueError(f'scientific artifact drift: {name}')
 OUT.mkdir(parents=True,exist_ok=True)
 for name,obj in one.items():(OUT/name).write_text(write(name,obj),encoding='utf-8')
 print(json.dumps({'artifact_count':len(one),'main_prompts':252,'minimal_edit_pairs':40},indent=2))
if __name__=='__main__':main()
