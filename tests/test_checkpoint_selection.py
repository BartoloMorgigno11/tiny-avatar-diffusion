from __future__ import annotations
import json
import pytest
from tiny_avatar_diffusion.checkpoint_selection import ALLOWED_SPLITS, build_manifest, image_name, select, write_atomic, load_or_initialize_progress, write_selection_markdown

def test_manifest_is_deterministic_and_validation_only(tmp_path):
    a=build_manifest(tmp_path/'a.json',count=2); b=build_manifest(tmp_path/'b.json',count=2)
    assert a==b and [x['split'] for x in a['entries']]==['ID_VAL','ID_VAL','COMP_VAL','COMP_VAL']
    assert a['sha256'] and a['entries'][0]['generation_seed']==2026

def test_forbidden_split_is_rejected():
    from tiny_avatar_diffusion.checkpoint_selection import _manifest_path
    with pytest.raises(ValueError): _manifest_path('ID_TEST')
    with pytest.raises(ValueError): _manifest_path('FINAL_OOD')

def test_selection_pass_conflict_and_not_run():
    base={15000:{'comp_val':{'kid':.2}},20000:{'comp_val':{'kid':.2}}}
    assert select(base)['selected_step']==20000
    base[20000]['comp_val']['kid']=.3; assert select(base)['status']=='CONFLICT'
    assert select({})['status']=='NOT_RUN'

def test_image_name_is_stable(): assert image_name({'image_id':'x'})=='x.png'

def test_progress_rejects_identity_change_and_selection_markdown(tmp_path):
    identity={'config_sha256':'a','manifest_sha256':'m','run_id':'r','checkpoints':{}}
    path=tmp_path/'progress.json'; progress=load_or_initialize_progress(path,identity); write_atomic(path,progress)
    assert load_or_initialize_progress(path,identity)['status']=='NOT_STARTED'
    with pytest.raises(ValueError): load_or_initialize_progress(path,{**identity,'config_sha256':'b'})
    output=tmp_path/'selection.md'; write_selection_markdown(output,{'status':'CONFLICT','selected_step':None,'weight_set':None},identity=identity,results={}); assert 'No checkpoint frozen' in output.read_text()
