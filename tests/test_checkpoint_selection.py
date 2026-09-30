from __future__ import annotations
import json
import pytest
import torch
from tiny_avatar_diffusion.checkpoint_selection import ALLOWED_SPLITS, build_manifest, image_name, select, write_atomic, load_or_initialize_progress, write_selection_markdown, execution_identity, expected_complete, tensor_to_png, compute_kid, sample_per_seed, generate_split, load_ema_checkpoint
from tiny_avatar_diffusion.text_conditioning import TextConditioningEncoder
from tiny_avatar_diffusion.unet import ConditionalTinyUNet

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

def test_identity_hash_is_stable_and_changes_with_checkpoint_or_batch(tmp_path):
    checkpoint=tmp_path/'x.pt'; checkpoint.write_bytes(b'x')
    a=execution_identity(run_id='r',manifest_sha256='m',checkpoints={5000:checkpoint},sampling_batch_size=8)
    b=execution_identity(run_id='r',manifest_sha256='m',checkpoints={5000:checkpoint},sampling_batch_size=8)
    assert a['config_sha256']==b['config_sha256']
    c=execution_identity(run_id='r',manifest_sha256='m',checkpoints={5000:checkpoint},sampling_batch_size=4)
    assert c['config_sha256']!=a['config_sha256']; checkpoint.write_bytes(b'y')
    assert execution_identity(run_id='r',manifest_sha256='m',checkpoints={5000:checkpoint},sampling_batch_size=8)['config_sha256']!=a['config_sha256']

def test_png_export_and_expected_set_validation(tmp_path):
    entry={'image_id':'one'}; tensor_to_png(torch.zeros(3,32,32),tmp_path/'one.png')
    assert expected_complete([entry],tmp_path)
    (tmp_path/'bad.png').write_bytes(b'x')
    with pytest.raises(ValueError): expected_complete([entry],tmp_path)

def test_cleanfid_adapter_uses_clean_mode(monkeypatch,tmp_path):
    seen={}
    def fake(*args,**kwargs): seen.update(kwargs); return .25
    from cleanfid import fid
    monkeypatch.setattr(fid,'compute_kid',fake)
    result=compute_kid(tmp_path/'real',tmp_path/'generated',batch_size=7,device=torch.device('cpu'))
    assert result['kid']==.25 and result['mode']=='clean' and seen['mode']=='clean' and seen['batch_size']==7

@pytest.mark.parametrize('status,phrase', [('PASS','Selected EMA'),('CONFLICT','No checkpoint frozen'),('NOT_RUN','Status')])
def test_selection_markdown_states(tmp_path,status,phrase):
    path=tmp_path/'x.md'; identity={'manifest_sha256':'m','config_sha256':'c'}
    write_selection_markdown(path,{'status':status,'selected_step':20000 if status=='PASS' else None,'weight_set':'EMA' if status=='PASS' else None},identity=identity,results={})
    assert phrase in path.read_text() and 'ID_TEST NOT USED' in path.read_text()

class _Text(torch.nn.Module):
    def forward(self, ids, masks): return ids.float().unsqueeze(-1)
class _Unet(torch.nn.Module):
    def forward(self,x,t,c,m): return torch.zeros_like(x)
class _Schedule:
    betas=torch.zeros(1)
    def p_sample_from_noise(self,x,t,e,noise,clip_denoised): assert clip_denoised; return {'sample':x+noise}

def test_production_seed_orchestration_is_batch_invariant():
    entries=[{'token_ids':[1,2], 'attention_mask':[1,1], 'generation_seed':seed} for seed in (1,2,3)]
    one=torch.cat([sample_per_seed(_Schedule(),_Text(),_Unet(),[entry],torch.device('cpu')) for entry in entries])
    grouped=sample_per_seed(_Schedule(),_Text(),_Unet(),entries,torch.device('cpu'))
    assert torch.equal(one,grouped) and not torch.equal(grouped[0],grouped[1])

def test_partial_generation_reuses_expected_pngs(monkeypatch,tmp_path):
    entries=[{'image_id':'a','token_ids':[1], 'attention_mask':[1], 'generation_seed':1},{'image_id':'b','token_ids':[1], 'attention_mask':[1], 'generation_seed':2}]
    tensor_to_png(torch.zeros(3,32,32),tmp_path/'a.png'); calls=[]
    monkeypatch.setattr('tiny_avatar_diffusion.checkpoint_selection.sample_per_seed',lambda *args,**kwargs: calls.append([e['image_id'] for e in args[3]]) or torch.zeros(len(args[3]),3,32,32))
    generate_split(entries=entries,text=None,unet=None,schedule=None,directory=tmp_path,batch_size=8,device=torch.device('cpu'))
    assert calls==[['b']] and expected_complete(entries,tmp_path)

def test_production_ema_loader_never_uses_raw(tmp_path):
    text,unet=TextConditioningEncoder(),ConditionalTinyUNet(dropout=0.0)
    raw_text={k:torch.zeros_like(v) for k,v in text.state_dict().items()}; raw_unet={k:torch.zeros_like(v) for k,v in unet.state_dict().items()}
    ema_text={k:torch.ones_like(v) if v.is_floating_point() else v.clone() for k,v in text.state_dict().items()}; ema_unet={k:torch.ones_like(v) if v.is_floating_point() else v.clone() for k,v in unet.state_dict().items()}
    path=tmp_path/'m.pt'; torch.save({'progress':{'step':5000},'text_encoder_state_dict':raw_text,'conditional_unet_state_dict':raw_unet,'ema_state_dict':{'text_encoder':ema_text,'conditional_unet':ema_unet}},path)
    loaded_text,loaded_unet=load_ema_checkpoint(path,expected_step=5000,device=torch.device('cpu'))
    assert torch.equal(loaded_text.token_embedding.weight,ema_text['token_embedding.weight']) and not torch.equal(loaded_text.token_embedding.weight,raw_text['token_embedding.weight'])
    assert torch.equal(next(loaded_unet.parameters()),ema_unet[next(iter(dict(unet.named_parameters())))])
    with pytest.raises(ValueError): load_ema_checkpoint(path,expected_step=10000,device=torch.device('cpu'))
