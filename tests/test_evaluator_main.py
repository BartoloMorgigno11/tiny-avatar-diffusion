from __future__ import annotations
import json,pytest,torch
from torch.utils.data import TensorDataset
from tiny_avatar_diffusion.evaluator_main import canonical_config,config_hash,identity,run
def test_config_stable_and_nominal_accounting():
 c=canonical_config();assert config_hash(c)==config_hash(c) and c["nominal_batches_per_epoch"]==736 and c["expected_optimizer_steps"]==36800
def test_new_run_refuses_existing(tmp_path,monkeypatch):
 import tiny_avatar_diffusion.evaluator_main as m
 monkeypatch.setattr(m,"preflight",lambda **_: {"config":canonical_config(),"clean":True});p=tmp_path/"run";p.mkdir()
 with pytest.raises(RuntimeError,match="refuses"):run(p,train_data=object(),val_data=object())

def test_tiny_orchestration_history_checkpoints_and_resume(tmp_path,monkeypatch):
 import tiny_avatar_diffusion.evaluator_main as m
 monkeypatch.setattr(m,"preflight",lambda **_: {"config":canonical_config(),"clean":True})
 monkeypatch.setattr(m,"weights",lambda: {k:torch.ones(n) for k,n in {"face_shape":7,"glasses":12,"hair_color":10,"face_color":11}.items()})
 monkeypatch.setattr(m,"loader",lambda *a,**k:[None])
 monkeypatch.setattr(m,"train_epoch",lambda *a,**k:1.0)
 values=iter([.1,.2,.15])
 monkeypatch.setattr(m,"evaluate",lambda *a,**k:{"val_loss":1.0,"val_macro_balanced_accuracy":next(values),"val_mean_macro_f1":.1,**{f"{n}_{metric}":.1 for n in ("face_shape","glasses","hair_color","face_color") for metric in ("accuracy","macro_f1","balanced_accuracy")}})
 p=tmp_path/"run"; first=run(p,epochs=2,train_data=object(),val_data=object());assert first["best_epoch"]==2 and (p/"latest.pt").exists() and len((p/"training_history.csv").read_text().splitlines())==3
 second=run(p,resume=True,epochs=3,train_data=object(),val_data=object());assert second["epochs_completed"]==3 and len((p/"training_history.csv").read_text().splitlines())==4
