from __future__ import annotations
import hashlib, json, random
import numpy as np
import pytest, torch
from PIL import Image
from tiny_avatar_diffusion.evaluator import (ARCHITECTURE_ID, CLASS_COUNTS, AttributeEvaluator, EvaluatorDataset, PREPROCESSING_ID, better_checkpoint, checkpoint_payload, class_weight_artifact, configure_main_determinism, make_optimizer, parameter_count, preprocess, preprocess_cached, seed_everything, validate_checkpoint, weighted_loss, write_class_weights)
from tiny_avatar_diffusion.evaluator_training import restore_checkpoint
from tiny_avatar_diffusion.evaluator_training import MAIN_LOADER_CONFIG

def test_architecture_shapes_parameter_count_and_no_dropout():
    model=AttributeEvaluator(); x=torch.randn(3,3,32,32); out=model(x)
    assert {k:v.shape for k,v in out.items()}=={"face_shape":(3,7),"glasses":(3,12),"hair_color":(3,10),"face_color":(3,11)}
    assert model.representation(x).shape==(3,128) and parameter_count()==293064
    assert not any(isinstance(m,torch.nn.Dropout) for m in model.modules())

def test_preprocessing_is_deterministic_float32_and_normalized():
    image=Image.frombytes("RGB",(1,1),bytes([0,127,255])); a=preprocess(image); b=preprocess(image)
    assert a.shape==(3,32,32) and a.dtype==torch.float32 and torch.equal(a,b) and torch.isfinite(a).all()
    assert torch.allclose(a[:,0,0],torch.tensor([-1.0,127/255*2-1,1.0]))

def test_preprocessing_golden_resize_before_float_conversion():
    image=Image.frombytes("RGB",(2,2),bytes([0,0,0, 255,0,0, 0,255,0, 0,0,255]))
    expected=image.convert("RGB").resize((32,32),resample=Image.Resampling.BILINEAR)
    expected_tensor=torch.frombuffer(bytearray(expected.tobytes()),dtype=torch.uint8).reshape(32,32,3).permute(2,0,1).to(torch.float32).div(255).mul(2).sub(1)
    assert torch.equal(preprocess(image),expected_tensor)

def test_main_determinism_requires_exact_cublas_workspace(monkeypatch):
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG",raising=False)
    with pytest.raises(RuntimeError,match="CUBLAS_WORKSPACE_CONFIG"):
        configure_main_determinism()
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    configure_main_determinism()
    assert torch.are_deterministic_algorithms_enabled()

def test_frozen_main_loader_defaults():
    assert MAIN_LOADER_CONFIG == {"num_workers":4,"pin_memory":True,"persistent_workers":True,"prefetch_factor":2}

def test_cached_preprocessing_and_dataset_have_no_archive_fallback(tmp_path):
    source=Image.frombytes("RGB",(2,2),bytes(range(12))*1)
    cached=source.convert("RGB").resize((32,32),resample=Image.Resampling.BILINEAR)
    cache=tmp_path/"cache";cache.mkdir();cached.save(cache/"sample.png",format="PNG")
    manifest=tmp_path/"manifest.csv";manifest.write_text("image_id,split,face_shape,glasses,hair_color,face_color\nsample,EVAL_TRAIN,0,0,0,0\n",encoding="utf-8")
    dataset=EvaluatorDataset("EVAL_TRAIN",manifest,cache); x,y=dataset[0]
    assert torch.equal(x,preprocess(source)) and y["glasses"].item()==0 and not hasattr(dataset,"_archives")
    (cache/"sample.png").unlink()
    with pytest.raises(FileNotFoundError,match="Missing evaluator cache item"):
        dataset[0]
    bad=Image.new("RGBA",(32,32));
    with pytest.raises(ValueError,match="RGB 32x32"): preprocess_cached(bad)

def test_class_weights_are_from_frozen_train_manifest_and_stable(tmp_path):
    artifact=class_weight_artifact(); assert artifact["split_manifest_sha256"]=="09fa2cc89caf3f9b2e72d97b3a4cda16a2c7d728ef895c9111fc2c1caa558ee9"
    assert artifact["total_train_examples"]==94200
    for name,part in artifact["attributes"].items():
        assert part["class_ids"]==list(range(CLASS_COUNTS[name])) and sum(part["train_counts"])==94200 and min(part["train_counts"])>0
        assert all(weight>0 and np.isfinite(weight) for weight in part["class_weights"])
        assert abs(sum(c*w for c,w in zip(part["train_counts"],part["class_weights"]))/94200-1)<1e-12
    a=write_class_weights(output_path=tmp_path/"a.json"); b=write_class_weights(output_path=tmp_path/"b.json")
    assert a==b

def test_optimizer_loss_determinism_and_checkpoint_roundtrip(tmp_path):
    seed_everything(); model=AttributeEvaluator(); optimizer=make_optimizer(model)
    assert isinstance(optimizer,torch.optim.AdamW) and optimizer.defaults["lr"]==1e-3 and optimizer.defaults["eps"]==1e-8
    seen=[p for g in optimizer.param_groups for p in g["params"]]; assert len(seen)==len({id(p) for p in seen})
    x=torch.randn(2,3,32,32); targets={name:torch.tensor([0,1]) for name in CLASS_COUNTS}; weights={name:torch.ones(count) for name,count in CLASS_COUNTS.items()}
    loss=weighted_loss(model(x),targets,weights); loss.backward(); optimizer.step(); assert torch.isfinite(loss)
    identity={"architecture":ARCHITECTURE_ID,"preprocessing":PREPROCESSING_ID,"seed":2026,"manifest":"x","weights":"y"}; payload=checkpoint_payload(model,optimizer,1,2,1,{"epoch":1.0},identity); path=tmp_path/"x.pt";torch.save(payload,path); loaded=torch.load(path,weights_only=False)
    validate_checkpoint(loaded,identity); assert loaded["completed_epoch"]==1 and loaded["global_optimizer_step"]==2
    with pytest.raises(ValueError):validate_checkpoint(loaded,{**identity,"seed":0})
    assert better_checkpoint({"epoch":2,"val_macro_balanced_accuracy":.5,"val_mean_macro_f1":.4,"val_loss":1.0},{"epoch":1,"val_macro_balanced_accuracy":.5,"val_mean_macro_f1":.4,"val_loss":1.0}) is False

def test_resume_restores_deterministic_second_step(tmp_path):
    identity={"architecture":ARCHITECTURE_ID,"seed":2026}; x=torch.ones(2,3,32,32); y={name:torch.tensor([0,1]) for name in CLASS_COUNTS}; w={name:torch.ones(count) for name,count in CLASS_COUNTS.items()}
    seed_everything(); uninterrupted=AttributeEvaluator(); opt_a=make_optimizer(uninterrupted); weighted_loss(uninterrupted(x),y,w).backward();opt_a.step(); opt_a.zero_grad(); payload=checkpoint_payload(uninterrupted,opt_a,1,1,1,None,identity); path=tmp_path/"resume.pt";torch.save(payload,path); weighted_loss(uninterrupted(x),y,w).backward();opt_a.step()
    seed_everything(); resumed=AttributeEvaluator();opt_b=make_optimizer(resumed); restore_checkpoint(path,resumed,opt_b,identity,torch.device("cpu")); weighted_loss(resumed(x),y,w).backward();opt_b.step()
    assert all(torch.equal(a,b) for a,b in zip(uninterrupted.parameters(),resumed.parameters()))
