from __future__ import annotations
import pytest
from tiny_avatar_diffusion.dr04_real_ceiling_evaluation import attribute_metrics, confusion_matrix, joint_metrics, macro_psr, ood_pair_metrics, _guard_ceiling

def test_metrics_confusion_and_baselines_use_numeric_class_order():
    metric=attribute_metrics([0,1,1,2],[0,2,1,2],3,[.2,.3,.5])
    assert confusion_matrix([0,1,1,2],[0,2,1,2],3)==[[1,0,0],[0,1,1],[0,0,1]]
    assert metric["support"]==[1,2,1] and metric["accuracy"]==.75 and metric["baselines"]["uniform_random_expected_accuracy"]==pytest.approx(1/3)
    assert metric["baselines"]["majority_accuracy"]==.5 and metric["baselines"]["frequency_aware_random_expected_accuracy"]==pytest.approx(.325)

def test_joint_and_ood_pair_metrics():
    base={f"{a}_correct":True for a in ("face_shape","glasses","hair_color","face_color")}; a={**base,"source_ood":"OOD1"}; b={**base,"source_ood":"OOD1","glasses_correct":False}; assert joint_metrics([a,b])["exact_attribute_match"]==.5
    records=[a]*99+[b]
    diagnostics=ood_pair_metrics(records+[{**base,"source_ood":"OOD2"}]*100+[{**base,"source_ood":"OOD3"}]*100+[{**base,"source_ood":"OOD4"}]*100)
    assert diagnostics[0]["both_correct"]==.99 and macro_psr(diagnostics)==pytest.approx(.9975)

def test_only_real_ceiling_splits_are_allowed():
    _guard_ceiling("REAL_ID_CEILING")
    with pytest.raises(ValueError,match="Only real ceiling"):_guard_ceiling("FINAL_OOD")
    with pytest.raises(ValueError,match="Only real ceiling"):_guard_ceiling("ID_TEST")
