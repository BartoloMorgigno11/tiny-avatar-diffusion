from __future__ import annotations
import importlib.util
from pathlib import Path
import torch

SPEC=importlib.util.spec_from_file_location("dr05_a2",Path(__file__).parents[1]/"scripts"/"dr05_diversity_metric_sanity.py"); audit=importlib.util.module_from_spec(SPEC);assert SPEC.loader is not None;SPEC.loader.exec_module(audit)

def test_bchw_transforms_have_exact_axes_ranges_and_no_input_mutation():
    source=torch.linspace(-.9,.8,4*3*32*32).reshape(4,3,32,32); before=source.clone(); color=audit.mild(source); shifted=audit.shift(source); structural=audit.structural(source)
    assert torch.equal(source,before) and all(x.shape==(4,3,32,32) and x.min()>=-1 and x.max()<=1 for x in (color,shifted,structural))
    assert torch.equal(color[:,1:],source[:,1:]) and torch.all(color[:,0]>source[:,0])
    assert torch.equal(shifted[:,:,:,0],source[:,:,:,0]) and torch.equal(shifted[:,:,:,1:],source[:,:,:,:-1]) and not torch.equal(shifted[:,:,1:,:],source[:,:,:-1,:])
    assert torch.equal(structural[:,:,:12],source[:,:,:12]) and torch.equal(structural[:,:,20:],source[:,:,20:]) and torch.equal(structural[:,:,:,:12],source[:,:,:,:12]) and torch.equal(structural[:,:,:,20:],source[:,:,:,20:])
    assert torch.all((structural[:,:,12:20,12:20]-source[:,:,12:20,12:20]).abs().sum(dim=(2,3))>0)
