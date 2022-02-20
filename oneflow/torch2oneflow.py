from email.mime import base
import torch
import oneflow as flow
from upcunet_v3 import *
import os

scale = 2
base_dir = "../torch"
srmodel = UpCunet2x()

parameters = torch.load(os.path.join(
    base_dir, 'weights_v3', 'up2x-latest-denoise3x.pth'))


new_parameters = dict()
for key, value in parameters.items():
    if "num_batches_tracked" not in key:
        val = value.detach().cpu().numpy()
        new_parameters[key] = val
srmodel.load_state_dict(new_parameters)
flow.save(srmodel.state_dict(), "weights/up2x-latest-denoise3x")
