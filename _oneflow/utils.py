import oneflow as flow
import numpy as np


def np2flow(np_frame, device):
    return flow.from_numpy(np.transpose(
        np_frame, (2, 0, 1))).unsqueeze(0).to(device) / 255.


def flow2np(tensor):
    return (np.transpose((tensor.data.squeeze().float() *
                          255.0).round().clamp_(0, 255).cpu().numpy(),
                         (1, 2, 0)))