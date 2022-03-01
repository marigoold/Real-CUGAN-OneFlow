import imp
import torch
import oneflow
import numpy as np
import time
from time import time as ttime
frame = np.random.randint(0, 255, size=[256, 256, 3])

frame=np.transpose(frame, (2, 0, 1))
oneflow.tensor(np.transpose(frame, (2, 0, 1))).unsqueeze(0).to("cuda") #/ 255.
frame_oneflow=oneflow.Tensor(frame)
frame_oneflow.cpu().numpy()
frame_oneflow.unsqueeze(0).to("cuda")
t0 = ttime()
for _ in range(1000):
    frame_oneflow=oneflow.from_numpy(frame)
frame_oneflow.cpu().numpy()
t1 = ttime()
print("oneflow use synthetic data : ",  t1 - t0)

torch.from_numpy(frame).unsqueeze(0).to("cuda").half() / 255

frame_torch= torch.from_numpy(frame)

frame_torch.cpu().numpy()
frame_torch.unsqueeze(0).to("cuda").half() / 255
torch.cuda.synchronize()
t0 = ttime()
for _ in range(1000):
    frame_torch= torch.from_numpy(frame)
torch.cuda.synchronize() 
t1 = ttime()
print("torch use synthetic data : ",  t1 - t0)