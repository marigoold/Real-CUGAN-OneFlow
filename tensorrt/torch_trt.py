import torch
from torch2trt import torch2trt
from upcunet_v3 import *
import time
import cv2
import sys
from time import time as ttime

path_module="../torch/weights_v3/up2x-latest-denoise3x.pth"
model = UpCunet2x()
weight = torch.load(path_module)
model.load_state_dict(weight)
model.eval().cuda()

# create example data
x = torch.rand(1, 3, 256, 256).cuda()
y = model(x)
t0 = ttime()
for _ in range(1000):
    with torch.no_grad():
        y = model(x)
y.cpu().numpy()    
t1 = ttime()
print( "done", t1 - t0)


# # # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])



# y_trt = model_trt(x)
# y = model(x)
# # # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))

# torch.save(model_trt.state_dict(), 'x2.pth')

from torch2trt import TRTModule
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('x2.pth'))

x = torch.ones((1, 3, 256, 256)).cuda()
for _ in range(10):
    y_trt = model_trt(x)
t0 = ttime()
for _ in range(1000):
    y_trt = model_trt(x)
t1 = ttime()
print( "done", t1 - t0)