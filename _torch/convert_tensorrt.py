import torch
from torch2trt import torch2trt
from upcunet_v3 import get_cunet

model = get_cunet(scale=2, denoise="3").cuda()

# create example data
x = torch.ones((1, 3, 128, 128)).cuda()

# convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x], max_batch_size=1)

y = model(x)
y_trt = model_trt(x)

# check the output against PyTorch
print(y)
print(y_trt)
print(torch.max(torch.abs(y - y_trt)))
