

# import torch
# from torch2trt import torch2trt
# from torchvision.models.alexnet import alexnet

# # create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()

# # create example data
# x = torch.ones((1, 3, 224, 224)).cuda()

# # convert to TensorRT feeding sample data as input
# model_trt = torch2trt(model, [x])


# y = model(x)
# y_trt = model_trt(x)

# # check the output against PyTorch
# print(torch.max(torch.abs(y - y_trt)))

import onnx
import onnx_tensorrt.backend as backend
import numpy as np

model = onnx.load("../torch/onnx/weights_v3.onnx")
engine = backend.prepare(model, device='CUDA:1')
input_data = np.random.random(size=(1, 3, 224, 224)).astype(np.float32)
output_data = engine.run(input_data)[0]
print(output_data)
print(output_data.shape)