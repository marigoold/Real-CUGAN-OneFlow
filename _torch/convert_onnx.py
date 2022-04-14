import torch
from upcunet_v3 import get_cunet

model = get_cunet(2, "3").cuda()

dummy_input = torch.randn(1, 3, 128, 128).cuda()
input_names = ["actual_input_1"]
output_names = ["output1"]

torch.onnx.export(model,
                  dummy_input,
                  "realcugan.onnx",
                  verbose=False,
                  input_names=input_names,
                  output_names=output_names)
