import oneflow as flow
from model import UpScalarGraph

upscaler = UpScalarGraph(
    scale=2,
    denoise="3",
    device="cuda",
    fp16=False,
    tensorrt=False,
    conv_cudnn_search=False,
)

upscaler_trt = UpScalarGraph(
    scale=2,
    denoise="3",
    device="cuda",
    fp16=False,
    tensorrt=False,
    conv_cudnn_search=False,
)

x = flow.randn(1, 3, 128, 128).cuda()
result = upscaler(x)
result_trt = upscaler_trt(x)
print(result)
print(result_trt)
print(flow.max(flow.abs(result - result_trt)))