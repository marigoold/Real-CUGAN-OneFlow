import oneflow as flow
from upcunet_v3 import get_cunet


class UpScalar(flow.nn.Module):

    def __init__(self, scale, denoise, device) -> None:
        super().__init__()
        self.model, self.weight_path = get_cunet(scale,
                                                 denoise,
                                                 return_weight_path=True)
        self.model = self.model.to(device).eval()

    def forward(self, input):
        return self.model(input)


class UpScalarGraph(flow.nn.Graph):

    def __init__(self,
                 scale,
                 denoise,
                 device,
                 fp16,
                 tensorrt,
                 conv_cudnn_search=False):
        super().__init__()
        self.model = UpScalar(scale, denoise, device)
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        if fp16:
            self.config.enable_amp(True)
            if tensorrt:
                self.model.config.enable_tensorrt()
                self.model.config.tensorrt.use_fp16()

        self.config.enable_cudnn_conv_heuristic_search_algo(conv_cudnn_search)

    def build(self, input):
        output = self.model(input)
        return output