import oneflow as flow
import oneflow.nn as nn



class EvalGraph(flow.nn.Graph):
    def __init__(self, model,fp16=True,conv_cudnn_search=False):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.model = model
        self.config.enable_cudnn_conv_heuristic_search_algo(conv_cudnn_search)
        # self.config.enable_tensorrt(True)
        # self.graph.config.enable_xla_jit(True)
        if fp16:
            self.config.enable_amp(True)

    def build(self, image):
        logits = self.model(image)
        return logits
