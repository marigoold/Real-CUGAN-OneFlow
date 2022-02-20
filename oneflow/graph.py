import oneflow as flow
import oneflow.nn as nn



class EvalGraph(flow.nn.Graph):
    def __init__(self, model,fp16=True):
        super().__init__()
        self.config.allow_fuse_add_to_output(True)
        self.config.allow_fuse_model_update_ops(True)
        self.model = model
        if fp16:
            self.config.enable_amp(True)

    def build(self, image):
        logits = self.model(image)
        return logits
