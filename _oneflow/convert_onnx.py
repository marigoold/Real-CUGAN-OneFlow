from oneflow_onnx.oneflow2onnx.flow2onnx import Export
from oneflow_onnx.oneflow2onnx.util import export_onnx_model
from model import UpScalarGraph
from utils import np2flow

graph = UpScalarGraph(scale=2,
                      denoise=None,
                      device="cpu",
                      fp16=False,
                      tensorrt=False,
                      conv_cudnn_search=False)

import cv2

img = np2flow(cv2.imread("test_imgs/test-img.jpeg"), "cpu")
result = graph(img)

export_onnx_model(graph,
                  external_data=False,
                  opset=None,
                  flow_weight_dir=graph.model.weight_path,
                  onnx_model_path="/tmp",
                  dynamic_batch_size=False)