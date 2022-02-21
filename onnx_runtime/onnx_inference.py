from pyexpat import model
import onnxruntime
import onnxruntime as rt
import numpy as np
import os
import cv2
from time import time as ttime



class onnx_inference:
    def __init__(self, model_path, cpu=False):
        self.model_file = model_path
        # providers = None will use available provider, for onnxruntime-gpu it will be "CUDAExecutionProvider"
        self.providers =  ['CUDAExecutionProvider']

        self.input_mean = 0.0
        self.input_std = 255
        self.session = None
        
    # input_size is (w,h), return error message, return None if success
    def check(self, test_img=None):
        #default is cfat
        if not os.path.exists(self.model_file):
            return "model_path not exists"
        print('use onnx-model:', self.model_file)
        try:
            session = onnxruntime.InferenceSession(
                self.model_file, providers=self.providers)
        except:
            return "load onnx failed"
        input_cfg = session.get_inputs()[0]
        input_shape = input_cfg.shape
        print('input-shape:', input_shape)
        if len(input_shape) != 4:
            return "length of input_shape should be 4"

        input_name = input_cfg.name
        outputs = session.get_outputs()
        output_names = []
        for o in outputs:
            output_names.append(o.name)
        if len(output_names) != 1:
            return "number of output nodes should be 1"
        self.session = session
        self.input_name = input_name
        self.output_names = output_names

        self.crop = None
        feat = self.forward(test_img)
        return None

    def forward(self, img):
        input_size = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0/self.input_std, input_size,
                                      (self.input_mean, self.input_mean, self.input_mean), swapRB=False)
        net_out = self.session.run(self.output_names, {self.input_name:blob})[0]
        net_out=np.squeeze(net_out)
        net_out = np.transpose(net_out*255, (1, 2, 0))
        return net_out


data = np.array(np.random.randn(224, 224, 3))
img = cv2.imread("test_img/test-img.jpg")
model = onnx_inference('weights/x2_v3.onnx')
print(model.check(img))

out = model.forward(img)

t0 = ttime()
for _ in range(1000):
    result = model.forward(img)
t1 = ttime()
print("done", t1 - t0)
cv2.imwrite("out.png",out)
