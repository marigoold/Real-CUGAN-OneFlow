import torch
from torch2trt import torch2trt
from upcunet_v3 import *
import cv2
from time import time as ttime
from torch2trt import TRTModule


def np2tensor( np_frame):
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).float() / 255


def tensor2np( tensor):
    return (np.transpose((tensor.data.squeeze().float()*255.0).round().clamp_(0, 255).byte().cpu().numpy(), (1, 2, 0)))


# image_path="../torch/input_dir1/2.png"
image_path="../test-img.jpeg"
# image = cv2.imread(image_path)[:, :, [2, 1, 0]]
image = cv2.imread(image_path)
image = cv2.resize(image, (64, 64))

x=np2tensor(image).cuda()

#x = torch.ones((1, 3, 256, 256)).cuda()
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('x2.pth'))
for _ in range(10):
    y_trt = model_trt(x)
t0 = ttime()
for _ in range(1000):
    y_trt = model_trt(x)
t1 = ttime()
print("done", t1 - t0)
result=tensor2np(y_trt)
cv2.imwrite("result.jpg",result)