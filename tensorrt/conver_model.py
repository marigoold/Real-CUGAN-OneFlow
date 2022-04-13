

import torch
from torch2trt import torch2trt
from upcunet_v3 import *
import cv2
from time import time as ttime

path_module = "../torch/weights_v3/up2x-latest-denoise3x.pth"
model = UpCunet2x()
weight = torch.load(path_module)
model.load_state_dict(weight)
model.eval().cuda()



def np2tensor( np_frame):
    return torch.from_numpy(np.transpose(np_frame, (2, 0, 1))).unsqueeze(0).float() / 255


def tensor2np( tensor):
    return (np.transpose((tensor.data.squeeze().float()*255.0).round().clamp_(0, 255).byte().cpu().numpy(), (1, 2, 0)))


# image_path="../torch/input_dir1/2.png"
image_path="../test-img.jpeg"
image = cv2.imread(image_path)[:, :, [2, 1, 0]]
image = cv2.imread(image_path)

x=np2tensor(image).cuda()
y = model(x)
result=tensor2np(y)
cv2.imwrite("result.jpg",result)



# # create example data
# x = torch.rand(1, 3, 64, 64).cuda()
# y = model(x)
# t0 = ttime()
# for _ in range(10):
#     with torch.no_grad():
#         y = model(x)
# y.cpu().numpy()
# t1 = ttime()
# print("done", t1 - t0)



# import torch
# from torch2trt import torch2trt
# from torchvision.models.alexnet import alexnet
 
# # create some regular pytorch model...
# model = alexnet(pretrained=True).eval().cuda()
 
# create example data
x = torch.ones((1, 3, 64, 64)).cuda()

y = model(x)
                      
# # convert to TensorRT feeding sample data as input
model_trt = torch2trt(model, [x])

y_trt = model_trt(x)

# check the output against PyTorch
print(y,y_trt)
print(torch.max(torch.abs(y - y_trt)))


torch.save(model_trt.state_dict(), 'x2.pth')