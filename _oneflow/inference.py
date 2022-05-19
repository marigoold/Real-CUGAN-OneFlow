from time import time
from model import UpScalarGraph
import cv2
import argparse
from pathlib import Path
import oneflow as flow
import oneflow.tensorrt
# from oneflow.contrib import *
import oneflow.profiler
from utils import np2flow, flow2np

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help="the path of input image")
    parser.add_argument('--scale',
                        type=int,
                        choices=[2, 3, 4],
                        help="the upscale ratio, must be one of (2, 3, 4)")
    parser.add_argument('--denoise', help="the level of denoise")
    parser.add_argument('--fp16',
                        action="store_true",
                        help="whether to use amp inference")
    parser.add_argument('--tensorrt',
                        action="store_true",
                        help="whether to use tensorrt")
    parser.add_argument('--output', help="the path of output image")

    args = parser.parse_args()

    upscaler = UpScalarGraph(
        scale=args.scale,
        denoise=args.denoise,
        device="cuda",
        fp16=args.fp16,
        tensorrt=args.tensorrt,
        conv_cudnn_search=False,
    )

    if not Path(args.input).exists():
        raise FileExistsError("The input image doesn't exist! ")
    image = np2flow(cv2.imread(args.input), "cuda")

    # for i in range(10):
    result = flow2np(upscaler(image))
    
    flow.tensorrt.write_int8_calibration("./int8_calibration") # int8_calibration目录需要手动创建
    # flow.tensorrt.cache_int8_calibration() # int8_calibration目录需要手动创建


    # oneflow.profiler.range_push('training_steps')
    # # oneflow.profiler.profiler_start()
    # start_time = time()
    # for i in range(100):
    #     result = flow2np(upscaler(image))
    # print(f"{time() - start_time:.4f}s")
    # oneflow.profiler.range_pop()
    # # oneflow.profiler.profiler_stop()

    cv2.imwrite(args.output, result)
