from model import UpScalarGraph
import cv2
import argparse
from pathlib import Path
import oneflow as flow
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
        tensorrt=False,
        conv_cudnn_search=False,
    )

    if not Path(args.input).exists():
        raise FileExistsError("The input image doesn't exist! ")
    image = cv2.imread(args.input)

    result = flow2np(upscaler(np2flow(image, "cuda")))
    cv2.imwrite(args.output, result)
