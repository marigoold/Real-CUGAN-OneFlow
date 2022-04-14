from upcunet_v3 import get_cunet
import cv2
import argparse
from pathlib import Path
from utils import np2torch, torch2np

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
    parser.add_argument('--output', help="the path of output image")

    args = parser.parse_args()

    upscaler = get_cunet(scale=args.scale, denoise=args.denoise).cuda()

    if not Path(args.input).exists():
        raise FileExistsError("The input image doesn't exist! ")
    image = np2torch(cv2.imread(args.input), "cuda")

    if args.fp16:
        upscaler = upscaler.half().cuda()
        image = image.half()

    result = torch2np(upscaler(image).float())

    cv2.imwrite(args.output, result)
