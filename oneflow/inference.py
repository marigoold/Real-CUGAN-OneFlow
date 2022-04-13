from upcunet_v3 import RealWaifuUpScaler
import cv2
import argparse
from pathlib import Path
import oneflow as flow

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
    upscaler = RealWaifuUpScaler(
        scale=args.scale,
        denoise=args.denoise,
        # weight_path=f"weights/oneflow/up{args.scale}x-latest-denoise3x",
        half=args.fp16,
        device="cuda:0",
        real_data=True,
        graph=True,
        # pretrained=True,
        profile=False,
        conv_cudnn_search=False,
    )

    if not Path(args.input).exists():
        raise FileExistsError("The input image doesn't exist! ")
    image = cv2.imread(args.input)
    result = upscaler(image, tile_mode=0)
    cv2.imwrite(args.output, result)
