from ast import arg
import torch
import oneflow as flow
from upcunet_v3 import get_cunet
from tqdm import tqdm
from pathlib import Path
import argparse


def convert(iter, model, target_dir):
    for state_dict in iter:
        torch_state_dict = torch.load(str(state_dict))
        flow_state_dict = dict()
        for key, value in torch_state_dict.items():
            if "num_batches_tracked" not in key:
                val = value.detach().cpu().numpy()
                flow_state_dict[key] = val
        model.load_state_dict(flow_state_dict)
        flow.save(model.state_dict(), str(Path(target_dir) / state_dict.stem))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-dir",
                        type=str,
                        help="the path of pytorch containing weights")

    parser.add_argument("--flow-dir",
                        type=str,
                        help="the path of flow containing weights")
    args = parser.parse_args()

    assert Path(args.torch_dir).exists() and Path(args.flow_dir).exists()

    for scale in [2, 3, 4]:
        bar = tqdm(list(Path(args.torch_dir).glob(f"up{scale}x*")),
                   desc=f"converting UpCuNet{scale}x")
        model = get_cunet(scale)
        convert(bar, model, args.flow_dir)
