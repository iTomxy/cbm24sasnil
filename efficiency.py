import argparse, os, json, time, timeit, socket
import torch
from thop import profile
from modules import build_model
from data import ts_spinelspelvic
from data.aug import val_trfm
from util import gpus_type, calc_stat

"""
Measure the efficiency of SASN-IL.
- FPS
- FLOPS

Notes
- My framework uses the same architecture as all naive baselines,
i.e. UNet with a segmentation head. So the they share the same efficiency
measurement result.
- The choice of dataset shall not affect the measurement. Here I just
use the test set of spineLSpelvic-small for testing.
"""

@torch.no_grad()
def measure_fps(args):
    # build model & augmentation, same as ./test.py
    model = build_model(args).cuda()
    val_trans = val_trfm(args)

    ds = ts_spinelspelvic.SliceDataset("test", "full", transform=val_trans,
        window=args.window, window_level=args.window_level, window_width=args.window_width, data_root=args.data_root)
    loader = torch.utils.data.DataLoader(ds, batch_size=1) # ensure bs=1

    sec_per_slice = []
    for i, batch in enumerate(loader):
        print(i, end='\r')
        images = batch["image"]

        tic = timeit.default_timer()
        model(images.cuda())
        sec_per_slice.append(timeit.default_timer() - tic)
        if len(sec_per_slice) >= args.max_n:
            break

    fps_list = [1.0 / x for x in sec_per_slice]
    with open("log/fps.json", "w") as f:
        json.dump({
            "when": time.asctime(time.gmtime()),
            "#slices": len(sec_per_slice),
            "hostname": socket.gethostname(), # on which machine
            "gpu": gpus_type(), # use what GPU
            "fps": calc_stat(fps_list),
            "second_per_slice": calc_stat(sec_per_slice),
        }, f, indent=1)


@torch.no_grad()
def measure_flops(args):
    # build model & augmentation, same as ./test.py
    model = build_model(args).cuda()
    val_trans = val_trfm(args)

    ds = ts_spinelspelvic.SliceDataset("test", "full", transform=val_trans,
        window=args.window, window_level=args.window_level, window_width=args.window_width, data_root=args.data_root)
    loader = torch.utils.data.DataLoader(ds, batch_size=1) # ensure bs=1

    for batch in loader:
        images = batch["image"]
        flops, params = profile(model, inputs=[images.cuda()])
        break

    # print(type(flops), type(params))
    with open("log/flops.json", "w") as f:
        json.dump({
            "when": time.asctime(time.gmtime()),
            "flops": flops,
            "#params": params,
        }, f, indent=1)


if "__main__" == __name__:
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--log_path', type=str, default="log/totalseg-spineLSpelvic-small/2nd-re_stu",
        help="use the saved config.json in it to build model & data augmentation.")
    parser.add_argument('-d', '--data_root', type=str, default="~/data/totalsegmentator")
    parser.add_argument('-n', '--max_n', type=int, default=10000,
        help="estimate inference time by averaging that of no more than `max_n` slices.")
    args = parser.parse_args()

    config_f = os.path.join(args.log_path, "config.json")
    with open(config_f, "r") as f:
        saved_args = argparse.Namespace(**json.load(f))

    saved_args.__dict__.update(args.__dict__)
    measure_fps(saved_args)
    measure_flops(saved_args)
