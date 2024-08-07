import argparse, os, os.path as osp, json, time
import numpy as np
import torch
from data import binarise_totalseg_label, val_trfm
# from util import *
from modules import build_model
from evaluation import test_3d
from config import test_args


args = test_args()
assert os.path.isfile(args.ckpt), args.ckpt
args.log_path = os.path.dirname(args.ckpt) or '.'
with open(os.path.join(args.log_path, "config.json"), "r") as f:
    _args = json.load(f)
    _args.update(args.__dict__)
    args.__dict__ = _args

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = build_model(args).to(device)

ckpt = torch.load(args.ckpt, map_location=device)
model.load_state_dict(ckpt[args.ckpt_key])

val_trans = val_trfm(args)

res, vol_wise_res = test_3d(args, model, args.dataset, args.split, val_trans, binarise_totalseg_label)

meta_info_dict = {
    "test_time": time.asctime(time.gmtime()),
    "epoch": ckpt.get("epoch", -1),
    "iter": ckpt.get("iter", -1),
    "global_step": ckpt.get("global_step", -1),
    "best_val_dice": ckpt.get("best_dice", -1),
    "args": args.__dict__,
}
res.update(meta_info_dict)

os.makedirs(args.log_path, exist_ok=True)
log_info = f"eval3d-{args.dataset}-{args.split}"
if "teacher" == args.ckpt_key:
    log_info += "-teacher"
log_info += "-" + osp.basename(args.ckpt).split(".pth")[0]

with open(os.path.join(args.log_path, log_info+".json"), "w") as f:
    json.dump(res, f, indent=1)

# volume-wise result
if args.vol_result:
    with open(os.path.join(args.log_path, log_info+"-volwise.json"), "w") as f:
        for s in vol_wise_res:
            f.write(s + '\n')
        f.write(json.dumps(meta_info_dict) + '\n')
