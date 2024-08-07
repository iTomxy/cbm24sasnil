import argparse, os, time

# processed from TotalSegmentator
SUPPORTED_DATASETS = (
    "totalseg-spine", "totalseg-spine-small",
    "totalseg-pelvic", "totalseg-pelvic-small",
    "totalseg-spineLSpelvic", "totalseg-spineLSpelvic-small"
)

# subsets of all classes of TotalSegmentator
TOTALSEG_CLS_SET = {
    "bone": set(list(range(25, 50+1)) + list(range(69, 78+1)) + list(range(91, 116+1))),
    "spine": set(list(range(26, 50+1))),
    "pelvic": (25, 77, 78),
}


def pos_int(v):
    """wrapper of int, raise error if `v <= 0`"""
    v = int(v)
    if v <= 0:
        raise argparse.ArgumentTypeError(f"Expect positive integer, got {v}")
    return v


def base_parser():
    """common args"""
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument('dataset', type=str, choices=SUPPORTED_DATASETS)
    # evaluation
    parser.add_argument('--include_bg', action="store_true", help="include bg (class 0) in evaluation")
    parser.add_argument('-m', '--metrics', type=str, nargs='+', default=[], help="if provided, only eval these metrics, else eval all")
    parser.add_argument('--eval_type', type=str, default="monai", choices=["monai", "medpy"])
    return parser


def init_train_args():
    parser = base_parser()
    # data
    parser.add_argument('--data_root', type=str, default=os.path.expanduser("~/sd10t"))
    parser.add_argument('-c', '--n_classes', type=int, default=1+1)
    # parser.add_argument('--n_dummy_classes', type=int, default=0, help="trick from ZhiHu")
    parser.add_argument('--slice_axis', type=int, default=2, help="slicing nii volume")
    parser.add_argument('--image_size', type=int, nargs='+', default=[224])
    parser.add_argument('--simple_resize', action="store_true", help="use torchvision.transforms.Resize instead of my ResizeZoomPad")
    parser.add_argument('-w', '--window', action="store_true", help="apply windowing")
    parser.add_argument('-wl', '--window_level', type=float, nargs='+', default=[1034])
    parser.add_argument('-ww', '--window_width', type=float, nargs='+', default=[2059])
    parser.add_argument('-p', '--partial', type=str, default='', choices=list(TOTALSEG_CLS_SET.keys()))

    # training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--backbone_lr_coef', type=float, default=1, help="can let the backbone update slower than seg heads")
    # parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=-1, help=">0 to clip gradient norm")
    # parser.add_argument('--w_full_dice', type=float, default=1)
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--decay_steps', type=int, nargs='+', default=[])
    parser.add_argument('-r', '--resume', type=str, default="")
    # parser.add_argument('--train_ts_label', action="store_true", help="train with TotalSegmentator labels (for upper-bound baseline)")
    # parser.add_argument('--train_full_label', action="store_true", help="train with full labels (for upper-bound baseline)")
    # parser.add_argument('--strong_aug', action="store_true", help="apply strong augmentation on student's input")
    parser.add_argument('--obviousness', type=int, default=0, help="if >0, only use slice with #bone pixel > this threshold for training")
    parser.add_argument('--topk_focal', type=float, default=-1, help="if >0, only pick this fraction of pixel with least loss for training")
    # parser.add_argument('--pseudo_agg_mode', type=str, default='bbox', choices=("bbox", "bbox_frac", "polygon", "convex"))
    # parser.add_argument('--update_teacher', action="store_true")
    # parser.add_argument('--teacher_weight', type=str, default="")
    # parser.add_argument('--teacher_update_iter', type=int, default=100)
    # parser.add_argument('--teacher_update_freq', type=pos_int, default=1)
    # parser.add_argument('--teacher_momentum', type=float, default=0.99)

    # find t0
    parser.add_argument('--t0_metr', type=str, default="iou", choices=("dice", "iou", "sensitivity", "specificity", "accuracy"))
    parser.add_argument('--t0_thres', type=float, default=0.85, help="tau in equation (1)")

    # validation
    parser.add_argument('--val_freq', type=int, default=1, help="<=0 to disable validation (and visualisation)")
    parser.add_argument('-v', '--vis', action="store_true", help="visualise prediction on validation")

    # evaluation
    # parser.add_argument('--eval_mask_modes', type=str, nargs='+', default=['none'], choices=EVAL_MASK_MODES)

    # misc
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--rm_old_ckpt', action="store_true", help="rm old epoch-wish ckpt")
    parser.add_argument('-o', '--log_path', type=str, default="log/{}".format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())))

    return parser.parse_args()


def test_args():
    parser = base_parser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('-s', '--split', type=str, default="test", choices=["test", "training", "validation"])
    parser.add_argument('-k', '--ckpt_key', type=str, default="model", choices=["model", "teacher"])
    parser.add_argument('--vol_result', action="store_true", help="also save volume-wise metrics")
    return parser.parse_args()
