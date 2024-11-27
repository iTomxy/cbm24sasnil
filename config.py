import argparse, os, time

# processed from TotalSegmentator
SUPPORTED_DATASETS = (
    "totalseg-spine", "totalseg-spine-small",
    "totalseg-pelvic", "totalseg-pelvic-small",
    "totalseg-spineLSpelvic", "totalseg-spineLSpelvic-small",
    "totalseg-spineCshoulder-small"
)

# subsets of all classes of TotalSegmentator
# classes of new TotalSegmentator v2
# totalsegmentator/map_to_binary.py/"total"
TOTALSEG_CLS_SET = {
    "bone": set(list(range(25, 50+1)) + list(range(69, 78+1)) + list(range(91, 116+1))),
    "spine": set(list(range(26, 50+1))),
    "pelvic": (25, 77, 78),
    "shoulder": list(range(69, 74+1)), # humerus, scapula, clavicula
}

# how to aggregate partial label & teacher prediction to form pseudo-label
AGGREGATE_MODE = ("bbox", "bbox_frac", "polygon", "convex")


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


def add_common_train_args(parser):
    # data
    parser.add_argument('--data_root', type=str, default=os.path.expanduser("~/sd10t"))
    # training
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--backbone_lr_coef', type=float, default=1, help="can let the backbone update slower than seg heads")
    parser.add_argument('--iter', type=int, default=100)
    parser.add_argument('--grad_clip', type=float, default=-1, help=">0 to clip gradient norm")
    parser.add_argument('-bs', '--batch_size', type=int, default=32)
    parser.add_argument('--decay_steps', type=int, nargs='+', default=[])
    parser.add_argument('-r', '--resume', type=str, default="")

    # validation
    parser.add_argument('--val_freq', type=int, default=1, help="<=0 to disable validation (and visualisation)")
    parser.add_argument('-v', '--vis', action="store_true", help="visualise prediction on validation")

    # misc
    parser.add_argument('--debug', action="store_true")
    parser.add_argument('--rm_old_ckpt', action="store_true", help="rm old epoch-wish ckpt")
    parser.add_argument('-o', '--log_path', type=str, default="log/{}".format(time.strftime("%Y%m%d-%H%M%S", time.gmtime())))


def stage1_args():
    parser = base_parser()
    add_common_train_args(parser)
    # data
    parser.add_argument('-c', '--n_classes', type=int, default=1+1)
    parser.add_argument('--slice_axis', type=int, default=2, help="slicing nii volume")
    parser.add_argument('--image_size', type=int, nargs='+', default=[224])
    parser.add_argument('--simple_resize', action="store_true", help="use torchvision.transforms.Resize instead of my ResizeZoomPad")
    parser.add_argument('-w', '--window', action="store_true", help="apply windowing")
    parser.add_argument('-wl', '--window_level', type=float, nargs='+', default=[1034])
    parser.add_argument('-ww', '--window_width', type=float, nargs='+', default=[2059])
    parser.add_argument('-p', '--partial', type=str, default='', choices=list(TOTALSEG_CLS_SET.keys()))

    # training
    parser.add_argument('--obviousness', type=int, default=0, help="if >0, only use slice with #bone pixel > this threshold for training")
    parser.add_argument('--topk_focal', type=float, default=-1, help="if >0, only pick this fraction of pixel with least loss for training")

    # find t0
    parser.add_argument('--t0_metr', type=str, default="iou", choices=("dice", "iou", "sensitivity", "specificity", "accuracy"))
    parser.add_argument('--t0_thres', type=float, default=0.85, help="tau in equation (1)")

    return parser.parse_args()


def stage2_args():
    parser = base_parser()
    add_common_train_args(parser)
    # pseudo-label
    parser.add_argument('--cl_pseudo_start', type=int, default=100, help="start iteration of using the proposed confidence learning based pseudo-label")
    parser.add_argument('--pseudo_agg_mode', type=str, default='bbox', choices=AGGREGATE_MODE)
    parser.add_argument('--resume_student', action="store_true", help="also initialise the student with the warmed-up initial teacher weight")

    # teacher
    parser.add_argument('--teacher_weight', type=str, default="", help="path to ckeckpoint of teacher from 1st stage")
    parser.add_argument('--ema', action="store_true", help="update teacher with EMA")
    parser.add_argument('--ema_start', type=int, default=100, help="start iteration of updating teacher")
    parser.add_argument('--ema_freq', type=pos_int, default=1, help="frequency of updating teacher")
    parser.add_argument('--ema_momentum', type=float, default=0.95)

    return parser.parse_args()


def test_args():
    parser = base_parser()
    parser.add_argument('ckpt', type=str)
    parser.add_argument('-s', '--split', type=str, default="test", choices=["test", "training", "validation"])
    parser.add_argument('-k', '--ckpt_key', type=str, default="model", choices=["model", "teacher"])
    parser.add_argument('--vol_result', action="store_true", help="also save volume-wise metrics")
    return parser.parse_args()
