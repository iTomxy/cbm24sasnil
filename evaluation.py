import os, random, math, json
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
import medpy
import medpy.metric.binary as mmb
import torch
from monai.networks.utils import one_hot
from monai.metrics import DiceMetric, MeanIoU, GeneralizedDiceScore, ConfusionMatrixMetric, HausdorffDistanceMetric, SurfaceDistanceMetric
from data import get_bone_data_files, get_bone_nii_list, BoneDataset, BoneDatasetNii


class SegEvaluator:
    """segmentation evaluation based on medpy.metric.binary
    It will calibrate some results from medpy.metric.binary (see __call__).
    It will also record the number of NaN in distance-based metrics that are
    caused by empty prediction or label.
    Usage:
        ```python
        # 2D image evaluation
        evaluator = SegEvaluator(N_CLASSES, IGNORE_BACKGROUND)
        for images, labels in loader:
            predictions = model(images)         # [bs, c, h, w]
            predictions = predictions.argmax(1) # -> [bs, h, w]
            for pred, lab in zip(predictions, labels):
                evaluator(pred=pred, y=lab) # eval each image
        print(evaluator.reduce())
        evaluator.reset()
        ```
    """

    METRICS = {
        "dice": mmb.dc,
        "iou": mmb.jc,
        "accuracy": lambda _B1, _B2: (_B1 == _B2).sum() / _B1.size,
        "sensitivity": mmb.sensitivity,
        "specificity": mmb.specificity,
        "hd": mmb.hd,
        "assd": mmb.assd,
        "hd95": mmb.hd95,
        "asd": mmb.asd
    }

    def __init__(self, n_classes, ignore_bg=False, select=[]):
        """
        Input:
            n_classes: int, assuming 0 is background
            ignore_bg: bool, ignore background or not in reduction
            select: List[str], list of name of metrics of interest,
                i.e. will only evaluate on these selected metrics if provided.
                Should be a subset of supported metrics (see METRICS).
        """
        self.n_classes = n_classes
        self.ignore_bg = ignore_bg

        self.metrics = {}
        _no_select = len(select) == 0
        for m, f in self.METRICS.items():
            if _no_select or m in select:
                self.metrics[m] = f

        self.reset()

    def reset(self):
        # records:
        #  - records[metr][c][i] = <metr> score of i-th datum on c-th class, or
        #  - records[metr][c] = # of NaN caused by empty pred/label
        self.records = {}
        for metr in self.metrics:
            # self.records[metr] = [[]] * self.n_classes # wrong
            self.records[metr] = [[] for _ in range(self.n_classes)]
        for metr in ("hd", "assd", "hd95", "asd"):
            if metr in self.metrics:
                self.records[f"empty_gt_{metr}"] = [0] * self.n_classes
                self.records[f"empty_pred_{metr}"] = [0] * self.n_classes

    def __call__(self, *, pred, y):
        """evaluates 1 prediction
        Input:
            pred: numpy.ndarray, typically [H, W] or [H, W, L], predicted class ID
            y: same as `pred`, label, ground-truth class ID
        """
        for c in range(self.n_classes):
            B_pred_c = (pred == c).astype(np.int64)
            B_c      = (y == c).astype(np.int64)
            pred_l0, pred_inv_l0, gt_l0, gt_inv_l0 = B_pred_c.sum(), (1 - B_pred_c).sum(), B_c.sum(), (1 - B_c).sum()
            for metr, fn in self.metrics.items():
                is_distance_metr = metr in ("hd", "assd", "hd95", "asd")
                if 0 == c and (self.ignore_bg or is_distance_metr):
                    # always ignore bg for distance metrics
                    a = np.nan
                elif 0 == gt_l0 and 0 == pred_l0 and metr in ("dice", "iou", "sensitivity"):
                    a = 1
                elif 0 == gt_inv_l0 and 0 == pred_inv_l0 and "specificity" == metr:
                    a = 1
                elif is_distance_metr and pred_l0 * gt_l0 == 0: # at least one party is all 0
                    if 0 == pred_l0 and 0 == gt_l0: # both are all 0
                        # nips23a&d, xmed-lab/GenericSSL
                        a = 0
                    else: # only one party is all 0
                        a = np.nan
                        if 0 == pred_l0:
                            self.records[f"empty_pred_{metr}"][c] += 1
                        else: # 0 == gt_l0
                            self.records[f"empty_gt_{metr}"][c] += 1
                else: # normal cases or that medpy can solve well
                    # try:
                    a = fn(B_pred_c, B_c)
                    # except:
                    #     a = np.nan

                self.records[metr][c].append(a)

    def reduce(self, prec=4):
        """calculate class-wise & overall average
        Input:
            prec: int, decimal precision
        Output:
            res: dict
                - res[<metr>]: float, overall average
                - res[<metr>_clswise]: List[float], class-wise average of each class
                - res[empty_pred|gt_<metr>]: int, overall #NaN caused by empty pred/label
                - res[empty_pred|gt_<metr>_clswise]: List[int], class-wise #NaN
        """
        res = {}
        for metr in self.records:
            if metr.startswith("empty_"):
                res[metr+"_clswise"] = self.records[metr]
                res[metr] = int(np.sum(self.records[metr]))
            else:
                CxN = np.asarray(self.records[metr], dtype=np.float32)
                nans = np.isnan(CxN)
                CxN[nans] = 0
                not_nans = ~nans

                # class-wise average
                cls_n = not_nans.sum(1) # [c]
                # cls_avg = np.where(cls_n > 0, CxN.sum(1) / cls_n, 0)
                _cls_n_denom = cls_n.copy()
                _cls_n_denom[0 == _cls_n_denom] = 1 # to avoid warning though not necessary
                cls_avg = np.where(cls_n > 0, CxN.sum(1) / _cls_n_denom, 0)
                res[f"{metr}_clswise"] = np.round(cls_avg, prec).tolist()

                # overall average
                ins_cls_n = not_nans.sum(0) # [n]
                # ins_avg = np.where(ins_cls_n > 0, CxN.sum(0) / ins_cls_n, 0)
                _ins_cls_n_denom = ins_cls_n.copy()
                _ins_cls_n_denom[0 == _ins_cls_n_denom] = 1 # to avoid warning though not necessary
                ins_avg = np.where(ins_cls_n > 0, CxN.sum(0) / _ins_cls_n_denom, 0)
                ins_n = (ins_cls_n > 0).sum()
                avg = ins_avg.sum() / ins_n if ins_n > 0 else 0
                res[metr] = float(np.round(avg, prec))

        return res


class SegEvaluatorMonai:
    """implemented with MONAI"""
    def __init__(self, n_classes, ignore_bg=False, select=[]):
        """ignore_bg: bool, for NON-distance-based metrics"""
        self.n_classes = n_classes
        self.overall_metrics, self.clswise_metrics = get_metrics(n_classes, not ignore_bg, select)
        self.reset()

    def reset(self):
        for k in self.overall_metrics:
            self.overall_metrics[k].reset()
        for k in self.clswise_metrics:
            self.clswise_metrics[k].reset()

    def __call__(self, *, pred, y):
        """evaluates 1 prediction
        Input:
            pred: torch.Tensor, typically [H, W] or [H, W, L], predicted class ID
            y: same as `pred`, label, ground-truth class ID
        """
        pred = one_hot(pred.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        y = one_hot(y.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        for k in self.overall_metrics:
            self.overall_metrics[k](y_pred=pred, y=y)
        for k in self.clswise_metrics:
            self.clswise_metrics[k](y_pred=pred, y=y)

    def reduce(self, prec=4):
        res = {}
        for k in self.overall_metrics:
            res[k] = metric_get_off(self.overall_metrics[k].aggregate(), prec)
        for k in self.clswise_metrics:
            res[k+"_cw"] = metric_get_off(self.clswise_metrics[k].aggregate(), prec)
        return res


def get_metrics(n_classes, include_bg=True, select=[]):
    """https://blog.csdn.net/HackerTom/article/details/133382705
    include_bg: bool, for NON-distance-based metrics
    """
    overall_metrics = {
        "dice": DiceMetric(include_background=include_bg, reduction="mean", get_not_nans=False, ignore_empty=True, num_classes=n_classes),
        "iou": MeanIoU(include_background=include_bg, reduction="mean", get_not_nans=False, ignore_empty=True),
        "sensitivity": ConfusionMatrixMetric(include_background=include_bg, metric_name='sensitivity', reduction="mean", compute_sample=True, get_not_nans=False),
        "specificity": ConfusionMatrixMetric(include_background=include_bg, metric_name='specificity', reduction="mean", compute_sample=True, get_not_nans=False),
        "accuracy": ConfusionMatrixMetric(include_background=include_bg, metric_name='accuracy', reduction="mean", get_not_nans=False),
        # distance based metrics: always does NOT include background
        "asd": SurfaceDistanceMetric(include_background=False, symmetric=False, reduction="mean", get_not_nans=False),
        "assd": SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean", get_not_nans=False),
        "hd": HausdorffDistanceMetric(include_background=False, reduction="mean", get_not_nans=False),
        "hd95": HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean", get_not_nans=False),
    }
    clswise_metrics = {
        "dice": DiceMetric(include_background=include_bg, reduction="mean_batch", get_not_nans=False, ignore_empty=True, num_classes=n_classes),
        "iou": MeanIoU(include_background=include_bg, reduction="mean_batch", get_not_nans=False, ignore_empty=True),
        "sensitivity": ConfusionMatrixMetric(include_background=include_bg, metric_name='sensitivity', reduction="mean_batch", compute_sample=True, get_not_nans=False),
        "specificity": ConfusionMatrixMetric(include_background=include_bg, metric_name='specificity', reduction="mean_batch", compute_sample=True, get_not_nans=False),
        "accuracy": ConfusionMatrixMetric(include_background=include_bg, metric_name='accuracy', reduction="mean_batch", get_not_nans=False),
        # distance based metrics: always does NOT include background
        "asd": SurfaceDistanceMetric(include_background=False, symmetric=False, reduction="mean_batch", get_not_nans=False),
        "assd": SurfaceDistanceMetric(include_background=False, symmetric=True, reduction="mean_batch", get_not_nans=False),
        "hd": HausdorffDistanceMetric(include_background=False, reduction="mean_batch", get_not_nans=False),
        "hd95": HausdorffDistanceMetric(include_background=False, percentile=95, reduction="mean_batch", get_not_nans=False),
    }
    if len(select) > 0:
        overall_metrics = {k: v for k, v in overall_metrics.items() if k in select}
        clswise_metrics = {k: v for k, v in clswise_metrics.items() if k in select}
    return overall_metrics, clswise_metrics


def metric_get_off(res, prec=6):
    if isinstance(res, list):
        assert len(res) == 1
        res = res[0]
    if "cuda" in res.device.type:
        res = res.cpu()
    if 0 == res.ndim:
        res = round(res.item(), prec)
    else:
        res = list(map(lambda x: round(x, prec), res.tolist()))
        if len(res) == 1:
            res = res[0]
    return res


@torch.no_grad()
def pred_volume(args, model, loader):
    """calculate prediction of a nii volume & its label volume
    Input:
        args: argparse.Namespace
        model: UnetAndClf
        loader: torch.utils.data.DataLoader
        within_bbox: bool, only evaluate within bbox
    Output:
        label_vol: [H, W, L], int, in {0, ..., #c - 1}
        pred_vol: (same as `label_vol`)
    """
    device = next(model.parameters()).device
    model.eval()
    pred_list, label_list = [], []
    for batch_data in loader:
        images = batch_data["image"].to(device)
        labels = batch_data["label"]
        feats, preds = model(images)
        outputs = preds[0].argmax(1, keepdim=False).cpu().numpy() # [bs, h, w]
        labels = labels[:, 0].numpy()
        pred_list.append(outputs)
        label_list.append(labels)

    pred_vol = np.vstack(pred_list).transpose(1, 2, 0)
    pred_vol = (1 == pred_vol).astype(np.uint8) # take care of unknown class (2)
    label_vol = np.vstack(label_list).transpose(1, 2, 0)
    # print(pred_vol.shape, label_vol.shape)

    return label_vol, pred_vol


def test_3d(args, model, dataset, split, trfm, bin_fn=None):
    """test/validate on a subset (multiple volumes)
    Input:
        args: argparse.Namespace
        model: UnetAndClf
        dataset: str, one of {verse19, ctpelvic1k}
        split: str, one of {training, validation, test}
        trfm: MultiCompose
        bin_fn: label binarisation function
    Output:
        res: dict, overall & class-wise metrics
        vol_wise_res: List[dict], overall & class-wise metrics of each volume
    """
    image_nii_list, label_nii_list, _, vol_id_list = get_bone_nii_list(dataset, split, data_root=args.data_root)
    print("#test data:", len(image_nii_list))
    model.eval()

    if "monai" == args.eval_type:
        evaluator = SegEvaluatorMonai(args.n_classes, ignore_bg=not args.include_bg, select=args.metrics)
        evaluator_vol = SegEvaluatorMonai(args.n_classes, ignore_bg=not args.include_bg, select=args.metrics)
    else:
        evaluator = SegEvaluator(args.n_classes, ignore_bg=not args.include_bg, select=args.metrics)
        evaluator_vol = SegEvaluator(args.n_classes, ignore_bg=not args.include_bg, select=args.metrics)

    vol_wise_res = [] # results of each volume
    for img_nii_f, lab_nii_f, vol_id in zip(image_nii_list, label_nii_list, vol_id_list):
        print(vol_id, end='\r')
        val_ds = BoneDatasetNii(img_nii_f, lab_nii_f, args.slice_axis, trfm, bin_fn,
            window=args.window, window_level=args.window_level, window_width=args.window_width)
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        label_vol, pred_vol = pred_volume(args, model, loader)
        if "monai" == args.eval_type:
            pred_vol = torch.LongTensor(pred_vol)
            label_vol = torch.LongTensor(label_vol)
        evaluator(pred=pred_vol, y=label_vol)
        # volume-wise result
        evaluator_vol(pred=pred_vol, y=label_vol)
        vw_res = {"volume": vol_id}
        vw_res.update(evaluator_vol.reduce())
        vol_wise_res.append(json.dumps(vw_res)) # will be written in one line
        evaluator_vol.reset() # reset for each volume
        if args.debug:
            break

    res = evaluator.reduce()
    return res, vol_wise_res
