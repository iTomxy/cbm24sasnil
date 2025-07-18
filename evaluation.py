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
# from data import get_bone_data_files, get_bone_nii_list, BoneDataset, BoneDatasetNii


class SegEvaluator:
    """numpy.ndarray based segmentation evaluation for semantic segmentation."""

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
    DISTANCE_BASED = ("hd", "assd", "hd95", "asd")

    def __init__(self, n_classes, bg_classes, ignore_classes=[], select=[]):
        """
        Input:
            n_classes: int, length of the softmax logit vector.
                For semantic/instance segmentation, this is the number of all classes.
                For part segmentation, this is the total number of all part categories from all object classes.
            bg_classes: int or List[int], class ID of the background class/es
                (or similar classes for all uncategorised classes).
                Typically, it is class 0.
            ignore_classes: int or List[int], ID of class/es to be ignored in evaluation.
            select: List[str], name list of metrics of interest
                Provide if you only want to evaluate on these selected metrics
                instead of all supported (see METRICS).
        """
        self.n_classes = n_classes
        if isinstance(bg_classes, int):
            bg_classes = (bg_classes,)
        self.bg_classes = bg_classes
        if isinstance(ignore_classes, int):
            ignore_classes = (ignore_classes,)
        self.ignore_classes = ignore_classes

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
        for metr in self.DISTANCE_BASED:
            if metr in self.metrics:
                self.records[f"empty_gt_{metr}"] = [0] * self.n_classes
                self.records[f"empty_pred_{metr}"] = [0] * self.n_classes

    def __call__(self, *, pred, y, spacing=None):
        """evaluates 1 prediction
        Input:
            pred: int numpy.ndarray, prediction (class ID after argmax) of one datum, not a batch
            y: same as `pred`, label (ground-truth class ID) of this datum
            spacing: float[] = None, len(spacing) = pred.ndim
        """
        for c in range(self.n_classes):
            B_pred_c = (pred == c).astype(np.int64)
            B_c      = (y == c).astype(np.int64)
            pred_l0, pred_inv_l0, gt_l0, gt_inv_l0 = B_pred_c.sum(), (1 - B_pred_c).sum(), B_c.sum(), (1 - B_c).sum()
            for metr, fn in self.metrics.items():
                is_distance_metr = metr in self.DISTANCE_BASED
                # if 0 == c and (self.ignore_bg or is_distance_metr):
                if c in self.ignore_classes or (is_distance_metr and c in self.bg_classes):
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
                    if is_distance_metr:
                        a = fn(B_pred_c, B_c, voxelspacing=spacing)
                    else:
                        a = fn(B_pred_c, B_c)
                    # except:
                    #     a = np.nan

                self.records[metr][c].append(a)

    def load_from_dict(self, vw_dict):
        """Useful when aggregating volume-wise results to an overall one.
        Assumes the dict structure to be as follows:
        {
            "<METRIC>_cw": List[float]
            "<other keys>": Any
        }
        Only keys of format `<METRIC>_cw` are used, while other keys are ignored.
        """
        for metr in vw_dict:
            if not metr.endswith("_cw") or metr.startswith("empty_"): # only use class-wise records
                continue
            cw_list = vw_dict[metr]
            assert len(cw_list) == self.n_classes
            metr = metr[:-3] # remove "_cw"
            for c, v in enumerate(cw_list):
                if c in self.ignore_classes or (metr in self.DISTANCE_BASED and c in self.bg_classes):
                    # always ignore bg for distance metrics
                    self.records[metr][c].append(np.nan)
                else:
                    self.records[metr][c].append(v)

    def reduce(self, prec=4):
        """calculate class-wise & overall average
        Input:
            prec: int, decimal precision
        Output:
            res: dict
                - res[<metr>]: float, overall average
                - res[<metr>_cw]: List[float], class-wise average of each class
                - res[empty_pred|gt_<metr>]: int, overall #NaN caused by empty pred/label
                - res[empty_pred|gt_<metr>_cw]: List[int], class-wise #NaN
        """
        res = {}
        for metr in self.records:
            if metr.startswith("empty_"):
                res[metr+"_cw"] = self.records[metr]
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
                res[f"{metr}_cw"] = np.round(cls_avg, prec).tolist()

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
    """implemented with MONAI
    Assuming `0` to be the background class.
    """

    DISTANCE_BASED = ("hd", "assd", "hd95", "asd")

    def __init__(self, n_classes, ignore_bg=False, select=[]):
        """ignore_bg: bool, for NON-distance-based metrics"""
        self.n_classes = n_classes
        self.overall_metrics, self.clswise_metrics = self.get_metrics(n_classes, not ignore_bg, select)
        self.reset()

    def get_metrics(self, n_classes, include_bg=True, select=[]):
        """instantiate MONAI segmentation measurements
        include_bg: bool, for NON-distance-based metrics
        Ref:
            - https://blog.csdn.net/HackerTom/article/details/133382705
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

    def reset(self):
        for k in self.overall_metrics:
            self.overall_metrics[k].reset()
        for k in self.clswise_metrics:
            self.clswise_metrics[k].reset()

    def __call__(self, *, pred, y, spacing=None):
        """evaluates 1 prediction
        Input:
            pred: torch.Tensor, typically [H, W] or [H, W, L], predicted class ID
            y: same as `pred`, label, ground-truth class ID
            spacing: float[] = None, len(spacing) = pred.ndim
        """
        if pred.dim() == 1:
            # NOTE Vectors of shape [L] are NOT natually supported (by distance-based metrics?).
            # I guess this is because it does not form a object surface.
            pred, y = pred.unsqueeze(1), y.unsqueeze(1) # [L] -> [L, 1], pretending [H, W]

        pred = one_hot(pred.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        y = one_hot(y.unsqueeze(0).unsqueeze(0), num_classes=self.n_classes, dim=1) # -> (B=1, C, H, W[, L])
        for k in self.overall_metrics:
            if k in self.DISTANCE_BASED:
                self.overall_metrics[k](y_pred=pred, y=y, spacing=spacing)
            else:
                self.overall_metrics[k](y_pred=pred, y=y)

        for k in self.clswise_metrics:
            if k in self.DISTANCE_BASED:
                self.clswise_metrics[k](y_pred=pred, y=y, spacing=spacing)
            else:
                self.clswise_metrics[k](y_pred=pred, y=y)

    def reduce(self, prec=4):
        res = {}
        for k in self.overall_metrics:
            # try:
            res[k] = self.metric_get_off(self.overall_metrics[k].aggregate(), prec)
            # except:
            #     print(k, type(self.overall_metrics[k]))

        for k in self.clswise_metrics:
            r = self.metric_get_off(self.clswise_metrics[k].aggregate(), prec)
            # Ensure the class-wise metrics are in list format
            # to be consistent with `Evaluator` above.
            if isinstance(r, (float, int)):
                r = [r]

            # In case of background(0) is excluded,
            # prepend a `0` to ensure the the length equals to #classes.
            # This assumes `0` is the background class.
            if len(r) != self.n_classes:
                assert len(r) == self.n_classes - 1
                r = [0] + r

            res[k+"_cw"] = r

        return res

    def metric_get_off(self, res, prec=6):
        """convert MONAI evaluation results to JSON-serializable format"""
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
def pred_volume(model, loader):
    """calculate prediction of a nii volume & its label volume
    Input:
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

    # Transpose so that the axis order, (LR, AP, IS), is preserved.
    pred_vol = np.vstack(pred_list).transpose(1, 2, 0)
    pred_vol = (1 == pred_vol).astype(np.uint8) # take care of unknown class (2)
    label_vol = np.vstack(label_list).transpose(1, 2, 0)
    # print(pred_vol.shape, label_vol.shape)

    return label_vol, pred_vol


def adjust_spacing(original_shape, new_shape, original_spacing):
    """Adjust the spacing according to resizing (assuming the axis order is perserved).
    Input:
        original_shape: int[3]
        new_shape: int[3]
        original_spacing: float[3]
    Output:
        new_spacing: float[3]
    """
    assert len(original_shape) == len(new_shape) == len(original_spacing)
    # Convert inputs to numpy arrays
    original_shape = np.array(original_shape)
    new_shape = np.array(new_shape)
    # Calculate scaling factors for each axis
    scale_factors = original_shape.astype(float) / new_shape.astype(float)
    # Calculate new spacing: new_spacing = original_spacing * scale_factor
    new_spacing = original_spacing * scale_factors
    return new_spacing


# def test_3d(args, model, dataset, split, trfm, bin_fn=None):
def test_3d(args, model, vol_dset_iterator):
    """test/validate on a subset (multiple volumes)
    Input:
        args: argparse.Namespace
        model: UnetAndClf
        vol_dset_iterator: iterator that yields a VolumeDataset for each volume id of a subset
    Output:
        res: dict, overall & class-wise metrics
        vol_wise_res: List[dict], overall & class-wise metrics of each volume
    """
    # image_nii_list, label_nii_list, _, vol_id_list = get_bone_nii_list(dataset, split, data_root=args.data_root)
    # print("#test data:", len(image_nii_list))
    model.eval()

    # class-wise evaluator
    if "monai" == args.eval_type:
        evaluator_vol = SegEvaluatorMonai(args.n_classes, ignore_bg=not args.include_bg, select=args.metrics)
    else:
        evaluator_vol = SegEvaluator(args.n_classes, 0, [] if args.include_bg else [0], select=args.metrics)
    # overall evaluator
    evaluator = SegEvaluator(args.n_classes, 0, [] if args.include_bg else [0], select=args.metrics)

    vol_wise_res = [] # results of each volume
    # for img_nii_f, lab_nii_f, vol_id in zip(image_nii_list, label_nii_list, vol_id_list):
    #     print(vol_id, end='\r')
    #     val_ds = BoneDatasetNii(img_nii_f, lab_nii_f, args.slice_axis, trfm, bin_fn,
    #         window=args.window, window_level=args.window_level, window_width=args.window_width)
    for vid, val_ds in vol_dset_iterator:
        print(vid, end='\r')
        loader = torch.utils.data.DataLoader(val_ds, batch_size=32, shuffle=False, pin_memory=torch.cuda.is_available())
        label_vol, pred_vol = pred_volume(model, loader)
        if "monai" == args.eval_type:
            pred_vol = torch.LongTensor(pred_vol)
            label_vol = torch.LongTensor(label_vol)
        # evaluator(pred=pred_vol, y=label_vol)
        # volume-wise result
        evaluator_vol(pred=pred_vol, y=label_vol, spacing=adjust_spacing(val_ds.shape, label_vol.shape, val_ds.spacing))
        vw_res = {"volume": vid, "metrics": evaluator_vol.reduce()}
        vol_wise_res.append(json.dumps(vw_res)) # will be written in one line
        evaluator_vol.reset() # reset for each volume
        evaluator.load_from_dict(vw_res["metrics"])
        del pred_vol, label_vol
        if args.debug:
            break

    res = evaluator.reduce()
    return res, vol_wise_res
