import os, random, csv, json, glob
import numpy as np
import cv2
from PIL import Image
import skimage
import medpy.io as medio
import nibabel as nib
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from util import seed_everything
from config import TOTALSEG_CLS_SET, AGGREGATE_MODE


def get_bone_data_files(dataset, subset, data_root="~/data"):
    assert subset in ("training", "test", "validation"), subset
    data_list_path = "datalist"
    split_csv_f = os.path.join(data_list_path, f"datalist_{dataset}_{subset}.csv")
    npz_list = []
    if os.path.isfile(split_csv_f):
        with open(split_csv_f, "r") as f:
            for line in csv.DictReader(f):
                npz_list.append(line["npz"])

        return npz_list

    print("make data list:", dataset, subset)
    os.makedirs(data_list_path, exist_ok=True)
    if "verse19" == dataset:
        data_path = os.path.expanduser(os.path.join(data_root, "verse/processed-verse19-slice-is"))
        for vid in os.listdir(os.path.join(data_path, subset)):
            npz_list.extend(glob.glob(os.path.join(data_path, subset, vid, "*.npz")))
    elif "ctpelvic1k" == dataset:
        data_path = os.path.expanduser(os.path.join(data_root, "ctpelvic1k"))
        split_f = os.path.join("datalist", f"splitting-{dataset}.json")
        slice_p = os.path.join(data_path, "processed-ctpelvic1k-slice-is")
        with open(split_f, "r") as f:
            split_dict = json.load(f)["splitting"]
        for sub_dset in split_dict:
            for vid in split_dict[sub_dset][subset]:
                npz_list.extend(glob.glob(os.path.join(data_path, vid, "*.npz")))
    elif dataset.startswith("totalseg-"):
        sub_dset = dataset.split('-')[1]
        data_path = os.path.expanduser(os.path.join(data_root, "totalsegmentator"))
        sub_dset_path = os.path.expanduser(os.path.join(data_path, sub_dset))
        split_f = os.path.join("datalist", f"splitting_compvol-{dataset[9:]}.json") # compatible with `tatalseg-spine-small`
        with open(split_f, "r") as f:
            split_dict = json.load(f)["splitting"]
        for vid in split_dict[subset]:
            npz_list.extend(glob.glob(os.path.join(sub_dset_path, vid, "slices_is", "*.npz")))

    assert len(npz_list) > 0
    with open(split_csv_f, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["npz"])
        writer.writeheader()
        for f in npz_list:
            writer.writerow({"npz": f})

    return npz_list


def get_bone_nii_list(dataset, subset="test", data_root="~/data"):
    """totalseg_label: bool, also include completed label inferred by Total Segmentator"""
    data_root = os.path.expanduser(data_root)
    assert os.path.isdir(data_root), data_root

    image_nii_list, label_nii_list, ts_label_nii_list, vol_id_list = [], [], [], []
    if "ctpelvic1k" == dataset:
        data_base = os.path.join(data_root, dataset)
        with open(os.path.join("datalist", f"splitting-{dataset}.json"), "r") as f:
            split_dict = json.load(f)["splitting"]
        data_dir = os.path.join(data_base, "processed-ctpelvic1k")
        ts_label_dir = os.path.join(data_base, "processed-ctpelvic1k-ts-bone-label")
        for sub_dset in split_dict:
            for fid in split_dict[sub_dset][subset]:
                image_nii_list.append(os.path.join(data_dir, fid + "_image.nii.gz"))
                label_nii_list.append(os.path.join(data_dir, fid + "_label.nii.gz"))
                ts_label_nii_list.append(os.path.join(ts_label_dir, fid + "_label_ts.nii.gz"))
                vol_id_list.append(fid)
    elif "verse19" == dataset:
        data_path = os.path.join(data_root, "verse", "processed-verse19", subset)
        ts_label_dir = os.path.join(data_root, "verse", "processed-verse19-ts-bone-label", subset)
        for f in os.listdir(data_path):
            if f.endswith("_image.nii.gz"):
                image_nii_list.append(os.path.join(data_path, f))
                label_nii_list.append(os.path.join(data_path, f[:-13] + "_label.nii.gz"))
                ts_label_nii_list.append(os.path.join(ts_label_dir, f[:-13] + "_label_ts.nii.gz"))
                vol_id_list.append(f[:-13])
    elif dataset.startswith("totalseg-"):
        data_base = os.path.join(data_root, "totalsegmentator")
        sub_dset = dataset.split('-')[1]
        with open(os.path.join("datalist", f"splitting_compvol-{dataset[9:]}.json"), "r") as f:
            split_dict = json.load(f)["splitting"]
        data_dir = os.path.join(data_base, sub_dset)
        for vid in split_dict[subset]:
            image_nii_list.append(os.path.join(data_dir, vid, "ct.nii.gz"))
            label_nii_list.append(os.path.join(data_dir, vid, "comb_label.nii.gz"))
            vol_id_list.append(vid)

        ts_label_nii_list = label_nii_list

    return image_nii_list, label_nii_list, ts_label_nii_list, vol_id_list


def std_image(image):
    """standardisation (0-1 scaling)"""
    denom = image.max() - image.min()
    if 0 == denom:
        denom = 1
    return (image - image.min()) / denom


def binarise_totalseg_label(lab, coi=TOTALSEG_CLS_SET["bone"]):
    """binarise TotalSegmentator label
    See: https://github.com/wasserth/TotalSegmentator#class-details
    lab: numpy.ndarray
    coi: set/tuple/list of int, classes ID regarded as positive
    """
    bin_lab = np.zeros_like(lab, dtype=np.uint8)
    for c in np.unique(lab):
        if c in coi:
            bin_lab[c == lab] = 1

    return bin_lab


class BoneDataset(torch.utils.data.Dataset):
    def __init__(self, npz_list, transform=None,
        window=True, window_level=[300], window_width=[290], # window CT image
        bin_fn=None, # label binarisation function
    ):
        """binary: bool, don't differentiate bones & only keep 1 `bone` class if True
        convex_hull: bool, calculate, augment and return binary convex hull mask
        """
        self.npz_list = npz_list
        self.transform = transform
        self.bin_fn = bin_fn
        if window:
            assert len(window_level) == len(window_width) > 0
        self.window = window
        self.window_level = window_level
        self.window_width = window_width
    def __len__(self):
        return len(self.npz_list)
    def __getitem__(self, index):
        with np.load(self.npz_list[index]) as _npz:
            img = _npz["image"].astype(np.float32)
            lab = _npz["label"]

        if self.window is not None:
            img_w_list = [
                torch.FloatTensor(np.clip((img - (wl - ww)) / (2 * ww), 0, 1)).unsqueeze(0)
                for wl, ww in zip(self.window_level, self.window_width)
            ]
            img_t = img_w_list[0] if 1 == len(self.window_level) else torch.cat(img_w_list, dim=0)
        else:
            img_t = torch.FloatTensor(img).unsqueeze(0)

        if self.bin_fn is not None:
            lab_t = torch.LongTensor(self.bin_fn(lab)).unsqueeze(0)
        else:
            lab_t = torch.LongTensor(lab).unsqueeze(0)

        batch = {"image": img_t, "label": lab_t}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


class BoneDatasetNii(torch.utils.data.Dataset):
    """for test set nii file"""
    def __init__(self, image_nii, label_nii, slice_axis, transform=None,
        bin_fn=None, # label binarisation function
        window=True, window_level=[300], window_width=[290], # window CT image
    ):
        """binary: bool, don't differentiate bones & only keep 1 `bone` class if True"""
        assert os.path.isfile(image_nii), image_nii
        assert os.path.isfile(label_nii), label_nii
        assert slice_axis in (0, 1, 2)
        image = nib.load(image_nii).get_fdata().astype(np.float32)
        label = nib.load(label_nii).get_fdata().astype(np.int32)
        assert image.shape == label.shape, f"image: {image.shape}, label: {label.shape}"
        if slice_axis != 0:
            image = np.moveaxis(image, slice_axis, 0)
            label = np.moveaxis(label, slice_axis, 0)
        if window:
            assert len(window_level) == len(window_width) > 0
            img_w_list = [
                torch.FloatTensor(np.clip((image - (wl - ww)) / (2 * ww), 0, 1)).unsqueeze(1) # [n, 1, h, w]
                for wl, ww in zip(window_level, window_width)
            ]
            self.images = img_w_list[0] if 1 == len(window_level) else torch.cat(img_w_list, dim=1) # [n, c, h, w]
        else:
            self.images = torch.FloatTensor(image).float().unsqueeze(1) # [n, 1, h, w]
        if bin_fn is not None:
            self.labels = torch.LongTensor(bin_fn(label)).unsqueeze(1) # [n, 1, h, w]
        else:
            self.labels = torch.LongTensor(label).unsqueeze(1) # [n, 1, h, w]
        # print(self.images.size(), self.labels.size(), end='\r')
        self.transform = transform
    def __len__(self):
        return self.images.size(0)
    def __getitem__(self, index):
        img = self.images[index] # [c=1, h, w]
        lab = self.labels[index]
        batch = {"image": img, "label": lab}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


class MultiCompose:
    """Extension of torchvision.transforms.Compose that accepts multiple inputs
    and ensures the same random seed is applied on each of these inputs at each transforms.
    This can be useful when simultaneously transforming images & segmentation masks.

    Usage:
        ```python
        ## 1. compatible with single input (just like torchvision.transforms.Compose)
        trfm = MultiCompose([
            transforms.Resize((224, 256), transforms.InterpolationMode.BILINEAR),
            transforms.RandomAffine(30, (0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(0.1, 0.2, 0.3, 0.4)
        ])
        aug_images = trfm(images)

        ## 2. sequential style
        seq_trfm = MultiCompose([
            # interpolation: image uses `bilinear`, label uses `nearest`
            [transforms.Resize((224, 256), transforms.InterpolationMode.BILINEAR),
             transforms.Resize((224, 256), transforms.InterpolationMode.NEAREST)],
            transforms.RandomAffine(30, (0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            # apply `ColorJitter` on image but not on label (thus `None`)
            (transforms.ColorJitter(0.1, 0.2, 0.3, 0.4), None),
        ])
        # apply augmentations on both `images` and `seg_labels`
        aug_images, aug_seg_labels = seq_trfm(images, seg_labels)

        ## 3. dict style
        dict_trfm = MultiCompose([
            # interpolation: image uses `bilinear`, label uses `nearest`
            {"image": transforms.Resize((224, 256), transforms.InterpolationMode.BILINEAR),
             "label": transforms.Resize((224, 256), transforms.InterpolationMode.NEAREST)},
            transforms.RandomAffine(30, (0.1, 0.1)),
            transforms.RandomHorizontalFlip(),
            # apply `ColorJitter` on image but not on label (lack here)
            {"image": transforms.ColorJitter(0.1, 0.2, 0.3, 0.4)},
        ])
        # apply augmentations on both `images` and `seg_labels`
        res = dict_trfm({"image": images, "label": seg_labels})
        aug_images = res["image"]
        aug_seg_labels = res["label"]
        ```
    """

    # numpy.random.seed range error:
    #   ValueError: Seed must be between 0 and 2**32 - 1
    MIN_SEED = 0 # - 0x8000_0000_0000_0000
    MAX_SEED = min(2**32 - 1, 0xffff_ffff_ffff_ffff)

    def __init__(self, transforms, seed=None):
        """
        transforms: list/tuple of:
            - transform object (for all inputs)
            - embedded list/tuple/dict of transform objects (for each input)
        seed: int, always use this seed if provided (deterministic for reproducibility)
        """
        self.transforms = transforms
        self.seed = seed

    def append(self, t):
        self.transforms.append(t)

    def extend(self, ts):
        assert isinstance(ts, (tuple, list))
        for t in ts:
            self.append(t)

    def call_sequential(self, *images):
        for t in self.transforms:
            if isinstance(t, (tuple, list)):
                # `<=` allows redundant transforms
                assert len(images) <= len(t), f"#inputs: {len(images)} v.s. #transforms: {len(self.transforms)}"
            else:
                t = [t] * len(images)

            _aug_images = []
            _seed = random.randint(MultiCompose.MIN_SEED, MultiCompose.MAX_SEED) if self.seed is None else self.seed
            for _im, _t in zip(images, t):
                seed_everything(_seed)
                _aug_images.append(_im if _t is None else _t(_im))

            images = _aug_images

        if len(images) == 1:
            images = images[0]
        return images

    def call_dict(self, images):
        for t in self.transforms:
            if not isinstance(t, dict):
                t = {k: t for k in images}

            _aug_images = {}
            _seed = random.randint(MultiCompose.MIN_SEED, MultiCompose.MAX_SEED) if self.seed is None else self.seed
            for k in images:
                seed_everything(_seed)
                _aug_images[k] = t[k](images[k]) if k in t and t[k] is not None else images[k]

            images = _aug_images

        return images

    def __call__(self, *images):
        if isinstance(images[0], dict):
            assert len(images) == 1
            return self.call_dict(images[0])
        else:
            return self.call_sequential(*images)


class ResizeZoomPad:
    """resize by zooming (to keep ratio aspect) & padding (to ensure size)
    Parameter:
        size: int or (int, int)
        interpolation: str / torchvision.transforms.functional.InterpolationMode
            can be {"nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"}
    """
    def __init__(self, size, interpolation="bilinear", pad_value=0):
        if isinstance(size, int):
            assert size > 0
            self.size = [size, size]
        elif isinstance(size, (tuple, list)):
            assert len(size) == 2 and size[0] > 0 and size[1] > 0
            self.size = size

        if isinstance(interpolation, str):
            assert interpolation.lower() in {"nearest", "bilinear", "bicubic", "box", "hamming", "lanczos"}
            interpolation = {
                "nearest": F.InterpolationMode.NEAREST,
                "bilinear": F.InterpolationMode.BILINEAR,
                "bicubic": F.InterpolationMode.BICUBIC,
                "box": F.InterpolationMode.BOX,
                "hamming": F.InterpolationMode.HAMMING,
                "lanczos": F.InterpolationMode.LANCZOS
            }[interpolation.lower()]
        self.interpolation = interpolation
        self.pad_value = pad_value

    def __call__(self, image):
        """image: [..., H, W] (e.g. [H, W], [C, H, W] or [N, C, H, W]) torch.Tensor"""
        dim_h, dim_w = image.ndim - 2, image.ndim - 1
        scale_h, scale_w = float(self.size[0]) / image.size(dim_h), float(self.size[1]) / image.size(dim_w)
        scale = min(scale_h, scale_w)
        tmp_size = [ # clipping to ensure size
            min(int(image.size(dim_h) * scale), self.size[0]),
            min(int(image.size(dim_w) * scale), self.size[1])
        ]
        image = F.resize(image, tmp_size, self.interpolation)
        assert image.size(dim_h) <= self.size[0] and image.size(dim_w) <= self.size[1]
        pad_h, pad_w = self.size[0] - image.size(dim_h), self.size[1] - image.size(dim_w)
        if pad_h > 0 or pad_w > 0:
            pad_left, pad_right = pad_w // 2, (pad_w + 1) // 2
            pad_top, pad_bottom = pad_h // 2, (pad_h + 1) // 2
            image = F.pad(image, (pad_left, pad_top, pad_right, pad_bottom), self.pad_value)
        return image


def aggregate_label(label_bp, pred, mode="bbox"):
    """aggregate given partial label & prediction
    label_bp: binary partial label, [H, W], in {0, 1}
    pred: prediction, [H, W], in {0, 1}
    """
    assert mode in ("bbox", "bbox_frac", "polygon", "convex")
    pseudo = (label_bp + pred > 0).astype(np.uint8)
    if label_bp.sum() < 40:
        return pseudo
    if "bbox_frac" == mode:
        _, lab_cc, _, _ = cv2.connectedComponentsWithStats(label_bp)
        for c in np.unique(lab_cc):
            if 0 == c: # background
                continue
            supp = np.where(c == lab_cc)
            u, d = supp[0].min(), supp[0].max()
            l, r = supp[1].min(), supp[1].max()
            pseudo[u: d+1, l: r+1] = label_bp[u: d+1, l: r+1]
    elif "bbox" == mode:
        supp = np.where(label_bp > 0)
        u, d = supp[0].min(), supp[0].max()
        l, r = supp[1].min(), supp[1].max()
        pseudo[u: d+1, l: r+1] = label_bp[u: d+1, l: r+1]
    elif "convex" == mode:
        chull = skimage.morphology.convex_hull_image(label_bp)
        pseudo[chull] = label_bp[chull]
    else: # polygon
        lx, ly = np.where(label_bp > 0)
        if lx.shape[0] > 0:
            rr, cc = skimage.draw.polygon(lx, ly)
            pseudo[rr, cc] = label_bp[rr, cc]

    return pseudo


def val_trfm(args):
    """only resize"""
    if args.simple_resize:
        _resize_trfm = {
            "image": transforms.Resize(args.image_size),
            "label": transforms.Resize(args.image_size, interpolation=InterpolationMode.NEAREST),
        }
    else:
        _resize_trfm = {
            "image": ResizeZoomPad(args.image_size),
            "label": ResizeZoomPad(args.image_size, "nearest"),
        }

    return MultiCompose([_resize_trfm])


def weak_trfm(args):
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2
    init_size = [s * 2 for s in args.image_size]

    if args.simple_resize:
        _resize_trfm = {
            "image": transforms.Resize(init_size),
            "label": transforms.Resize(init_size, interpolation=InterpolationMode.NEAREST),
        }
    else:
        _resize_trfm = {
            "image": ResizeZoomPad(init_size),
            "label": ResizeZoomPad(init_size, "nearest"),
        }

    trfm1 = MultiCompose([
        _resize_trfm,
        transforms.RandomCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        {
            "image": transforms.RandomAffine(degrees=45, translate=(0.1, 0.1)),
            "label": transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
        },
    ])

    # pixel transforms for image (asserts image #channels in {1, 3})
    trfm2 = transforms.RandomChoice([
        transforms.RandomAdjustSharpness(2),
        transforms.GaussianBlur((5, 9)),
    ])

    return trfm1, trfm2


def weak_trfm2(args):
    """separate some transforms that should be inverted"""
    # common transforms for image & label
    if len(args.image_size) == 1:
        args.image_size = args.image_size * 2
    init_size = [s * 2 for s in args.image_size]

    if args.simple_resize:
        _resize_trfm = {
            "image": transforms.Resize(init_size, interpolation=InterpolationMode.BILINEAR),
            "label": transforms.Resize(init_size, interpolation=InterpolationMode.NEAREST),
        }
    else:
        _resize_trfm = {
            "image": ResizeZoomPad(init_size, "bilinear"),
            "label": ResizeZoomPad(init_size, "nearest"),
        }

    trfm1 = MultiCompose([
        _resize_trfm,
        transforms.RandomCrop(args.image_size),
    ])

    # spatial transforms for student input & teacher prediction
    trfm2 = MultiCompose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        {
            "image": transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), interpolation=InterpolationMode.BILINEAR),
            "label": transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
            "pseudo": transforms.RandomAffine(degrees=45, translate=(0.1, 0.1), interpolation=InterpolationMode.NEAREST),
        }
    ])

    # pixel transforms for image
    trfm3 = transforms.RandomChoice([
        transforms.RandomAdjustSharpness(2),
        transforms.GaussianBlur((5, 9)),
    ])

    return trfm1, trfm2, trfm3


def strong_trfm(args):
    pass
