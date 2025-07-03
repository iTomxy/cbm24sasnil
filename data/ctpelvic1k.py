import os, csv, json, glob, time, random
import numpy as np
import nibabel as nib
import torch
from .common import binarise_label
from .ts_spinelspelvic import TOTALSEG_CLS_SET
from .common import binarise_label, get_split_vids as _get_split_vids, unlabel

"""CTPelvic1K
- partial: only spine/vertebrae (training)
- full: all bones (test)

To complement the missing bone labels in original CTPelvic1K dataset,
keep all other bone predictions from the TotalSegmentator `total` subtask prediction
than sacrum and ilium (= hip), i.e. excluding `pelvic`.
"""

CTPELVIC1K_CLS_SET = {
    "background": (0,),
    "sacrum": (1,),
    "hip": (2, 3),
    "lumbar_vertebra": (5,)
}
SPLIT_JSON = os.path.join("datalist", "splitting-ctpelvic1k.json")
# classes to be excluded from TotalSegmentator prediction
CLS_EXCLUDE_FROM_TS = frozenset(list(TOTALSEG_CLS_SET["pelvic"]) + list(TOTALSEG_CLS_SET["spine"]))


def get_split_vids(split):
    return _get_split_vids(split, SPLIT_JSON)


def get_slice_list(subset, data_root="~/data/ctpelvic1k"):
    assert subset in ("training", "test", "validation"), subset
    split_csv_f = os.path.join("datalist", f"datalist_ctpelvic1k_{subset}.csv")
    npz_list = []
    if os.path.isfile(split_csv_f):
        with open(split_csv_f, "r") as f:
            for line in csv.DictReader(f):
                npz_list.append(line["npz"])

        return npz_list

    print("make data list: ctpelvic1k", subset)
    data_root = os.path.expanduser(data_root)
    for vid in get_split_vids(subset):
        print(vid, os.path.join(data_root, "slice_is", vid, "*.npz"), end='\r')
        npz_list.extend(glob.glob(os.path.join(data_root, "slice_is", vid, "*.npz")))

    assert len(npz_list) > 0
    with open(split_csv_f, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["npz"])
        writer.writeheader()
        for f in npz_list:
            writer.writerow({"npz": f})

    return npz_list


def combine_n_binarise(label, ts_label, mode):
    """combine original ctpelvic1k label & TotalSegmentator prediction"""
    assert mode in ("partial", "full", "as-is"), mode
    # map pengwin label to TotalSegmentator
    lab_p2ts = np.zeros_like(label, dtype=np.int32)
    lab_p2ts[1 == label] = 25 # sacrum
    lab_p2ts[2 == label] = 78 # right hip
    lab_p2ts[3 == label] = 77 # left hip
    lab_p2ts[4 == label] = 27 # lumbar vertebra -> vertebrae_L5
    # remove pelvic (sacrum & hip) & spine from TS prediction
    ts_lab_exc = unlabel(ts_label, CLS_EXCLUDE_FROM_TS)
    # combine
    label_comb = np.where(lab_p2ts > 0, lab_p2ts, ts_lab_exc)
    # binarise
    if "partial" == mode:
        label_comb = binarise_label(label_comb, TOTALSEG_CLS_SET["spine"])
    elif "full" == mode:
        label_comb = binarise_label(label_comb, TOTALSEG_CLS_SET["bone"])

    return label_comb


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, split, bin_mode, transform=None,
        window=True, window_level=[300], window_width=[290], # window CT image
        data_root="~/data/ctpelvic1k",
    ):
        assert split in ("training", "test", "validation"), split
        assert bin_mode in ("partial", "full", "as-is"), bin_mode
        self.npz_list = get_slice_list(split, data_root)
        self.transform = transform
        self.bin_mode = bin_mode
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
            lab_ts = _npz["total"]

        if self.window is not None:
            img_w_list = [
                torch.FloatTensor(np.clip((img - (wl - ww)) / (2 * ww), 0, 1)).unsqueeze(0)
                for wl, ww in zip(self.window_level, self.window_width)
            ]
            img_t = img_w_list[0] if 1 == len(self.window_level) else torch.cat(img_w_list, dim=0)
        else:
            img_t = torch.FloatTensor(img).unsqueeze(0)

        lab = combine_n_binarise(lab, lab_ts, self.bin_mode)
        lab_t = torch.LongTensor(lab).unsqueeze(0)
        batch = {"image": img_t, "label": lab_t}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


class VolumeDataset(torch.utils.data.Dataset):
    """for test set nii file"""
    def __init__(self, volume_id, bin_mode, transform=None,
        window=True, window_level=[300], window_width=[290], # window CT image
        data_root="~/data/ctpelvic1k",
    ):
        assert bin_mode in ("partial", "full", "as-is"), bin_mode
        self.volume_id = volume_id
        data_root = os.path.expanduser(data_root)
        image_nii = nib.load(os.path.join(data_root, "reorient_LPS", "{}_image.nii.gz".format(volume_id)))
        label_nii = nib.load(os.path.join(data_root, "reorient_LPS", "{}_label.nii.gz".format(volume_id)))
        ts_pred_nii = nib.load(os.path.join(data_root, "ts_pred", "{}-total.nii.gz".format(volume_id)))

        self.spacing = tuple(map(float, image_nii.header.get_zooms())) # spacing of LPS

        image = image_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.int32)
        ts_pred = ts_pred_nii.get_fdata().astype(np.int32)
        assert image.shape == label.shape, f"image: {image.shape}, label: {label.shape}"
        # Record volume shape here, BEFORE the axis moving below.
        # This will be used to adjust the spacing according to the resizing in augmentation.
        self.shape = image.shape

        # The orientation is RAS, and loading with nibabel won't change this.
        # See: https://github.com/iTomxy/data/blob/master/totalsegmentator/sieve.py
        # So slice along axis-2. Now move axis-2 to axis-0, latet slice along axis-0.
        image = np.moveaxis(image, 2, 0) # [H, W, L] -> [L, H, W]
        label = np.moveaxis(label, 2, 0)
        ts_pred = np.moveaxis(ts_pred, 2, 0)

        # self.raw_images_np = image # (un-windowed raw image) for canny edge
        if window:
            assert len(window_level) == len(window_width) > 0
            img_w_list = [
                torch.FloatTensor(np.clip((image - (wl - ww)) / (2 * ww), 0, 1)).unsqueeze(1) # [n, 1, h, w]
                for wl, ww in zip(window_level, window_width)
            ]
            self.images = img_w_list[0] if 1 == len(window_level) else torch.cat(img_w_list, dim=1) # [n, c, h, w]
        else:
            self.images = torch.FloatTensor(image).float().unsqueeze(1) # [n, 1, h, w]

        label = combine_n_binarise(label, ts_pred, bin_mode)
        self.labels = torch.LongTensor(label).unsqueeze(1) # [n, 1, h, w]

        print(self.images.size(), self.labels.size(), end='\r')
        self.transform = transform

    def __len__(self):
        return self.images.size(0)

    def __getitem__(self, index):
        img = self.images[index] # [c, h, w]
        lab = self.labels[index]
        batch = {"image": img, "label": lab}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


def iter_vol_dataset(
    vid_list, bin_mode, transform=None,
    window=True, window_level=[300], window_width=[290], data_root="~/data/ctpelvic1k"
):
    """Yield a `VolumeDataset` for each volume id in `vid_list`. Used in testing."""
    for vid in vid_list:
        yield vid, VolumeDataset(vid, bin_mode, transform, window, window_level, window_width, data_root)
