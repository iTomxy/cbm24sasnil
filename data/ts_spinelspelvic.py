import os, csv, glob
import numpy as np
import nibabel as nib
import torch
from .common import binarise_label, get_split_vids as _get_split_vids

"""
TotalSegmentator subset: spineLSpelvic-small
- partial: only spine/vertebrae (training)
- full: all bones (test)
"""

# classes of new TotalSegmentator v2
# totalsegmentator/map_to_binary.py/"total"
TOTALSEG_CLS_SET = {
    "bone": frozenset(list(range(25, 50+1)) + list(range(69, 78+1)) + list(range(91, 116+1))),
    "spine": frozenset(list(range(26, 50+1))),
    "pelvic": frozenset([25, 77, 78]),
    "shoulder": frozenset(range(69, 74+1)), # humerus, scapula, clavicula
}
SPLIT_JSON = os.path.join("datalist", "splitting_compvol-spineLSpelvic-small.json")


def get_split_vids(split):
    return _get_split_vids(split, SPLIT_JSON)


def get_slice_list(subset, data_root="~/data/totalsegmentator"):
    assert subset in ("training", "test", "validation"), subset
    split_csv_f = os.path.join("datalist", f"datalist_spineLSpelvic-small_{subset}.csv")
    npz_list = []
    if os.path.isfile(split_csv_f):
        with open(split_csv_f, "r") as f:
            for line in csv.DictReader(f):
                npz_list.append(line["npz"])

        return npz_list

    print("make data list: totalseg-spineLSpelvic-small", subset)
    data_root = os.path.expanduser(data_root)
    for vid in get_split_vids(subset):
        npz_list.extend(glob.glob(os.path.join(data_root, "spineLSpelvic", vid, "slices_is", "*.npz")))

    assert len(npz_list) > 0
    with open(split_csv_f, "w", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["npz"])
        writer.writeheader()
        for f in npz_list:
            writer.writerow({"npz": f})

    return npz_list


class SliceDataset(torch.utils.data.Dataset):
    def __init__(self, split, bin_mode, transform=None,
        window=True, window_level=[300], window_width=[290], # window CT image
        data_root="~/data/totalsegmentator",
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

        if self.window is not None:
            img_w_list = [
                torch.FloatTensor(np.clip((img - (wl - ww)) / (2 * ww), 0, 1)).unsqueeze(0)
                for wl, ww in zip(self.window_level, self.window_width)
            ]
            img_t = img_w_list[0] if 1 == len(self.window_level) else torch.cat(img_w_list, dim=0)
        else:
            img_t = torch.FloatTensor(img).unsqueeze(0)

        if "partial" == self.bin_mode:
            lab = binarise_label(lab, coi=TOTALSEG_CLS_SET["spine"])
        elif "full" == self.bin_mode:
            lab = binarise_label(lab, coi=TOTALSEG_CLS_SET["bone"])

        lab_t = torch.LongTensor(lab).unsqueeze(0)
        batch = {"image": img_t, "label": lab_t}
        if self.transform is not None:
            batch = self.transform(batch)
        return batch


class VolumeDataset(torch.utils.data.Dataset):
    """for test set nii file"""
    def __init__(self, volume_id, bin_mode, transform=None,
        window=True, window_level=[300], window_width=[290], # window CT image
        data_root="~/data/totalsegmentator",
    ):
        assert bin_mode in ("partial", "full", "as-is"), bin_mode
        self.volume_id = volume_id
        data_dir = os.path.expanduser(os.path.join(data_root, "spineLSpelvic"))
        image_nii = nib.load(os.path.join(data_dir, volume_id, "ct.nii.gz"))
        label_nii = nib.load(os.path.join(data_dir, volume_id, "comb_label.nii.gz"))

        self.spacing = tuple(map(float, image_nii.header.get_zooms().tolist())) # spacing of RAS

        image = image_nii.get_fdata().astype(np.float32)
        label = label_nii.get_fdata().astype(np.int32)
        assert image.shape == label.shape, f"image: {image.shape}, label: {label.shape}"
        # Record volume shape here, BEFORE the axis moving below.
        # This will be used to adjust the spacing according to the resizing in augmentation.
        self.shape = image.shape

        # The orientation is RAS, and loading with nibabel won't change this.
        # See: https://github.com/iTomxy/data/blob/master/totalsegmentator/sieve.py
        # So slice along axis-2. Now move axis-2 to axis-0, latet slice along axis-0.
        image = np.moveaxis(image, 2, 0) # [H, W, L] -> [L, H, W]
        label = np.moveaxis(label, 2, 0)

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

        if "full" == bin_mode:
            label = binarise_label(label, coi=TOTALSEG_CLS_SET["bone"])
        elif "partial" == bin_mode:
            label = binarise_label(label, coi=TOTALSEG_CLS_SET["spine"])

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
    window=True, window_level=[300], window_width=[290], data_root="~/data/totalsegmentator"
):
    """Yield a `VolumeDataset` for each volume id in `vid_list`. Used in testing."""
    for vid in vid_list:
        yield vid, VolumeDataset(vid, bin_mode, transform, window, window_level, window_width, data_root)
