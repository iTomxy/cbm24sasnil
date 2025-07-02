import os, json
import numpy as np
import torch

"""
Some common functions.
"""

def binarise_label(lab, coi):
    """binarise label: {0: not in `coi`, 1: in `coi`}
    lab: numpy.ndarray
    coi: set/tuple/list of int, classes ID regarded as positive
    """
    bin_lab = np.zeros_like(lab, dtype=np.uint8)
    for c in np.unique(lab):
        if c in coi:
            bin_lab[c == lab] = 1

    return bin_lab


def get_split_vids(split, split_f):
    with open(split_f, "r") as f:
        split_dict = json.load(f)["splitting"]
    return split_dict[split]


def unlabel(label, coi):
    """erase specified class/es label by setting zero
    label: int numpy.ndarray or torch.LongTensor
    coi: List[int], class IDs to erase
    """
    if isinstance(label, np.ndarray):
        new_label = label.copy()
    elif isinstance(label, torch.Tensor):
        new_label = label.clone()
    for c in coi:
        new_label[c == label] = 0
    return new_label
