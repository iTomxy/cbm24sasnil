import os, re, random, math, socket, time, timeit, functools
import packaging.version
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn.functional as nnF


def get_palette(n_classes, pil_format=True):
    """ Returns the color map for visualizing the segmentation mask.
    Example:
        ```python
        palette = get_palette(n_classes, True)
        seg_mask = seg_model(image) # int, [H, W], in [0, n_classes]
        seg_img = PIL.Image.fromarray(seg_mask)
        seg_img.putpalette(palette)
        seg_img.convert("RGB").save("seg.jpg")
        ```
    Args:
        n_classes: int, number of classes
        pil_format: bool, whether in format suitable for `PIL.Image.putpalette`.
            see: https://pillow.readthedocs.io/en/stable/reference/ImagePalette.html
    Returns:
        palette: [(R_i, G_i, B_i)] if `pil_format` is False, or
            [R1, G1, B1, R2, G2, B2, ...] if `pil_format` is True
    """
    n = n_classes
    palette = [0] * (n * 3)
    for j in range(0, n):
        lab = j
        palette[j * 3 + 0] = 0
        palette[j * 3 + 1] = 0
        palette[j * 3 + 2] = 0
        i = 0
        while lab:
            palette[j * 3 + 0] |= (((lab >> 0) & 1) << (7 - i))
            palette[j * 3 + 1] |= (((lab >> 1) & 1) << (7 - i))
            palette[j * 3 + 2] |= (((lab >> 2) & 1) << (7 - i))
            i += 1
            lab >>= 3

    if pil_format:
        return palette

    res = []
    for i in range(0, len(palette), 3):
        res.append(tuple(palette[i: i+3]))
    return res


def color_seg(label, n_classes=0, palette=None):
    """put colour on the segmentation mask
    Input:
        label: [H, W], int numpy.ndarray
        n_classes: int, num of classes (including background), inferred from `label` if not provided.
        palette: [Ri, Gi, Bi] for i = 1, ..., n. (returned from `get_palette`) Use it if provided.
    Output:
        label_rgb: [H, W, 3], PIL.Image
    """
    if n_classes < 1:
        n_classes = math.ceil(np.max(label)) + 1
    if palette is not None:
        assert len(palette) >= 3 * n_classes
    label_rgb = Image.fromarray(label.astype(np.int32)).convert("L")
    label_rgb.putpalette(get_palette(n_classes) if palette is None else palette)
    return label_rgb.convert("RGB")


def blend_seg(image, label, n_classes=0, alpha=0.7, rescale=False, transparent_bg=True, save_file=""):
    """blend image & pixel-level label/prediction
    Input:
        image: [H, W] or [H, W, C], int numpy.ndarray, in [0, 255]
        label: [H, W], int numpy.ndarray
        n_classes: int, num of classes (including background), inferred from `label` if not provided
        alpha: float in (0, 1)
        rescale: bool, normalise & scale to [0, 255] if True
        transparent_bg: bool, don't colour (i.e. use original image pixel value for) background pixels if True
        save_file: str, path to save the blended image
    Output:
        blended_image: PIL.Image
    """
    if rescale:
        denom = image.max() - image.min()
        if 0 != denom:
            image = (image - image.min()) / denom * 255
        image = np.clip(image, 0, 255).astype(np.uint8)
    img_pil = Image.fromarray(image).convert("RGB")
    lab_pil = color_seg(label, n_classes)
    blended_image = Image.blend(img_pil, lab_pil, alpha)
    if transparent_bg:
        blended_image = Image.fromarray(np.where(
            (0 == label)[:, :, np.newaxis],
            np.asarray(img_pil),
            np.asarray(blended_image)
        ))
    if save_file:
        blended_image.save(save_file)
    return blended_image


def seed_everything(seed=42):
    """pytorch version seed everything
    https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def human_time(seconds, prec=1):
    """transfer seconds to human readable time string
    Input:
        - seconds: float, time in second
        - prec: int, decimal precision of second to show
    Output:
        - str
    """
    prec = max(0, prec)
    seconds = round(seconds, prec)
    minutes = int(seconds // 60)
    hours = minutes // 60
    days = hours // 24

    seconds %= 60
    minutes %= 60
    hours %= 24

    str_list = []
    if days > 0:
        str_list.append("{:d}d".format(days))
    if hours > 0:
        str_list.append("{:d}h".format(hours))
    if minutes > 0:
        str_list.append("{:d}m".format(minutes))
    if seconds > 0:
        str_list.append("{0:.{1}f}s".format(seconds, prec))

    return ' '.join(str_list) if len(str_list) > 0 else "0s"


class tic_toc:
    """timer with custom message"""

    def __init__(self, message="time used"):
        self.msg = message

    def __enter__(self):
        self.tic = timeit.default_timer()

    def __exit__(self, exc_type, exc_val, exc_tb):
        n_second = timeit.default_timer() - self.tic
        print("{}:".format(self.msg), human_time(n_second))

    def __call__(self, f):
        """enable to use as a context manager
        ```python
        @tic_toc("foo")
        def bar:
            pass
        ```
        https://stackoverflow.com/questions/9213600/function-acting-as-both-decorator-and-context-manager-in-python
        """
        @functools.wraps(f)
        def decorated(*args, **kwargs):
            with self:
                return f(*args, **kwargs)
        return decorated


def infiter(it):
    """wrap an iterator to be infinitely looping"""
    while True:
        for x in it:
            yield x


def compact_image_grid(image_list, exact=False):
    """adaptively arrange images in a compactest 2D grid (for better visualisation)
    Input:
        image_list: list of images in format of [h, w] or [h, w, c] numpy.ndarray
        exact: bool, subjest to #grids = #images or not.
            If False, #grids > #images may happen for a more compact view.
    Output:
        grid: [H, W] or [H, W, c], compiled images
    """
    n = len(image_list)
    if 1 == n:
        return image_list[0]

    # max image resolution
    max_h, max_w = 0, 0
    for im in image_list:
        h, w = im.shape[:2]
        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # find compactest layout
    nr, nc = 1, n
    min_peri = nr * max_h + nc * max_w # 1 row
    for r in range(2, n + 1):
        if exact and n % r != 0:
            continue
        c = math.ceil(n / r)
        assert r * c >= n and r * (c - 1) <= n
        peri = r * max_h + c * max_w
        if peri < min_peri:
            nr, nc, min_peri = r, c, peri
    assert nr * nc >= n

    grid_shape = (nr * max_h, nc * max_w) + image_list[0].shape[2:]
    grid = np.zeros(grid_shape, dtype=image_list[0].dtype)
    for i, img in enumerate(image_list):
        r, c = i // nc, i % nc
        h, w = img.shape[:2]
        grid[r*max_h: r*max_h+h, c*max_w: c*max_w+w] = img

    return grid


def bin_seg_err_map(*, y, pred):
    """visaulise error map of binary segmentation prediction
    y: int[H, W], in {0, 1}, label
    pred: int[H, W], in {0, 1}, prediction
    """
    err_map_palette = [0, 0, 0] + [0, 255, 0] + [255, 0, 0] + [0, 0, 255] # tn(k), tp(g), fp(r), fn(b)
    err_map = np.zeros_like(y)
    err_map[(0 == pred) & (0 == y)] = 0 # tn, balcK
    err_map[(1 == pred) & (1 == y)] = 1 # tp, Green
    err_map[(1 == pred) & (0 == y)] = 2 # fp, Red
    err_map[(0 == pred) & (1 == y)] = 3 # fn, Blue
    return np.asarray(color_seg(err_map, palette=err_map_palette))#, dtype=np.uint8)


if "__main__" == __name__:
    p = free_port()
    print(p, type(p)) # int
