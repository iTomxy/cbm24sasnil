This repository contains an *unofficial, partial* implementation of paper:

> Huihui Zhou, Yan Wang, Benyan Zhang, Chunhua Zhou, Maxim S. Vonsky, Lubov B. Mitrofanova, Duowu Zou, and Qingli Li. 2024. Unsupervised domain adaptation for histopathology image segmentation with incomplete labels. Comput. Biol. Med. 171, C (Mar 2024). https://doi.org/10.1016/j.compbiomed.2024.108226

I only implement its incomplete label correction stage as a baseline of incomplete label segmentation task.
Also, I have NOT tested this implementation on its original dataset.

# Backbone

The original paper uses [DAFormer](https://github.com/lhoyer/DAFormer) as its backbone,
which is develop upon [MMSegmentation](https://github.com/open-mmlab/mmsegmentation).
But it is hard to hack.
Thus I instead use UNet from [MONAI](https://github.com/Project-MONAI/MONAI) as backbone.

# Data

See [iTomxy/data/totalsegmentator](https://github.com/iTomxy/data/tree/master/totalsegmentator).

# Dependencies

- [monai](https://github.com/Project-MONAI/MONAI)
- [medpy](https://github.com/loli/medpy) >= 0.5.2
- [nibabel](https://github.com/nipy/nibabel)
- ~~[invertransforms](https://github.com/gregunz/invertransforms)~~
- [scikit-image](https://github.com/scikit-image/scikit-image)

# Train & Test

```shell
bash run.sh
```

# Citation

If you find this repository helpful,
please consider citing it as follows:

```
@software{Liang_Reimplementation_of_SASN-IL,
    author = {Liang, Tianyou},
    title = {Reimplementation of SASN-IL},
    url = {https://github.com/iTomxy/cbm24sasnil},
    version = {2024.8.17}
}
```

You can also export this BibTeX or APA format citation string via the *Cite this repository* at the right-side panel of this GitHub repository page.

You may also want to cite the original paper, see [here](https://www.sciencedirect.com/science/article/pii/S001048252400310X).
