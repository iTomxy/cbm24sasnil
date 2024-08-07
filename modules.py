import warnings
from collections import OrderedDict
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

import monai
from monai.networks.blocks.convolutions import Convolution, ResidualUnit
from monai.networks.layers.factories import Act, Norm
from monai.networks.layers.simplelayers import SkipConnection


class StegoSegHead(nn.Module):
    """`DinoFeaturizer` in https://github.com/mhamilton723/STEGO/blob/master/src/modules.py"""
    def __init__(self, feature_dim, n_classes):
        super(StegoSegHead, self).__init__()
        self.cluster1 = torch.nn.Conv2d(feature_dim, n_classes, (1, 1))
        self.cluster2 = torch.nn.Sequential(
            torch.nn.Conv2d(feature_dim, feature_dim, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(feature_dim, n_classes, (1, 1)))
    def forward(self, x):
        return self.cluster1(x) + self.cluster2(x)


class UnetAndClf(nn.Module):
    """use UNet as feature extractor & stack another classifier layer onto it"""
    def __init__(self, unet, feature_dim, n_classes, n_seg_head=1):
        super(UnetAndClf, self).__init__()
        self.unet = unet
        self.seg_heads = nn.ModuleList([StegoSegHead(feature_dim, n_classes) for _ in range(n_seg_head)])
    def forward(self, x):
        fea = self.unet(x)
        ys = [sh(fea) for sh in self.seg_heads]
        return fea, ys


class IntermediateLayerGetter:
    """wrap a torch.nn.Module to get its intermedia layer outputs
    From: https://github.com/sebamenabar/Pytorch-IntermediateLayerGetter
    Usage:
        ```python
        getter = IntermediateLayerGetter(network, {
            "<module_name>": "<return_key>",
            ...
        })
        inter_output_dict, final_output = getter(input)
        for return_key, return_value in inter_output_dict.items():
            print(return_key, return_value.size())
        ```
    """

    def __init__(self, model, return_layers):
        """
        model: torch.nn.module, the PyTorch module to call
        return_layers: dict, {<module_name>: <return_key>}
            <module_name> specifies whose output you want to get,
            <return_key> specifies how you want to call this output.
        """
        self._model = model
        print("getter model:", type(model), next(model.parameters()).device)
        self.return_layers = return_layers

    def __call__(self, *args, **kwargs):
        """
        Input:
            (the same as how you call the original module)
        Output:
            ret: OrderedDict, {<return_key>: <return_value>}
                In case a submodule is called more than once, <return_value> will be a list.
            output: tensor, final output
        """
        ret = OrderedDict()
        handles = []
        for name, new_name in self.return_layers.items():
            def hook(module, input, output, new_name=new_name):
                if new_name in ret:
                    if type(ret[new_name]) is list:
                        ret[new_name].append(output)
                    else:
                        ret[new_name] = [ret[new_name], output]
                else:
                    ret[new_name] = output

            try:
                layer = self._model.get_submodule(name)
                h = layer.register_forward_hook(hook)
            except AttributeError as e:
                raise AttributeError(f'Module {name} not found')

            handles.append(h)

        output = self._model(*args, **kwargs)

        for h in handles:
            h.remove() # removes the corresponding added hook

        return ret, output


def build_model(args):
    assert args.window and len(args.window_level) > 0
    channels = tuple(map(lambda x: 4 * x, (16, 32, 64, 128, 256)))
    unet = monai.networks.nets.UNet(
        spatial_dims=2,
        in_channels=len(args.window_level),
        out_channels=channels[0],
        channels=channels,
        strides=(2, 2, 2, 2),
        num_res_units=2,
    )
    model = UnetAndClf(unet, channels[0], args.n_classes, 1)
    return model
