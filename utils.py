from torch import nn
from collections import OrderedDict
from addict import Dict
import pytorch_msssim as py_ms
from math import log10
import torch
from PIL import Image
import torchvision.utils as vutils
import numpy as np


def psnr(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return 10 * log10(1 / nn.MSELoss()(Ft_p, IFrame))
    else:
        return 10 * log10(1 / nn.MSELoss()(Ft_p, IFrame).item())


def ssim(Ft_p, IFrame, outputTensor=False):
    if outputTensor:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True)
    else:
        return py_ms.ssim(Ft_p, IFrame, data_range=1, size_average=True).item()


def make_layers(block):
    layers = []
    for layer_name, v in block.items():
        if "pool" in layer_name:
            layer = nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])
            layers.append((layer_name, layer))
        elif "deconv" in layer_name:
            transposeConv2d = nn.ConvTranspose2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4],
            )
            layers.append((layer_name, transposeConv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(
                    (
                        "leaky_" + layer_name,
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
        elif "conv" in layer_name:
            conv2d = nn.Conv2d(
                in_channels=v[0],
                out_channels=v[1],
                kernel_size=v[2],
                stride=v[3],
                padding=v[4],
            )
            layers.append((layer_name, conv2d))
            if "relu" in layer_name:
                layers.append(("relu_" + layer_name, nn.ReLU(inplace=True)))
            elif "leaky" in layer_name:
                layers.append(
                    (
                        "leaky_" + layer_name,
                        nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    )
                )
        else:
            raise NotImplementedError
    return nn.Sequential(OrderedDict(layers))


def flatten_opts(opts):
    """Flattens a multi-level addict.Dict or native dictionnary into a single
    level native dict with string keys representing the keys sequence to reach
    a value in the original argument.

    d = addict.Dict()
    d.a.b.c = 2
    d.a.b.d = 3
    d.a.e = 4
    d.f = 5
    flatten_opts(d)
    >>> {
        "a.b.c": 2,
        "a.b.d": 3,
        "a.e": 4,
        "f": 5,
    }

    Args:
        opts (addict.Dict or dict): addict dictionnary to flatten

    Returns:
        dict: flattened dictionnary
    """
    values_list = []

    def p(d, prefix="", vals=[]):
        for k, v in d.items():
            if isinstance(v, (Dict, dict)):
                p(v, prefix + k + ".", vals)
            elif isinstance(v, list):
                if v and isinstance(v[0], (Dict, dict)):
                    for i, m in enumerate(v):
                        p(m, prefix + k + "." + str(i) + ".", vals)
                else:
                    vals.append((prefix + k, str(v)))
            else:
                if isinstance(v, Path):
                    v = str(v)
                vals.append((prefix + k, v))

    p(opts, vals=values_list)
    return dict(values_list)


def text_to_array(text, width=640, height=40):
    """
    Creates a numpy array of shape height x width x 3 with
    text written on it using PIL

    Args:
        text (str): text to write
        width (int, optional): Width of the resulting array. Defaults to 640.
        height (int, optional): Height of the resulting array. Defaults to 40.

    Returns:
        np.ndarray: Centered text
    """
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (width, height), (255, 255, 255))
    try:
        font = ImageFont.truetype("UnBatang.ttf", 25)
    except OSError:
        font = ImageFont.load_default()

    d = ImageDraw.Draw(img)
    text_width, text_height = d.textsize(text)
    h = 40 // 2 - 3 * text_height // 2
    w = width // 2 - text_width
    d.text((w, h), text, font=font, fill=(30, 30, 30))
    return np.array(img)


def all_texts_to_array(texts, width=640, height=40):
    """
    Creates an array of texts, each of height and width specified
    by the args, concatenated along their width dimension

    Args:
        texts (list(str)): List of texts to concatenate
        width (int, optional): Individual text's width. Defaults to 640.
        height (int, optional): Individual text's height. Defaults to 40.

    Returns:
        list: len(texts) text arrays with dims height x width x 3
    """
    return [text_to_array(text, width, height) for text in texts]


def all_texts_to_tensors(texts, width=640, height=40):
    """
    Creates a list of tensors with texts from PIL images

    Args:
        texts (list(str)): texts to write
        width (int, optional): width of individual texts. Defaults to 640.
        height (int, optional): height of individual texts. Defaults to 40.

    Returns:
        list(torch.Tensor): len(texts) tensors 3 x height x width
    """
    arrays = all_texts_to_array(texts, width, height)
    arrays = [array.transpose(2, 0, 1) for array in arrays]
    return [torch.tensor(array) for array in arrays]


def upload_images(
    image_outputs, epoch, exp=None, im_per_row=4, rows_per_log=10, legends=[],
):
    """
    Save output image

    Args:
        image_outputs (list(torch.Tensor)): all the images to log
        im_per_row (int, optional): umber of images to be displayed per row.
            Typically, for a given task: 3 because [input prediction, target].
            Defaults to 3.
        rows_per_log (int, optional): Number of rows (=samples) per uploaded image.
            Defaults to 5.
        comet_exp (comet_ml.Experiment, optional): experiment to use.
            Defaults to None.
    """
    nb_per_log = im_per_row * rows_per_log
    n_logs = len(image_outputs) // nb_per_log + 1

    header = None
    if len(legends) == im_per_row and all(isinstance(t, str) for t in legends):
        header_width = max(im.shape[-1] for im in image_outputs)
        headers = all_texts_to_tensors(legends, width=header_width)
        header = torch.cat(headers, dim=-1)

    for logidx in range(n_logs):
        ims = image_outputs[logidx * nb_per_log : (logidx + 1) * nb_per_log]
        if not ims:
            continue
        ims = torch.stack([im.squeeze() for im in ims]).squeeze()
        image_grid = vutils.make_grid(
            ims, nrow=im_per_row, normalize=True, scale_each=True, padding=0
        )

        if header is not None:
            image_grid = torch.cat([header.to(image_grid.device), image_grid], dim=1)

        image_grid = image_grid.permute(1, 2, 0).cpu().numpy()
        exp.log_image(
            Image.fromarray((image_grid * 255).astype(np.uint8)),
            name=f"{str(epoch)}_#{logidx}",
        )


def init_weights(net, init_type="normal", init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """

    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (
            classname.find("Conv") != -1 or classname.find("Linear") != -1
        ):
            if debug:
                print(classname)
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    "initialization method [%s] is not implemented" % init_type
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif (
            classname.find("BatchNorm2d") != -1
        ):  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(
    net,
    init_type="normal",
    init_gain=0.02,
    gpu_ids=[],
    debug=False,
    initialize_weights=True,
):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert torch.cuda.is_available()
        net.to(gpu_ids[0])
        # if not amp:
        # net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs for non-AMP training
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
        print("Model weights initialized!")
    return net
