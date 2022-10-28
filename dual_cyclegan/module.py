# -*- coding: utf-8 -*-

"""Residual block modules.

References:
    - https://github.com/r9y9/wavenet_vocoder
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

from logging import getLogger

import torch
import torch.nn as nn

logger = getLogger(__name__)


# Adapted from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
def init_weights(net, init_type="normal", init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, "weight") and (("Conv" in classname) or ("Linear" in classname)):
            if init_type == "normal":
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == "xavier_normal":
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == "kaiming_normal":
                nn.init.kaiming_normal_(m.weight.data, a=0, mode="fan_in")
            elif init_type == "orthogonal":
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    f"initialization method [{init_type}] is not implemented"
                )
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif "BatchNorm" in classname:
            # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)  # apply the initialization function <init_func>


class Conv1d(torch.nn.Conv1d):
    """Conv1d module with customized initialization."""

    def __init__(self, *args, **kwargs):
        """Initialize Conv1d module."""
        super(Conv1d, self).__init__(*args, **kwargs)

    def reset_parameters(self):
        """Reset parameters."""
        torch.nn.init.kaiming_normal_(self.weight, nonlinearity="relu")
        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0.0)


class Conv1d1x1(Conv1d):
    """1x1 Conv1d with customized initialization."""

    def __init__(self, in_channels, out_channels, bias=True):
        """Initialize 1x1 Conv1d module."""
        super(Conv1d1x1, self).__init__(
            in_channels, out_channels, kernel_size=1, padding=0, dilation=1, bias=bias
        )


class Conv1dGLU(nn.Module):
    """Gated linear unit

    Language Modeling with Gated Convolutional Networks
    https://arxiv.org/abs/1612.08083

    The implementation is based on the WaveCycleGAN paper.
    "WaveCycleGAN: Synthetic-to-natural speech waveform conversion using
    cycle-consistent adversarial networks"
    https://arxiv.org/abs/1809.10288

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=None,
        dilation=1,
        norm_layer=None,
        bias=True,
        *args,
        **kwargs,
    ):
        super(Conv1dGLU, self).__init__()
        if padding is None:
            if kernel_size % 2 == 0:
                padding = kernel_size // 2 * dilation
                self.trim_last = dilation
            else:
                padding = (kernel_size - 1) // 2 * dilation
                self.trim_last = 0
        else:
            self.trim_last = 0

        self.pad = nn.ReflectionPad1d(padding)
        self.conv = Conv1d(
            in_channels,
            out_channels * 2,
            kernel_size,
            padding=0,
            dilation=dilation,
            bias=bias,
            *args,
            **kwargs,
        )

        # Apply normalization layer to the output of conv
        if norm_layer is None:
            self.norm_layer = None
        elif norm_layer.lower() == "batch_norm":
            self.norm_layer = nn.BatchNorm1d(out_channels * 2)
        elif norm_layer.lower() == "instance_norm":
            # Enable trainable parameters by default
            self.norm_layer = nn.InstanceNorm1d(out_channels * 2, affine=True)
        else:
            logger.warning(f"{norm_layer} is not supported as norm layer.")
            self.norm_layer = None

    def forward(self, x):
        """Forward

        Args:
            x (Tensor): (B, C, T)

        Returns:
            Tensor: output

        """
        x = self.conv(self.pad(x))
        x = x[:, :, : -self.trim_last] if self.trim_last > 0 else x
        x = self.norm_layer(x) if self.norm_layer is not None else x

        a, b = x.split(x.size(1) // 2, dim=1)
        x = a * torch.sigmoid(b)

        return x


class ResidualBlock(nn.Module):
    """WaveCycleGAN2-based residual block module.

    "WaveCycleGAN2: Time-domain Neural Post-filter for Speech Waveform Generation"
    https://arxiv.org/abs/1904.02892

    """

    def __init__(
        self,
        channels=32,
        kernel_size=15,
        dilation=1,
        norm_layer="instance_norm",
    ):
        super(ResidualBlock, self).__init__()
        """Initialize ResidualBlock module.

        Args:
            channels (int): Number of hidden channels.
            kernel_size (int): Kernel size.
            dilation (int): Dilation size.
            norm_layer (str): Normalization layer in residual blocks.

        """
        self.block = nn.Sequential(
            Conv1dGLU(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                norm_layer=norm_layer,
            ),
            Conv1dGLU(
                channels,
                channels,
                kernel_size,
                dilation=dilation,
                norm_layer=norm_layer,
            ),
        )

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input (B, C, T).

        Returns:
            Tensor: Output tensor (B, C, T).

        """
        return x + self.block(x)
