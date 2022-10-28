# -*- coding: utf-8 -*-

"""Dual-CycleGAN modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

from logging import getLogger

import torch
from torch import nn
from torchaudio.transforms import Resample, Spectrogram

from dual_cyclegan.module import Conv1d, Conv1dGLU, ResidualBlock, init_weights

logger = getLogger(__name__)


class DualCycleGANGenerator(nn.Module):
    """DualCycleGAN's Generator module."""

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        num_blocks=6,
        channels=64,
        kernel_size=15,
        norm_layer="instance_norm",
        use_resample_net=False,
        resample_params={
            "orig_freq": 8000,
            "new_freq": 8000,
            "lowpass_filter_width": 151,
        },
        use_weight_norm=True,
        init="kaiming_normal",
    ):
        """Initialize DualCycleGAN Generator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            num_blocks (int): Number of residual blocks.
            channels (int): Number of hidden channels.
            kernel_size (int): Kernel size.
            norm_layer (str): Normalization layer in residual blocks.
            use_resample_net (bool): Whether to resample input signal.
            resample_params (int): Params for the resampling layer.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(DualCycleGANGenerator, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_blocks = num_blocks
        self.channels = channels
        self.kernel_size = kernel_size
        assert kernel_size % 2 == 1, "only odd kernel size is supported"
        self.use_weight_norm = use_weight_norm

        # resampling layer based on sinc-interpolation
        if use_resample_net:
            assert (
                resample_params["orig_freq"] != resample_params["new_freq"]
            ), "Same sampling rates are given though use_resample_net is True"
            self.resample_net = Resample(**resample_params)
        else:
            assert (
                resample_params["orig_freq"] == resample_params["new_freq"]
            ), "Different sampling rates are given though use_resample_net is False"
            self.resample_net = None

        # define first convolution layer
        self.first_conv = Conv1dGLU(
            in_channels,
            channels,
            kernel_size,
            norm_layer=None,
        )

        # define residual blocks
        conv_layers = []
        for block in range(num_blocks):
            dilation = 2 if block == 0 else 4
            conv_layers.append(
                ResidualBlock(
                    channels,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    norm_layer=norm_layer,
                )
            )
        self.conv_layers = nn.Sequential(*conv_layers)

        # define output layers
        self.last_conv = nn.Sequential(
            nn.ReflectionPad1d((kernel_size - 1) // 2),
            Conv1d(
                channels,
                out_channels,
                kernel_size=kernel_size,
                padding=0,
            ),
        )

        init_weights(self, init)

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signal (B, in_channels, T).

        Returns:
            Tensor: Output tensor (B, out_channels, T).

        """
        # resample input signal
        if self.resample_net is not None:
            x = self.resample_net(x)

        x = self.first_conv(x)
        x = self.conv_layers(x)
        x = self.last_conv(x)

        return x

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                logger.debug(f"Weight norm is removed from {m}.")
                nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class TimeDomainDiscriminator(nn.Module):
    """Parallel WaveGAN's discriminator module.

    "Parallel WaveGAN: A fast waveform generation model based on
    generative adversarial networks with multi-resolution spectrogram"
    https://arxiv.org/abs/1910.11480

    """

    def __init__(
        self,
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        layers=10,
        conv_channels=64,
        dilation_factor=2,
        nonlinear_activation="LeakyReLU",
        nonlinear_activation_params={"negative_slope": 0.2},
        bias=True,
        use_weight_norm=True,
    ):
        """Initialize TimeDomainDiscriminator module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            kernel_size (int): Kernel size.
            layers (int): Number of conv layers.
            conv_channels (int): Number of channels in conv layers.
            dilation_factor (int): Dilation factor. For example, if dilation_factor = 2,
                the dilation will be 2, 4, 8, ..., and so on.
            nonlinear_activation (str): Nonlinear function after each conv.
            nonlinear_activation_params (dict): Nonlinear function parameters.
            bias (bool): Whether to use bias parameter in conv.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(TimeDomainDiscriminator, self).__init__()
        assert (kernel_size - 1) % 2 == 0, "Not support even number kernel size."
        assert dilation_factor > 0, "Dilation factor must be > 0."
        self.conv_layers = nn.ModuleList()
        conv_in_channels = in_channels
        for i in range(layers - 1):
            if i == 0:
                dilation = 1
            else:
                dilation = i if dilation_factor == 1 else dilation_factor**i
                conv_in_channels = conv_channels
            padding = (kernel_size - 1) // 2 * dilation
            conv_layer = [
                Conv1d(
                    conv_in_channels,
                    conv_channels,
                    kernel_size=kernel_size,
                    padding=padding,
                    dilation=dilation,
                    bias=bias,
                ),
                getattr(nn, nonlinear_activation)(
                    inplace=True, **nonlinear_activation_params
                ),
            ]
            self.conv_layers += conv_layer

        padding = (kernel_size - 1) // 2
        self.last_conv_layer = Conv1d(
            conv_in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=bias,
        )

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signals (B, 1, C).

        Returns:
            Tensor: Output tensor (B, 1, T).
            List: Latent feature mappings of each layer.

        """
        fmaps = []
        for i, f in enumerate(self.conv_layers):
            x = f(x)
            if return_fmaps and i > 0:  # ignore the first feature map
                fmaps += [x]
        x = self.last_conv_layer(x)

        if return_fmaps:
            return x, fmaps
        else:
            return x

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, nn.Conv1d):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)


class SpecDomainDiscriminator(nn.Module):
    """NU-GAN's discriminiator module.

    "NU-GAN: High resolution neural upsampling with GAN"
    https://arxiv.org/abs/2010.11362

    """

    def __init__(
        self,
        stft_params={
            "n_fft": 512,
            "hop_length": 80,
            "win_length": 512,
            "power": 1.0,
        },
        groups=[1, 4, 16, 64, 256],
        n_layers=3,
        kernel_size=4,
        stride=2,
        bias=True,
        summarize=True,
        use_weight_norm=True,
        use_spectral_norm=False,
    ):
        """Initialize SpecDomainDiscriminator module.

        Args:
            stft_params (dict); Params for STFT.
            groups (list): List of group sizes.
            n_layers (int): Number of conv layers.
            kernel_size (int): Kernel size.
            stride (int): Stride size.
            bias (bool): Whether to use bias parameter in conv.
            summarize (bool): Whether to summarize outputs of sub-discriminators.
            use_weight_norm (bool): Whether to use weight norm.
                If set to true, it will be applied to all of the conv layers.
            use_spectral_norm (bool): Whether to use spectral norm.
                If set to true, it will be applied to all of the conv layers.

        """
        super(SpecDomainDiscriminator, self).__init__()
        self.wav2spec = Spectrogram(**stft_params)
        self.groups = groups
        self.n_layers = n_layers
        self.nets = nn.ModuleDict()
        in_channels = stft_params.n_fft // 2
        for group in groups:
            for n in range(n_layers):
                self.nets[f"{group}_{n}"] = nn.Sequential(
                    Conv1d(
                        in_channels,
                        in_channels,
                        kernel_size=kernel_size,
                        stride=stride,
                        groups=group,
                        bias=bias,
                    ),
                    nn.LeakyReLU(0.2, True),
                )
            self.nets[f"{group}_{n_layers}"] = Conv1d(
                in_channels, group, kernel_size=kernel_size
            )

        # summarize all multi-band outputs with a fully connected layer
        if summarize:
            self.summarizer = nn.Linear(sum(groups), 1, bias=bias)
        else:
            self.summarizer = None

        if use_weight_norm and use_spectral_norm:
            raise ValueError("Either use use_weight_norm or use_spectral_norm.")

        # apply weight norm
        if use_weight_norm:
            self.apply_weight_norm()

        # apply spectral norm
        if use_spectral_norm:
            self.apply_spectral_norm()

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signals (B, C, T).

        Returns:
            Tensor: Output tensor (B, C', T).
            List: Latent feature mappings of each layer.

        """
        # remove 0-th frequency components
        s = self.wav2spec(x.squeeze(1))[:, 1:]
        s = torch.log(torch.clamp(s, 1e-5))

        outs, fmaps = [], []
        for group in self.groups:
            x = s
            for n in range(self.n_layers + 1):
                x = self.nets[f"{group}_{n}"](x)
                if return_fmaps:
                    fmaps.append(x)
            outs.append(x)
        out = torch.cat(outs, dim=1)

        if self.summarizer is not None:
            out = self.summarizer(out.transpose(1, 2)).transpose(1, 2)

        if return_fmaps:
            return out, fmaps
        else:
            return out

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.utils.weight_norm(m)
                logger.debug(f"Weight norm is applied to {m}.")

        self.apply(_apply_weight_norm)

    def apply_spectral_norm(self):
        """Apply spectral normalization module from all of the layers."""

        def _apply_spectral_norm(m):
            if isinstance(m, (nn.Conv1d, nn.Linear)):
                nn.utils.spectral_norm(m)
                logger.debug(f"Spectral norm is applied to {m}.")

        self.apply(_apply_spectral_norm)


class DualCycleGANDiscriminator(nn.Module):
    """DualCycleGAN's Discriminator module.

    Multi-domain discriminators.
    Parallel WaveGAN's discrminiator is used as the time domain D.
    NU-GAN's discrminiator is used as the spectral domain D.

    """

    def __init__(
        self,
        time_params={
            "in_channels": 1,
            "out_channels": 1,
            "kernel_size": 3,
            "layers": 10,
            "conv_channels": 64,
            "dilation_factor": 1,
            "use_weight_norm": True,
        },
        spec_params={
            "stft_params": {
                "n_fft": 512,
                "hop_length": 80,
                "win_length": 512,
                "power": 1.0,
            },
            "groups": [1, 4, 16, 64, 256],
            "n_layers": 3,
            "kernel_size": 4,
            "stride": 2,
            "summarize": True,
            "use_weight_norm": True,
            "use_spectral_norm": False,
        },
    ):
        """Initialize DualCycleGANDiscriminator module.

        Args:
            time_params (dict): Params for time-domain discriminator.
            spec_params (dict): Params for spectral-domain discriminator.

        """
        super(DualCycleGANDiscriminator, self).__init__()

        self.timeD = TimeDomainDiscriminator(**time_params)
        self.specD = SpecDomainDiscriminator(**spec_params)

    def forward(self, x, return_fmaps=False):
        """Calculate forward propagation.

        Args:
            x (Tensor): Input signals (B, 1, T).

        Returns:
            List: Output scales.
            List: Latent feature mappings of each layer.

        """
        if return_fmaps:
            out1, fmaps1 = self.timeD(x, return_fmaps=return_fmaps)
            out2, fmaps2 = self.specD(x, return_fmaps=return_fmaps)
            return [out1, out2], fmaps1 + fmaps2
        else:
            out1 = self.timeD(x)
            out2 = self.specD(x)
            return [out1, out2]
