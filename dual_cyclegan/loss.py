# -*- coding: utf-8 -*-

"""Loss criteria modules.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel


def stft(x, fft_size, hop_size, win_length, window, center=True, onesided=True):
    """Perform STFT and convert to magnitude spectrogram.

    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.

    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).

    """
    x_stft = torch.stft(
        x,
        fft_size,
        hop_size,
        win_length,
        window,
        center=center,
        onesided=onesided,
        return_complex=False,
    )
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    return torch.sqrt(torch.clamp(real**2 + imag**2, min=1e-7)).transpose(2, 1)


class MelSpectralLoss(nn.Module):
    """Mel-spectral loss module."""

    def __init__(
        self,
        sample_rate,
        fft_size=1024,
        hop_size=120,
        win_length=1024,
        window="hann_window",
        n_mels=80,
        fmin=0,
        fmax=None,
    ):
        """Initialize MelSpectralLoss module.

        Args:
            sample_rate (int): Sampling rate.
            n_fft (int): FFT points.
            hop_length (int): Hop length.
            win_length (Optional[int]): Window length.
            window (str): Window type.
            n_mels (int): Number of mel basis.
            fmin (Optional[int]): Minimum frequency for mel-filter bank.
            fmax (Optional[int]): Maximum frequency for mel-filter bank.

        """
        super(MelSpectralLoss, self).__init__()
        self.fft_size = fft_size
        self.hop_size = hop_size
        self.win_length = win_length if win_length is not None else fft_size
        self.register_buffer("window", getattr(torch, window)(win_length))
        melmat = librosa_mel(
            sr=sample_rate,
            n_fft=fft_size,
            n_mels=n_mels,
            fmin=fmin,
            fmax=fmax if fmax is not None else sample_rate / 2,
        ).T
        self.register_buffer("melmat", torch.from_numpy(melmat).float())

    def wave_to_log_mel(self, x):
        """Calculate log mel-spectrogram from waveform.

        Args:
            x (Tensor): waveform tensor (B, 1, T).

        Returns:
            Tensor: Log mel-spectrogram.

        """
        x_mag = stft(x, self.fft_size, self.hop_size, self.win_length, self.window)
        x_mel = torch.matmul(x_mag, self.melmat)

        return torch.log(torch.clamp(x_mel, min=1e-7))

    def forward(self, x, y):
        """Calculate mel-spectral loss.

        Args:
            x (Tensor): Generated waveform tensor (B, 1, T).
            y (Tensor): Groundtruth waveform tensor (B, 1, T).

        Returns:
            Tensor: Loss value.

        """
        x = x.squeeze(1)
        y = y.squeeze(1)

        mel_loss = F.l1_loss(
            self.wave_to_log_mel(x),
            self.wave_to_log_mel(y),
        )

        return mel_loss


class AdversarialLoss(nn.Module):
    """Generative adversarial networks loss module."""

    def __init__(
        self,
        average_by_discriminators=False,
        loss_type="mse",
    ):
        """Initialize AdversarialLoss module."""
        super(AdversarialLoss, self).__init__()
        self.average_by_discriminators = average_by_discriminators
        assert loss_type in ["mse", "hinge"], f"{loss_type} is not supported."
        if loss_type == "mse":
            self.adv_criterion = self._mse_adv_loss
            self.real_criterion = self._mse_real_loss
            self.fake_criterion = self._mse_fake_loss
        else:
            self.adv_criterion = self._hinge_adv_loss
            self.real_criterion = self._hinge_real_loss
            self.fake_criterion = self._hinge_fake_loss

    def forward(self, p_fakes, p_reals=None):
        """Calculate forward propagation.

        Args:
            p_fakes (list): List of discriminator outputs for fake samples.
            p_reals (list): List of discriminator outputs for real samples.

        Returns:
            Tensor: Loss values.

        """
        if p_reals is None:
            # generator adversarial loss
            if isinstance(p_fakes, (tuple, list)):
                adv_loss = 0.0
                for p_fake in p_fakes:
                    adv_loss += self.adv_criterion(p_fake)
                if self.average_by_discriminators:
                    adv_loss /= len(p_fakes)
            else:
                adv_loss = self.adv_criterion(p_fakes)
            return adv_loss
        else:
            # discriminator adversarial loss
            if isinstance(p_fakes, (tuple, list)):
                fake_loss, real_loss = 0.0, 0.0
                for p_fake, p_real in zip(p_fakes, p_reals):
                    fake_loss += self.fake_criterion(p_fake)
                    real_loss += self.real_criterion(p_real)
                if self.average_by_discriminators:
                    fake_loss /= len(p_fakes)
                    real_loss /= len(p_reals)
            else:
                fake_loss = self.fake_criterion(p_fakes)
                real_loss = self.real_criterion(p_reals)
            return fake_loss, real_loss

    def _mse_adv_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_real_loss(self, x):
        return F.mse_loss(x, x.new_ones(x.size()))

    def _mse_fake_loss(self, x):
        return F.mse_loss(x, x.new_zeros(x.size()))

    def _hinge_adv_loss(self, x):
        return -x.mean()

    def _hinge_real_loss(self, x):
        return -torch.mean(torch.min(x - 1, x.new_zeros(x.size())))

    def _hinge_fake_loss(self, x):
        return -torch.mean(torch.min(-x - 1, x.new_zeros(x.size())))


class FeatMatchLoss(nn.Module):
    """Feature matching loss module."""

    def __init__(self, average_by_layers=False):
        """Initialize FeatMatchLoss module."""
        super(FeatMatchLoss, self).__init__()
        self.average_by_layers = average_by_layers

    def forward(self, fmap_fakes, fmap_reals):
        """Calcualate feature matching loss.

        Args:
            fmap_fakes (list): List of list of discriminator outputs
                calcuated from generater outputs.
            fmap_reals (list): List of list of discriminator outputs
                calcuated from groundtruth samples.

        Returns:
            Tensor: Loss value.

        """
        fm_loss = 0.0
        for fm_fake, fm_real in zip(fmap_fakes, fmap_reals):
            fm_loss += F.l1_loss(fm_fake, fm_real.detach())
        if self.average_by_layers:
            fm_loss /= len(fmap_fakes)

        return fm_loss
