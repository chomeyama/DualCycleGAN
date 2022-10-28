# -*- coding: utf-8 -*-

"""Dataset module.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import os

import soundfile as sf
from hydra.utils import to_absolute_path
from torch.utils.data import Dataset


class TrainAudioDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        audio_list_L,
        audio_list_H,
        audio_second_threshold=0.5,
        sample_rate_threshold_L=16000,
        sample_rate_threshold_H=48000,
        return_utt_id=False,
    ):
        """Initialize dataset.

        Args:
            audio_list_L (str): Path to file listing paths to low-resolution audios.
            audio_list_H (str): Path to file listing paths to high-resolution audios.
            audio_second_threshold (float): Threshold to remove short audios.
            sample_rate_threshold_L (float): Minimum sampling rate of low-resolution audios.
            sample_rate_threshold_H (float): Minimum sampling rate of high-resolution audios.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of audio files
        with open(audio_list_L, "r") as f:
            audio_files_L = [file_path.replace("\n", "") for file_path in f.readlines()]
        with open(audio_list_H, "r") as f:
            audio_files_H = [file_path.replace("\n", "") for file_path in f.readlines()]

        # filter by threshold
        idxs1, idxs2 = [], []
        for idx, file_path in enumerate(audio_files_L):
            audio, sr = sf.read(to_absolute_path(file_path))
            if (
                sample_rate_threshold_L <= sr
                and audio_second_threshold <= audio.shape[0] / sr
            ):
                idxs1.append(idx)
        for idx, file_path in enumerate(audio_files_H):
            audio, sr = sf.read(to_absolute_path(file_path))
            if (
                sample_rate_threshold_H <= sr
                and audio_second_threshold <= audio.shape[0] / sr
            ):
                idxs2.append(idx)

        self.audio_files_L = [audio_files_L[idx] for idx in idxs1]
        self.audio_files_H = [audio_files_H[idx] for idx in idxs2]
        self.utt_ids1 = [
            os.path.splitext(os.path.basename(f))[0] for f in self.audio_files_L
        ]
        self.utt_ids2 = [
            os.path.splitext(os.path.basename(f))[0] for f in self.audio_files_H
        ]
        self.return_utt_id = return_utt_id

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only when return_utt_id = True).
            ndarray: Audio signal (T, ).
            int: Sampling rate.

        """
        # allow duplication for the lessor
        idx1 = idx % len(self.audio_files_L)
        idx2 = idx % len(self.audio_files_H)
        utt_id1 = self.utt_ids1[idx1]
        utt_id2 = self.utt_ids2[idx2]
        audio1, sr1 = sf.read(self.audio_files_L[idx1])
        audio2, sr2 = sf.read(self.audio_files_H[idx2])

        if self.return_utt_id:
            items = (utt_id1, audio1, sr1), (utt_id2, audio2, sr2)
        else:
            items = (audio1, sr1), (audio2, sr2)

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return max(len(self.audio_files_L), len(self.audio_files_H))


class InferAudioDataset(Dataset):
    """PyTorch compatible audio and mel dataset."""

    def __init__(
        self,
        audio_list,
        sample_rate_threshold=16000,
        return_utt_id=False,
    ):
        """Initialize dataset.

        Args:
            audio_list (str): Path to file listing paths to input audios.
            sample_rate_threshold (float): Minimum sampling rate of input audios.
            return_utt_id (bool): Whether to return the utterance id with arrays.

        """
        # find all of audio files
        with open(audio_list, "r") as f:
            audio_files = [file_path.replace("\n", "") for file_path in f.readlines()]

        # filter by threshold
        idxs = []
        for idx, file_path in enumerate(audio_files):
            _, sr = sf.read(to_absolute_path(file_path))
            if sample_rate_threshold <= sr:
                idxs.append(idx)

        self.audio_files = [audio_files[idx] for idx in idxs]
        self.utt_ids = [
            os.path.splitext(os.path.basename(f))[0] for f in self.audio_files
        ]
        self.return_utt_id = return_utt_id

    def __getitem__(self, idx):
        """Get specified idx items.

        Args:
            idx (int): Index of the item.

        Returns:
            str: Utterance id (only when return_utt_id = True).
            ndarray: Audio signal (T,).
            int: Sampling rate.

        """
        utt_id = self.utt_ids[idx]
        audio, sr = sf.read(self.audio_files[idx % len(self.audio_files)])

        if self.return_utt_id:
            items = utt_id, audio, sr
        else:
            items = audio, sr

        return items

    def __len__(self):
        """Return dataset length.

        Returns:
            int: The length of dataset.

        """
        return len(self.audio_files)
