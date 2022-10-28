# -*- coding: utf-8 -*-

"""Inference script.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import os
from logging import getLogger
from os.path import join
from time import time

import hydra
import librosa
import numpy as np
import soundfile as sf
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig
from tqdm import tqdm

from dual_cyclegan.dataset import InferAudioDataset

logger = getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="infer")
def main(config: DictConfig) -> None:
    """Run inference process"""

    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Inference is performed on {device}.")

    # setup models
    model = {}
    state_dict = torch.load(config.checkpoint_path)
    for name in ["G1", "G3"]:
        model[name] = hydra.utils.instantiate(config.model[name])
        model[name].load_state_dict(state_dict["model"][name])
        model[name].remove_weight_norm()
        model[name].eval().to(device)
    logger.info(f"Loaded model parameters from {config.checkpoint_path}")

    # get dataset
    dataset = InferAudioDataset(
        audio_list=to_absolute_path(config.data.audio_list_L.eval),
        return_utt_id=True,
    )
    logger.info(f"The number of features to be decoded = {len(dataset)}.")

    # create directory to output results
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # start inference
    total_rtf = 0.0
    with torch.no_grad(), tqdm(dataset, desc="[decode]") as pbar:
        for idx, (utt_id, x, sr) in enumerate(pbar, 1):
            # setup input audio
            assert sr >= config.data.sample_rate_L
            if sr > config.data.sample_rate_L:
                x = librosa.resample(x, orig_sr=sr, target_sr=config.data.sample_rate_L)
            x = torch.FloatTensor(x).view(1, 1, -1).to(device)  # (1, 1, T)

            # perform inference
            start = time()
            y = model["G3"](model["G1"](x))
            rtf = (time() - start) / (y.size(-1) / config.data.sample_rate_H)
            pbar.set_postfix({"RTF": rtf})
            total_rtf += rtf

            # save output audio as PCM 16 bit wav file
            out_path = join(out_dir, utt_id + ".wav")
            y = np.clip(y.view(-1).cpu().numpy(), -1, 1)
            sf.write(out_path, y, config.data.sample_rate_H, "PCM_16")

    # report average RTF
    logger.info(
        f"Finished processing of {idx} utterances (RTF = {total_rtf / idx:.03f})."
    )


if __name__ == "__main__":
    main()
