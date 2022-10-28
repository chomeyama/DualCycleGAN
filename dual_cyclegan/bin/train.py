# -*- coding: utf-8 -*-

"""Training script.

References:
    - https://github.com/kan-bayashi/ParallelWaveGAN

"""

import itertools
import math
import os
import random
import sys
from collections import defaultdict
from logging import getLogger
from os.path import join

import hydra
import librosa
import matplotlib
import numpy as np
import torch
import torch.multiprocessing as mp
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dual_cyclegan.dataset import TrainAudioDataset

# set to avoid matplotlib error in CLI environment
matplotlib.use("Agg")
logger = getLogger(__name__)


class Trainer(object):
    """Customized trainer module for Dual-CycleGAN training."""

    def __init__(
        self,
        steps,
        epochs,
        data_loader,
        sampler,
        model,
        criterion,
        optimizer,
        scheduler,
        config,
        rank=0,
        device=torch.device("cpu"),
    ):
        """Initialize trainer.

        Args:
            steps (int): Initial global steps.
            epochs (int): Initial global epochs.
            data_loader (dict): Dict of data loaders. It must contrain "train" and "dev" loaders.
            model (dict): Dict of models. It must contrain "generator" and "discriminator" models.
            criterion (dict): Dict of criterions. It must contrain "stft" and "mse" criterions.
            optimizer (dict): Dict of optimizers.
                It must contrain "generator" and "discriminator" optimizers.
            scheduler (dict): Dict of schedulers.
                It must contrain "generator" and "discriminator" schedulers.
            config (dict): Config dict loaded from yaml format configuration file.
            device (torch.deive): Pytorch device instance.

        """
        self.steps = steps
        self.epochs = epochs
        self.data_loader = data_loader
        self.sampler = sampler
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.rank = rank
        self.device = device
        self.writer = SummaryWriter(to_absolute_path(config.out_dir))
        self.finish_train = False
        self.total_train_loss = defaultdict(float)
        self.total_eval_loss = defaultdict(float)
        self.total_grad_norm = defaultdict(float)

    def run(self):
        """Run training."""
        self.tqdm = tqdm(
            initial=self.steps, total=self.config.train.max_train_steps, desc="[train]"
        )
        while True:
            # train one epoch
            self._train_epoch()

            # check whether training is finished
            if self.finish_train:
                break

        self.tqdm.close()
        logger.info("Finished training.")

    def save_checkpoint(self, checkpoint_path):
        """Save checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be saved.

        """
        state_dict = {
            "optimizer": {
                "generator": self.optimizer["generator"].state_dict(),
                "discriminator": self.optimizer["discriminator"].state_dict(),
            },
            "scheduler": {
                "generator": self.scheduler["generator"].state_dict(),
                "discriminator": self.scheduler["discriminator"].state_dict(),
            },
            "steps": self.steps,
            "epochs": self.epochs,
        }
        state_dict["model"] = {}
        for name in ["G1", "G2", "G3", "G4", "D1", "D2", "D3"]:
            if self.config.train.n_gpus > 1:
                state_dict["model"][name] = self.model[name].module.state_dict()
            else:
                state_dict["model"][name] = self.model[name].state_dict()

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            os.makedirs(os.path.dirname(checkpoint_path))
        torch.save(state_dict, checkpoint_path)

    def load_checkpoint(self, checkpoint_path, load_only_params=False):
        """Load checkpoint.

        Args:
            checkpoint_path (str): Checkpoint path to be loaded.
            load_only_params (bool): Whether to load only model parameters.

        """
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        for name in ["G1", "G2", "G3", "G4", "D1", "D2", "D3"]:
            if self.config.train.n_gpus > 1:
                self.model[name].module.load_state_dict(state_dict["model"][name])
            else:
                self.model[name].load_state_dict(state_dict["model"][name])
        if not load_only_params:
            self.steps = state_dict["steps"]
            self.epochs = state_dict["epochs"]
            self.optimizer["generator"].load_state_dict(
                state_dict["optimizer"]["generator"]
            )
            self.optimizer["discriminator"].load_state_dict(
                state_dict["optimizer"]["discriminator"]
            )
            self.scheduler["generator"].load_state_dict(
                state_dict["scheduler"]["generator"]
            )
            self.scheduler["discriminator"].load_state_dict(
                state_dict["scheduler"]["discriminator"]
            )
        else:
            logger.info("Loaded only model parameters")

    def _pretrain_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y, z = batch
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        #######################
        #      Generator      #
        #######################
        # S_LR -> T_LR -> S_LR
        x1 = self.model["G1"](x)
        x2 = self.model["G2"](x1)
        # T_LR -> S_LR -> T_LR
        z1 = self.model["G2"](z)
        z2 = self.model["G1"](z1)
        # T_LR -> T_HR -> T_LR
        z3 = self.model["G3"](z)
        z4 = self.model["G4"](z3)
        # T_HR -> T_LR -> T_HR
        y1 = self.model["G4"](y)
        y2 = self.model["G3"](y1)

        # initialize
        fm_L_loss = None
        fm_H_loss = None
        idt_S_loss = None
        idt_T_loss = None
        idt_L_loss = None
        idt_H_loss = None
        gen_loss = 0.0

        # adversarial loss
        p_fake_S = self.model["D1"](z1)
        p_fake_T = self.model["D2"](x1)
        if self.config.train.lambda_fm > 0:
            p_fake_L, fmaps_fake_L = self.model["D2"](y1, return_fmaps=True)
            p_fake_H, fmaps_fake_H = self.model["D3"](z3, return_fmaps=True)
        else:
            p_fake_L = self.model["D2"](y1, return_fmaps=False)
            p_fake_H = self.model["D3"](z3, return_fmaps=False)
        adv_S_loss = self.criterion["adversarial"](p_fake_S)
        adv_T_loss = self.criterion["adversarial"](p_fake_T)
        adv_L_loss = self.criterion["adversarial"](p_fake_L)
        adv_H_loss = self.criterion["adversarial"](p_fake_H)
        adv_loss = adv_S_loss + adv_T_loss + adv_L_loss + adv_H_loss
        gen_loss += self.config.train.lambda_adv * adv_loss

        # feature matching loss
        if self.config.train.lambda_fm > 0:
            with torch.no_grad():
                _, fmaps_real_L = self.model["D2"](z, return_fmaps=True)
                _, fmaps_real_H = self.model["D3"](y, return_fmaps=True)
            # NOTE: the first argument must be the fake samples
            fm_L_loss = self.criterion["feat_match"](fmaps_fake_L, fmaps_real_L)
            fm_H_loss = self.criterion["feat_match"](fmaps_fake_H, fmaps_real_H)
            feat_loss = fm_L_loss + fm_H_loss
            gen_loss += self.config.train.lambda_fm * feat_loss

        # cycle consistency loss
        cyc_S_loss = self.criterion["cycle_L"](x, x2)
        cyc_T_loss = self.criterion["cycle_L"](z, z2)
        cyc_L_loss = self.criterion["cycle_L"](z, z4)
        cyc_H_loss = self.criterion["cycle_H"](y, y2)
        cyc_loss = cyc_S_loss + cyc_T_loss + cyc_L_loss + cyc_H_loss
        gen_loss += self.config.train.lambda_cyc * cyc_loss

        # identity mapping loss
        if self.steps < self.config.train.identity_loss_until:
            z_id = self.model["G1"](z)
            x_id = self.model["G2"](x)
            idt_S_loss = self.criterion["identity_L"](x, x_id)
            idt_T_loss = self.criterion["identity_L"](z, z_id)
            # NOTE: z and y are parallel data
            z_id = self.model["G3"](z)
            y_id = self.model["G4"](y)
            idt_L_loss = self.criterion["identity_L"](z, y_id)
            idt_H_loss = self.criterion["identity_H"](y, z_id)
            idt_loss = idt_S_loss + idt_T_loss + idt_L_loss + idt_H_loss
            gen_loss += self.config.train.lambda_idt * idt_loss

        # update generators
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        for name in ["G1", "G2", "G3", "G4"]:
            if self.config.train.optim.generator.clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model[name].parameters(),
                    self.config.train.optim.generator.clip_norm,
                )
                if not torch.isfinite(grad_norm):
                    logger.info(
                        f"[{name}] Grad norm is not finite ({grad_norm}) at step {self.steps}"
                    )
                self.total_grad_norm[f"grad_norm/{name}"] += grad_norm
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        p_fake_S = self.model["D1"](z1.detach())
        p_real_S = self.model["D1"](x)
        p_fake_T = self.model["D2"](x1.detach())
        p_real_T = self.model["D2"](z)  # = p_real_L
        p_fake_L = self.model["D2"](y1.detach())
        p_fake_H = self.model["D3"](z3.detach())
        p_real_H = self.model["D3"](y)

        # discriminator loss
        # NOTE: the first argument must be the fake samples
        fake_S_loss, real_S_loss = self.criterion["adversarial"](p_fake_S, p_real_S)
        fake_T_loss, real_T_loss = self.criterion["adversarial"](p_fake_T, p_real_T)
        fake_L_loss, real_L_loss = self.criterion["adversarial"](p_fake_L, p_real_T)
        fake_H_loss, real_H_loss = self.criterion["adversarial"](p_fake_H, p_real_H)
        fake_loss = fake_S_loss + (fake_T_loss + fake_L_loss) / 2 + fake_H_loss
        real_loss = real_S_loss + (real_T_loss + real_L_loss) / 2 + real_H_loss
        dis_loss = fake_loss + real_loss

        # update discriminator
        self.optimizer["discriminator"].zero_grad()
        dis_loss.backward()
        for name in ["D1", "D2", "D3"]:
            if self.config.train.optim.discriminator.clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model[name].parameters(),
                    self.config.train.optim.discriminator.clip_norm,
                )
                if not torch.isfinite(grad_norm):
                    logger.info(
                        f"[{name}] Grad norm is not finite ({grad_norm}) at step {self.steps}"
                    )
                self.total_grad_norm[f"grad_norm/{name}"] += grad_norm
        self.optimizer["discriminator"].step()
        self.scheduler["discriminator"].step()

        results = dict(
            adv_S_loss=adv_S_loss,
            adv_T_loss=adv_T_loss,
            adv_L_loss=adv_L_loss,
            adv_H_loss=adv_H_loss,
            fm_L_loss=fm_L_loss,
            fm_H_loss=fm_H_loss,
            cyc_S_loss=cyc_S_loss,
            cyc_T_loss=cyc_T_loss,
            cyc_L_loss=cyc_L_loss,
            cyc_H_loss=cyc_H_loss,
            idt_S_loss=idt_S_loss,
            idt_T_loss=idt_T_loss,
            idt_L_loss=idt_L_loss,
            idt_H_loss=idt_H_loss,
            gen_loss=gen_loss,
            fake_S_loss=fake_S_loss,
            fake_T_loss=fake_T_loss,
            fake_L_loss=fake_L_loss,
            fake_H_loss=fake_H_loss,
            real_S_loss=real_S_loss,
            real_T_loss=real_T_loss,
            real_L_loss=real_L_loss,
            real_H_loss=real_H_loss,
            dis_loss=dis_loss,
        )
        results = {k: float(v) for k, v in results.items() if v is not None}
        for k, v in results.items():
            self.total_train_loss[f"train/{k}"] += v

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _finetune_step(self, batch):
        """Train model one step."""
        # parse batch
        x, y, z = batch
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        #######################
        #      Generator      #
        #######################
        # domain adaptation
        with torch.no_grad():
            x = self.model["G1"](x).detach()
        # just train resampling cyclegan
        x1 = self.model["G3"](x)
        x2 = self.model["G4"](x1)
        y1 = self.model["G4"](y)
        y2 = self.model["G3"](y1)

        # initialize
        gen_loss = 0.0

        # adversarial loss
        p_fake_L = self.model["D2"](y1)
        p_fake_H = self.model["D3"](x1)
        adv_L_loss = self.criterion["adversarial"](p_fake_L)
        adv_H_loss = self.criterion["adversarial"](p_fake_H)
        adv_loss = adv_L_loss + adv_H_loss
        gen_loss += self.config.train.lambda_adv * adv_loss

        # cycle consistency loss
        cyc_L_loss = self.criterion["cycle_L"](x, x2)
        cyc_H_loss = self.criterion["cycle_H"](y, y2)
        cyc_loss = cyc_L_loss + cyc_H_loss
        gen_loss += self.config.train.lambda_cyc * cyc_loss

        # update generators
        self.optimizer["generator"].zero_grad()
        gen_loss.backward()
        for name in ["G3", "G4"]:
            if self.config.train.optim.generator.clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model[name].parameters(),
                    self.config.train.optim.generator.clip_norm,
                )
                if not torch.isfinite(grad_norm):
                    logger.info(
                        "[{}] Grad norm is not finite ({}) at step {}".format(
                            name, grad_norm, self.steps
                        )
                    )
                self.total_grad_norm[f"grad_norm/{name}"] += grad_norm
        self.optimizer["generator"].step()
        self.scheduler["generator"].step()

        #######################
        #    Discriminator    #
        #######################
        p_real_L = self.model["D2"](z)
        p_fake_L = self.model["D2"](y1.detach())
        p_fake_H = self.model["D3"](x1.detach())
        p_real_H = self.model["D3"](y)

        # discriminator loss
        # NOTE: the first argument must be the fake samples
        fake_L_loss, real_L_loss = self.criterion["adversarial"](p_fake_L, p_real_L)
        fake_H_loss, real_H_loss = self.criterion["adversarial"](p_fake_H, p_real_H)
        fake_loss = fake_L_loss + fake_H_loss
        real_loss = real_L_loss + real_H_loss
        dis_loss = fake_loss + real_loss

        # update discriminator
        self.optimizer["discriminator"].zero_grad()
        dis_loss.backward()
        for name in ["D2", "D3"]:
            if self.config.train.optim.discriminator.clip_norm > 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.model[name].parameters(),
                    self.config.train.optim.discriminator.clip_norm,
                )
                if not torch.isfinite(grad_norm):
                    logger.info(
                        "[{}] Grad norm is not finite ({}) at step {}".format(
                            name, grad_norm, self.steps
                        )
                    )
                self.total_grad_norm[f"grad_norm/{name}"] += grad_norm
        self.optimizer["discriminator"].step()
        self.scheduler["discriminator"].step()

        results = dict(
            adv_L_loss=adv_L_loss,
            adv_H_loss=adv_H_loss,
            cyc_L_loss=cyc_L_loss,
            cyc_H_loss=cyc_H_loss,
            gen_loss=gen_loss,
            fake_L_loss=fake_L_loss,
            fake_H_loss=fake_H_loss,
            real_L_loss=real_L_loss,
            real_H_loss=real_H_loss,
            dis_loss=dis_loss,
        )
        results = {k: float(v) for k, v in results.items() if v is not None}
        for k, v in results.items():
            self.total_train_loss[f"train/{k}"] += v

        # update counts
        self.steps += 1
        self.tqdm.update(1)
        self._check_train_finish()

    def _train_epoch(self):
        """Train model one epoch."""
        for train_steps_per_epoch, batch in enumerate(
            self.data_loader["train_no_dev"], 1
        ):
            # train one step
            if self.steps < self.config.train.finetune_start_steps:
                self._pretrain_step(batch)
            else:
                self._finetune_step(batch)

            # check interval
            if self.rank == 0:
                self._check_log_interval()
                self._check_eval_interval()
                self._check_save_interval()

            # check whether training is finished
            if self.finish_train:
                return

        # update
        self.epochs += 1
        self.train_steps_per_epoch = train_steps_per_epoch
        logger.info(
            f"(Steps: {self.steps}) Finished {self.epochs} epoch training "
            f"({self.train_steps_per_epoch} steps per epoch)."
        )

        # needed for shuffle in distributed training
        if self.config.train.n_gpus > 1:
            self.sampler["train_no_dev"].set_epoch(self.epochs)

    @torch.no_grad()
    def _eval_pretrain_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y, z = batch
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        #######################
        #      Generator      #
        #######################
        # S_LR -> T_LR -> S_LR
        x1 = self.model["G1"](x)
        x2 = self.model["G2"](x1)
        # T_LR -> S_LR -> T_LR
        z1 = self.model["G2"](z)
        z2 = self.model["G1"](z1)
        # T_LR -> T_HR -> T_LR
        z3 = self.model["G3"](z)
        z4 = self.model["G4"](z3)
        # T_HR -> T_LR -> T_HR
        y1 = self.model["G4"](y)
        y2 = self.model["G3"](y1)

        # initialize
        fm_L_loss = None
        fm_H_loss = None
        idt_S_loss = None
        idt_T_loss = None
        idt_L_loss = None
        idt_H_loss = None
        gen_loss = 0.0

        # adversarial loss
        p_fake_S = self.model["D1"](z1)
        p_fake_T = self.model["D2"](x1)
        if self.config.train.lambda_fm > 0:
            p_fake_L, fmaps_fake_L = self.model["D2"](y1, return_fmaps=True)  # LR
            p_fake_H, fmaps_fake_H = self.model["D3"](z3, return_fmaps=True)  # HR
        else:
            p_fake_L = self.model["D2"](y1, return_fmaps=False)  # LR
            p_fake_H = self.model["D3"](z3, return_fmaps=False)  # HR
        adv_S_loss = self.criterion["adversarial"](p_fake_S)
        adv_T_loss = self.criterion["adversarial"](p_fake_T)
        adv_L_loss = self.criterion["adversarial"](p_fake_L)
        adv_H_loss = self.criterion["adversarial"](p_fake_H)
        adv_loss = adv_S_loss + adv_T_loss + adv_L_loss + adv_H_loss
        gen_loss += self.config.train.lambda_adv * adv_loss

        # feature matching loss
        if self.config.train.lambda_fm > 0:
            _, fmaps_real_L = self.model["D2"](z, return_fmaps=True)
            _, fmaps_real_H = self.model["D3"](y, return_fmaps=True)
            # NOTE: the first argument must be the fake samples
            feat_match_L_loss = self.criterion["feat_match"](fmaps_fake_L, fmaps_real_L)
            feat_match_H_loss = self.criterion["feat_match"](fmaps_fake_H, fmaps_real_H)
            feat_match_loss = feat_match_L_loss + feat_match_H_loss
            gen_loss += self.config.train.lambda_fm * feat_match_loss

        # cycle consistency loss
        cyc_S_loss = self.criterion["cycle_L"](x, x2)
        cyc_T_loss = self.criterion["cycle_L"](z, z2)
        cyc_L_loss = self.criterion["cycle_L"](z, z4)
        cyc_H_loss = self.criterion["cycle_H"](y, y2)
        cyc_loss = cyc_S_loss + cyc_T_loss + cyc_L_loss + cyc_H_loss
        gen_loss += self.config.train.lambda_cyc * cyc_loss

        # identity mapping loss
        if self.steps < self.config.train.identity_loss_until:
            z_id = self.model["G1"](z)
            x_id = self.model["G2"](x)
            idt_S_loss = self.criterion["identity_L"](x, x_id)
            idt_T_loss = self.criterion["identity_L"](z, z_id)
            # NOTE: z and y are parallel data
            z_id = self.model["G3"](z)
            y_id = self.model["G4"](y)
            idt_L_loss = self.criterion["identity_L"](z, y_id)
            idt_H_loss = self.criterion["identity_H"](y, z_id)
            idt_loss = idt_S_loss + idt_T_loss + idt_L_loss + idt_H_loss
            gen_loss += self.config.train.lambda_idt * idt_loss

        #######################
        #    Discriminator    #
        #######################
        # cyclegan-1
        p_fake_S = self.model["D1"](z1)
        p_real_S = self.model["D1"](x)
        p_fake_T = self.model["D2"](x1)
        p_real_T = self.model["D2"](z)  # = p_real_L
        # cyclegan-2
        p_fake_L = self.model["D2"](y1)
        p_fake_H = self.model["D3"](z3)
        p_real_H = self.model["D3"](y)

        # discriminator loss
        # NOTE: the first argument must be the fake samples
        fake_S_loss, real_S_loss = self.criterion["adversarial"](p_fake_S, p_real_S)
        fake_T_loss, real_T_loss = self.criterion["adversarial"](p_fake_T, p_real_T)
        fake_L_loss, real_L_loss = self.criterion["adversarial"](p_fake_L, p_real_T)
        fake_H_loss, real_H_loss = self.criterion["adversarial"](p_fake_H, p_real_H)
        fake_loss = fake_S_loss + (fake_T_loss + fake_L_loss) / 2 + fake_H_loss
        real_loss = real_S_loss + (real_T_loss + real_L_loss) / 2 + real_H_loss
        dis_loss = fake_loss + real_loss

        results = dict(
            adv_S_loss=adv_S_loss,
            adv_T_loss=adv_T_loss,
            adv_L_loss=adv_L_loss,
            adv_H_loss=adv_H_loss,
            fm_L_loss=fm_L_loss,
            fm_H_loss=fm_H_loss,
            cyc_S_loss=cyc_S_loss,
            cyc_T_loss=cyc_T_loss,
            cyc_L_loss=cyc_L_loss,
            cyc_H_loss=cyc_H_loss,
            idt_S_loss=idt_S_loss,
            idt_T_loss=idt_T_loss,
            idt_L_loss=idt_L_loss,
            idt_H_loss=idt_H_loss,
            gen_loss=gen_loss,
            fake_S_loss=fake_S_loss,
            fake_T_loss=fake_T_loss,
            fake_L_loss=fake_L_loss,
            fake_H_loss=fake_H_loss,
            real_S_loss=real_S_loss,
            real_T_loss=real_T_loss,
            real_L_loss=real_L_loss,
            real_H_loss=real_H_loss,
            dis_loss=dis_loss,
        )
        results = {k: float(v) for k, v in results.items() if v is not None}
        for k, v in results.items():
            self.total_eval_loss[f"eval/{k}"] += v

    @torch.no_grad()
    def _eval_finetune_step(self, batch):
        """Evaluate model one step."""
        # parse batch
        x, y, z = batch
        x = x.to(self.device)
        y = y.to(self.device)
        z = z.to(self.device)

        #######################
        #      Generator      #
        #######################
        # domain adaptation
        x = self.model["G1"](x)
        # resampling
        x1 = self.model["G3"](x)
        x2 = self.model["G4"](x1)
        y1 = self.model["G4"](y)
        y2 = self.model["G3"](y1)

        # initialize
        gen_loss = 0.0

        # adversarial loss
        p_fake_L = self.model["D2"](y1)
        p_fake_H = self.model["D3"](x1)
        adv_L_loss = self.criterion["adversarial"](p_fake_L)
        adv_H_loss = self.criterion["adversarial"](p_fake_H)
        adv_loss = adv_L_loss + adv_H_loss
        self.total_eval_loss["eval/adversarial_L_loss"] += adv_L_loss.item()
        self.total_eval_loss["eval/adversarial_H_loss"] += adv_H_loss.item()
        self.total_eval_loss["eval/adversarial_loss"] += adv_loss.item()
        gen_loss += self.config.train.lambda_adv * adv_loss

        # cycle consistency loss
        cyc_L_loss = self.criterion["cycle_L"](x, x2)
        cyc_H_loss = self.criterion["cycle_H"](y, y2)
        cyc_loss = cyc_L_loss + cyc_H_loss
        self.total_eval_loss["eval/cycle_L_loss"] += cyc_L_loss.item()
        self.total_eval_loss["eval/cycle_H_loss"] += cyc_H_loss.item()
        self.total_eval_loss["eval/cycle_loss"] += cyc_loss.item()
        gen_loss += self.config.train.lambda_cyc * cyc_loss

        self.total_eval_loss["eval/generator_loss"] += gen_loss.item()

        #######################
        #    Discriminator    #
        #######################
        p_real_L = self.model["D2"](z)
        p_fake_L = self.model["D2"](y1)
        p_fake_H = self.model["D3"](x1)
        p_real_H = self.model["D3"](y)

        # discriminator loss
        # NOTE: the first argument must be the fake samples
        fake_L_loss, real_L_loss = self.criterion["adversarial"](p_fake_L, p_real_L)
        fake_H_loss, real_H_loss = self.criterion["adversarial"](p_fake_H, p_real_H)
        fake_loss = fake_L_loss + fake_H_loss
        real_loss = real_L_loss + real_H_loss
        dis_loss = fake_loss + real_loss

        results = dict(
            adv_L_loss=adv_L_loss,
            adv_H_loss=adv_H_loss,
            cyc_L_loss=cyc_L_loss,
            cyc_H_loss=cyc_H_loss,
            gen_loss=gen_loss,
            fake_L_loss=fake_L_loss,
            fake_H_loss=fake_H_loss,
            real_L_loss=real_L_loss,
            real_H_loss=real_H_loss,
            dis_loss=dis_loss,
        )
        results = {k: float(v) for k, v in results.items() if v is not None}
        for k, v in results.items():
            self.total_eval_loss[f"eval/{k}"] += v

    def _eval_epoch(self):
        """Evaluate model one epoch."""
        logger.info(f"(Steps: {self.steps}) Start evaluation.")
        # change mode
        for key in self.model.keys():
            self.model[key].eval()

        # calculate loss for each batch
        for eval_steps_per_epoch, batch in enumerate(
            tqdm(self.data_loader["dev"], desc="[eval]"), 1
        ):
            # eval one step
            if self.steps < self.config.train.finetune_start_steps:
                self._eval_pretrain_step(batch)
            else:
                self._eval_finetune_step(batch)

            # save intermediate result
            if eval_steps_per_epoch == 1:
                self._genearete_and_save_intermediate_result(batch)

        logger.info(
            f"(Steps: {self.steps}) Finished evaluation "
            f"({eval_steps_per_epoch} steps per epoch)."
        )

        # average loss
        for key in self.total_eval_loss.keys():
            self.total_eval_loss[key] /= eval_steps_per_epoch
            logger.info(
                f"(Steps: {self.steps}) {key} = {self.total_eval_loss[key]:.4f}."
            )

        # record
        self._write_to_tensorboard(self.total_eval_loss)

        # reset
        self.total_eval_loss = defaultdict(float)

        # restore mode
        for key in self.model.keys():
            self.model[key].train()

    @torch.no_grad()
    def _genearete_and_save_intermediate_result(self, batch):
        """Generate and save intermediate result."""
        # delayed import to avoid error related backend error
        import matplotlib.pyplot as plt

        # use the only 0-th sample
        x, y, z = batch
        x = x[0:1].to(self.device)
        y = y[0:1].to(self.device)

        # generate
        x1 = self.model["G1"](x)
        x2 = self.model["G3"](x1)
        x3 = self.model["G4"](x2)
        x4 = self.model["G2"](x3)

        # save audio to tfboard
        self.writer.add_audio(
            "audio_real_S_LR",
            x.view(-1).cpu().numpy().T,
            self.steps,
            self.config.data.sample_rate_L,
        )
        self.writer.add_audio(
            "audio_fake_T_LR",
            x1.view(-1).cpu().numpy().T,
            self.steps,
            self.config.data.sample_rate_L,
        )
        self.writer.add_audio(
            "audio_fake_T_HR",
            x2.view(-1).cpu().numpy().T,
            self.steps,
            self.config.data.sample_rate_H,
        )
        self.writer.add_audio(
            "audio_recn_S_LR",
            x4.view(-1).cpu().numpy().T,
            self.steps,
            self.config.data.sample_rate_L,
        )
        self.writer.add_audio(
            "audio_real_T_HR",
            y.view(-1).cpu().numpy().T,
            self.steps,
            self.config.data.sample_rate_H,
        )

    def _write_to_tensorboard(self, loss):
        """Write to tensorboard."""
        for key, value in loss.items():
            self.writer.add_scalar(key, value, self.steps)

    def _check_save_interval(self):
        if self.steps % self.config.train.checkpoint_interval == 0:
            self.save_checkpoint(
                to_absolute_path(
                    join(self.config.out_dir, "checkpoints" f"checkpoint-{self.steps}steps.pkl")
                )
            )
            logger.info(f"Successfully saved checkpoint @ {self.steps} steps.")

    def _check_eval_interval(self):
        if self.steps % self.config.train.eval_interval_steps == 0:
            self._eval_epoch()

    def _check_log_interval(self):
        if self.steps % self.config.train.log_interval_steps == 0:
            for key in self.total_train_loss.keys():
                self.total_train_loss[key] /= self.config.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_train_loss[key]:.4f}."
                )
            for key in self.total_grad_norm.keys():
                self.total_grad_norm[key] /= self.config.train.log_interval_steps
                logger.info(
                    f"(Steps: {self.steps}) {key} = {self.total_grad_norm[key]:.4f}."
                )
            self._write_to_tensorboard(self.total_train_loss)
            self._write_to_tensorboard(self.total_grad_norm)

            # reset
            self.total_train_loss = defaultdict(float)
            self.total_grad_norm = defaultdict(float)

    def _check_train_finish(self):
        if self.steps >= self.config.train.max_train_steps:
            self.finish_train = True


class Collater(object):
    """Customized collater for Pytorch DataLoader in training."""

    def __init__(
        self,
        batch_max_steps=8000,
        sample_rate_L=16000,
        sample_rate_H=48000,
    ):
        """Initialize customized collater for PyTorch DataLoader.

        Args:
            batch_max_steps (int): The maximum length of batch.
            sample_rate_L (int) : Sampling rate of low-resolution audios.
            sample_rate_H (int) : Sampling rate of high-resolution audios.

        """
        self.batch_max_steps = batch_max_steps
        self.sample_rate_L = sample_rate_L
        self.sample_rate_H = sample_rate_H

        # Currently only support divisible pair of sampling rates
        assert sample_rate_H % sample_rate_L == 0
        self.sr_factor = sample_rate_H // sample_rate_L

    def __call__(self, batch):
        """Convert into batch tensors.

        Args:
            batch (list): list of audio.

        Returns:
            Tensor: Low-resolution source domain audio batch (B, 1, T).
            Tensor: High-resolution target domain audio batch (B, 1, T * sr_factor).
            Tensor: Low-resolution target domain audio batch (B, 1, T).

        """
        # initialize batch to be returned
        x_batch = []
        y_batch = []
        z_batch = []

        # make batch with random cut
        for (audio1, sr1), (audio2, sr2) in batch:

            # process for audio1 from dataset A
            sr_rate = math.ceil(sr1 / self.sample_rate_L)
            start1 = np.random.randint(0, len(audio1) - self.batch_max_steps * sr_rate)
            audio1 = audio1[start1 : start1 + self.batch_max_steps * sr_rate]
            x = librosa.resample(audio1, orig_sr=sr1, target_sr=self.sample_rate_L)
            assert x.shape[0] >= self.batch_max_steps
            x = x[: self.batch_max_steps]

            # process for audio2 from dataset B
            sr_rate = math.ceil(sr2 / self.sample_rate_H)
            start2 = np.random.randint(
                0, len(audio2) - self.batch_max_steps * self.sr_factor * sr_rate
            )
            audio2 = audio2[
                start2 : start2 + self.batch_max_steps * self.sr_factor * sr_rate
            ]
            y = librosa.resample(audio2, orig_sr=sr2, target_sr=self.sample_rate_H)
            assert y.shape[0] >= self.batch_max_steps * self.sr_factor
            y = y[: self.batch_max_steps * self.sr_factor]
            # create parallel low-respolution audio
            z = librosa.resample(
                y, orig_sr=self.sample_rate_H, target_sr=self.sample_rate_L, fix=True
            )

            # append to batches
            x_batch.append(x)
            y_batch.append(y)
            z_batch.append(z)

        # convert each batch to tensor, asuume that each item in batch has the same length
        x_batch = torch.FloatTensor(np.array(x_batch)).unsqueeze(1)
        y_batch = torch.FloatTensor(np.array(y_batch)).unsqueeze(1)
        z_batch = torch.FloatTensor(np.array(z_batch)).unsqueeze(1)
        return x_batch, y_batch, z_batch


def run(rank, world_size, use_cuda, config):
    """Run training process."""

    distributed = False
    if not use_cuda:
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
        # effective when using fixed size inputs
        # see https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(rank)
        # setup for distributed training
        # see example: https://github.com/NVIDIA/apex/tree/master/examples/simple/distributed
        if world_size > 1:
            distributed = True
            os.environ["MASTER_ADDR"] = "localhost"
            os.environ["MASTER_PORT"] = "29500"
            torch.distributed.init_process_group(
                backend="nccl", init_method="env://", rank=rank, world_size=world_size
            )

    # suppress logger for distributed training
    if rank != 0:
        sys.stdout = open(os.devnull, "w")

    # create directory to output results
    out_dir = to_absolute_path(config.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # save entire configs to output directory
    with open(join(out_dir, "config.yaml"), "w") as f:
        OmegaConf.save(config, f)

    # get dataset and dataloader
    dataset, sampler, data_loader = {}, {}, {}
    collater = Collater(
        batch_max_steps=config.data.batch_max_steps,
        sample_rate_L=config.data.sample_rate_L,
        sample_rate_H=config.data.sample_rate_H,
    )
    for phase in ["train_no_dev", "dev"]:
        dataset[phase] = TrainAudioDataset(
            audio_list_L=config.data.audio_list_L[phase],
            audio_list_H=config.data.audio_list_H[phase],
            audio_second_threshold=config.data.batch_max_steps
            / config.data.sample_rate_L,
            sample_rate_threshold_L=config.data.sample_rate_L,
            sample_rate_threshold_H=config.data.sample_rate_H,
            return_utt_id=False,
        )
        if distributed:
            # setup sampler for distributed training
            from torch.utils.data.distributed import DistributedSampler

            sampler[phase] = DistributedSampler(
                dataset[phase],
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
        else:
            sampler[phase] = None
        data_loader[phase] = DataLoader(
            dataset[phase],
            shuffle=False if distributed else True,
            batch_size=config.data.batch_size,
            collate_fn=collater,
            num_workers=config.data.num_workers,
            sampler=sampler[phase],
            pin_memory=config.data.pin_memory,
        )

    # define models
    model = {
        "G1": hydra.utils.instantiate(config.model.G1).to(device),
        "G2": hydra.utils.instantiate(config.model.G2).to(device),
        "G3": hydra.utils.instantiate(config.model.G3).to(device),
        "G4": hydra.utils.instantiate(config.model.G4).to(device),
        "D1": hydra.utils.instantiate(config.model.D1).to(device),
        "D2": hydra.utils.instantiate(config.model.D2).to(device),
        "D3": hydra.utils.instantiate(config.model.D3).to(device),
    }

    # wrap model for distributed training
    if distributed:
        for name in ["G1", "G2", "G3", "G4", "D1", "D2", "D3"]:
            # I don't know why but 'broadcast_buffers=False' avoid failure of gradient computation
            model[name] = DistributedDataParallel(model[name], broadcast_buffers=False)

    # define criterions
    criterion = {
        "adversarial": hydra.utils.instantiate(config.train.adversarial_loss).to(
            device
        ),
        "feat_match": hydra.utils.instantiate(config.train.feat_match_loss).to(device),
        "cycle_L": hydra.utils.instantiate(config.train.cycle_L_loss).to(device),
        "cycle_H": hydra.utils.instantiate(config.train.cycle_H_loss).to(device),
        "identity_L": hydra.utils.instantiate(config.train.identity_L_loss).to(device),
        "identity_H": hydra.utils.instantiate(config.train.identity_H_loss).to(device),
    }

    # define optimizers and schedulers
    optimizer = {
        "generator": hydra.utils.instantiate(
            config.train.optim.generator.optimizer,
            params=itertools.chain(
                model["G1"].parameters(),
                model["G2"].parameters(),
                model["G3"].parameters(),
                model["G4"].parameters(),
            ),
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.optim.discriminator.optimizer,
            params=itertools.chain(
                model["D1"].parameters(),
                model["D2"].parameters(),
                model["D3"].parameters(),
            ),
        ),
    }
    scheduler = {
        "generator": hydra.utils.instantiate(
            config.train.optim.generator.lr_scheduler,
            optimizer=optimizer["generator"],
        ),
        "discriminator": hydra.utils.instantiate(
            config.train.optim.discriminator.lr_scheduler,
            optimizer=optimizer["discriminator"],
        ),
    }

    # show settings
    for name in ["G1", "G2", "G3", "G4", "D1", "D2", "D3"]:
        logger.info(model[name])
    logger.info(optimizer["generator"])
    logger.info(optimizer["discriminator"])
    logger.info(scheduler["generator"])
    logger.info(scheduler["discriminator"])
    for key, val in criterion.items():
        logger.info(f"{key}: {val}")

    # define trainer
    trainer = Trainer(
        steps=0,
        epochs=0,
        data_loader=data_loader,
        sampler=sampler,
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        rank=rank,
        device=device,
    )

    # resume from checkpoint
    if config.train.resume.checkpoint_path is not None:
        logger.info(
            f"Resume training from checkpoint: {config.train.resume.checkpoint_path}"
        )
        trainer.load_checkpoint(
            config.train.resume.checkpoint_path, config.train.resume.load_only_params
        )

    # run training loop
    try:
        trainer.run()
    finally:
        trainer.save_checkpoint(join(out_dir, "checkpoints", f"checkpoint-{trainer.steps}steps.pkl"))
        logger.info(f"Successfully saved checkpoint @ {trainer.steps}steps.")


@hydra.main(version_base=None, config_path="config", config_name="train")
def main(config: DictConfig) -> None:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        from torch.backends import cudnn

        cudnn.benchmark = config.cudnn.benchmark
        cudnn.deterministic = config.cudnn.deterministic
        logger.info(f"cudnn.deterministic: {cudnn.deterministic}")
        logger.info(f"cudnn.benchmark: {cudnn.benchmark}")

    world_size = config.train.n_gpus
    n_gpus_available = torch.cuda.device_count()
    if n_gpus_available > 1:
        if n_gpus_available < config.train.n_gpus:
            print(
                f"You require {world_size} GPUs but only {n_gpus_available} GPUs are available."
            )
            sys.exit(0)
        print(
            f"You have available {n_gpus_available} GPUs but using {world_size} GPUs."
        )

    # initialize seed
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    mp.spawn(run, args=(world_size, use_cuda, config), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
