#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 12 14:39:46 2024

@author: louis
"""

import torch
from lightning.pytorch import LightningModule
from model.utils.abs_models import AbsSpeechModel, AbsReverbModel, OracleModel, AbsJointLossModule
from lightning.pytorch.utilities import grad_norm
import contextlib


class JointModel(LightningModule):
    def __init__(
        self,
        speech_model: AbsSpeechModel | None = None,
        reverb_model: AbsReverbModel | None = None,
        joint_loss_module: AbsJointLossModule | None = None,
        speech_model_ckpt_path: str | None = None,
        reverb_model_ckpt_path: str | None = None,
        train_speech_model: bool = True,
        train_reverb_model: bool = True,
        # additional_val_losses: list[nn.Module] = [],
    ):
        super().__init__()
        self.speech_model = speech_model
        self.reverb_model = reverb_model
        self.joint_loss_module = joint_loss_module

        self.strict_loading = False

        if self.speech_model is not None and self.joint_loss_module is not None:
            self.speech_model.output_domain = self.joint_loss_module.SPEECH_INPUT_DOMAIN
        if self.reverb_model is not None and self.joint_loss_module is not None:
            self.reverb_model.output_domain = self.joint_loss_module.REVERB_INPUT_DOMAIN
        # self.additional_val_losses = additional_val_losses
        self.train_speech_model = train_speech_model
        self.train_reverb_model = train_reverb_model

        if self.speech_model is not None and speech_model_ckpt_path is not None:
            self.speech_model.load_state_dict_from_joint_model(speech_model_ckpt_path)
        if self.reverb_model is not None and reverb_model_ckpt_path is not None:
            self.reverb_model.load_state_dict_from_joint_model(reverb_model_ckpt_path)

        # https://github.com/Lightning-AI/pytorch-lightning/pull/18951
        if self.speech_model is not None and not self.train_speech_model:
            self.speech_model.freeze()
            self.speech_model.eval()
        if self.reverb_model is not None and not self.train_reverb_model:
            self.reverb_model.freeze()
            self.reverb_model.eval()
        if self.joint_loss_module is not None:
            self.joint_loss_module.freeze()
            self.joint_loss_module.eval()
        if self.speech_model is not None and self.reverb_model is not None and self.joint_loss_module is None:
            raise ValueError(
                "Ambiguity on which model to train: Both speech model and reverb model are provided but not joint_loss_module"
            )

    def predict_dry_speech(self, y):
        pred = self.speech_model(y)
        s_hat = self.speech_model.get_time(pred)
        return s_hat

    def forward(self, y):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._setup_log()

        y, (s, h, rir_properties) = batch
        if self.speech_model is not None:
            with torch.no_grad() if not self.train_speech_model else contextlib.nullcontext():
                s_hat, loss_speech = self.speech_model.training_step((y, s), batch_idx=batch_idx).values()
        else:
            s_hat, loss_speech = None, None
        if self.reverb_model is not None:
            with torch.no_grad() if not self.train_reverb_model else contextlib.nullcontext():
                h_hat, loss_reverb = self.reverb_model.training_step(
                    (y, (h, rir_properties)), batch_idx=batch_idx
                ).values()
        else:
            h_hat, loss_reverb = None, None

        # loss computation
        if self.joint_loss_module is not None and h_hat is not None and s_hat is not None:
            joint_loss = self.joint_loss_module.training_step(
                s_hat=s_hat,
                h_hat=h_hat,
                s=s,
                h=h,
                y=y,
                batch_idx=batch_idx,
            )
        elif self.speech_model is not None:
            joint_loss = loss_speech
        elif self.reverb_model is not None:
            joint_loss = loss_reverb
        else:
            joint_loss = None
        return joint_loss

    def _setup_log(self):
        # Needed to have nested lightningmodules
        if self.speech_model is not None:
            self.speech_model._current_fx_name = self._current_fx_name
        if self.reverb_model is not None:
            self.reverb_model._current_fx_name = self._current_fx_name
        if self.joint_loss_module is not None:
            self.joint_loss_module._current_fx_name = self._current_fx_name

    def validation_step(self, batch, batch_idx):
        if batch_idx == 0:
            self._setup_log()

        y, (s, h, rir_properties) = batch
        if self.speech_model is not None:  # and not isinstance(self.speech_model, OracleModel):
            s_hat, loss_speech = self.speech_model.validation_step((y, s), batch_idx=batch_idx).values()
        else:
            s_hat, loss_speech = None, None
        if self.reverb_model is not None:  # and not isinstance(self.reverb_model, OracleModel):
            h_hat, loss_reverb = self.reverb_model.validation_step(
                (y, (h, rir_properties)), batch_idx=batch_idx
            ).values()
        else:
            h_hat, loss_reverb = None, None

        if self.joint_loss_module is not None and h_hat is not None and s_hat is not None:
            joint_loss = self.joint_loss_module.validation_step(
                s_hat=s_hat,
                h_hat=h_hat,
                s=s,
                h=h,
                y=y,
                batch_idx=batch_idx,
            )
        elif self.speech_model is not None:
            joint_loss = loss_speech
        elif self.reverb_model is not None:
            joint_loss = loss_reverb
        else:
            joint_loss = None
        return joint_loss

    def on_test_epoch_start(self):
        from model.utils.metrics import all_speech_metrics

        self.speech_model.metrics = torch.nn.ModuleList(all_speech_metrics).to(self.device)

    def on_train_start(self):
        if self.speech_model is not None:
            self.speech_model.on_train_start()
        if self.reverb_model is not None:
            self.reverb_model.on_train_start()

    def test_step(self, batch, batch_idx):
        return self.validation_step(batch, batch_idx)

    # def on_before_optimizer_step(self, optimizer):
    #     norms = grad_norm(self, norm_type=2)
    #     self.log_dict(norms, on_epoch=True)


if __name__ == "__main__":
    from model.losses.rereverberation_loss import TimeFrequencyRereverberationLoss
    from model.speech_models.fullsubnet import FullSubNet

    m = JointModel(
        speech_model=FullSubNet(), reverb_model=OracleModel(), joint_loss_module=TimeFrequencyRereverberationLoss()
    )
