#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contains all abstract classes for speech or reverb models.

See `framework_details.md`.

"""

import os
import warnings
import torch
import numpy as np
from collections import OrderedDict
from lightning.pytorch import LightningModule
from torch import nn
from abc import ABC, abstractmethod
from model.utils.default_stft_istft import default_istft_module, default_stft_module
from model.utils.metrics import IgnoreLastSamplesMetricWrapper, DifferenceMetric
from model.utils.tensor_ops import crop_or_zero_pad_to_target_len
from model.utils.run_management import get_best_checkpoint
from enum import Enum
from lightning.pytorch.trainer.states import RunningStage


class SignalDomain(Enum):
    """Enum for signal domains used as input of joint loss."""

    TIME = 0
    STFT = 1
    # CRM=3


class ModelType(Enum):
    """Enum of the main components of the model.joint_model.JointModel."""

    DRY_SPEECH = "dry_speech_model"
    REVERB = "reverb_model"
    REREVERBERATION_LOSS = "rereverberation_loss"  # convolutive model and reverberation matching loss


class FirstLevelModule(LightningModule):
    """
    Handle metrics, logging, and STFT and ISTFT computation.

    See `framework_details.md`.
    """

    output_domain = SignalDomain.TIME  # default

    def __init__(self, metrics: list[nn.Module] = []):
        super().__init__()
        self.metrics = nn.ModuleList(metrics)
        self.stft_module = default_stft_module
        self.istft_module = default_istft_module
        self.fs = 16000
        self.metrics_all_batches = dict()

    def num_batches_for_current_trainer_stage(self, dataloader_idx=0):
        if self.trainer.state.stage == RunningStage.TRAINING:
            return self.trainer.num_training_batches
        if self.trainer.state.stage == RunningStage.VALIDATING:
            return self.trainer.num_val_batches[dataloader_idx]
        if self.trainer.state.stage == RunningStage.TESTING:
            return self.trainer.num_test_batches[dataloader_idx]
        if self.trainer.state.stage == RunningStage.SANITY_CHECKING:
            return self.trainer.num_sanity_val_batches[dataloader_idx]
        if self.trainer.state.stage == RunningStage.PREDICTING:
            return self.trainer.num_predict_batches[dataloader_idx]

    def log_for_future_aggregate(self, name, value, batch_idx, *args, dataloader_idx=0, **kwargs):
        if name not in self.metrics_all_batches or batch_idx == 0:
            size = (self.num_batches_for_current_trainer_stage(dataloader_idx=dataloader_idx),)
            self.metrics_all_batches[name] = torch.full(size=size, fill_value=torch.nan, device=self.device)
        self.metrics_all_batches[name][batch_idx] = value.detach()
        self.log(name, value, *args, **kwargs)

    def log_audios(self, waveform_batches_dict, scale=True):
        tensorboard = self.logger.experiment
        for k, v in waveform_batches_dict.items():
            wav = v[(0,) * (v.ndim - 1)]
            if wav.numel() > 0:
                if scale:
                    wav = wav / abs(wav).max()
                tensorboard.add_audio(k, wav, global_step=self.global_step, sample_rate=self.fs)
        tensorboard.close()

    def log_metrics_and_audios(self, pred_time_domain, target, batch_idx, dataloader_idx=0, input=None):
        for metric in self.metrics:
            metric_name = (
                "val_"
                + self.MODEL_TYPE.value
                + "_"
                + (
                    str(metric)
                    if isinstance(metric, (IgnoreLastSamplesMetricWrapper, DifferenceMetric))
                    else type(metric).__name__
                )
            )
            try:
                self.log_for_future_aggregate(
                    metric_name, metric(pred_time_domain, target), batch_idx=batch_idx, on_epoch=True
                )
                if self.MODEL_TYPE == ModelType.DRY_SPEECH and input is not None:
                    # Metric between input (wet speech) and target (dry speech)
                    self.log_for_future_aggregate(
                        metric_name + "_input", metric(input, target), batch_idx=batch_idx, on_epoch=True
                    )
            except:
                pass
        # log audios
        if batch_idx == 0:
            self.log_audios(
                {
                    self.MODEL_TYPE.value + "_target": target if self.MODEL_TYPE == ModelType.DRY_SPEECH else target[0],
                    self.MODEL_TYPE.value + "_prediction": pred_time_domain,
                }
            )
        if batch_idx == self.num_batches_for_current_trainer_stage(dataloader_idx=dataloader_idx) - 1:
            self.aggregate_metrics()

    def aggregate_metrics(self):
        if self.logger.log_dir is None:
            logdir = "lightning_logs/debug"
            warnings.warn(f"No logger.log_dir found, setting logdir to {logdir}")
        else:
            logdir = self.logger.log_dir
        metrics_folder = os.path.join(logdir, f"latest_results")
        os.makedirs(metrics_folder, exist_ok=True)
        for k, metrics_tensor in self.metrics_all_batches.items():
            self.log(k + "_median", torch.median(metrics_tensor))
            self.log(k + "_std", torch.std(metrics_tensor))
            np.save(
                os.path.join(metrics_folder, "".join(ki for ki in k if (ki.isalnum() or ki == "_")) + ".npy"),
                metrics_tensor.detach().cpu().numpy(),
            )
        self.metrics_all_batches.clear()

    def log_loss(self, loss, batch_idx=None):
        if loss is not None:
            self.log_for_future_aggregate(
                self._current_fx_name + "_" + self.MODEL_TYPE.value + "_loss", loss, batch_idx=batch_idx, on_epoch=True
            )

    def on_validation_epoch_end(self):
        self.aggregate_metrics()

    def on_test_epoch_end(self):
        self.aggregate_metrics()

    def on_train_epoch_end(self):
        self.aggregate_metrics()


class AbsSpeechOrReverbModel(ABC, FirstLevelModule):
    """
    Abstract parent class for SpeechModel or ReverbModel to handle training_step.

    See `framework_details.md`.
    """

    # Default stft and istft
    def __init__(self, metrics: list[nn.Module] = [], crop_input_to_target: bool = False):
        super().__init__(metrics=metrics)
        self.crop_input_to_target = crop_input_to_target

    @abstractmethod
    def internal_loss(self, pred, target):
        raise NotImplementedError()

    @abstractmethod
    def forward(self, input):
        """
        Forward pass

        Takes wet signal as input and returns the prediction, and all temporary tensors which are used to compute the internal loss.

        See `framework_details.md`.

        Parameters
        ----------
        input : torch.Tensor
            time-domain wet signal.

        Returns
        -------
        pred :
            prediction and all temporary tensors used to compute the internal loss

        """
        pred = NotImplemented
        return pred

    def get_time(self, pred, **kwargs):
        """See `framework_details.md`."""
        return pred

    def get_stft(self, pred, **kwargs):
        """See `framework_details.md`."""
        return self.stft_module(pred)

    def _get_pred_in_domain(self, pred, domain, **kwargs):
        if domain == SignalDomain.STFT:
            return self.get_stft(pred, **kwargs)
        if domain == SignalDomain.TIME:
            return self.get_time(pred, **kwargs)

    def _get_pred_good_domain(self, pred, **kwargs):
        return self._get_pred_in_domain(pred, self.output_domain, **kwargs)

    def _get_pred_all_domains(self, pred, domains, **kwargs):
        return {domain: self._get_pred_in_domain(pred, domain, **kwargs) for domain in domains}

    def training_step(self, batch, batch_idx):
        """
        Perform training step.

        See `framework_details.md`.

        Parameters
        ----------
        batch : tuple
            - If speech model: (y, s)
            - If reverb model: (y, (h, rir properties which are returned by the dataset).
        batch_idx : int
            batch_idx.

        Returns
        -------
        dict
            - "pred" :
                - If speech model, estimated dry speech (in apropriate domain for joint_loss_module).
                - If reverb model, estimated rir (in apropriate domain for joint_loss_module).
            - "loss" : Internal loss
        """
        input, target = batch
        if self.crop_input_to_target:
            input = input[..., : target.size(-1)]
        pred = self.forward(input)
        loss = self.internal_loss(pred, target)
        self.log_loss(loss, batch_idx=batch_idx)
        pred_good_domain = self._get_pred_good_domain(
            pred, length=(target.size(-1) if self.MODEL_TYPE == ModelType.DRY_SPEECH else None)
        )
        return {"pred": pred_good_domain, "loss": loss}

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        """
        Perform validation step.

        See `framework_details.md`.

        Parameters
        ----------
        batch : tuple
            - If speech model: (y, s)
            - If reverb model: (y, (h, rir properties which are returned by the dataset).
        batch_idx : int
            batch_idx.

        Returns
        -------
        dict
            - "pred" :
                - If speech model, estimated dry speech (in apropriate domain for joint_loss_module).
                - If reverb model, estimated rir (in apropriate domain for joint_loss_module).
            - "loss" : Internal loss
        """
        input, target = batch
        if self.crop_input_to_target:
            input = input[..., : target.size(-1)]
        pred = self.forward(input)
        loss = self.internal_loss(pred, target)
        self.log_loss(loss, batch_idx=batch_idx)

        # Metrics are supposed to all be time domain
        pred_time_domain = self.get_time(
            pred, length=(target.size(-1) if self.MODEL_TYPE == ModelType.DRY_SPEECH else None)
        )
        if self.output_domain != SignalDomain.TIME:
            pred_good_domain = self._get_pred_good_domain(pred)
        else:
            pred_good_domain = pred_time_domain

        self.log_metrics_and_audios(
            pred_time_domain=pred_time_domain,
            target=target,
            input=input,
            batch_idx=batch_idx,
            dataloader_idx=dataloader_idx,
        )
        return {"pred": pred_good_domain, "loss": loss}

    def load_state_dict_from_joint_model(self, ckpt_path: str):
        """
        Loads the state-dict of a speech or a reverb model from the checkpoint path of a JointModel.

        Parameters
        ----------
        ckpt_path : str
            - If ckeckpoint path: loads from this specific checkpoint.
            - If directory: Seeks the best checkpoint  in the directory according to a logged monitor.
            The monitor mode depends on ModelType. For speech, monitor mode is max, for reverb, monitor mode is min.
            See `this doc <https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.callbacks.ModelCheckpoint.html#lightning.pytorch.callbacks.ModelCheckpoint>`_

        Returns
        -------
        None.

        """
        if os.path.isdir(ckpt_path):
            print("dir is given as ckpt_path, loading best checkpoint")
            if self.MODEL_TYPE == ModelType.DRY_SPEECH:
                monitor_mode = max
            else:
                monitor_mode = min
            ckpt_path = get_best_checkpoint(ckpt_path, monitor_mode=monitor_mode)
            print(f"Best_checkpoint found {ckpt_path}")
        state_dict = torch.load(ckpt_path, weights_only=False)["state_dict"]
        new_dict = OrderedDict()
        for k, v in state_dict.items():
            if self.MODEL_TYPE == ModelType.DRY_SPEECH and k.startswith("speech_model."):
                new_dict[k[13:]] = v
            # Ignore other modules (reverb model and joint_loss_module)
            if self.MODEL_TYPE == ModelType.REVERB and k.startswith("reverb_model."):
                new_dict[k[13:]] = v
            # else don't append the keys to the state dict
        self.load_state_dict(new_dict, strict=False)


class AbsSpeechModel(AbsSpeechOrReverbModel):
    """
    Abstract parent class for Speech Model.

    See `framework_details.md`.
    """

    MODEL_TYPE = ModelType.DRY_SPEECH


class AbsReverbModel(AbsSpeechOrReverbModel):
    """
    Abstract parent class for Reverb Model.

    See `framework_details.md`.
    """

    MODEL_TYPE = ModelType.REVERB


class OracleModel(ABC, FirstLevelModule):
    @abstractmethod
    def training_step(self, batch, batch_idx): ...

    def validation_step(self, batch, batch_idx):
        return self.training_step(batch, batch_idx)

    def internal_loss(self, pred, target):
        return None

    def forward(self, input):
        raise RuntimeError("Oracle model is not supposed to be used at inference")


class OracleSpeechModel(OracleModel, AbsSpeechModel):
    """
    Oracle speech model.

    See `framework_details.md`.
    """

    def training_step(self, batch, batch_idx):
        input, target = batch
        pred = target
        pred_good_domain = self._get_pred_good_domain(
            pred, length=(target.size(-1) if self.MODEL_TYPE == ModelType.DRY_SPEECH else None)
        )
        return {"pred": pred_good_domain, "loss": None}


class OracleReverbModel(OracleModel, AbsReverbModel):
    """
    Oracle reverb model.

    See `framework_details.md`.
    """

    def __init__(self, metrics: list[nn.Module] = [], target_len: int = 32000, crop_input_to_target: bool = False):
        super().__init__(metrics=metrics)
        self.crop_input_to_target = crop_input_to_target
        self.target_len = target_len

    def training_step(self, batch, batch_idx):
        _, (target, _) = batch
        target_cropped = crop_or_zero_pad_to_target_len(target, self.target_len)
        return {"pred": target_cropped, "loss": None}


class OracleParametersReverbModel(OracleReverbModel):
    """
    Abstract parent for reverb model with oracle parameters.

    See `framework_details.md`.
    """

    def training_step(self, batch, batch_idx):
        y, (h, rir_properties) = batch
        h_hat = self.convert_rir(h, rir_properties)
        return {"pred": h_hat, "loss": None}


class AbsJointLossModule(ABC, FirstLevelModule):
    """
    Abstract parent for joint loss.

    See `framework_details.md`.
    """

    def __init__(self, metrics: list[nn.Module] = []):
        super().__init__(metrics=metrics)

    @property
    @abstractmethod
    def SPEECH_INPUT_DOMAIN():
        return SignalDomain.TIME

    @property
    @abstractmethod
    def REVERB_INPUT_DOMAIN():
        return SignalDomain.TIME

    @abstractmethod
    def forward(self, s_hat, h_hat, s, h, y):
        loss = NotImplemented
        return loss

    def training_step(self, s_hat, h_hat, s, h, y, batch_idx):
        loss = self.forward(s_hat, h_hat, s, h, y)
        self.log_loss(loss, batch_idx=batch_idx)
        if True or not loss.isfinite():  # Disabled check in order to get better perf (no CPU-GPU sync)
            return loss
        return None

    def validation_step(self, s_hat, h_hat, s, h, y, batch_idx):
        raise NotImplementedError()
