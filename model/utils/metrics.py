#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 15:47:48 2024

@author: louis
"""
import torch
from torch import nn
from model.utils.tensor_ops import energy_to_db
from nnAudio import features
import torchmetrics.audio


class IgnoreLastSamplesMetricWrapper(nn.Module):
    def __init__(self, base_metric: nn.Module, num_ignored_first_samples: int = 0, num_ignored_last_samples: int = 256):
        super().__init__()
        self.base_metric = base_metric
        self.num_ignored_last_samples = num_ignored_last_samples
        self.num_ignored_first_samples = num_ignored_first_samples

    def forward(self, pred, target):
        if self.num_ignored_last_samples > 0:
            return self.base_metric(
                pred[..., self.num_ignored_first_samples : -self.num_ignored_last_samples],
                target[..., self.num_ignored_first_samples : -self.num_ignored_last_samples],
            )
        else:
            return self.base_metric(
                pred[..., self.num_ignored_first_samples :],
                target[..., self.num_ignored_first_samples :],
            )

    def __str__(self):
        return f"{type(self.base_metric).__name__}[{self.num_ignored_first_samples}:-{self.num_ignored_last_samples}]"


class DifferenceMetric(nn.Module):
    def __init__(self, property_to_measure):
        super().__init__()
        self.property_to_measure = property_to_measure

    def forward(self, pred, target):
        if isinstance(target, tuple):
            target = target[0]
        pred = pred[..., : target.size(-1)]
        target = target[..., : pred.size(-1)]
        pred_property = self.property_to_measure(pred)
        target_property = self.property_to_measure(target)
        pred_property_finite = pred_property[torch.logical_and(pred_property.isfinite(), target_property.isfinite())]
        target_property_finite = target_property[
            torch.logical_and(pred_property.isfinite(), target_property.isfinite())
        ]
        return (pred_property_finite - target_property_finite).abs().mean()

    def __str__(self):
        return f"{type(self.property_to_measure).__name__} difference"


from model.reverb_models.drr import DRR


def edc(rir):
    power = rir.abs().square()
    return torch.flip(torch.cumsum(torch.flip(power, (-1,)), -1), (-1,))


class EDR(nn.Module):
    def __init__(self, return_dB=True, scale="global", n_fft=512, sr=16000):
        super().__init__()
        self.spec_module = features.STFT(
            n_fft=n_fft, hop_length=n_fft // 8, freq_scale="log", sr=sr, fmin=10, fmax=sr / 2, output_format="Magnitude"
        )
        self.return_dB = return_dB
        self.scale = scale
        self.reestimate_from_oracle_rir = True

    def forward(self, rir):
        rir_tf = self.spec_module(rir)
        edr = edc(rir_tf)
        if self.scale.lower() == "bandwise":
            edr = edr / edr[..., 0, None]
        elif self.scale.lower() == "global":
            edr = edr / edr[..., 0].mean(axis=-1, keepdim=True)[..., None]
        else:
            pass
        if self.return_dB:
            edr = energy_to_db(edr)
        return edr


class SRMRWrapper(torchmetrics.audio.SpeechReverberationModulationEnergyRatio):
    def forward(self, pred, target):
        return super().forward(pred)


drr_difference = DifferenceMetric(DRR())
edr_difference = DifferenceMetric(EDR())


dar_metrics = [drr_difference, edr_difference]

all_speech_metrics = [
    IgnoreLastSamplesMetricWrapper(
        base_metric=torchmetrics.audio.ScaleInvariantSignalDistortionRatio(), num_ignored_last_samples=256
    ),
    IgnoreLastSamplesMetricWrapper(
        base_metric=torchmetrics.audio.ShortTimeObjectiveIntelligibility(fs=16000, extended=True),
        num_ignored_last_samples=256,
    ),
    IgnoreLastSamplesMetricWrapper(
        base_metric=torchmetrics.audio.PerceptualEvaluationSpeechQuality(fs=16000, mode="wb"),
        num_ignored_last_samples=256,
    ),
    IgnoreLastSamplesMetricWrapper(
        base_metric=SRMRWrapper(
            fs=16000, n_cochlear_filters=23, low_freq=125, min_cf=4, max_cf=128, fast=True, norm=False
        ),
        num_ignored_last_samples=256,
    ),
]
