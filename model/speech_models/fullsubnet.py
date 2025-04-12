#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 16:54:56 2024

@author: louis
"""


import sys
import os
import torch
from torch import nn
from model.utils.abs_models import AbsSpeechModel

# Add FullSubNet's audio_zen to path
sys.path.append(os.path.join(os.path.dirname(__file__), "FullSubNet"))
from model.speech_models.FullSubNet.audio_zen.acoustics.mask import decompress_cIRM, build_complex_ideal_ratio_mask
from model.speech_models.FullSubNet.recipes.dns_interspeech_2020.fullsubnet.model import Model as OriginalFullSubNet
from torchmetrics.audio import (
    ShortTimeObjectiveIntelligibility,
    ScaleInvariantSignalDistortionRatio,
)


class FullSubNet(AbsSpeechModel):
    def __init__(
        self,
        metrics: list[nn.Module] = [
            ShortTimeObjectiveIntelligibility(fs=16000),
            ScaleInvariantSignalDistortionRatio(),
        ],
        num_freqs=257,
        # look_ahead=2,
        look_ahead=5,
        sequence_model="LSTM",
        fb_num_neighbors=0,
        sb_num_neighbors=15,
        fb_output_activate_function="ReLU",
        sb_output_activate_function=False,
        fb_model_hidden_size=512,
        sb_model_hidden_size=384,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=1,
        # num_groups_in_drop_band=2,
        weight_init=False,
    ):
        super().__init__(metrics=metrics, crop_input_to_target=True)
        self.original_fullsubnet = OriginalFullSubNet(
            num_freqs=num_freqs,
            look_ahead=look_ahead,
            sequence_model=sequence_model,
            fb_num_neighbors=fb_num_neighbors,
            sb_num_neighbors=sb_num_neighbors,
            fb_output_activate_function=fb_output_activate_function,
            sb_output_activate_function=sb_output_activate_function,
            fb_model_hidden_size=fb_model_hidden_size,
            sb_model_hidden_size=sb_model_hidden_size,
            norm_type=norm_type,
            num_groups_in_drop_band=num_groups_in_drop_band,
            # num_groups_in_drop_band=2,
            weight_init=weight_init,
        )
        self.loss_function = nn.MSELoss()

    def forward(self, input):
        Y = self.stft_module(input)
        noisy_mag, noisy_real, noisy_imag = Y.abs(), Y.real, Y.imag
        cRM_compressed = self.original_fullsubnet(noisy_mag)
        return cRM_compressed.permute(0, 2, 3, 1), (noisy_real, noisy_imag)

    def get_stft(self, pred, **kwargs):
        cRM, (noisy_real, noisy_imag) = pred
        cRM_decompressed = decompress_cIRM(cRM)
        noisy_real = noisy_real[:, 0, ...]
        noisy_imag = noisy_imag[:, 0, ...]
        # enhanced_real = cRM[:, 0, None, ...] * noisy_real - cRM[:, 1, None, ...] * noisy_imag
        # enhanced_imag = cRM[:, 1, None, ...] * noisy_real + cRM[:, 0, None, ...] * noisy_imag
        # enhanced_stft = (enhanced_real + 1j * enhanced_imag)[..., 0, :, :]
        # cRM_complex = torch.view_as_complex(cRM.permute(0, 2, 3, 1).contiguous())
        # enhanced_stft = cRM_complex * torch.complex(noisy_real, noisy_imag)[:, 0, ...]
        enhanced_real = cRM_decompressed[..., 0] * noisy_real - cRM_decompressed[..., 1] * noisy_imag
        enhanced_imag = cRM_decompressed[..., 1] * noisy_real + cRM_decompressed[..., 0] * noisy_imag
        enhanced_stft = torch.complex(enhanced_real, enhanced_imag)
        return enhanced_stft.unsqueeze(-3)  # unsqueeze to match B, C, F, T shape

    def get_time(self, pred, length=None):
        enhanced_stft = self.get_stft(pred)
        return self.istft_module(enhanced_stft, length=length)

    def internal_loss(self, pred, target):
        cRM, (noisy_real, noisy_imag) = pred
        S = self.stft_module(target)
        clean_real, clean_imag = S.real, S.imag
        cIRM = build_complex_ideal_ratio_mask(
            noisy_real=noisy_real[:, 0, ...],
            noisy_imag=noisy_imag[:, 0, ...],
            clean_real=clean_real[:, 0, ...],
            clean_imag=clean_imag[:, 0, ...],
        )  # [B, F, T, 2]
        # cRM = cRM.permute(0, 2, 3, 1)
        loss = self.loss_function(cRM, cIRM)
        return loss


if __name__ == "__main__":
    model = FullSubNet(
        num_freqs=257,
        # look_ahead=2,
        look_ahead=5,
        sequence_model="LSTM",
        fb_num_neighbors=0,
        sb_num_neighbors=15,
        fb_output_activate_function="ReLU",
        sb_output_activate_function=False,
        fb_model_hidden_size=512,
        sb_model_hidden_size=384,
        norm_type="offline_laplace_norm",
        num_groups_in_drop_band=1,
        # num_groups_in_drop_band=2,
        weight_init=False,
    )
