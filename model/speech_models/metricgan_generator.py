#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 14:34:32 2024

@author: louis
"""

import torch
from model.utils.abs_models import AbsSpeechModel
from speechbrain.lobes.models.MetricGAN_U import EnhancementGenerator
from speechbrain.processing.features import STFT, spectral_magnitude, ISTFT


class MetricGANGenerator(AbsSpeechModel):
    def __init__(self, metrics: list[torch.nn.Module] = [], speechbrain_ckpt_path=""):
        # self.specific_stft_module = STFT(
        # sample_rate=16000, win_length=32, hop_length=16, n_fft=512, window_fn=torch.hann_window
        # )
        super().__init__(metrics=metrics, crop_input_to_target=True)
        self.model = EnhancementGenerator(input_size=257, hidden_size=200, num_layers=2, lin_dim=300, dropout=0)
        self.min_mask = 0.2
        # self.specific_istft_module = ISTFT(sample_rate=16000, n_fft=512, win_length=32, window_fn=hann_window)

        self.speechbrain_ckpt_path = speechbrain_ckpt_path
        self.load_speechbrain_ckpt()

    # def forward(self):
    # ptit mask, ptit clamp

    def load_speechbrain_ckpt(self):
        if self.speechbrain_ckpt_path:
            print("loading checkpoint: ", self.speechbrain_ckpt_path)
            self.model.load_state_dict(torch.load(self.speechbrain_ckpt_path))

    def forward(self, y):
        noisy_stft = self.stft_module(y)
        noisy_spec = noisy_stft.abs().squeeze(-3).transpose(-1, -2)
        mask = self.model(noisy_spec, lengths=torch.ones(noisy_stft.size(0), device=noisy_spec.device))
        mask = mask.clamp(min=self.min_mask)
        mask_reshaped = mask.unsqueeze(-3).transpose(-1, -2)
        predict_spec = torch.mul(mask_reshaped, noisy_stft)
        return predict_spec

    def get_stft(self, pred, **kwargs):
        return pred

    def get_time(self, pred, length=None):
        return self.istft_module(pred, length=length)

    def internal_loss(self, pred, target):
        target_stft = self.stft_module(target)
        return torch.nn.functional.mse_loss(pred.abs(), target_stft.abs())
