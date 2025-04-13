#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 17 12:12:02 2024

@author: louis
"""
import sys
import os

sys.path.append(os.getcwd())

from model.utils.abs_models import AbsJointLossModule
import torch
from torch import nn
import torchaudio
from model.utils.tensor_ops import fftconvolve
from model.utils.default_stft_istft import default_stft_parameters, default_stft_module
import math
from model.utils.abs_models import SignalDomain, ModelType


class RereverberationLoss(AbsJointLossModule):
    MODEL_TYPE = ModelType.REREVERBERATION_LOSS

    def __init__(
        self,
        base_loss: nn.Module,
        metrics: list[nn.Module] = [],
    ):
        super().__init__(metrics=metrics)
        self.base_loss = base_loss


class TimeDomainRereverberationLoss(RereverberationLoss):
    SPEECH_INPUT_DOMAIN = SignalDomain.TIME
    REVERB_INPUT_DOMAIN = SignalDomain.TIME

    def __init__(self, base_loss: nn.Module, metrics: list[nn.Module] = []):
        super().__init__(base_loss=base_loss, metrics=metrics)
        self.fft_convolver = torchaudio.transforms.FFTConvolve()

    def forward(self, s_hat, h_hat, s, h, y):
        y_hat = self.fft_convolver(s_hat, h_hat)
        y = y[..., : y_hat.size(-1)]
        y_hat = y_hat[..., : y.size(-1)]
        y = y.expand(*y.shape[:-2], y_hat.shape[-2], *y.shape[-1:])
        # assert y.shape == y_hat.shape
        return self.base_loss(y_hat, y)


class TimeFrequencyRereverberationLoss(RereverberationLoss):
    SPEECH_INPUT_DOMAIN = SignalDomain.STFT
    REVERB_INPUT_DOMAIN = SignalDomain.TIME

    def __init__(
        self,
        base_loss: nn.Module,
        num_absolute_cross_bands: int = 32,
        metrics: list[nn.Module] = [],
        # n_fft: int = default_stft_parameters["n_fft"],
        # hop_length: int = default_stft_parameters["hop_length"],
        # analysis_window_fn: object = default_stft_parameters["window_fn"],
        # synthesis_window_fn: object = torch.ones,
    ):
        super().__init__(base_loss=base_loss, metrics=metrics)
        self.num_absolute_cross_bands = num_absolute_cross_bands
        self.n_fft = self.stft_module.n_fft
        self.hop_length = self.stft_module.hop_length
        self.register_buffer("analysis_window", self.stft_module.window)
        self.synthesis_window_fn = torch.ones

        self.num_noncausal_coeffs = math.ceil(self.n_fft / self.hop_length) - 1
        self.register_buffer("phi", self._compute_phi())

    def _compute_phi(self):
        synthesis_window = self.synthesis_window_fn(self.n_fft)
        F_prime = self.num_absolute_cross_bands * 2 + 1

        exp_mult_in_sum = torch.exp(
            2j
            * torch.pi
            * torch.outer(
                torch.arange(self.n_fft),
                torch.arange(-self.num_absolute_cross_bands, self.num_absolute_cross_bands + 1),
            )
            / self.n_fft
        )

        phi_sum = torch.zeros(F_prime, 2 * self.n_fft - 1, dtype=exp_mult_in_sum.dtype)
        for m in range(-self.n_fft + 1, self.n_fft):
            internal_sum = torch.zeros(F_prime, dtype=torch.complex64)
            for n in range(0, self.n_fft):
                # for n in range(max(0, -m), min(N,N-m)):
                if 0 <= m + n < self.n_fft:
                    internal_sum += self.analysis_window[n] * synthesis_window[m + n] * exp_mult_in_sum[n, :]
                else:
                    pass
            phi_sum[..., m + self.n_fft - 1] = internal_sum

        f_array = torch.arange(self.n_fft // 2 + 1)
        f_prime = f_array[:, None].expand(-1, F_prime) + torch.arange(
            -self.num_absolute_cross_bands, self.num_absolute_cross_bands + 1
        )[None, :].expand(self.n_fft // 2 + 1, -1)

        f_prime_outer_n = torch.einsum("ij,t -> ijt", f_prime, torch.arange(-self.n_fft + 1, self.n_fft))
        exp_f_prime_outer_n = torch.exp(+2j * torch.pi * f_prime_outer_n / self.n_fft)
        phi = torch.einsum("ijt, jt->ijt", exp_f_prime_outer_n, phi_sum)
        return phi / self.n_fft

    def _construct_S_unfolded(self, S):
        if self.num_absolute_cross_bands > 0:
            S_padded = torch.cat(
                (
                    # torch.zeros_like(S[..., 1 : self.num_absolute_cross_bands + 1, :]),
                    S[..., 1 : self.num_absolute_cross_bands + 1, :].flip(-2).conj(),
                    S,
                    S[..., -self.num_absolute_cross_bands - 1 : -1, :].flip(-2).conj(),
                ),
                dim=-2,
            )
        else:
            S_padded = S
        S_unfolded = S_padded.unfold(-2, 2 * self.num_absolute_cross_bands + 1, 1).mT
        return S_unfolded

    def _compute_ctf(self, h):
        h_phi = fftconvolve(h[..., None, None, :], self.phi[(h.ndim - 1) * [None]])
        ctf = torch.nn.functional.pad(h_phi, (-((self.n_fft - 1) % self.hop_length), 0))[..., :: self.hop_length]
        return ctf

    def _compute_reverberant_spectrogram(self, dry_spectrogram, impulse_response):
        S_hat_unfolded = self._construct_S_unfolded(dry_spectrogram)
        ctf = self._compute_ctf(impulse_response)
        Y_hat = fftconvolve(S_hat_unfolded, ctf).sum(axis=-2)[..., self.num_noncausal_coeffs :]
        return Y_hat

    def forward(self, s_hat, h_hat, s, h, y):
        # s_hat in tf domain
        Y_hat = self._compute_reverberant_spectrogram(s_hat, h_hat)
        Y = self.stft_module(y)
        Y_hat = Y_hat[..., : Y.size(-1)]
        Y = Y[..., : Y_hat.size(-1)]
        Y = Y.expand(*Y.shape[:-3], Y_hat.shape[-3], *Y.shape[-2:])
        # assert Y_hat.shape == Y.shape
        return self.base_loss(Y_hat, Y)

    def validation_step(self, s_hat, h_hat, s, h, y, batch_idx):
        Y_hat = self._compute_reverberant_spectrogram(s_hat, h_hat)
        Y = self.stft_module(y)
        Y_hat = Y_hat[..., : Y.size(-1)]
        Y = Y[..., : Y_hat.size(-1)]
        y_hat = self.istft_module(Y_hat)
        self.log_metrics_and_audios(pred_time_domain=y_hat, target=y, batch_idx=batch_idx)
        loss = self.base_loss(Y_hat, Y)
        self.log_loss(loss, batch_idx=batch_idx)
        return loss


def test_tf_rereverberation_loss():
    import matplotlib.pyplot as plt
    from datasets import (
        WSJSimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
    )
    from model.utils.default_stft_istft import default_stft_module, default_istft_module
    from torchmetrics.audio import SignalNoiseRatio
    from model.utils.metrics import IgnoreLastSamplesMetricWrapper
    import scipy.signal
    import tqdm.auto as tqdm

    torch.manual_seed(0)
    rir_dataset = SynthethicRirDataset(query="rt_60>0.8")
    WSJDataset("data/speech", "test")
    # rir_dataset = SynthethicRirDataset(query='rt_60 > 0.5 and rt_60 < 0.6')
    # rir_dataset = SynthethicRirDataset(query="rt_60 > 0. and rt_60 < 0.3")
    data_module = WSJSimulatedRirDataModule(rir_dataset=rir_dataset, dry_signal_target_len=49151)
    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()

    NUM_BATCHES = 10
    NUM_ABSOLUTE_CROSS_BANDS = [2, 4, 8, 16]

    # device = torch.device("cuda")
    device = torch.device("cpu")

    if device.type == "cuda":
        forward_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)

    default_stft_module = default_stft_module.to(device=device)
    default_istft_module = default_istft_module.to(device=device)
    metric = IgnoreLastSamplesMetricWrapper(SignalNoiseRatio(), num_ignored_last_samples=256 + 16383).to(device=device)

    results = torch.zeros(len(NUM_ABSOLUTE_CROSS_BANDS), NUM_BATCHES, device=device)
    for i_nacb, nacb in enumerate(tqdm.tqdm(NUM_ABSOLUTE_CROSS_BANDS, leave=True)):
        loss_module = TimeFrequencyRereverberationLoss(torch.dist, num_absolute_cross_bands=nacb).to(device=device)
        for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
            if batch_idx >= NUM_BATCHES:
                break
            y, (s, h, _) = batch
            y = y.to(device=device)
            s = s.to(device=device)
            h = h.to(device=device)
            # s = s.to(dtype=torch.float64)
            # h = h.to(dtype=torch.float64)
            # s = torch.zeros_like(s)
            # s[..., 0] = 1
            # s[...] = torch.tensor(scipy.signal.chirp(torch.arange(s.size(-1)) / 16000, f0=20, f1=8000, t1=s.size(-1) / 16000))
            # # s=torch.ones_like(s)
            # h = torch.zeros_like(h)
            # h[..., 0] = 1
            # y = fftconvolve(s, h)
            S = default_stft_module(s)

            if device.type == "cuda":
                forward_start.record()
            Y_hat = loss_module._compute_reverberant_spectrogram(S, h)
            if device.type == "cuda":
                forward_end.record()
                torch.cuda.synchronize()
                print(f"forward timing {forward_start.elapsed_time(forward_end):.2f} ms")
            y_hat = default_istft_module(Y_hat, length=y.size(-1))
            snr = metric(y_hat, y)
            results[i_nacb, batch_idx] = snr
    print(results.mean(axis=-1))
    plt.close("all")
    plt.figure()
    plt.plot(NUM_ABSOLUTE_CROSS_BANDS, results.mean(axis=-1))
    # plt.plot((y_hat).squeeze(), label="$\hat{y}$")
    # plt.plot((y).squeeze(), label="$y$")
    # plt.legend()
    plt.show()


if __name__ == "__main__":
    test_tf_rereverberation_loss()
