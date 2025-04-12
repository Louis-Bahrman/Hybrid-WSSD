#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 15:36:16 2024

@author: louis
"""
from torchaudio.transforms import Spectrogram as OriginalSpectrogram, InverseSpectrogram as OriginalInverseSpectrogram
import torch
from model.utils.tensor_ops import zero_pad


default_stft_parameters = dict(
    n_fft=512,
    hop_length=256,
    win_length=512,
    window_fn=torch.hann_window,
    center=True,
)


class Spectrogram(OriginalSpectrogram):
    def forward(self, waveform):
        if not self.center:
            raise NotImplementedError()
        waveform_padded = torch.nn.functional.pad(waveform, (self.n_fft // 2, self.n_fft // 2))
        X = super().forward(waveform_padded)
        return X[..., 1:-1]


class InverseSpectrogramCOLA(OriginalInverseSpectrogram):
    def forward(self, spectrogram, length=None):
        if not self.center:
            raise NotImplementedError()
        # pack batch as in original
        # spectrogram = torch.nn.functional.pad(spectrogram, (0, 1))
        shape = spectrogram.size()
        spectrogram = spectrogram.reshape(-1, shape[-2], shape[-1])

        expected_waveform_length = self.n_fft + self.hop_length * (shape[-1] - 1)
        c = torch.fft.irfft(spectrogram, dim=-2)
        waveform = torch.nn.functional.fold(
            c,
            output_size=(1, expected_waveform_length),
            kernel_size=(1, self.n_fft),
            dilation=1,
            padding=0,
            stride=(1, self.hop_length),
        )
        waveform = waveform[..., self.n_fft // 2 :].squeeze(-3, -2)
        if length is not None:
            waveform = zero_pad(waveform, length)

        # unpack batch
        waveform = waveform.reshape(shape[:-2] + waveform.shape[-1:])

        return waveform


default_stft_module = Spectrogram(**default_stft_parameters, power=None)

default_istft_module = InverseSpectrogramCOLA(**default_stft_parameters)


# %% Using nnAudio

USE_NNAUDIO = False

import nnAudio.features

default_stft_parameters_nnaudio = default_stft_parameters.copy()
default_stft_parameters_nnaudio.pop("window_fn")
default_stft_parameters_nnaudio |= dict(window="hann", iSTFT=True)


class NNAudioSpectrogram(nnAudio.features.STFT):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.n_fft = kwargs.get("n_fft")
        self.hop_length = kwargs.get("hop_length")
        self.win_length = kwargs.get("win_length")
        if "hann" in kwargs.get("window"):
            delattr(self, "window")
            self.register_buffer("window", torch.hann_window(self.n_fft))
        else:
            raise NotImplementedError()

    def forward(self, x):
        res = torch.view_as_complex(super().forward(x))
        if x.ndim == 3:
            res = res.unsqueeze(1)
        return res


class NNAudioInverseSpectrogram(torch.nn.Module):
    def __init__(self, stft_module: torch.nn.Module):
        super().__init__()
        self.stft_module = stft_module

    def forward(self, X, length=None):
        if X.ndim == 4 and X.size(1) == 1:
            needs_unsqueeze = True
            X = X.squeeze(1)
        else:
            needs_unsqueeze = False
        res = self.stft_module.inverse(torch.view_as_real(X), length=length, refresh_win=False)
        if needs_unsqueeze:
            return res.unsqueeze(1)
        return res


if USE_NNAUDIO:
    nnAudio_stft_module = NNAudioSpectrogram(**default_stft_parameters_nnaudio)
    default_stft_module = nnAudio_stft_module
    default_istft_module = NNAudioInverseSpectrogram(nnAudio_stft_module)
    default_stft_parameters = default_stft_parameters_nnaudio

# %% Test


def test_stft_istft_single_batch():
    from datasets import WSJDataset
    import matplotlib.pyplot as plt
    import functools
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio
    from model.utils.metrics import IgnoreLastSamplesMetricWrapper

    torch.manual_seed(0)
    dataset = WSJDataset("data/speech", "test")
    s = dataset[23]
    s = s[..., :40000]
    n_fft = 512
    # hop_length, analysis_window, synthesis_window = 512, torch.ones(n_fft), torch.ones(n_fft)
    # hop_length, analysis_window, synthesis_window = 256, torch.hamming_window(n_fft, periodic=True), torch.ones(n_fft)
    hop_length, analysis_window, synthesis_window = 256, torch.hann_window(n_fft, periodic=True), torch.ones(n_fft)

    center = True
    stft_module_torch = functools.partial(
        torch.stft,
        n_fft=n_fft,
        hop_length=hop_length,
        window=analysis_window,
        return_complex=True,
        onesided=True,
        center=center,
    )
    istft_module_torch = functools.partial(
        torch.istft, n_fft=n_fft, hop_length=hop_length, window=analysis_window, center=center
    )

    s_hat = default_istft_module(stft_module_torch(s), length=s.size(-1))
    s_stft_istft = istft_module_torch(stft_module_torch(s))

    print(
        IgnoreLastSamplesMetricWrapper(ScaleInvariantSignalDistortionRatio(), num_ignored_last_samples=256)(
            s_hat[..., : s.size(-1)], s[..., : s_hat.size(-1)]
        )
    )

    plt.close("all")
    plt.figure()
    plt.plot(s.squeeze(), label="ground truth")
    plt.plot(s_stft_istft.squeeze(), label="target")
    plt.plot(s_hat.squeeze(), label="pred")
    plt.legend()


def test_stft_istft_several_batches():
    from datasets import (
        WSJSimulatedRirDataModule,
        SynthethicRirDataset,
        WSJDataset,
    )
    from torchmetrics.audio import SignalNoiseRatio
    from model.utils.metrics import IgnoreLastSamplesMetricWrapper
    import tqdm.auto as tqdm
    from model.utils.default_stft_istft import default_stft_module, default_istft_module

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

    # device = torch.device("cuda")
    device = torch.device("cpu")

    default_stft_module = default_stft_module.to(device=device)
    default_istft_module = default_istft_module.to(device=device)
    metric = IgnoreLastSamplesMetricWrapper(SignalNoiseRatio(), num_ignored_last_samples=256).to(device=device)

    results = torch.zeros(NUM_BATCHES, device=device)
    for batch_idx, batch in enumerate(tqdm.tqdm(loader)):
        if batch_idx >= NUM_BATCHES:
            break
        y, (s, h, _) = batch
        y = y.to(device=device)
        s = s.to(device=device)
        h = h.to(device=device)
        s_hat = default_istft_module(default_stft_module(s), length=s.size(-1))
        results[batch_idx] = metric(s_hat[..., : s.size(-1)], s[..., : s_hat.size(-1)])
    print(results.mean())


if __name__ == "__main__":
    test_stft_istft_several_batches()
