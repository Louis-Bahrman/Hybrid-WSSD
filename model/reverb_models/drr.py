#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 17:08:06 2025

@author: louis
"""

import math
import torch
import nnAudio.features
import torchaudio
from model.utils.tensor_ops import energy_to_db
from torch import nn
import lightning.pytorch as L


class DRR(nn.Module):
    def __init__(self, ms_after_peak: float = 2.5, fs=16000, assume_peak_at_beginning=True):
        super().__init__()
        self.fs = fs
        self.ms_after_peak = ms_after_peak
        self.direct_path_duration_seconds = self.ms_after_peak / 1000
        self.num_samples_after_peak = round(self.direct_path_duration_seconds * self.fs)
        if not assume_peak_at_beginning:
            raise NotImplementedError()

    def direct_energy(self, h):
        return h[..., : self.num_samples_after_peak].abs().square().sum(axis=-1, keepdim=True)

    def reverberant_energy(self, h):
        return h[..., self.num_samples_after_peak :].abs().square().sum(axis=-1, keepdim=True)

    def forward(self, h):
        return energy_to_db(self.direct_energy(h) / self.reverberant_energy(h))


# TODO ajouter bandwise DRR


class DRREstimator(nn.Module): ...


class MackDRREstimator(DRREstimator):
    def __init__(
        self,
        single_mask: bool = True,
        fs: int = 16000,
        frame_length_ms: float = 32.0,
        hop_length_ms: float = 10.0,
        epsilon=1e-8,
        hidden_size: int = 300,
        bandwise: bool = False,
        oracle_drr_module: nn.Module | None = None,
    ):
        super().__init__()
        self.single_mask = single_mask
        self.fs = fs
        n_fft = int(frame_length_ms / 1000 * self.fs)
        self.num_bands = n_fft // 2 + 1
        hop_length = int(hop_length_ms / 1000 * self.fs)
        # self.spec_module = torchaudio.transforms.Spectrogram(
        #     n_fft=n_fft,
        #     hop_length=hop_length,
        #     power=2,
        # )
        self.spec_module = nnAudio.features.stft.STFT(n_fft=n_fft, hop_length=hop_length, output_format="Magnitude")
        self.epsilon = epsilon
        self.bandwise = bandwise
        self.bilstm = nn.LSTM(
            input_size=self.num_bands,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True,
            num_layers=2,
            dropout=0.5,
        )
        self.linear = nn.Linear(in_features=hidden_size * 2, out_features=(2 - self.single_mask) * self.num_bands)
        self.oracle_drr_module = oracle_drr_module
        self.internal_loss = nn.MSELoss()

    def forward(self, y):
        y_normalized = y / y.abs().max(dim=-1, keepdim=True).values
        Y_linear = self.spec_module(y_normalized)
        Y_log = (Y_linear.abs() + self.epsilon).log()
        if Y_linear.ndim == 4:
            # there is a channel dim and it messes up with lstm
            Y_log = Y_log[:, 0, :, :]
        bilstm_output = self.bilstm(Y_log.transpose(-1, -2))[0]
        # breakpoint()
        linear_output = self.linear(bilstm_output)
        mask = linear_output.sigmoid().transpose(-1, -2)
        if self.single_mask:
            X_1 = Y_linear * mask.sqrt()
            X_2 = Y_linear * (1.0 - mask).sqrt()
        else:
            X_1 = Y_linear * mask[..., : self.num_bands, :]
            X_2 = Y_linear * mask[..., self.num_bands :, :]
        if self.bandwise:
            dim_for_sum = -1
        else:
            dim_for_sum = (-1, -2)
        drr = energy_to_db(
            X_1.square().sum(dim=dim_for_sum, keepdim=True)
            / (X_2.square().sum(dim=dim_for_sum, keepdim=True) + self.epsilon)
        )
        return drr  # .squeeze(-1)


def compare_DRR_sampling_rate(num_batches=20):
    import tqdm.auto as tqdm
    from datasets import EARSReverbDataModule, WSJ1SimulatedRirDataModule, SynthethicRirDataset
    import scipy.stats
    import matplotlib.pyplot as plt
    from model.reverb_models.polack import PolackAnalysis, DirectToPolackRatio
    import lightning.pytorch as L

    L.seed_everything(12)

    data_module = EARSReverbDataModule(batch_size=10, return_rir=True, resample_rir=False, enable_caching_val=False)
    drr_48 = DRR(fs=48000, ms_after_peak=2.5, assume_peak_at_beginning=True)
    drr_16 = DRR(fs=16000, ms_after_peak=2.5, assume_peak_at_beginning=True)

    data_module.prepare_data()
    data_module.setup()
    loader = data_module.train_dataloader()
    drrs_48, drrs_16 = [], []
    for i, (y, (x, h_48, rir_properties)) in enumerate(tqdm.tqdm(loader, total=min(len(loader), num_batches))):
        if i >= num_batches:
            break
        h_16 = data_module.dataset_train.resampling_transform(h_48)
        drrs_16.append(drr_16(h_16))
        drrs_48.append(drr_48(h_48))

    drrs_16 = torch.cat(drrs_16).squeeze().cpu()
    drrs_48 = torch.cat(drrs_48).squeeze().cpu()
    print("MSE", nn.MSELoss()(drrs_16, drrs_48))
    print("MAE", nn.L1Loss()(drrs_16, drrs_48))
    print(
        "Pearson correlation",
        round(scipy.stats.pearsonr(drrs_48.numpy(), drrs_16.numpy()).statistic, 2),
    )
    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(8, 8))
    fig_title = "DRRs at 16kHz vs 48kHz"
    fig.suptitle(fig_title)
    upper_graph_lim = max(max(drrs_16), max(drrs_48)) + 1
    lower_graph_lim = min(min(drrs_16), min(drrs_48)) - 1
    # lower_graph_lim = min(min(preds_dpr), min(preds_drr)) - 1
    ax.set_aspect("equal")
    ax.plot(drrs_48, drrs_16, ".")
    ax.set_xlim(lower_graph_lim, upper_graph_lim)
    ax.set_ylim(lower_graph_lim, upper_graph_lim)
    ax.plot([lower_graph_lim, upper_graph_lim], [lower_graph_lim, upper_graph_lim], "--", color="tab:red")
    ax.set_ylabel("DRR @16kHz (dB)")
    ax.set_xlabel("DRR @48kHz (dB)")
    fig.savefig(f"res/2025_01_30/{fig_title}.pdf")
    plt.show()


def compare_direct_to_polack_or_reverberant(start_polack_only_after_direct: bool, num_batches=20):
    import tqdm.auto as tqdm
    from datasets import EARSReverbDataModule, WSJ1SimulatedRirDataModule, SynthethicRirDataset
    import scipy.stats
    import matplotlib.pyplot as plt
    from model.reverb_models.polack import PolackAnalysis, DirectToPolackRatio
    import lightning.pytorch as L

    if start_polack_only_after_direct:
        direct_path_duration_ms = 2.5
    else:
        direct_path_duration_ms = 0.0

    res_dict = dict()
    for data_module_name in ("Synthethic RIRs", "RIRs from EARS @48kHz", "RIRs from EARS @16kHz"):
        L.seed_everything(12)
        if "ears" in data_module_name.lower():
            if "48" in data_module_name.lower():
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=False, enable_caching_val=False
                )
                polack_analysis = PolackAnalysis(
                    rir_length=192000, fs=48000, direct_path_duration_ms=direct_path_duration_ms
                )
                dpr = DirectToPolackRatio(
                    polack_analysis=polack_analysis, fs=48000, ms_after_peak=2.5, assume_peak_at_beginning=True
                )
                drr = DRR(fs=48000, ms_after_peak=2.5, assume_peak_at_beginning=True)
            else:
                data_module = EARSReverbDataModule(
                    batch_size=10, return_rir=True, resample_rir=True, enable_caching_val=False
                )
                polack_analysis = PolackAnalysis(
                    rir_length=64000, fs=16000, direct_path_duration_ms=direct_path_duration_ms
                )
                dpr = DirectToPolackRatio(
                    polack_analysis=polack_analysis, fs=16000, ms_after_peak=2.5, assume_peak_at_beginning=True
                )
                drr = DRR(fs=16000, ms_after_peak=2.5, assume_peak_at_beginning=True)
        else:
            data_module = WSJ1SimulatedRirDataModule(
                rir_dataset=SynthethicRirDataset(rir_root="./data/rirs_v2"),
                dry_signal_target_len=49151,
                batch_size=10,
            )
            polack_analysis = PolackAnalysis(
                rir_length=16383, fs=16000, direct_path_duration_ms=direct_path_duration_ms
            )
            dpr = DirectToPolackRatio(
                polack_analysis=polack_analysis, fs=48000, ms_after_peak=2.5, assume_peak_at_beginning=True
            )
            drr = DRR(fs=16000, ms_after_peak=2.5, assume_peak_at_beginning=True)

        data_module.setup()
        loader = data_module.train_dataloader()
        preds_drr, preds_dpr = [], []
        for i, (y, (x, h, rir_properties)) in enumerate(tqdm.tqdm(loader, total=min(len(loader), num_batches))):
            if i >= num_batches:
                break
            preds_drr.append(drr(h))
            preds_dpr.append(dpr(h))
        preds_drr = torch.cat(preds_drr).squeeze().cpu()
        preds_dpr = torch.cat(preds_dpr).squeeze().cpu()
        print("MSE", nn.MSELoss()(preds_drr, preds_dpr))
        print("MAE", nn.L1Loss()(preds_drr, preds_dpr))
        print(
            "Pearson correlation",
            round(scipy.stats.pearsonr(preds_dpr.numpy(), preds_drr.numpy()).statistic, 2),
        )
        res_dict[data_module_name] = (preds_dpr, preds_drr)

    fig, axs = plt.subplots(1, len(res_dict), sharex=True, sharey=True, figsize=(30, 10))
    fig.suptitle(f"Polack analysis starts only after direct path={start_polack_only_after_direct}")
    upper_graph_lim = max(max((max(preds_dpr), max(preds_drr)) for preds_dpr, preds_drr in res_dict.values())) + 1
    lower_graph_lim = min(min((min(preds_dpr), min(preds_drr)) for preds_dpr, preds_drr in res_dict.values())) - 1
    # lower_graph_lim = min(min(preds_dpr), min(preds_drr)) - 1
    for ax, (data_module_name, (preds_dpr, preds_drr)) in zip(axs, res_dict.items()):
        ax.set_title(data_module_name)
        ax.set_aspect("equal")
        ax.plot(preds_drr[preds_dpr.sort().indices], preds_dpr.sort().values, ".")
        ax.set_xlim(lower_graph_lim, upper_graph_lim)
        ax.set_ylim(lower_graph_lim, upper_graph_lim)
        ax.plot([lower_graph_lim, upper_graph_lim], [lower_graph_lim, upper_graph_lim], "--", color="tab:red")
        ax.set_ylabel("Direct to Polack Ratio (dB)")
        ax.set_xlabel("Direct to Reverberant Ratio (dB)")
    fig.savefig(
        f"res/2025_01_30/direct_to_reverberant_vs_polack_ratios_start_polack_only_after_direct_{start_polack_only_after_direct}.pdf"
    )
    plt.show()


if __name__ == "__main__":
    import torch

    compare_DRR_sampling_rate()
    # compare_direct_to_polack_or_reverberant(start_polack_only_after_direct=True)
    # compare_direct_to_polack_or_reverberant(start_polack_only_after_direct=False)
    # res = MackDRREstimator()(torch.rand(8, 1, 48000))
