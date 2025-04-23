#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 11:41:33 2024

@author: louis
"""

from model.utils.abs_models import AbsReverbModel, OracleParametersReverbModel
from model.utils.tensor_ops import energy_to_db, db_to_energy, Filterbank, tuple_to_device
import torch
import torchaudio
from torch import nn
import math
import warnings
from model.reverb_models.early_echoes import FixedTimeEarlyEnd
from model.reverb_models.drr import DRR, DRREstimator
import matplotlib.pyplot as plt

# %% Anaysis


def edc(rir: torch.Tensor):
    """
    Energy decay curve.

    Parameters
    ----------
    rir : torch.Tensor
        RIR. Peak should be at index 0.

    Returns
    -------
    torch.Tensor
        Energy Decay Curve.

    """
    power = rir.abs().square()
    return torch.flip(torch.cumsum(torch.flip(power, (-1,)), -1), (-1,))


def tau_from_rt_60(rt_60_in_seconds: float | torch.Tensor, fs: int = 1):
    """Compute energy decay tau from RT60."""
    # fs=1 for tau in seconds
    return rt_60_in_seconds * fs / (3 * math.log(10))


def sigma_from_polack_energy(polack_energy, rt_60, integral_lower_bound=0.0, integral_upper_bound=math.inf, fs=1):
    # rt_60 needs to have the same unit (samples or seconds) as upper and lower bounds for it to work
    tau = tau_from_rt_60(rt_60, fs=fs)
    return torch.sqrt(
        polack_energy
        * 2
        / tau
        / (torch.exp(-2 * integral_lower_bound / tau) - torch.exp(-2 * integral_upper_bound / tau))
    )


def polack_energy(sigma, rt_60, integral_lower_bound=0.0, integral_upper_bound=math.inf, fs=1):
    # rt_60 needs to have the same unit (samples or seconds) as upper and lower bounds for it to work
    tau = tau_from_rt_60(rt_60, fs=fs)
    return (
        sigma**2 * tau / 2 * (torch.exp(-2 * integral_lower_bound / tau) - torch.exp(-2 * integral_upper_bound / tau))
    )


# def simple_linear_regression(x, y, force_origin=False):
#     if force_origin:
#         return torch.mean(x * y) / torch.mean(x**2), 0
#     cov = torch.cov(torch.stack((x, y)))[1, 0]
#     # Compute the variance of x
#     x_var = torch.var(x)
#     # Compute the slope and intercept of the regression line
#     slope = cov / x_var
#     intercept = y.mean() - slope * x.mean()
#     return slope, intercept


# def solve_edc_60dB(edc_db_scaled, force_origin=False, return_linear_approx=False, min_energy=-70.0, max_energy=0.0):
#     if force_origin:
#         indexes_for_regression = torch.where(edc_db_scaled > min_energy)[0]
#     else:
#         indexes_for_regression = torch.where(
#             torch.logical_and(edc_db_scaled <= max_energy, edc_db_scaled > min_energy)
#         )[0]
#     breakpoint()
#     y_for_regression = edc_db_scaled[indexes_for_regression]
#     x_for_regression = indexes_for_regression.float()
#     slope, intercept = simple_linear_regression(x_for_regression, y_for_regression, force_origin=force_origin)
#     rt_60 = -(intercept + 60) / slope
#     if return_linear_approx:
#         return rt_60, (x_for_regression, slope * x_for_regression + intercept)
#     return rt_60


class PolackAnalysis(nn.Module):
    def __init__(
        self,
        regress: bool = True,  # Always True, also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,
        rir_length: int = 16383,
        fs: int = 16000,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        direct_path_duration_ms: float = 2.5,
    ):
        super().__init__()
        if not regress:
            raise NotImplementedError()
        self.regression_min_energy = regression_min_energy
        self.regression_max_energy = regression_max_energy
        self.intersect_zero = intersect_zero
        self.rt_60_depends_from_slope_only = rt_60_depends_from_slope_only
        self.fs = fs
        self.direct_path_duration_ms = direct_path_duration_ms
        self.direct_path_duration_seconds = self.direct_path_duration_ms / 1000
        self.direct_path_duration_samples = round(self.direct_path_duration_seconds * self.fs)

        self.register_buffer("orig_indexes_for_regression", torch.arange(rir_length) / self.fs)

    def forward(self, h):
        edc_h = edc(h)  # + self.epsilon
        edc_db = energy_to_db(edc_h)
        total_energy = edc_db[..., 0, None]
        edc_db_scaled = edc_db - total_energy
        begin_polack_in_samples = torch.argmax(1 * (edc_db_scaled <= self.regression_max_energy), dim=-1, keepdim=True)
        end_polack_in_samples = torch.argmin(1 * (edc_db_scaled > self.regression_min_energy), dim=-1, keepdim=True)
        energy_mask = torch.logical_and(
            edc_db_scaled <= self.regression_max_energy, edc_db_scaled > self.regression_min_energy
        )
        # Clip the mask
        energy_mask[..., : self.direct_path_duration_samples + 1] = False
        begin_polack_in_samples.clamp_(min=self.direct_path_duration_samples + 1)

        edc_db_scaled[~energy_mask] = torch.nan
        indexes_for_regression = self.orig_indexes_for_regression[(h.ndim - 1) * (None,)].repeat(
            *h.shape[:-1], 1
        )  # repeat also copies so it works
        indexes_for_regression[~energy_mask] = torch.nan
        if self.intersect_zero:
            numerator = (indexes_for_regression * edc_db_scaled).nanmean(dim=-1, keepdims=True)
            denominator = (indexes_for_regression**2).nanmean(dim=-1, keepdims=True)
            slope = numerator / denominator
            rt_60 = -60 / slope
        else:
            edc_db_scaled_mean = edc_db_scaled.nanmean(dim=-1, keepdims=True)
            indexes_for_regression_mean = indexes_for_regression.nanmean(-1, keepdims=True)
            numerator = (
                (indexes_for_regression - indexes_for_regression_mean) * (edc_db_scaled - edc_db_scaled_mean)
            ).nansum(dim=-1, keepdims=True)
            denominator = ((indexes_for_regression - indexes_for_regression_mean) ** 2).nansum(dim=-1, keepdims=True)
            slope = numerator / denominator
            intercept = edc_db_scaled_mean - slope * indexes_for_regression_mean
            # rt_60 = -60 / (slope * self.fs)
            if self.rt_60_depends_from_slope_only:
                rt_60 = -60 / slope
            else:
                db_regress_init = (self.regression_max_energy - intercept) / slope
                db_regress_end = (self.regression_min_energy - intercept) / slope
                rt_60 = (
                    -60 / (self.regression_min_energy - self.regression_max_energy) * (db_regress_end - db_regress_init)
                )
        if getattr(self, "plot", False):
            import matplotlib.pyplot as plt

            if self.intersect_zero:
                intercept = torch.zeros_like(slope)
            edc_db_scaled_full = edc_db - total_energy
            num_axes = edc_db_scaled_full.shape[-2] if edc_db_scaled_full.shape[-2] < 32 else 10
            fig, axs = plt.subplots(
                nrows=num_axes,
                ncols=1,
                squeeze=False,
                figsize=(6, 4 * num_axes),
            )
            for i_axis in range(num_axes):
                ax = axs[axs.shape[0] - 1 - i_axis, 0]
                channel_or_band = round(i_axis / num_axes * edc_db_scaled_full.shape[-2])
                # channel or band depends on whether it is scaled or not
                ax.plot(
                    self.orig_indexes_for_regression,
                    edc_db_scaled_full[((0,) * (edc_db_scaled_full.ndim - 2) + (channel_or_band,))],
                    color="tab:blue",
                )
                indexes_to_plot = indexes_for_regression[
                    ((0,) * (indexes_for_regression.ndim - 2) + (channel_or_band,))
                ]
                slope_for_ax = slope[(0,) * (slope.ndim - 2) + (channel_or_band,)]
                intercept_for_ax = intercept[(0,) * (intercept.ndim - 2) + (channel_or_band,)]
                regression_line = slope_for_ax * indexes_to_plot + intercept_for_ax
                regression_line_full = slope_for_ax * self.orig_indexes_for_regression + intercept_for_ax
                ax.plot(indexes_to_plot, regression_line, color="tab:orange")
                ax.plot(
                    self.orig_indexes_for_regression,
                    regression_line_full,
                    "--",
                    color="tab:orange",
                )
                rt_60_for_plot = float(rt_60[(0,) * (rt_60.ndim - 2) + (channel_or_band,)])
                ax.vlines(rt_60_for_plot, -80, 1, label=f"RT60 = {rt_60_for_plot:.2f} s")
                ax.set_ylim(-80, 1)
                ax.set_xlabel("time")
                ax.set_ylabel("Energy (dB)")
                channel_or_band_str = "channel" if edc_db_scaled_full.ndim == 3 else "band"
                ax.set_title(
                    f"EDC (dB) of {channel_or_band_str} {channel_or_band}, (intercept in 0={self.intersect_zero})"
                )
                ax.legend()
            fig.tight_layout()
            # fig.show()
        # Actually, valid_energy == self.regression_max_energy - regression_min_energy but in dB
        # energy_to_db(-h_valid_energy.square().nansum(dim=-1,keepdim=True)+h_invalid_energy.square().nansum(dim=-1,keepdim=True)) - total_energy
        # valid_energy = 10 ** ((self.regression_max_energy - self.regression_min_energy) / 10)
        valid_energy = edc_h.gather(dim=-1, index=begin_polack_in_samples) - edc_h.gather(
            dim=-1, index=end_polack_in_samples
        )
        # we need to compute sigma in samples because the valid_energy is computed in Power \times samples
        sigma = sigma_from_polack_energy(
            valid_energy,
            rt_60=rt_60,
            integral_lower_bound=begin_polack_in_samples,
            integral_upper_bound=end_polack_in_samples,
            fs=self.fs,
        )
        return sigma, rt_60


class DirectToPolackRatio(DRR):
    def __init__(
        self,
        polack_analysis: nn.Module | None = None,
        fs: int = 16000,
        ms_after_peak: float = 2.5,
        assume_peak_at_beginning=True,
    ):
        super().__init__(
            fs=fs,
            ms_after_peak=ms_after_peak,
            assume_peak_at_beginning=assume_peak_at_beginning,
        )
        self.polack_analysis = polack_analysis

    def forward(self, h, sigma=None, rt_60=None):
        if sigma is None or rt_60 is None:
            analyzed_sigma, analyzed_rt_60 = self.polack_analysis(h)
            if sigma is None:
                sigma = analyzed_sigma
            if rt_60 is None:
                rt_60 = analyzed_rt_60
        estimated_polack_energy = polack_energy(
            sigma, rt_60, integral_lower_bound=self.num_samples_after_peak, integral_upper_bound=math.inf, fs=self.fs
        )
        direct_energy = self.direct_energy(h)
        return energy_to_db(direct_energy / estimated_polack_energy)


# %% Synthesis


class ReverberationTimeShortening(nn.Module):
    def __init__(self, sr=16000, time_after_max=0.0025, assume_peak_at_beginning: bool = True):
        super().__init__()
        self.sr = sr
        self.time_after_max = time_after_max
        if not assume_peak_at_beginning:
            raise NotImplementedError()

    def forward(self, rir, original_T60, target_T60):
        """Shorten reverberation time of a RIR.

        See this paper for more details:
            Speech Dereverberation With a Reverberation Time Shortening Target
            https://arxiv.org/abs/2204.08765

        Args:
            rir: given RIR.
            original_T60: the rt60 of the given RIR.
            target_T60: the target rt60.
            sr: sampling rate. Defaults to 16000.
            time_after_max: the time after the maximum of the RIR. Defaults to 0.002.

        Returns:
            The shortened RIR and the window.

        Cite:
            @article{zhou2022single,
                title={Single-Channel Speech Dereverberation using Subband Network with A Reverberation Time Shortening Target},
                author={Zhou, Rui and Zhu, Wenye and Li, Xiaofei},
                journal={arXiv preprint arXiv:2204.08765},
                year={2022}
            }
        """
        assert rir.squeeze().ndim == 1, "rir must be a 1D array."

        q = 3 / (target_T60 * self.sr) - 3 / (original_T60 * self.sr)
        idx_max = 0
        N1 = int(idx_max + self.time_after_max * self.sr)
        win = torch.empty_like(rir)
        win[..., :N1] = 1
        win[..., N1:] = 10 ** (-q * torch.arange(rir.shape[-1] - N1))
        rir_shortened = rir * win
        # return rir , win
        return rir_shortened


class PolackSynthesis(nn.Module):
    def __init__(
        self,
        early_echoes_masking_module: nn.Module,
        rir_length: int = 16383,
        fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
    ):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.rir_length = rir_length
        self.fs = fs
        self.register_buffer("T", torch.arange(self.rir_length) / self.fs)
        self.positive_valued = positive_valued
        self.num_polack_draws = num_polack_draws
        self.fixed_sigma = fixed_sigma

    def expand_v_for_polack_draws(self, v):
        return v.expand(*v.shape[:-2], self.num_polack_draws, -1)

    def compute_v(self, sigma, rt_60, early_reverb_mask):
        if isinstance(early_reverb_mask, torch.Tensor) and early_reverb_mask.ndim == 1:
            early_reverb_mask = early_reverb_mask[(sigma.ndim - 1) * (None,)].expand(*sigma.shape[:-1], -1)
        tau = tau_from_rt_60(rt_60)
        if self.fixed_sigma is not None:
            sigma = self.fixed_sigma
        T_divided_by_tau = self.T / tau
        v = sigma * torch.exp(-T_divided_by_tau)
        v[early_reverb_mask] = 0
        v[~v.isfinite()] = 0  # In case sigma and tau wrongly estimated (because of early echoes and very short rir)
        v = self.expand_v_for_polack_draws(v)
        # ax_ext.plot(v[0].squeeze(), label="variance")
        return v

    def forward(self, sigma, rt_60, rir_properties=dict(), peak_amplitude=1.0):
        early_reverb_mask = self.early_echoes_masking_module.compute_early_mask(rir_properties)
        v = self.compute_v(sigma, rt_60, early_reverb_mask=early_reverb_mask)
        rir = torch.randn_like(v) * v
        rir[..., 0] = peak_amplitude
        if self.positive_valued:
            rir = rir.abs()
        return rir


# %% Analysis-synthesis


class RirToPolack(torch.nn.Module):
    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        regress: bool = True,  # also makes it differentiable and batched
        regression_max_energy: float = -5.0,
        regression_min_energy: float = -25.0,
        analysis_rir_length: int = 16383,
        synthesis_rir_length: int = 16383,
        intersect_zero: bool = False,
        rt_60_depends_from_slope_only: bool = True,  # whether to substract the intersect in the computation of the RT60. Should be true for synthesis using Polack's model
        analysis_normalization_method: str | None = None,
        synthesis_normalization_method: str | None = None,
        analysis_fs: int = 16000,
        synthesis_fs: int = 16000,
        positive_valued: bool = False,
        num_polack_draws: int = 1,
        fixed_sigma: float | None = None,
        move_all_direct_energy_to_peak: bool = False,  # should be true for blind
        direct_path_duration_ms=2.5,
    ):
        super().__init__()
        self.oracle_polack_analysis = PolackAnalysis(
            regress=regress,
            regression_max_energy=regression_max_energy,
            regression_min_energy=regression_min_energy,
            rir_length=analysis_rir_length,
            fs=analysis_fs,
            intersect_zero=intersect_zero,
            rt_60_depends_from_slope_only=rt_60_depends_from_slope_only,
            direct_path_duration_ms=direct_path_duration_ms,
        )
        self.polack_synthesis = PolackSynthesis(
            early_echoes_masking_module=early_echoes_masking_module,
            rir_length=synthesis_rir_length,
            fs=synthesis_fs,
            positive_valued=positive_valued,
            num_polack_draws=num_polack_draws,
            fixed_sigma=fixed_sigma,
        )
        self.analysis_normalization_method = analysis_normalization_method
        self.synthesis_normalization_method = synthesis_normalization_method
        self.oracle_drr_module_for_analysis = DirectToPolackRatio(
            None, fs=analysis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.oracle_drr_module_for_synthesis = DirectToPolackRatio(
            None, fs=synthesis_fs, ms_after_peak=direct_path_duration_ms, assume_peak_at_beginning=True
        )
        self.move_all_direct_energy_to_peak = move_all_direct_energy_to_peak

    def normalize_rir(self, h, stage):
        # Normalize h
        if stage == "analysis":
            normalization_method = self.analysis_normalization_method
            drr_module = self.oracle_drr_module_for_analysis
        else:
            normalization_method = self.synthesis_normalization_method
            drr_module = self.oracle_drr_module_for_synthesis

        if normalization_method is None or normalization_method == "" or "none" in normalization_method.lower():
            return h
        if "peak" in normalization_method.lower():
            return h / (h[..., 0].abs().unsqueeze(-1))
        if "direct" in normalization_method.lower() and "energy" in normalization_method.lower():
            return h / drr_module.direct_energy(h).sqrt()
        if "total" in normalization_method.lower() and "energy" in normalization_method.lower():
            return h / h.abs().square().sum(dim=-1, keepdims=True).sqrt()
        if "rms" in normalization_method.lower():
            return h / h.abs().square().mean(dim=-1, keepdims=True).sqrt()
        else:
            raise ValueError()

    def convert_rir(self, h, rir_properties):
        h_normalized = self.normalize_rir(h, stage="analysis")
        if getattr(self, "plot", False):
            fig, ax = plt.subplots(1, 1)
            ax.plot(
                self.oracle_polack_analysis.orig_indexes_for_regression.cpu().numpy(),
                h_normalized[0].squeeze().detach().cpu().numpy(),
                label="Target RIR (normalized)",
            )

        normalized_peak_from_analysis = h_normalized[..., 0].abs()
        sigma, rt_60 = self.oracle_polack_analysis(h_normalized)
        target_drr_db = self.oracle_drr_module_for_analysis(h_normalized, sigma=sigma, rt_60=rt_60)
        # convert from dB
        target_drr_linear = db_to_energy(target_drr_db)
        # compute reverberant energy at target fs under polack's model
        target_reverberant_energy_at_target_fs = polack_energy(
            sigma,
            rt_60,
            integral_lower_bound=self.oracle_drr_module_for_analysis.direct_path_duration_seconds,
            fs=self.polack_synthesis.fs,
        )
        # compute the target direct energy
        target_direct_energy_at_target_fs = target_drr_linear * target_reverberant_energy_at_target_fs
        if self.move_all_direct_energy_to_peak:
            peak_for_synthesis = target_direct_energy_at_target_fs.sqrt()[..., 0]
        else:
            peak_for_synthesis = normalized_peak_from_analysis
        h_hat = self.polack_synthesis(sigma, rt_60, rir_properties, peak_for_synthesis)
        h_hat_normalized = self.normalize_rir(h_hat, stage="synthesis")
        # print(
        #     f"DRR: {target_drr_db[0].squeeze().item():.2f} -> {energy_to_db(self.oracle_drr_module_for_synthesis.direct_energy(h_hat_normalized)/self.oracle_drr_module_for_synthesis.reverberant_energy(h_hat_normalized))[0].squeeze().item():.2f}"
        # )
        if getattr(self, "plot", False):
            ax.plot(
                self.polack_synthesis.T.cpu().numpy(),
                h_hat_normalized[(0,) * (h_hat.ndim - 1)].squeeze().cpu().numpy(),
                label="Synthesized RIR (normalized)",
            )
            ax.legend()
        return h_hat_normalized

    def forward(self, h, rir_properties):
        return self.convert_rir(h, rir_properties)


class OraclePolackAnalysisSynthesis(RirToPolack, OracleParametersReverbModel): ...



# %% Only RT_60 extraction


class PolackAnalysisRT60Only(PolackAnalysis):
    def forward(self, h):
        _, rt_60 = super().forward(h)
        return rt_60


class PolackToISMRT60Distance(nn.Module):
    def __init__(
        self,
        rt_60_estimator: nn.Module = PolackAnalysisRT60Only(),
    ):
        super().__init__()
        self.rt_60_estimator = rt_60_estimator

    def forward(self, pred, target):
        pred_rt_60 = self.rt_60_estimator(target[0])
        target_rt_60 = target[1]["rt_60"]
        return (pred_rt_60 - target_rt_60).abs().mean()


