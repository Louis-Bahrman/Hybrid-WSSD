#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  3 15:14:51 2023

@author: louis
"""

import torch
import math
from torchaudio.functional import fftconvolve as fftconvolve_torchaudio, filtfilt
from torchaudio.functional.functional import _check_shape_compatible, _check_convolve_mode, _apply_convolve_mode
import scipy
import itertools
from lightning.pytorch.utilities import move_data_to_device


def power_to_db(x):
    #     if x == 0:
    #         return -torch.inf
    return 10 * torch.log10(x)


energy_to_db = power_to_db


def db_to_power(x_db):
    return torch.pow(10, x_db / 10)


db_to_energy = db_to_power


def db_to_amplitude(x_db):
    return db_to_power(x_db).sqrt()


def signal_power_db(signal):
    return power_to_db((torch.abs(signal) ** 2).mean())


def complex_circular_gaussian_noise(mean=0 + 0j, std=math.sqrt(2), **kwargs):
    return torch.complex(torch.normal(mean.real, std, **kwargs), torch.normal(mean.imag, std, **kwargs))


def awgn(original_signal, target_snr_db):
    """
    Adds white gaussian noise whose variance is adjusted to fit a given signal-to-noise ratio

    Parameters
    ----------
    original_signal : np.array
        original signal.
    target_snr_db : float
        target signal-to-noise ratio (in db).

    Returns
    -------
    noisy_signal : np.array
        noisy signal s.t. snr_db(noisy_signal, original_signal) = target_snr_db.

    """
    Px_db = signal_power_db(original_signal)
    var = db_to_power(Px_db - target_snr_db)
    std = torch.sqrt(var)
    if original_signal.dtype.is_complex:
        return complex_circular_gaussian_noise(original_signal, std)
    return torch.normal(original_signal, std)


def arange_last_dim_like(a):
    return torch.arange(a.shape[-1], dtype=a.dtype, device=a.device).expand(*a.shape[:-1], -1)


def zero_pad(a, target_len, dim=-1):
    assert dim == -1
    return torch.nn.functional.pad(a, (0, target_len - a.shape[-1]), mode="constant", value=0)


def crop_or_zero_pad_to_target_len(a, target_len, dim=-1):
    # functional.pad seems to also be able to crop
    return zero_pad(a, target_len, dim=dim)


def fftconvolve_complex(x: torch.Tensor, y: torch.Tensor, mode: str = "full") -> torch.Tensor:
    r"""
    Same as torchaudio.functional.fftconvolve but for complex_valued tensors
    """
    _check_shape_compatible(x, y)
    _check_convolve_mode(mode)

    n = x.size(-1) + y.size(-1) - 1
    fresult = torch.fft.fft(x, n=n) * torch.fft.fft(y, n=n)
    result = torch.fft.ifft(fresult, n=n)
    return _apply_convolve_mode(result, x.size(-1), y.size(-1), mode)


def fftconvolve(x: torch.Tensor, y: torch.Tensor, mode: str = "full", dim=-1) -> torch.Tensor:
    """
    Wrapper around torchaudio.fftconvolve if the inputs aren't real

    Parameters
    ----------
    x : torch.Tensor
        DESCRIPTION.
    y : torch.Tensor
        DESCRIPTION.
    mode : str, optional
        DESCRIPTION. The default is "full".

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    x_transposed = x.transpose(-1, dim)
    y_transposed = y.transpose(-1, dim)
    if x.dtype.is_complex or y.dtype.is_complex:
        res = fftconvolve_complex(x_transposed, y_transposed, mode)
    else:
        res = fftconvolve_torchaudio(x_transposed, y_transposed, mode)
    return res.transpose(-1, dim)


def correlate(x, k, **kwargs):
    return fftconvolve(x, k.flip(-1).conj(), **kwargs)


def test_correlate():
    import scipy

    x = torch.rand(64, dtype=torch.complex64)
    k = torch.rand(64, dtype=torch.complex64)
    print(torch.dist(correlate(x, k), torch.tensor(scipy.signal.correlate(x.numpy(), k.numpy()))))


def toeplitz_transposed_product_fft(x: torch.Tensor, k: torch.Tensor, zero_pad: bool = False) -> torch.Tensor:
    if x.dtype.is_complex or k.dtype.is_complex:
        raise NotImplementedError("There might be a conjugation error. Use the correlation instead")
    # Performs X.T@k where X is toeplitz of coefs x
    res = fftconvolve((x[..., : k.shape[-1]]).flip(-1), k)[..., -k.shape[-1] :]
    if zero_pad:
        return zero_pad(res, x.shape[-1] + k.shape[-1] - 1)
    return res


def toeplitz(x):
    # returns the toeplitz matrix which 1st row is x
    # res = torch.zeros(x.shape + (x.shape[-1],), device=x.device, dtype=x.dtype)
    # for i in range(x.shape[-1]):
    #     res[..., i:] = x[..., :i]
    res = torch.tril(torch.stack([torch.roll(x, shifts=i, dims=-1) for i in range(x.shape[-1])], dim=-1))
    return res


def test_toeplitz():
    import scipy

    x = torch.rand(512)
    x_toeplitz = toeplitz(x)
    x_scipy_toeplitz = torch.tensor(scipy.linalg.toeplitz(x.numpy(), torch.zeros_like(x).numpy()))
    assert torch.allclose(x_toeplitz, x_scipy_toeplitz), torch.dist(x_scipy_toeplitz, x_toeplitz)


def deconvolve_corr(y, x, **kwargs_deconvolve_toeplitz):
    # from https://media.ed.ac.uk/media/Topic+73A+Application+of+Cross-Correlation+to+System+Identification/1_e6662yx1
    x_autocorr = correlate(x, x)
    # x_autocorr = x_autocorr[..., x.shape[-1] - 1 : x.shape[-1] + output_len - 1]  # Only in Jonathan's version
    # x_autocorr = crop_or_zero_pad_to_target_len(x_autocorr, output_len)
    # Rxx = toeplitz(x_autocorr)
    xy_corr = correlate(y, x)
    # xy_corr = xy_corr[..., x.shape[-1] - 1 : x.shape[-1] + output_len - 1]  # only in Jonathan's version
    xy_corr = crop_or_zero_pad_to_target_len(xy_corr, x_autocorr.shape[-1])  # Only in my version
    # xy_corr = crop_or_zero_pad_to_target_len(xy_corr, output_len)  # Only in my version
    # res = torch.linalg.solve_triangular(Rxx, xy_corr[..., None], upper=False)[..., 0] # Bcp moins précis, erreurs énormes
    return deconvolve_toeplitz(xy_corr, x_autocorr, **kwargs_deconvolve_toeplitz)


def deconvolve_toeplitz(y, x, output_len=None, solver="qr"):
    # solve_lstsq to use least-squares
    if output_len is None:
        output_len = y.shape[-1] - x.shape[-1] + 1
    x_toeplitz = toeplitz(zero_pad(x, y.size(-1)))
    if "qr" in solver or "lstsq" in solver or "tri" in solver:
        x_toeplitz = x_toeplitz[..., :output_len]
    if "qr" in solver:
        return solve_lstsq_qr(x_toeplitz, y)
    if "lstsq" in solver:
        return torch.linalg.lstsq(x_toeplitz, y).solution
    if "tri" in solver:
        raise RuntimeWarning("solve_triangular is very unstable")
        return torch.linalg.solve_triangular(
            x_toeplitz[..., :output_len, :output_len],
            y[..., :output_len, None],
            upper=False,
        ).squeeze(-1)
    if "solve" in solver:
        return torch.linalg.solve(x_toeplitz, y)[..., :output_len]
    raise ValueError("unsupported solver")
    # return torch.linalg.solve(toeplitz(x), y[..., : x.size(-1)])[..., : y.size(-1) - x.size(-1) + 1]


def solve_lstsq_qr(A, y):
    # Does the same as torch.linalg.lstsq but faster backward
    Q, R = torch.linalg.qr(A, mode="reduced")
    rhs = Q.mH @ y.unsqueeze(-1)
    return torch.linalg.solve_triangular(R, rhs, upper=True).squeeze(-1)


def deconvolve_fourier(y, x, output_len=None, epsilon=1e-8, use_autocorr=True):
    if output_len is None:
        output_len = y.shape[-1] - x.shape[-1] + 1
    Y = torch.fft.fft(y, y.size(-1))
    X = torch.fft.fft(x, y.size(-1))
    if use_autocorr:
        numerator = Y * X.conj()
        denominator = X * X.conj()
    else:
        numerator = Y
        denominator = X
    return torch.fft.ifft(numerator / (denominator + epsilon))[..., :output_len]


def test_deconvolve_fourier(): ...


def batched_inner_product_3d(x, y, keepdim=False):
    res = torch.linalg.vecdot(x, y)
    if keepdim:
        return res[..., None]
    return res


def white_noise_same_std(t, last_dim_new_size=None):
    if last_dim_new_size is None:
        last_dim_new_size = t.shape[-1]
    noise = torch.randn(t.shape[:-1] + (last_dim_new_size,), device=t.device, dtype=t.dtype)
    return noise * t.std(axis=-1, keepdim=True)


def test_outer_with_lag():
    # Used in FCP
    F = 257
    T = 128
    a = torch.rand(F, T)
    b = torch.rand(F, T)
    b_toeplitz = toeplitz(b.transpose(-1, -2)).permute((-2, -1, -3))
    assert b_toeplitz.shape == (F, F, T)
    assert all(((b_toeplitz[..., t] == b_toeplitz[..., t].tril()).all() for t in range(b_toeplitz.shape[-1])))
    res = a * b_toeplitz
    assert all(
        (res[..., f, f_prime, :] == a[..., f_prime, :] * b[..., f - f_prime, :]).all()
        for f in range(F)
        for f_prime in range(F)
        if f - f_prime >= 0
    )


def compare_deconvolution_methods():
    import itertools

    B, C, F, T = 1, 1, 257, 128
    Tk = 64
    x = torch.rand((B, C, F, T), dtype=torch.complex64)
    k = torch.rand((B, C, F, Tk), dtype=torch.complex64)
    # x = torch.arange(8).to(dtype=torch.complex64) + 1
    # k = torch.arange(4).to(dtype=torch.complex64) + 2
    y = fftconvolve(x, k)
    x = awgn(x, 10)
    for deconvolver, solver in itertools.product(
        (deconvolve_corr, deconvolve_toeplitz),
        [
            "qr",
            "lstsq",
            # "tri",
            # "solve",
        ],
    ):
        k_hat = deconvolver(y, x, solver=solver)
        print(deconvolver.__name__, solver + ":", torch.dist(k_hat, k))
    k_deconvolved_fourier = deconvolve_fourier(y, x)
    print("Fourier", torch.dist(k, k_deconvolved_fourier))


def nansum_complex(x, dim=-1, keepdim=False):
    # here we don't care, itshould be the case that both real and imag are nans
    return torch.complex(x.real.nansum(dim=dim, keepdim=keepdim), x.imag.nansum(dim=dim, keepdim=keepdim))


def tuple_to_device(t, device=torch.device("cpu")):
    if isinstance(t, torch.Tensor):
        return t.to(device=device)
    if isinstance(t, dict):
        return {k: tuple_to_device(v) for k, v in t.items()}
    if t is None:
        return t
    return tuple(tuple_to_device(ti, device=device) for ti in t)


tuple_to_device = move_data_to_device


def autocorrelation(signal, dim: int = -1, return_half: bool = True):
    signal_fourier = torch.fft.fft(signal, n=2 * signal.shape[-1] - 1, dim=dim)
    autocorr_fourier = torch.fft.ifft(signal_fourier * signal_fourier.conj(), dim=dim)
    if return_half:
        if dim != -1:
            raise NotImplementedError()
        return autocorr_fourier[..., : signal.shape[-1] + 1]
    return torch.fft.fftshift(autocorr_fourier, dim=dim)


class Filterbank(torch.nn.Module):
    def __init__(
        self,
        num_evenly_spaced_filters: int | None,
        filter_type: str = "butter",
        filter_order: int = 4,
        fs: int = 16000,
    ):
        super().__init__()
        self.num_evenly_spaced_filters = num_evenly_spaced_filters
        self.filter_type = filter_type
        self.filter_order = filter_order
        self.fs = fs
        filter_cutoffs = torch.linspace(0, self.fs / 2, self.num_evenly_spaced_filters + 1)
        self.band_edges = list(itertools.pairwise(filter_cutoffs))
        if self.filter_type.lower() != "butter":
            raise NotImplementedError()
        filter_coeffs = tuple(zip(*(self.butter_filter(lowcut, highcut) for lowcut, highcut in self.band_edges)))
        maxlen_a = max(len(a) for a in filter_coeffs[1])
        maxlen_b = max(len(b) for b in filter_coeffs[0])
        self.register_buffer(
            "filter_coeffs_a", torch.stack(tuple(zero_pad(torch.tensor(a), maxlen_a) for a in filter_coeffs[1]))
        )
        self.register_buffer(
            "filter_coeffs_b", torch.stack(tuple(zero_pad(torch.tensor(b), maxlen_b) for b in filter_coeffs[0]))
        )

    def butter_filter(self, lowcut, highcut):
        if lowcut == 0:
            return scipy.signal.butter(self.filter_order, highcut, btype="low", output="ba", fs=self.fs)
        if highcut == self.fs / 2:
            return scipy.signal.butter(self.filter_order, lowcut, btype="high", output="ba", fs=self.fs)
        return scipy.signal.butter(self.filter_order, (lowcut, highcut), btype="band", output="ba", fs=self.fs)

    def forward(self, signal):
        # [... t] -> [..., num_filt, t]
        return filtfilt(
            signal.unsqueeze(-2).expand(*signal.shape[:-1], len(self.filter_coeffs_a), -1).to(dtype=torch.double),
            self.filter_coeffs_a,
            self.filter_coeffs_b,
        ).to(dtype=signal.dtype)

    def inverse(self, filterbanked_signal):
        # [..., num_filt, t] ->  [... t]
        assert filterbanked_signal.size(-2) == self.num_evenly_spaced_filters
        return torch.sum(filterbanked_signal, dim=-2)

    def plot_filters(self):
        import matplotlib.pyplot as plt
        import numpy as np

        fig, ax1 = plt.subplots(tight_layout=True)

        ax1.set_title(f"Frequency Response")
        # ax1.axvline(self.fs, color='black', linestyle=':', linewidth=0.8)
        ax1.set_ylabel("Amplitude in dB")
        ax1.set_xlabel("f")
        w, frequency_response_all_filters = np.stack(
            [
                scipy.signal.freqz(filter_coeff_b.numpy(), filter_coeff_a.numpy(), fs=self.fs, worN=4096)
                for (filter_coeff_a, filter_coeff_b) in zip(self.filter_coeffs_a, self.filter_coeffs_b)
            ],
            axis=-1,
        )
        ax1.plot(w, 20 * np.log10(abs(frequency_response_all_filters)))
        ax1.set_ylim(-100, 10)


def test_filterbank():
    n = torch.randn(100, 1, 10000)
    fb = Filterbank(num_evenly_spaced_filters=20)
    n_fb = fb(n)
    n_hat = fb.inverse(n_fb)
    from torchmetrics.audio import ScaleInvariantSignalDistortionRatio

    sisdr = ScaleInvariantSignalDistortionRatio()(n_hat, n)
    print(f"Analysis-synthesis SISDR: {sisdr}")


def OMP(dictionnary, y, n_iter):
    residual = y.clone()
    set_of_indices = torch.full_like(y[..., :n_iter], -1, dtype=torch.long)  # -1 means that it is empty
    for iteration in range(n_iter):
        scalar_product = torch.einsum("...ti,...t->...i", dictionnary, residual)
        current_idx = torch.argmax(scalar_product.abs(), dim=-1)
        set_of_indices[..., iteration] = current_idx
        dict_subset = dictionnary.gather(
            -1, set_of_indices[..., None, : iteration + 1].expand(*set_of_indices.shape[:-1], dictionnary.size(-2), -1)
        )
        optimal_gains = solve_lstsq_qr(dict_subset, y)
        residual = y - torch.einsum("...ti, ...i -> ...t", dict_subset, optimal_gains)
    return set_of_indices, optimal_gains


def test_omp():
    raise NotImplementedError()


class Polynomial(torch.nn.Module):
    def __init__(
        self,
        coeffs: list[float] | tuple[float] | torch.Tensor | None = None,
        deg: int | None = None,
        trainable: bool = False,
    ) -> None:
        """


        Parameters
        ----------
        coeffs : list[float] | torch.Tensor | None, optional
            Coefficients of the polynomial. If None, defaults to identity polynomial of degree deg. The default is None.
        deg : int | None, optional
            Degree of the identity polynomial if coeffs is not specified. The default is None.
        trainable : bool, optional
            Whether the coeffs should be considered as buffers or parameters. The default is False.

        Raises
        ------
        ValueError
            If coeffs and deg are not specified.

        """

        super().__init__()
        self.trainable = trainable
        if coeffs is None and deg is None:
            raise ValueError("Please specify either coeffs or deg")
        if coeffs is None:
            coeffs = torch.zeros(deg + 1)
            coeffs[1] = 1.0
        if self.trainable:
            self.register_parameter("coeffs", torch.nn.Parameter(torch.as_tensor(coeffs)))
        else:
            self.register_buffer("coeffs", torch.as_tensor(coeffs))

    @property
    def deg(self):
        return len(self.coeffs) - 1

    @property
    def degree(self):
        return self.deg

    def __repr__(self):
        return f"Polynomial(coeffs={self.coeffs})"

    def forward(self, x):
        if self.coeffs.ndim > 1:
            raise NotImplementedError(
                "Batching of coeffs in forward not supported yet (need to define wether pointwise or not)"
            )
        x_expanded = x.unsqueeze(-1).expand(*(-1,) * x.ndim, *self.coeffs.shape)
        X = x_expanded ** arange_last_dim_like(x_expanded)
        y = torch.sum(X * self.coeffs[(None,) * x.ndim], dim=-1)
        return y

    def fit(self, x, y, solver="lstsq", flatten: bool = True, inplace: bool = True):
        coeffs = polyfit(x, y, deg=self.deg, solver=solver, flatten=flatten)
        if inplace:
            self.coeffs = coeffs
        return coeffs

    # def __add__(self, other):
    #     return Polynomial(self.coeffs + other.coeffs, trainable=self.trainable or other.trainable)

    # def __iadd__(self, other):
    #     self.coeffs += other.coeffs

    # def __neg__(self):
    #     return Polynomial(-self.coeffs, trainable=self.trainable)

    # def __eq__(self, other):
    #     return self.coeffs == other.coeffs


def polyfit(x, y, deg: int = 2, solver="lstsq", flatten: bool = True):
    if solver.lower() not in ["lstsq", "qr"]:
        raise NotImplementedError()
    if flatten:
        x_flat = x.flatten()
        y_flat = y.flatten()
    else:
        x_flat = x
        y_flat = y
    x_expanded = x_flat.unsqueeze(-1).expand(*(-1,) * x_flat.ndim, deg + 1)
    X = x_expanded ** arange_last_dim_like(x_expanded)
    if "qr" in solver.lower():
        return solve_lstsq_qr(X, y_flat)
    else:
        return torch.linalg.lstsq(X, y_flat).solution


def test_polyfit(deg=4, snr_db=10, flatten: bool = True):
    import matplotlib.pyplot as plt

    x = 4 * torch.rand(8, 2, 1000) - 2
    if flatten:
        x = x.flatten()
    coeffs = 4 * torch.randn(deg + 1)
    x_expanded = x.unsqueeze(-1).expand(*(-1,) * x.ndim, len(coeffs))
    X = x_expanded ** arange_last_dim_like(x_expanded)
    y = torch.sum(X * coeffs[(None,) * x.ndim], dim=-1)
    y_noisy = awgn(y, target_snr_db=snr_db)
    estimated_coeffs = polyfit(x, y_noisy, deg=deg, flatten=flatten)
    if flatten:
        plt.figure()
        plt.plot(x, y, ".", label="target")
        plt.plot(x, y_noisy, ".", label="noisy")
        y_hat = torch.sum(X * estimated_coeffs[(None,) * x.ndim], dim=-1)
        plt.plot(x, y_hat, ".", label="estimated")
        plt.legend()
    print(f"coeffs: real: {coeffs}, estimated: {estimated_coeffs}")


def test_polynomial():
    import matplotlib.pyplot as plt

    poly = Polynomial(deg=3)
    x = 4 * torch.rand(4) - 2
    y = poly(x)
    assert torch.allclose(x, y)

    deg = 3
    coeffs = 4 * torch.randn(deg + 1)
    poly = Polynomial(coeffs=coeffs)
    print(poly)
    x = 4 * torch.rand(1, 1, 1000) - 2
    y = poly(x)
    y_noisy = awgn(y, target_snr_db=10)
    estimated_coeffs = poly.fit(x, y_noisy)
    print(poly)
    plt.figure()
    plt.plot(x[(0,) * (x.ndim - 1)], y[(0,) * (y.ndim - 1)], ".", label="target")
    plt.plot(x[(0,) * (x.ndim - 1)], y_noisy[(0,) * (y_noisy.ndim - 1)], ".", label="noisy")
    poly_hat = Polynomial(coeffs=estimated_coeffs)
    y_hat = poly_hat(x)
    plt.plot(x[(0,) * (x.ndim - 1)], y_hat[(0,) * (y_hat.ndim - 1)], ".", label="estimated")


if __name__ == "__main__":
    compare_deconvolution_methods()
    # test_correlate()
    # test_deconvolve_toeplitz()
    test_outer_with_lag()
    # test_toeplitz()
    # test_deconvolve_corr()
    test_filterbank()
    # test_omp()
    test_polyfit()
    test_polynomial()
