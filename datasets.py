#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

This file contains all utilities for dataset management

"""

import math
import copy
import pyroomacoustics as pra
import numpy as np
import itertools
import torch
from torchaudio.datasets import LIBRISPEECH
import torchaudio.transforms
import lightning as L

import tqdm.auto as tqdm
import os
import pandas as pd
import soundfile as sf
import glob

from torch.utils.data import random_split
from model.utils.tensor_ops import (
    crop_or_zero_pad_to_target_len,
    energy_to_db,
    db_to_amplitude,
    zero_pad,
)

# %% utils


def limit_dataset_size(dataset: torch.utils.data.Dataset, limit_size: float | int = 1.0):
    """
    Create a subset of the `dataset` of size `limit_size`.

    Behaviour similar to L.Trainer.overfit_batches but used only for training sets
    (unlike `L.Trainer.overfit_batches` which applies a similar split to validation and test sets)

    This also solves the problem of `l.Trainer.limit_train_batches` which does not create a deterministic subset.

    Parameters
    ----------
    dataset : torch.utils.data.Dataset
        Any dataset, such as EarsReverbDataset
    limit_size : float | int, optional:
        - If >1: number of samples of the dataset
        - If <1: uses a fraction of the dataset
        - The default is 1.0, which returns the dataset itself

    Raises
    ------
    ValueError
        if `limit_size` > `len(dataset)`.

    Returns
    -------
    torch.utils.data.Dataset
        Subset of target size.

    """
    if not isinstance(limit_size, (int, float)):
        raise ValueError("Only float or int limit_sizes are supported")
    if limit_size == 1.0 or limit_size == len(dataset):
        return dataset
    if isinstance(limit_size, float) and limit_size < 1.0:
        num_selected_indices = round(len(dataset) * limit_size)
    else:
        limit_size = int(limit_size)
        if limit_size > len(dataset):
            raise ValueError("limit_size cannot be greater than the len of the dataset")
        num_selected_indices = limit_size
    print(20 * "-" + "\n" + f"Using {num_selected_indices}/{len(dataset)} of the dataset\n" + 20 * "-")
    indices = range(num_selected_indices)
    return torch.utils.data.Subset(dataset, indices)


# %% Synthethic RIR
class SynthethicRirDataset(torch.utils.data.Dataset):
    """Dataset of RIRs synthesized using the Image Source Method (ISM) from PyroomAcoustics."""

    def __init__(
        self,
        rir_root: str = "./data/rirs_v2",  # root of the RIR folder
        num_new_rooms: int = 0,  # number of new rooms to generate
        room_dim_range: tuple[float, float] = (
            5.0,
            10.0,
        ),  # width and length range of the rooms in meters
        room_height_range: tuple[float, float] = (
            2.5,
            4.0,
        ),  # height range of the room in meters
        rt60_range: tuple[float, float] = (0.2, 1.0),  # range of RT_60
        num_sources_per_room: int = 1,  # number of sources. If source_mic_distance is defined, force this to 1.
        num_mics_per_room: int = 16,  # number of mics in the room
        min_distance_to_wall: float = 0.5,  # min distance to wall to stay in the limits of ISM method
        mic_height_range: tuple[float, float] = (
            0.7,
            2,
        ),  # also used for source placement
        fs: int = int(16e3),  # Sample rate of the generated RIR
        query: str = "",  # Used in pandas.Dataframe.filter
        return_properties: list[str] | None = [
            "volume",
            "shoebox_length",
            "shoebox_width",
            "shoebox_height",
            "rt_60",
        ],  # properties to return at sampling
        source_mic_distance_range: tuple[float, float] | None = (
            0.75,
            2.5,
        ),  # None for no constraints on distance
    ):
        """
        Initialize a Synthethic RIR dataset.

        RIRs are sampled using the Image source method (ISM) using pyroomacoustics.

        If `num_new_rooms > 1`, also generates rooms, else only samples from existing ones.

        Parameters
        ----------
        rir_root : str, optional
            The root directory where the RIR data is stored. Defaults to "./data/rirs_v2".

        num_new_rooms : int, optional
            The number of new rooms to generate. Defaults to 0.

        room_dim_range : tuple of float, optional
            A tuple specifying the minimum and maximum width and length (in meters) of the rooms.
            Defaults to (5.0, 10.0).

        room_height_range : tuple of float, optional
            A tuple specifying the minimum and maximum height (in meters) of the room. Defaults to (2.5, 4.0).

        rt60_range : tuple of float, optional
            A tuple specifying the range of RT60 (reverberation time) values for the rooms.
            Defaults to (0.2, 1.0).

        num_sources_per_room : int, optional
            The number of sound sources per room. If the `source_mic_distance_range` is defined, this value is forced to 1. Defaults to 1.

        num_mics_per_room : int, optional
            The number of microphones per room. Defaults to 16.

        min_distance_to_wall : float, optional
            The minimum distance to the wall to stay within the limits of the ISM method. Defaults to 0.5.

        mic_height_range : tuple of float, optional
            A tuple specifying the minimum and maximum height (in meters) for microphone placement.
            Also used for the placement of sources. Defaults to (0.7, 2.0).

        fs : int, optional
            The sample rate (in Hz) of the generated RIRs. Defaults to 16kHz (16000 Hz).

        query : str, optional
            A query string used for filtering data within a pandas DataFrame.
            Example: `query="rt_60 > 0.8"` returns a subset of the dataset with a rt_60 > 0.8.
            Defaults to an empty string.

        return_properties : list of str or None, optional
            A list of properties to return during the sampling process, such as room volume, shoebox dimensions,
            and reverberation time (RT60). If set to None, no properties will be returned. Defaults to:
            ["volume", "shoebox_length", "shoebox_width", "shoebox_height", "rt_60"].

        source_mic_distance_range : tuple of float or None, optional
            A tuple specifying the minimum and maximum distance (in meters) between the sound source and the microphones.
            If None, there are no constraints on the distance. Defaults to (0.75, 2.5).

        """
        # parse data to construct path
        self.rir_root = rir_root
        self.num_new_rooms = num_new_rooms
        self.room_dim_range = room_dim_range
        self.room_height_range = room_height_range
        self.rt60_range = rt60_range
        self.num_sources_per_room = num_sources_per_room
        self.num_mics_per_room = num_mics_per_room
        self.min_distance_to_wall = min_distance_to_wall
        self.mic_height_range = mic_height_range
        self.fs = fs
        self.query = query
        self.return_properties = return_properties or []
        self.source_mic_distance_range = source_mic_distance_range

        self.rir_properties = None
        self.filtered_rir_properties = None

        if self.num_new_rooms == 0:
            self._read_rir_csv()
            self._filter_rir_properties()

        if self.source_mic_distance_range is not None:
            # we force new behaviour
            self.num_mics_per_room = self.num_sources_per_room * self.num_mics_per_room
            self.num_sources_per_room = 1

    @property
    def rir_csv_path(self):
        return os.path.join(self.rir_root, "properties.csv")

    @property
    def num_filtered_rooms(self):
        return len(self.filtered_rir_properties["room_idx"].unique())

    @property
    def num_total_rooms(self):
        return len(self.rir_properties["room_idx"].unique())

    @property
    def num_filtered_rirs(self):
        return len(self.filtered_rir_properties.index)

    @property
    def num_rirs(self):
        return len(self.rir_properties.index)

    def _room_path(self, room_idx):
        return os.path.join(self.rir_root, f"room_{room_idx}")

    def _rir_path(self, room_idx, rir_idx_in_room):
        return os.path.join(self._room_path(room_idx), f"rir_{rir_idx_in_room}.wav")

    def _read_rir_csv(self):
        self.rir_properties = pd.read_csv(self.rir_csv_path, index_col="rir_global_idx")

    def _write_rir_csv(self):
        self.rir_properties.to_csv(self.rir_csv_path, float_format="%.3f")

    def _filter_rir_properties(self):
        if self.query is not None and self.query != "":
            self.filtered_rir_properties = self.rir_properties.query(self.query)
            # if len(self.filtered_rir_properties) == 0:
            #     raise ValueError("filtering returned empty dataset, try widening the search")
        else:
            self.filtered_rir_properties = self.rir_properties

    def generate_data_if_needed(self):
        if self.num_new_rooms > 0:
            self._generate_data()

    def _valid_position_range(self, shoebox_dim):
        return (
            [
                self.min_distance_to_wall,
                self.min_distance_to_wall,
                self.mic_height_range[0],
            ],
            [
                *(shoebox_dim - self.min_distance_to_wall)[:2],
                self.mic_height_range[1],
            ],
        )
        # return np.vstack((np.vstack((2 * [self.min_distance_to_wall], shoebox_dim[:2])).T, self.mic_height_range))
        #     zip(2 * [self.min_distance_to_wall], shoebox_dim[:2] - self.min_distance_to_wall, self.mic_height_range)
        # )

    def _sample_uniform_positions(self, shoebox_dim, num_positions=1):
        return self.rng.uniform(*self._valid_position_range(shoebox_dim), size=(num_positions, 3))

    def _is_valid_position(self, position, shoebox_dim):
        eps = 1e-5
        return (
            (self.min_distance_to_wall - eps <= position).all()
            and (position <= shoebox_dim - self.min_distance_to_wall + eps).all()
            and (self.mic_height_range[0] - eps <= position[..., -1]).all()
            and (position[..., -1] <= self.mic_height_range[1] + eps).all()
        )

    def _generate_data(self):
        self.rng = np.random.default_rng()
        os.makedirs(self.rir_root, exist_ok=True)
        if not os.path.isfile(self.rir_csv_path):
            # generate file

            self.rir_properties = pd.DataFrame(
                columns=[
                    "rir_global_idx",
                    "rir_path",
                    # Room properties
                    "room_idx",
                    "shoebox_length",
                    "shoebox_width",
                    "shoebox_height",
                    "volume",
                    "rt_60",
                    "absorption",
                    # rir_properties
                    "rir_idx_in_room",
                    "source_idx",
                    "mic_idx",
                    "source_x",
                    "source_y",
                    "source_z",
                    "mic_x",
                    "mic_y",
                    "mic_z",
                    "source_mic_distance",
                ]
            ).set_index("rir_global_idx")
            self._write_rir_csv()
            # self.rooms_properties=pd.DataFrame(columns=["room_idx","room_path", "shoebox_dim", "volume", "rt_60", "absorption"]).set_index("room_idx")
            # self._write_rooms_csv()
        # self._read_rooms_csv()
        self._read_rir_csv()
        current_num_new_rooms = 0
        rir_global_idx = self.rir_properties.index.max() + 1 if self.num_rirs > 0 else 1

        room_idx = self.rir_properties["room_idx"].max() + 1 if self.num_rirs > 0 else 1

        with tqdm.tqdm(total=self.num_new_rooms) as pbar:
            while current_num_new_rooms < self.num_new_rooms:
                shoebox_dim = np.zeros(3)
                shoebox_dim[:2] = self.rng.uniform(*self.room_dim_range, 2)
                shoebox_dim[2] = self.rng.uniform(*self.room_height_range)
                rt60 = self.rng.uniform(*self.rt60_range)
                try:
                    absorption, max_order = pra.inverse_sabine(rt60, shoebox_dim)
                except ValueError:
                    # Room too large for rt60
                    pass
                else:
                    room = pra.ShoeBox(
                        shoebox_dim,
                        self.fs,
                        absorption=absorption,
                        max_order=max_order,
                        use_rand_ism=True,
                    )
                    volume = np.prod(room.shoebox_dim)
                    room_path = self._room_path(room_idx)
                    assert not os.path.isdir(room_path), "Error in room_idx generation"
                    # add sources
                    source_pos = self._sample_uniform_positions(shoebox_dim, num_positions=self.num_sources_per_room)
                    assert self._is_valid_position(source_pos, shoebox_dim)
                    for sp in source_pos:
                        room.add_source(sp)

                    # add mic old behaviour: Uniform sampling
                    if self.source_mic_distance_range is None:
                        # old behiaviour
                        mic_pos = self._sample_uniform_positions(shoebox_dim, num_positions=self.num_mics_per_room)
                        assert self._is_valid_position(mic_pos, shoebox_dim)
                        room.add_microphone_array(mic_pos.T)
                    # new behaviour: Sampling of position with rejection if the distance is not within range
                    else:
                        mic_idx = 0
                        while mic_idx < self.num_mics_per_room:
                            mic_pos_within_room = False
                            source_mic_distance = self.rng.uniform(*self.source_mic_distance_range)
                            while not mic_pos_within_room:
                                normal_3d = self.rng.normal(size=3)
                                source_mic_vector = source_mic_distance * normal_3d / np.linalg.norm(normal_3d)
                                assert np.allclose(
                                    np.linalg.norm(source_mic_vector),
                                    source_mic_distance,
                                )
                                mic_pos = source_pos + source_mic_vector
                                mic_pos_within_room = self._is_valid_position(mic_pos, shoebox_dim)
                            mic_idx += 1
                            room.add_microphone(mic_pos.T)

                    room.compute_rir()

                    os.mkdir(room_path)
                    for rir_idx_in_room, (source_idx, mic_idx) in enumerate(
                        itertools.product(
                            range(self.num_sources_per_room),
                            range(self.num_mics_per_room),
                        )
                    ):
                        source_pos = room.sources[source_idx].position
                        mic_pos = room.mic_array.R.T[mic_idx]
                        source_mic_distance = np.linalg.norm(source_pos - mic_pos)
                        rir_path = self._rir_path(room_idx, rir_idx_in_room)
                        rir_properties_dict = {
                            "rir_path": rir_path,
                            # Room properties
                            "room_idx": room_idx,
                            "shoebox_length": shoebox_dim[0],
                            "shoebox_width": shoebox_dim[1],
                            "shoebox_height": shoebox_dim[2],
                            "volume": volume,
                            "rt_60": rt60,
                            "absorption": absorption,
                            # rir_properties
                            "rir_idx_in_room": rir_idx_in_room,
                            "source_idx": source_idx,
                            "mic_idx": mic_idx,
                            "source_x": source_pos[0],
                            "source_y": source_pos[1],
                            "source_z": source_pos[2],
                            "mic_x": mic_pos[0],
                            "mic_y": mic_pos[1],
                            "mic_z": mic_pos[2],
                            "source_mic_distance": source_mic_distance,
                        }

                        self.rir_properties.loc[rir_global_idx] = rir_properties_dict
                        rir = room.rir[mic_idx][source_idx]
                        sf.write(rir_path, rir, self.fs)
                        rir_global_idx += 1

                    room_idx += 1
                    current_num_new_rooms += 1
                    self._write_rir_csv()
                    pbar.update(1)

        self._write_rir_csv()
        self._filter_rir_properties()

    def __len__(self):
        return self.num_filtered_rirs

    def __getitem__(self, idx):
        """
        Return RIR and rir_properties
        -------
        waveform : torch.Tensor
            Shape [1,n] where n is the length (in samples of the RIR).
        other_properties : Dict
            Dict containing the rir_properties.

        """
        # Use iloc
        rir_row = self.filtered_rir_properties.iloc[idx]
        rir_path = rir_row["rir_path"]
        waveform, sample_rate = torchaudio.load(rir_path)
        if sample_rate != self.fs:
            raise ValueError(f"sample rate should be {self.fs}, but got {sample_rate}")
        other_properties = {k: torch.tensor(v).unsqueeze(0) for k, v in rir_row[self.return_properties].items()}
        return waveform, other_properties

    def random_split_by_rooms(self, *proportions):
        """
        Perform random splitting.

        Unlike `torch.utils.data.random_split`,
        ensures that RIRs from the same room are in the same subset

        Parameters
        ----------
        *proportions : float
            proportions of val and test (<1.).
            The proportion of train is automatically computed

        Returns
        -------
        subsets : torch.utils.data.Dataset
            train, val, and test subsets.

        """
        # Train test splits is implemented here since we use room information
        unique_room_idxs = self.filtered_rir_properties["room_idx"].unique()

        # We use torch random split function which is easier and also works with integers
        proportions = (1 - sum(proportions), *proportions)
        rooms_of_each_subset = random_split(unique_room_idxs, proportions)

        subsets = []
        # Only a shallow copy is needed, we will only modify the query method, not the dataframe
        for rooms_of_subset in rooms_of_each_subset:
            subset = copy.copy(self)
            subset.add_filter_to_query(f"room_idx.isin({list(rooms_of_subset)})")
            subset._filter_rir_properties()
            subsets.append(subset)
        return subsets

    def add_filter_to_query(self, filter_to_add):
        if self.query == "":
            self.query = filter_to_add
        else:
            self.query += " & " + filter_to_add


# %% RIR transforms


class ConvolveDryWithEarly(torch.nn.Module):
    """Convolves a dry signal with early reverberation."""

    def __init__(self, early_echoes_masking_module: torch.nn.Module):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.convolution_transform = torchaudio.transforms.FFTConvolve(mode="full")

    def forward(self, x, h, rir_properties):
        h_early = self.early_echoes_masking_module.rir_to_early(h[None, ...].clone(), rir_properties)[0, ...]
        return self.convolution_transform(x, h_early)[..., : x.size(-1)]


class RandomModifyDRR(torch.nn.Module):
    """
    Applies a gain uniformly sampled within a dB range on the first n samples of the rir.

    This causes the direct to reverberant ratio (DRR) to change.
    """

    def __init__(self, min_db_gain=-12, max_db_gain=3, apply_on_first_n_samples=80):
        super().__init__()
        self.min_db_gain = min_db_gain
        self.max_db_gain = max_db_gain
        self.apply_on_first_n_samples = apply_on_first_n_samples

    def forward(self, h, *args):
        assert torch.allclose(h[..., 0].abs(), torch.ones_like(h[..., 0]))
        gain_multiplier_db = (self.max_db_gain - self.min_db_gain) * torch.rand(h.shape[:-1]) + self.min_db_gain
        orig_energy_h_db = energy_to_db(
            h[..., : self.apply_on_first_n_samples].abs().square().sum(dim=-1, keepdim=True)
        )
        gain_db = gain_multiplier_db.unsqueeze(-1)  # + orig_energy_h_db
        gain_linear = db_to_amplitude(gain_db)
        h[..., : self.apply_on_first_n_samples] = gain_linear * h[..., : self.apply_on_first_n_samples]
        return h


class NormalizeEnergy(torch.nn.Module):
    """RMS normalization of the RIR."""

    def forward(self, h):
        h_energy = h.abs().square().sum(dim=-1, keepdim=True)
        return h / h_energy.sqrt()


class DARRirAugmentation(torch.nn.Module):
    """
    RandomModyfyDRR and NormalizeEnergy augmentations.

    Augmentations used in `Differentiable Artificial Reverberation <https://doi.org/10.1109/TASLP.2022.3193298>`_.

    """

    def __init__(self):
        super().__init__()
        self.submodules = torch.nn.Sequential(RandomModifyDRR(), NormalizeEnergy())

    def forward(self, h, *args):
        return self.submodules(h)


class RIRToLate(torch.nn.Module):
    """
    Returns the late reverberation only of a rir.

    The direct path should be at the first sample of the RIR

    """

    def __init__(
        self,
        early_echoes_masking_module: torch.nn.Module,
        include_direct_path: bool = True,  # whether to include the peak of the RIR
    ):
        super().__init__()
        self.early_echoes_masking_module = early_echoes_masking_module
        self.include_direct_path = include_direct_path

    def forward(self, h, rir_properties):
        return self.early_echoes_masking_module.rir_to_late(
            h.clone()[None, ...],
            rir_properties,
            include_direct_path=self.include_direct_path,
        )[0, ...]


# %% Dry and wet datasets and datamodules


def normalize_sox(x: torch.Tensor, sample_rate: int = 16000):
    # dependant on samplerate so should be avoided
    return torchaudio.sox_effects.apply_effects_tensor(x, sample_rate=sample_rate, effects=[["norm"]])[0]


def normalize_max(x: torch.Tensor, target_max: float = 0.5):
    # 0.5 instead of sth ike 0.98 in order to not saturate when convolving with k
    return x / x.abs().max() * target_max


def remove_silent_windows(x: torch.Tensor, silence_power: float = -20.0, window_len: int = 1024):
    x_split = x.split(window_len, dim=-1)
    x_split_nonsilent = [t for t in x_split if energy_to_db(t.norm() ** 2) > silence_power]
    return torch.cat(x_split_nonsilent, dim=-1)


class AudioDatasetConvolvedWithRirDataset(torch.utils.data.Dataset):
    """
    For each audio signal, picks a random rir and convolve.

    return format (wet, (dry, rir, rir_properties))

    """

    def __init__(
        self,
        audio_dataset,
        rir_dataset,
        dry_signal_target_len: int | None = 32767,
        rir_target_len: int | None = 16383,
        align_and_scale_to_direct_path=True,
        dry_signal_start_index=16000,  # None for random start
        pre_associate=False,
        convolve_here: bool = True,  # else convolve afterwards on GPU
        resampling_transform: torch.nn.Module | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
        rir_transforms_for_wet: torch.nn.Module | None = None,  # Applied on both h and y
        dry_only_transforms: torch.nn.Module | None = None,  # Applied on s only, not on y
        rir_only_transforms: torch.nn.Module | None = None,  # applied on h only (not on y or s)
        index_according_to_rir_dataset: bool = False,
        enable_caching: bool = True,
        limit_size: int | float = 1.0,
    ):
        """
        Instantiate a dataset of dry audios convolved with RIRs.

        Parameters
        ----------
        audio_dataset : torch.utils.data.Dataset
            Dataset of dry audios.
        rir_dataset : torch.utils.data.Dataset
            Dataset of RIRs.
        dry_signal_target_len : int | None, optional
            If set, dry signal is cropped or zero-padded to this length. The default is 32767.
        rir_target_len : int | None, optional
            RIR is cropped or zero-padded to this length. The default is 16383.
        align_and_scale_to_direct_path : bool, optional
            Whether to start the RIR at the direct path and set its amplitude to 1.

            The default is True.
        dry_signal_start_index : int, optional
            First sample of the dry signal returned.
            If not set, randomly selected in `(0, len(dry_signal)-dry_signal_target_len)`.
            The default is 16000.
        pre_associate : bool, optional
            Whether to use dynamic mixing

            - If true: no dynamic mixing, each audio will be convolved deterministically with the same RIR at each time it is sampled
            - If false: dynamic mixing, every time an audio is sampled, it will be convolved with a randomly-selected RIR

            The default is False.
        convolve_here : bool, optional
            Whether the convolution between dry and wet should be performed on the CPU or GPU.

            - if True: convolution performed on CPU

            - If False: convolution is delayed to the GPU, in the method `on_after_batch_transfer`
            The wet output of __getitem__ will be a tensor full of `nan`

            The default is True.
        resampling_transform : torchaudio.transforms.Resample | None, optional
            Resampling transform applied on both RIR and dry speech before convolving them.
            The default is None.
        normalize : bool, optional
            Whether to normalize the dry signal before convolution. The default is True.
        ignore_silent_windows : bool, optional
            Whether to remove silences in the dry signal before convolving it with the RIR.
            If True, splits the dry audio in 1024-samples long segments.
            For each segment, checks whether its total energy is greater than 20 dB.
            If not, the segment is discarded.
            The default is True.
        rir_transforms_for_wet : torch.nn.Module | None, optional
            RIR transforms only applied on wet signal, not on the dry signal.
            The default is None.

            Warning, not thoroughly tested
        dry_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the dry signal only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        rir_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the RIR only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        index_according_to_rir_dataset : bool, optional
            Whether to index the dataset by the dry signal or the RIR index.

            - if True: Indexes according to the RIR dataset.
            The length of the convolved dataset will be the length of the RIR dataset.

            - if False: Indexes according to the dry audios dataset.
            The length of the convolved dataset will be the length of `audio_dataset`.

            The default is False.
        enable_caching : bool, optional
            Whether to cache audios and RIRs in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            The default is True.
        limit_size : int | float, optional
            Limit of the size of both RIR and dry audio datasets.
            See `limit_dataset_size`.
            The default is 1.0.

        Raises
        ------
        NotImplementedError
            If one of `rir_transforms_for_wet`, `dry_only_transforms`, or `rir_only_transforms`
            is set and `convolve_here is False` or.the returned RIR properties are not empty.
        """
        self.limit_size = limit_size
        self.audio_dataset = limit_dataset_size(audio_dataset, limit_size=self.limit_size)
        self.rir_dataset = limit_dataset_size(rir_dataset, limit_size=self.limit_size)
        self.pre_associate = pre_associate and len(self.rir_dataset) > 0
        if self.pre_associate:
            self.pre_association = torch.randint(len(self.rir_dataset), size=(len(self.audio_dataset),))
        self.dry_signal_target_len = dry_signal_target_len
        self.rir_target_len = rir_target_len
        self.align_and_scale_to_direct_path = align_and_scale_to_direct_path
        self.dry_signal_start_index = dry_signal_start_index
        self.convolve_here = convolve_here
        self.resampling_transform = resampling_transform
        self.normalize = normalize
        self.normalize_op = normalize_max
        self.ignore_silent_windows = ignore_silent_windows
        self.rir_transforms_for_wet = rir_transforms_for_wet
        self.dry_only_transforms = dry_only_transforms
        self.rir_only_transforms = rir_only_transforms
        self.index_according_to_rir_dataset = index_according_to_rir_dataset
        if index_according_to_rir_dataset and self.pre_associate:
            self.pre_association = torch.randint(len(self.audio_dataset), size=(len(self.rir_dataset),))

        if (self.dry_only_transforms is not None or self.rir_only_transforms is not None) and not self.convolve_here:
            raise NotImplementedError()

        assert self.normalize or not self.ignore_silent_windows, "need to normalize in order to ignore_silent_windows"

        if self.convolve_here:
            self.convolution_transform = torchaudio.transforms.FFTConvolve(mode="full")

        self.enable_caching = enable_caching
        if self.enable_caching and (
            self.rir_transforms_for_wet is not None
            or self.rir_only_transforms is not None
            or self.rir_target_len is None
            or not (self.rir_dataset.return_properties is None or len(self.rir_dataset.return_properties) == 0)
        ):
            raise NotImplementedError()
        if self.enable_caching:
            self.rir_cache = torch.full((len(rir_dataset), 1, self.rir_target_len), torch.nan)
            self.dry_cache = dict()
            self._cache_all_audios()

    def __len__(self):
        if self.index_according_to_rir_dataset:
            return len(self.rir_dataset)
        return len(self.audio_dataset)

    def get_rir(self, rir_idx):
        try:
            if not self.enable_caching:
                raise KeyError()
            rir_aligned_scaled_cropped = self.rir_cache[rir_idx]
            if rir_aligned_scaled_cropped.isnan().any():
                raise KeyError()
            rir_properties = dict()
        except KeyError:
            rir, rir_properties = self.rir_dataset[rir_idx]
            if self.resampling_transform:
                rir = self.resampling_transform(rir)
            # Align and scale dry and RIR to direct path
            if self.align_and_scale_to_direct_path:
                peak_index = torch.argmax(torch.abs(rir))
                rir_peak = rir[..., peak_index]
                rir_aligned = rir[..., peak_index:]
                rir_aligned_scaled = rir_aligned / rir_peak
                if "rt_60" in rir_properties.keys():
                    raise NotImplementedError("I don't know if it is really needed")
                    rir_properties["rt_60"] -= peak_index / self.rir_dataset.fs
            else:
                rir_aligned_scaled = rir
            if self.rir_target_len is not None:
                rir_aligned_scaled_cropped = crop_or_zero_pad_to_target_len(
                    rir_aligned_scaled, target_len=self.rir_target_len
                )
            else:
                rir_aligned_scaled_cropped = rir_aligned_scaled
            if self.enable_caching:
                self.rir_cache[rir_idx] = rir_aligned_scaled_cropped
        return rir_aligned_scaled_cropped, rir_properties

    def get_dry(self, audio_idx):
        try:
            # if not use cache skip to default loading
            if not self.enable_caching:
                raise KeyError()
            x_full = self.dry_cache[audio_idx]
        except KeyError:
            x_full = self.audio_dataset[audio_idx]

            if self.normalize:
                x_full = self.normalize_op(x_full)
            if self.ignore_silent_windows:
                x_full = remove_silent_windows(x_full)

            if self.resampling_transform:
                x_full = self.resampling_transform(x_full)
            if self.enable_caching:
                # put in cache
                self.dry_cache[audio_idx] = x_full
        return x_full

    def __getitem__(self, idx):
        if self.index_according_to_rir_dataset:
            rir_idx = idx
            if self.pre_associate:
                audio_idx = int(self.pre_association[rir_idx])
            else:
                audio_idx = int(torch.randint(len(self.audio_dataset), (1,)))
        else:
            audio_idx = idx
            # Pick RIR
            if self.pre_associate:
                rir_idx = int(self.pre_association[audio_idx])
            else:
                rir_idx = int(torch.randint(len(self.rir_dataset), (1,)))

        # Pick dry, use cache if necessary
        x_full = self.get_dry(audio_idx)

        # get rir, use cache if necessary
        rir_aligned_scaled_cropped, rir_properties = self.get_rir(rir_idx)

        # get section of dry signal
        if self.dry_signal_start_index is None:
            start_index = torch.randint(
                max(1, x_full.shape[-1] - self.dry_signal_target_len),
                size=(1,),
            )[0]
        else:
            start_index = self.dry_signal_start_index
            if start_index >= x_full.size(-1):
                print("tensor not long enough to be cropped, skipping")
                return self.__getitem__(idx + 1)
        x = x_full[..., start_index:]
        # Crop for linear convolution
        if self.dry_signal_target_len is not None:
            x_cropped = crop_or_zero_pad_to_target_len(x, target_len=self.dry_signal_target_len)
        else:
            x_cropped = x
        x_cropped = x_cropped - x_cropped.mean()
        # Perform convolution on align rir and non scaled audio
        # We use x_cropped and not x_scaled_cropped for convolution because we don't want to scale 2 times
        if self.rir_transforms_for_wet is not None:
            rir_transformed_for_wet = self.rir_transforms_for_wet(rir_aligned_scaled_cropped.clone(), rir_properties)
        else:
            rir_transformed_for_wet = rir_aligned_scaled_cropped.clone()
        if self.convolve_here:
            y = self.convolution_transform(rir_transformed_for_wet, x_cropped)
        else:
            y = torch.full(
                x.shape[:-1] + (x_cropped.shape[-1] + rir_transformed_for_wet.shape[-1] - 1,),
                fill_value=torch.nan,
            )
        if self.dry_only_transforms is not None:
            x_transformed = self.dry_only_transforms(x_cropped, rir_aligned_scaled_cropped.clone(), rir_properties)
        else:
            x_transformed = x_cropped
        if self.rir_only_transforms is not None:
            rir_only_transformed = self.rir_only_transforms(rir_aligned_scaled_cropped.clone(), rir_properties)
        else:
            rir_only_transformed = rir_aligned_scaled_cropped.clone()
        return y, (x_transformed, rir_only_transformed, rir_properties)

    def _cache_all_audios(self):
        print("caching dry signals")
        for dry_idx in tqdm.trange(len(self.audio_dataset)):
            _ = self.get_dry(dry_idx)
        print("caching RIRs")
        for rir_idx in tqdm.trange(len(self.rir_dataset)):
            _ = self.get_rir(rir_idx)

    def export(self, base_path: str, fs=16000, crop: int | None = None):
        """
        Export the dataset to files

        Parameters
        ----------
        base_path :
            Path to export the dataset to.
        fs : TYPE, optional
            Sampling rate. The default is 16000.
        crop : int | None, optional
            whether to crop signals to a given length. The default is None.

        """
        os.makedirs(os.path.join(base_path, "wet"))
        os.makedirs(os.path.join(base_path, "dry"))
        os.makedirs(os.path.join(base_path, "rir"))
        properties = []
        for idx in tqdm.trange(len(self)):
            y, (s, h, rir_properties) = self.__getitem__(idx)
            # print(y.abs().max())
            scaling_factor = 1 / torch.maximum(y.abs().max(), s.abs().max())
            y = scaling_factor * y
            s = scaling_factor * s
            properties.append({k: v.item() for k, v in rir_properties.items()})
            if crop is not None:
                y = y[..., :crop]
                s = s[..., :crop]
            torchaudio.save(
                os.path.join(base_path, "wet", str(idx) + ".wav"),
                y,
                sample_rate=fs,
            )
            torchaudio.save(
                os.path.join(base_path, "dry", str(idx) + ".wav"),
                s,
                sample_rate=fs,
            )
            torchaudio.save(
                os.path.join(base_path, "rir", str(idx) + ".wav"),
                h,
                sample_rate=fs,
            )
        df = pd.DataFrame(properties)
        df.index.name = "idx"
        df.to_csv(os.path.join(base_path, "properties.csv"), float_format="%.3f")


class AudioDatasetConvolvedWithRirDatasetDataModule(L.LightningDataModule):
    """
    Abstract class to combine audio dataset and RIR dataset.

    You should inherit from this class and define your own "split_audio_dataset" method
    """

    def __init__(
        self,
        rir_dataset: torch.utils.data.Dataset,  # Dataset used at train and val
        rir_dataset_test: torch.utils.data.Dataset | None = None,  # for testing only
        batch_size: int = 8,
        audio_root: str = "./data/speech",
        dry_signal_target_len: int = 32767,
        rir_target_len: int = 16383,
        align_and_scale_to_direct_path: bool = True,
        dry_signal_start_index_train: int | None = None,
        dry_signal_start_index_val_test: int | None = 16000,
        proportion_val_audio: float | None = 0.1,  # Switch behaviour to use ready-made val split or not
        proportion_val_rir: float = 0.1,
        num_workers: int = 8,
        convolve_on_gpu: bool = False,
        resampling_transform: torch.nn.Module | None = None,
        num_distinct_rirs_per_batch: int | None = None,
        normalize: bool = True,
        ignore_silent_windows: bool = True,
        rir_transforms_for_wet: torch.nn.Module | None = None,
        rir_only_transforms: torch.nn.Module | None = None,
        dry_only_transforms: torch.nn.Module | None = None,
        index_according_to_rir_dataset: bool = False,
        dynamic_mixing_at_training: bool = True,
        prefetch_factor: int = 2,
        enable_caching_train: bool = False,
        enable_caching_val: bool = False,
        limit_training_size: float | int = 1.0,
    ):
        """
        Instantiate `AudioDatasetConvolvedWithRirDatasetDataModule`.

        Parameters
        ----------
        rir_dataset :
            RIR dataset used at train and val
        rir_dataset_test :
            RIR dataset used for testing only
        batch_size :
            batch_size
        audio_root :
            Dry signal dataset root
        dry_signal_target_len : int | None, optional
           If set, dry signal is cropped or zero-padded to this length. The default is 32767.
        rir_target_len : int | None, optional
            RIR is cropped or zero-padded to this length. The default is 16383.
        align_and_scale_to_direct_path : bool, optional
            Whether to start the RIR at the direct path and set its amplitude to 1.
            The default is True.
        dry_signal_start_index_train : int, optional
            First sample of the dry signal returned.
            If not set, randomly selected in `[0,len(dry_signal)-dry_signal_target_len]`.
            The default is None.
        dry_signal_start_index_val_test : int, optional
            Dry signal start on val and test sets
            If not set, randomly selected in `(0, len(dry_signal)-dry_signal_target_len)`.
            The default is 16000.
        proportion_val_audio :
            Proportion of the dry audio training set used for validation
        proportion_val_rir :
            Proportion of the RIR training set used for validation
        num_workers : int
            See `torch.utils.data.DataLoader`
        convolve_on_gpu : bool, optional
            Whether the convolution between dry and wet should be performed on the CPU or GPU.

            - if False: convolution performed on CPU

            - If True: convolution is delayed to the GPU, in the method `on_after_batch_transfer`

            The default is True.
        resampling_transform : torchaudio.transforms.Resample | None, optional
            Resampling transform applied on both RIR and dry speech before convolving them.
            The default is None.
        num_distinct_rirs_per_batch :
            If set, several dry audios will be convolved with the same RIR inside a batch.
            See `on_before_batch_transfer` method.
            If None, defaults to regular dynamic mixing.
            Warning, not thoroughly tested.
            Defaults to None.
        normalize : bool, optional
            Whether to normalize the dry signal before convolution. The default is True.
        ignore_silent_windows : bool, optional
            Whether to remove silences in the dry signal before convolving it with the RIR.
            If True, splits the dry audio in 1024-samples long segments.
            For each segment, checks whether its total energy is greater than 20 dB.
            If not, the segment is discarded.
            The default is True.
        rir_transforms_for_wet : torch.nn.Module | None, optional
            RIR transforms only applied on wet signal, not on the dry signal.
            The default is None.

            Warning, not thoroughly tested
        dry_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the dry signal only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        rir_only_transforms : torch.nn.Module | None, optional
            Transforms applied on the RIR only, after the wet signal has been created.
            The default is None.

            Warning, not thoroughly tested
        index_according_to_rir_dataset : bool, optional
            Whether to index the dataset by the dry signal or the RIR index.

            - if True: Indexes according to the RIR dataset.
            The length of the convolved dataset will be the length of the RIR dataset.

            - if False: Indexes according to the dry audios dataset.
            The length of the convolved dataset will be the length of `audio_dataset`.

            The default is False.
        dynamic_mixing_at_training : bool, optional
            Whether to use dynamic mixing at training. Defaults to True.
        prefetch_factor:
            See `torch.utils.data.DataLoader`
        enable_caching_train : bool, optional
            Whether to cache audios and RIRs from the training set in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            The default is False.
        enable_caching_val : bool, optional
            Whether to cache audios and RIRs from the val set in the CPU RAM for faster access.
            If so, `__init__` might take some time.
            Caching is never enabled for test, since only one pass of the dataset is done.
            The default is False.
        limit_training_size : int | float, optional
            Limit of the size of both RIR and dry audio datasets at training.
            See `limit_dataset_size`.
            The default is 1.0.

        """
        super().__init__()
        self.save_hyperparameters(
            ignore=[
                "rir_dataset",
                "rir_dataset_test",
                "rir_further_transforms",
                "dry_only_transforms",
                "rir_only_transforms",
                "rir_transforms_for_wet",
            ]
        )
        self.save_hyperparameters(ignore=[], logger=False)

        self.rir_dataset = rir_dataset
        self.rir_dataset_test = rir_dataset_test
        if not os.path.isdir(self.hparams.audio_root):
            os.makedirs(self.hparams.audio_root)

        self.convolution_transform = torchaudio.transforms.FFTConvolve(mode="full")

    def prepare_data(self):
        if hasattr(self.rir_dataset, "generate_data_if_needed"):
            self.rir_dataset.generate_data_if_needed()

    def split_audio_dataset(self, stage=None):
        raise NotImplementedError()

    def setup(self, stage=None):
        self.split_audio_dataset()
        assert hasattr(self, "dry_train") and hasattr(self, "dry_val") and hasattr(self, "dry_test")
        self.generate_convolved_datasets()

    def generate_convolved_datasets(self):
        self.rir_dataset_train, self.rir_dataset_val = self.rir_dataset.random_split_by_rooms(
            self.hparams.proportion_val_rir
        )
        # Create the custom 'merged' datasets
        self.dataset_train = AudioDatasetConvolvedWithRirDataset(
            self.dry_train,
            self.rir_dataset_train,
            pre_associate=not self.hparams.dynamic_mixing_at_training,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_train,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
            rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
            dry_only_transforms=self.hparams.dry_only_transforms,
            rir_only_transforms=self.hparams.rir_only_transforms,
            index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
            enable_caching=self.hparams.enable_caching_train,
            limit_size=self.hparams.limit_training_size,
        )
        self.dataset_val = AudioDatasetConvolvedWithRirDataset(
            self.dry_val,
            self.rir_dataset_val,
            pre_associate=True,
            dry_signal_target_len=self.hparams.dry_signal_target_len,
            rir_target_len=self.hparams.rir_target_len,
            align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
            dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
            convolve_here=not self.hparams.convolve_on_gpu
            and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
            resampling_transform=self.hparams.resampling_transform,
            normalize=self.hparams.normalize,
            ignore_silent_windows=self.hparams.ignore_silent_windows,
            rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
            dry_only_transforms=self.hparams.dry_only_transforms,
            rir_only_transforms=self.hparams.rir_only_transforms,
            index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
            enable_caching=self.hparams.enable_caching_val,
            limit_size=1.0,
        )
        if self.rir_dataset_test is not None:
            self.dataset_test = AudioDatasetConvolvedWithRirDataset(
                self.dry_test,
                self.rir_dataset_test,
                pre_associate=True,
                dry_signal_target_len=self.hparams.dry_signal_target_len,
                rir_target_len=self.hparams.rir_target_len,
                align_and_scale_to_direct_path=self.hparams.align_and_scale_to_direct_path,
                dry_signal_start_index=self.hparams.dry_signal_start_index_val_test,
                convolve_here=not self.hparams.convolve_on_gpu
                and self.num_distinct_rirs_per_batch == self.hparams.batch_size,
                resampling_transform=self.hparams.resampling_transform,
                normalize=self.hparams.normalize,
                ignore_silent_windows=self.hparams.ignore_silent_windows,
                rir_transforms_for_wet=self.hparams.rir_transforms_for_wet,
                dry_only_transforms=self.hparams.dry_only_transforms,
                rir_only_transforms=self.hparams.rir_only_transforms,
                index_according_to_rir_dataset=self.hparams.index_according_to_rir_dataset,
                enable_caching=False,
                limit_size=1.0,
            )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=True,
            prefetch_factor=(self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=(self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def test_dataloader(self):
        assert self.rir_dataset_test is not None
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=(self.hparams.prefetch_factor if self.hparams.num_workers > 0 else None),
            persistent_workers=True if self.hparams.num_workers > 0 else False,
        )

    def predict_dataloader(self):
        return self.test_dataloader()

    @property
    def num_distinct_rirs_per_batch(self):
        if self.hparams.num_distinct_rirs_per_batch is None:
            return self.hparams.batch_size
        else:
            return self.hparams.num_distinct_rirs_per_batch

    def on_before_batch_transfer(self, batch, dataloader_idx):
        # If num_distinct_rirs_per_batch is set
        if isinstance(batch, (list, tuple)) and self.num_distinct_rirs_per_batch < self.hparams.batch_size:
            if self.hparams.batch_size % self.num_distinct_rirs_per_batch != 0:
                raise RuntimeError(
                    f"num_distinct_rirs_per_batch={self.num_distinct_rirs_per_batch} should divide batch size={batch.shape[0]}"
                )
            num_rirs_repeats = self.hparams.batch_size // self.num_distinct_rirs_per_batch
            batch[1][1] = batch[1][1][: self.num_distinct_rirs_per_batch, ...]
            if not self.hparams.convolve_on_gpu:
                batch[0] = self.convolution_transform(
                    batch[1][0],
                    batch[1][1].repeat_interleave(num_rirs_repeats, dim=0),
                )
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        # to perform convolution on gpu
        # https://lightning.ai/docs/pytorch/stable/data/datamodule.html#on-after-batch-transfer
        if self.hparams.convolve_on_gpu and isinstance(batch, (list, tuple)):
            if self.num_distinct_rirs_per_batch < self.hparams.batch_size:
                # Check correctness
                num_rirs_repeats = self.hparams.batch_size // self.num_distinct_rirs_per_batch
                batch[0] = self.convolution_transform(
                    batch[1][0],
                    batch[1][1].repeat_interleave(num_rirs_repeats, dim=0),
                )
            else:
                batch[0] = self.convolution_transform(batch[1][0], batch[1][1])
        # assert batch[0].isfinite().all()
        return batch


# %% WSJ + synthethic


class WSJDataset(torch.utils.data.Dataset):
    """WSJ0"""

    EXPECTED_SAMPLERATE = 16000
    TRAIN_TEST_DISKS = {
        "train": range(1, 13),
        "test": range(14, 16),
    }

    @property
    def wav_root(self):
        return os.path.join(
            self.audio_root,
            "WSJ",
            "WSJ0_wav_mic" + str(self.mic_number),
            self.subset,
        )

    @property
    def sphere_root(self):
        return os.path.join(self.audio_root, "WSJ", "csr_1")

    @property
    def base_path(self):
        if self.wav:
            return self.wav_root
        else:
            return self.sphere_root

    def _check_base_path_exists(self):
        if not os.path.isdir(self.base_path):
            raise ValueError("Path does not exist or is not WSJ0")

    def __init__(
        self,
        audio_root: str = "./data/speech",
        subset: str = "train",
        mic_number: int = 1,
        wav: bool = True,
    ):
        """
        Instantiate WSJ.

        Parameters
        ----------
        audio_root : str, optional
            DESCRIPTION. The default is "./data/speech".
        subset : str, optional
            "train" or "test". The default is "train".
        mic_number : int, optional
            DESCRIPTION. The default is 1.
        wav : bool, optional
            Whether to load the audio as a wav file (True) or the native sphere format (False).
            The default is True.

        """
        self.audio_root = audio_root
        self.subset = subset
        self.mic_number = mic_number
        self.wav = wav
        self.paths_list = []

        self._check_base_path_exists()

        if self.wav:
            self.paths_list = sorted(glob.glob(os.path.join(self.base_path, "*.wav")))
        else:
            for i_disk in self.TRAIN_TEST_DISKS[subset]:
                self.paths_list.extend(
                    sorted(
                        glob.glob(
                            os.path.join(
                                self.base_path,
                                "11-" + str(i_disk) + ".1",
                                "**",
                                "*.wv" + str(self.mic_number),
                            ),
                            recursive=True,
                        )
                    )
                )

    @property
    def len_hours(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths_list):
            total_len += torchaudio.info(path).num_frames
        return total_len / self.EXPECTED_SAMPLERATE / 3600

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        assert fs == self.EXPECTED_SAMPLERATE
        return x

    def export_to_wav(self, new_path=None):
        if new_path is None:
            new_path = self.wav_root
        os.makedirs(new_path)
        for i in tqdm.trange((len(self))):
            x = self[i]
            torchaudio.save(os.path.join(new_path, f"{i}.wav"), x, self.EXPECTED_SAMPLERATE)


WSJ0Dataset = WSJDataset


class WSJ1Dataset(torch.utils.data.Dataset):
    """
    WSJ1 dataset.

    Test tasks are Hub 1 and 2, and spokes S1 to S4 and S9.
    S5 to S8 are excluded since they are noisy.

    """

    EXPECTED_SAMPLERATE = 16000
    TRAIN_TEST_DISKS = {
        "train": range(1, 32),
        "test": range(33, 35),
    }
    # we exclude s5-8 which deal with noisy data
    RELEVANT_TEST_TASKS = [
        "h1",
        "h2",
        "s1",
        "s2",
        "s3",
        "s4",
        "s9",
    ]

    @property
    def wav_root(self):
        return os.path.join(
            self.audio_root,
            "WSJ",
            "WSJ1_wav_mic" + str(self.mic_number),
            self.subset,
        )

    @property
    def sphere_root(self):
        return os.path.join(self.audio_root, "WSJ", "csr_2_comp")

    @property
    def base_path(self):
        if self.wav:
            return self.wav_root
        else:
            return self.sphere_root

    def _check_base_path_exists(self):
        if not os.path.isdir(self.base_path):
            raise ValueError("Path does not exist or is not WSJ1")

    def __init__(
        self,
        audio_root: str = "./data/speech",
        subset: str = "train",
        mic_number: int = 1,
        wav: bool = True,
    ):
        """
        Instantiate WSJ1 dataset.

        The test tasks are Hub 1 and 2, and spokes S1 to S4 and S9. S5 to S8 are excluded since they are noisy.

        Parameters
        ----------
        audio_root : str, optional
            Root in which WSJ folder can be found. The default is "./data/speech".
        subset : str, optional
            "train" or "test". The default is "train".
        mic_number : int, optional
            Mic number, 1 for headset (not reverberant). The default is 1.
        wav : bool, optional
            Whether data has been exported to wav beforehand. Else tries to load native sphere format. The default is True.

        """
        self.audio_root = audio_root
        self.subset = subset
        self.mic_number = mic_number
        self.wav = wav
        self.paths_list = []

        self._check_base_path_exists()

        if self.wav:
            self.paths_list = sorted(glob.glob(os.path.join(self.base_path, "*.wav")))
        else:
            if "train" in subset.lower():
                with open(
                    os.path.join(
                        self.sphere_root,
                        "13-32.1/wsj1/doc/indices/wsj1/train/tr_s_wv1.ndx",
                    )
                ) as f:
                    self.paths_list.extend([self.process_line(line) for line in f if not line.startswith(";;")])
                with open(
                    os.path.join(
                        self.sphere_root,
                        "13-32.1/wsj1/doc/indices/wsj1/train/tr_l_wv1.ndx",
                    )
                ) as f:
                    self.paths_list.extend([self.process_line(line) for line in f if not line.startswith(";;")])
                # self.paths_list = list(dict.fromkeys(self.paths_list))  # remove duplicates
            else:
                # https://catalog.ldc.upenn.edu/docs/LDC94S13A/csrnov93.html
                relevant_index_files = os.listdir(os.path.join(self.sphere_root, "13-32.1/wsj1/doc/indices/wsj1/eval"))
                relevant_index_files = [
                    ndx for ndx in relevant_index_files if ndx.startswith(tuple(self.RELEVANT_TEST_TASKS))
                ]

                for ndx_file in relevant_index_files:
                    with open(
                        os.path.join(
                            self.sphere_root,
                            "13-32.1/wsj1/doc/indices/wsj1/eval",
                            ndx_file,
                        )
                    ) as f:
                        self.paths_list.extend([self.process_line(line) for line in f if not line.startswith(";;")])
                # remove duplicates
                self.paths_list = list(dict.fromkeys(self.paths_list))

    def process_line(self, line):
        # wrong disk between 32 and 33 in test set
        if line.startswith("13_32_1:wsj1/si_et_") and not line.startswith("13_32_1:wsj1/si_et_s9"):
            line_list = list(line)
            line_list[4] = "3"
            line = "".join(line_list)
        if line.startswith("13_33_1:wsj1/si_et_s9"):
            line_list = list(line)
            line_list[4] = "2"
            line = "".join(line_list)
        line = line.rstrip()
        disk, end = line.split(":")
        d1, d2, d3 = disk.split("_")
        return self.sphere_root + "/" + d1 + "-" + d2 + ".1" + "/" + end

    @property
    def len_hours(self):
        total_len = 0
        for path in tqdm.tqdm(self.paths_list):
            total_len += torchaudio.info(path).num_frames
        return total_len / self.EXPECTED_SAMPLERATE / 3600

    def __len__(self):
        return len(self.paths_list)

    def __getitem__(self, index):
        path = self.paths_list[index]
        x, fs = torchaudio.load(path)
        assert fs == self.EXPECTED_SAMPLERATE
        return x

    def export_to_wav(self, new_path=None):
        if new_path is None:
            new_path = self.wav_root
        os.makedirs(new_path)
        for i in tqdm.trange((len(self))):
            x = self[i]
            torchaudio.save(os.path.join(new_path, f"{i}.wav"), x, self.EXPECTED_SAMPLERATE)


class WSJSimulatedRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    NUM_OF_VALS = NotImplemented

    def split_audio_dataset(self, stage=None):
        self.wsj_train_full = WSJDataset(self.hparams.audio_root, subset="train")
        self.dry_test = WSJDataset(self.hparams.audio_root, subset="test")
        if self.hparams.proportion_val_audio is not None:
            proportions_wsj = (
                1 - self.hparams.proportion_val_audio,
                self.hparams.proportion_val_audio,
            )
            self.dry_train, self.dry_val = random_split(self.wsj_train_full, proportions_wsj)
        else:
            self.dry_val = torch.utils.data.Subset(self.wsj_train_full, range(self.NUM_OF_VALS + 1))
            self.dry_train = torch.utils.data.Subset(
                self.wsj_train_full,
                range(self.NUM_OF_VALS + 1, len(self.wsj_train_full)),
            )


class WSJ1SimulatedRirDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    NUM_OF_VALS = NotImplemented

    def split_audio_dataset(self, stage=None):
        self.wsj_train_full = WSJ1Dataset(self.hparams.audio_root, subset="train", wav=True)
        self.dry_test = WSJ1Dataset(self.hparams.audio_root, subset="test", wav=True)
        if self.hparams.proportion_val_audio is not None:
            proportions_wsj = (
                1 - self.hparams.proportion_val_audio,
                self.hparams.proportion_val_audio,
            )
            self.dry_train, self.dry_val = random_split(self.wsj_train_full, proportions_wsj)
        else:
            self.dry_val = torch.utils.data.Subset(self.wsj_train_full, range(self.NUM_OF_VALS + 1))
            self.dry_train = torch.utils.data.Subset(
                self.wsj_train_full,
                range(self.NUM_OF_VALS + 1, len(self.wsj_train_full)),
            )


# %% Paired Data


class PairedDataset(torch.utils.data.Dataset):
    """
    Paired dataset

    Can be the result of `AudioDatasetConvolvedWithRirDataset.export`.
    Is meant to be used for testing.
    """

    def __init__(self, path="./data/test_wsj1_same"):
        self.path = path
        self.df = pd.read_csv(os.path.join(path, "properties.csv"))

    def __len__(self):
        return len(os.listdir(os.path.join(self.path, "wet")))

    def __getitem__(self, index):
        y, _ = torchaudio.load(os.path.join(self.path, "wet", str(index) + ".wav"))
        s, _ = torchaudio.load(os.path.join(self.path, "dry", str(index) + ".wav"))
        h, _ = torchaudio.load(os.path.join(self.path, "rir", str(index) + ".wav"))
        rir_properties = {
            k: torch.tensor(v).unsqueeze(0) for k, v in self.df.loc[index].to_dict().items() if k != "idx"
        }
        return y, (s, h, rir_properties)


class PairedDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    """
    DataModule associated with paired dataset

    Can be the result of `AudioDatasetConvolvedWithRirDataset.export`.
    Only defines test dataset.
    """

    def __init__(self, path="./data/paired_test_same", *args, **kwargs):
        super(L.LightningDataModule, self).__init__()
        self.path = path

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        self.dataset_test = PairedDataset(self.path)

    def train_dataloader(self):
        raise NotImplementedError()

    def val_dataloader(self):
        raise NotImplementedError()

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.dataset_test,
            batch_size=1,
            num_workers=8,
            pin_memory=True,
            drop_last=False,
        )

    def on_before_batch_transfer(self, batch, dataloader_idx):
        return batch

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return batch


# %% Additional functions


class NoDataModule(AudioDatasetConvolvedWithRirDatasetDataModule):
    def __init__(self, **kwargs):
        pass


def reset_batch_size(dataloader, new_batch_size):
    """
    Resets the batch size of a dataloader.

    Since this attribute cannot be modified inplace, a new dataloader is returned.

    Parameters
    ----------
    dataloader : torch.utils.data.DataLoader
        Dataloader.
    new_batch_size : int
        New batch size.

    Returns
    -------
    torch.utils.data.DataLoader
        new dataloader with new batch size.

    """
    return torch.utils.data.DataLoader(
        dataset=dataloader.dataset,
        batch_size=new_batch_size,
        num_workers=dataloader.num_workers,
        pin_memory=dataloader.pin_memory,
        drop_last=dataloader.drop_last,
        timeout=dataloader.timeout,
        worker_init_fn=dataloader.worker_init_fn,
        multiprocessing_context=dataloader.multiprocessing_context,
        generator=dataloader.generator,
        prefetch_factor=dataloader.prefetch_factor,
        persistent_workers=dataloader.persistent_workers,
        pin_memory_device=dataloader.pin_memory_device,
    )
