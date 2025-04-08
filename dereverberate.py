#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dereverberation script to process a file"""

import os
import torch
import torchaudio
import argparse
import contextlib
import yaml
import glob
import warnings

with contextlib.redirect_stdout(None):
    from model.joint_model import JointModel
    from datasets import AudioDatasetConvolvedWithRirDatasetDataModule
    from cli import MyCli
    from model.utils.run_management import get_best_checkpoint

UNSUPERVISED_SUPPORT = False
if UNSUPERVISED_SUPPORT:  # whether to use also Gretsi Article
    LOG_ROOT = "icassp_and_gretsi_logs"
else:
    LOG_ROOT = "icassp_logs"


def instantiate_model_only(config_path, ckpt_path: str | None = None):
    # https://github.com/Lightning-AI/pytorch-lightning/issues/17447
    # latest_valid_config=
    config = yaml.load(open(config_path, "r"), Loader=yaml.FullLoader)
    args_dict = {
        key: value
        for key, value in config.items()
        if key in ("model", "seed_everything")
    } | {"data": {"class_path": "datasets.NoDataModule"}}
    args_dict["model"]["reverb_model"] = None
    args_dict["model"]["joint_loss_module"] = None
    if ckpt_path is None:
        print("Using best checkpoint")
        ckpt_path = get_best_checkpoint(
            os.path.dirname(config_path), monitor_mode=max
        )
    args_dict["model"]["speech_model_ckpt_path"] = ckpt_path
    cli = MyCli(
        model_class=JointModel,
        datamodule_class=AudioDatasetConvolvedWithRirDatasetDataModule,
        subclass_mode_model=False,
        subclass_mode_data=True,
        run=False,
        args=args_dict,
    )
    model = cli.model

    return model


def test_load_model():
    model = instantiate_model_only(
        # config_path="remote_logs/old_logs/FullSubNet_dry_wsj1/version_4_test_wsj1_nonoise/version_0/config.yaml",
        config_path="./icassp_and_gretsi_logs/FullSubNet_dry_wsj1/version_4/config.yaml",
    )


def tast_load_all_model():
    import glob

    config_paths = glob.glob(
        os.path.join("./icassp_and_gretsi_logs", "**", "config.yaml"),
        recursive=True,
    )
    for config_path in config_paths:
        model = instantiate_model_only(
            # config_path="remote_logs/old_logs/FullSubNet_dry_wsj1/version_4_test_wsj1_nonoise/version_0/config.yaml",
            config_path=config_path,
        )


def model_predict_wav(
    model,
    input_audio="./examples/wet.wav",
    output_audio=None,
    use_cuda_if_available: bool = True,
):
    if output_audio is None:
        output_audio = os.path.join(
            os.path.dirname(input_audio),
            "".join(os.path.basename(input_audio).split(".")[:-1])
            + "_predicted_dry.wav",
        )
    if use_cuda_if_available and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    y, fs = torchaudio.load(input_audio)
    if model.speech_model.fs != fs:
        raise ValueError(
            "Expected audio samplerate to be {model.speech_model.fs}, input audio is at {fs}"
        )
    y = y.to(device=device)
    model.freeze()
    model.eval()
    model = model.to(device=device)
    s = model.predict_dry_speech(y[None, ...])
    s /= s.abs().max()
    print(f"saving audio to {output_audio}")
    torchaudio.save(output_audio, s[0, ...].cpu(), sample_rate=fs)
    return s


def get_log_path(model_variant, supervision_variant):
    if any(mv in model_variant.lower() for mv in ("fsn", "fullsubnet")):
        model_title = "FullSubNet"
        begin = "FSN"
        # Handle case of "FullSubNet_dry_wsj1"
        if "strong" in supervision_variant:
            begin = "FullSubNet"
    elif any(
        mv in model_variant.lower() for mv in ("bi_lstm", "bilstm", "bi-lstm")
    ):
        model_title = "BiLSTM"
        begin = "bilstm"
    else:
        raise ValueError("Model variant not supported")
    if "strong" in supervision_variant.lower():
        supervision_title = "with strong supervision"
        end = "dry_wsj1"
    elif any(
        sv in supervision_variant.lower() for sv in ("weak", "rt_60", "rt60")
    ):
        supervision_title = "with weak supervision of only RT60"
        end = "train_polack_meanearly_meansigma_wsj1_newschedule"
    elif any(
        sv in supervision_variant.lower()
        for sv in ("none", "auto", "unsupervised")
    ):
        if not UNSUPERVISED_SUPPORT:
            raise NotImplementedError("Unsupported")
        supervision_title = "without any supervision (acoustic information estimated externally)"
        end = "prego_notrain"
    else:
        raise ValueError("Model variant not supported")
    print(f"Using {model_title} trained {supervision_title}")
    return os.path.join(LOG_ROOT, begin + "_" + end)


def get_config_file(log_path):
    config_files = glob.glob(
        os.path.join(log_path, "**", "config.yaml"), recursive=True
    )
    if len(config_files) != 1:
        raise RuntimeError("Expected one and only one config file")
    return config_files[0]


def parse_args():
    parser = argparse.ArgumentParser(description="Dereverberate an audio file")
    parser.add_argument(
        "input_audio", help="input (reverberant) audio path", type=str
    )
    supervision_variant_help = (
        "Supervision variant, "
        + "must be either 'strong' (Wet/Dry pairs), "
        + "or 'weak' (RT60)"
    )
    if UNSUPERVISED_SUPPORT:
        supervision_variant_help = (
            supervision_variant_help
            + " or 'unsupervised' (External RT60 estimation as in Gretsi submission).",
        )
        default_supervision_variant = "unsupervised"
    else:
        supervision_variant_help = supervision_variant_help + "."
        default_supervision_variant = "weak"
    parser.add_argument(
        "--supervision_variant",
        "-s",
        help=f"{supervision_variant_help}. Default: {default_supervision_variant}",
        type=str,
        default=default_supervision_variant,
    )
    parser.add_argument(
        "--model_variant",
        "-m",
        help="Model variant: FullSubNet or BiLSTM. Default: FullSubNet",
        type=str,
        default="FullSubNet",
    )
    parser.add_argument(
        "--use_cuda_if_available",
        "--cuda",
        help="Use CUDA if CUDA accelerator is available.",
        default=True,
        action=argparse.BooleanOptionalAction,
    )
    parser.add_argument(
        "--output_audio",
        "-o",
        help="output (predicted dry) audio path. Default: '[INPUT_AUDIO]_predicted_dry.wav'",
        default=None,
    )
    args = parser.parse_args()
    return args


def dereverberate_audio(
    input_audio,
    model_variant="FSN",
    supervision_variant="weak",
    output_audio=None,
    use_cuda_if_available: bool = True,
):
    if not os.path.exists(input_audio):
        raise ValueError(f"input audio not found {input_audio}")
    log_path = get_log_path(
        model_variant=model_variant, supervision_variant=supervision_variant
    )
    if not os.path.isdir(log_path):
        print(log_path)
        raise NotImplementedError("wrong arguments not catched")
    config_file = get_config_file(log_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = instantiate_model_only(config_file)
        model_predict_wav(
            model,
            input_audio=input_audio,
            output_audio=output_audio,
            use_cuda_if_available=use_cuda_if_available,
        )


if __name__ == "__main__":
    args = parse_args()
    dereverberate_audio(**vars(args))


# tast_load_all_model()
