#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 10:37:33 2025

@author: louis
"""

import os
import glob
import shutil
import torch


def get_all_version_folders_containing_checkpoint(base_path="./lightning_logs/taslp"):
    checkpoint_paths = glob.glob(os.path.join(base_path, "**", "checkpoints"), recursive=True)
    return [os.path.abspath(os.path.join(cp, os.pardir)) for cp in checkpoint_paths]


def get_latest_valid_version(base_path):
    # valid if contains ckpt folder
    version_paths = os.listdir(base_path)
    valid_version_paths = []
    for version_path in version_paths:
        abs_version_path = os.path.abspath(os.path.join(base_path, version_path))
        if "checkpoints" in os.listdir(abs_version_path):
            valid_version_paths.append(abs_version_path)
    if not valid_version_paths:
        return ""
    return max(valid_version_paths, key=os.path.getmtime)


def get_latest_checkpoint(base_dir):
    if base_dir.endswith(".ckpt"):
        base_dir = os.path.dirname(base_dir)
    if not "version_" in base_dir:
        base_dir = get_latest_valid_version(base_dir)
    if not base_dir.endswith("checkpoints"):
        base_dir = os.path.join(base_dir, "checkpoints")
    all_checkpoints = [os.path.abspath(os.path.join(base_dir, ckpt_path)) for ckpt_path in os.listdir(base_dir)]
    return max(all_checkpoints, key=os.path.getmtime)


def get_best_checkpoint(base_dir, monitor_mode):
    latest_checkpoint = get_latest_checkpoint(base_dir)
    d = torch.load(latest_checkpoint)
    model_checkpoint_callbacks = {k: v for k, v in d["callbacks"].items() if "ModelCheckpoint" in k}
    if len(model_checkpoint_callbacks) > 1:
        raise RuntimeError("Not that many callbacks expected")
    model_checkpoint_callback = next(iter(model_checkpoint_callbacks.values()))
    best_k_models = model_checkpoint_callback["best_k_models"]
    if len(best_k_models) == 0:
        print("No checkpoint dict ranked by monitor found, defaulting to latest checkpoint")
        best_ckpt = latest_checkpoint
    else:
        best_ckpt = monitor_mode(best_k_models, key=best_k_models.get)
        print(f"best_monitor_value: {best_k_models[best_ckpt].item()}")
    # correct if latest_checkpoint is computed from a sshfs-mounted dir
    best_ckpt_in_latest_ckpt_dir = os.path.join(os.path.dirname(latest_checkpoint), os.path.basename(best_ckpt))
    return best_ckpt_in_latest_ckpt_dir


def config_to_new_version(config_path):
    # copy config to old
    # new config
    ...


def move_all_except_lastest_version_containing_checkpoint_to_trash(
    base_path="./lightning_logs/taslp",
    trash_path="./lightning_logs/trash",
    dry_run=True,
    clear_all_except_config_and_best_checkpoint: bool = False,
    clear_all_except_config_and_best_checkpoint_monitor_mode=max,
    delete_test_versions: bool = True,
    backup_all_ckpts: bool = True,  # false for less disk space usage but very dagerous
):
    # find all "version_*" dirs (in all loggers)
    version_paths = sorted(glob.glob(os.path.join(base_path, "**", "version_*"), recursive=True))
    # We handle test versions differently
    if delete_test_versions:
        for vp in version_paths:
            if "test" in vp:
                logger_path = os.path.abspath(os.path.join(vp, os.pardir))
                logger_name = os.path.relpath(logger_path, base_path)
                source_dir = vp
                if os.path.isdir(source_dir):
                    trash_dest = os.path.join(trash_path, logger_name) + "_test"
                    print(f"moving {source_dir} to {trash_dest}")
                    if not dry_run:
                        try:
                            os.makedirs(trash_dest, exist_ok=True)
                            shutil.move(source_dir, trash_dest)
                        except Exception as e:
                            print(e)
                            breakpoint()
    version_paths = sorted(glob.glob(os.path.join(base_path, "**", "version_*"), recursive=True))
    # Extract all logger_paths
    logger_paths = set(os.path.abspath(os.path.join(vp, os.pardir)) for vp in version_paths)
    for logger_path in logger_paths:
        # get logger_name
        logger_name = os.path.relpath(logger_path, base_path)
        trash_dest = os.path.join(trash_path, logger_name)
        # get latest version
        latest_valid_version = get_latest_valid_version(base_path=logger_path)
        for version in os.listdir(logger_path):
            source_dir = os.path.join(logger_path, version)
            if version != os.path.basename(latest_valid_version):
                if not dry_run:
                    os.makedirs(trash_dest, exist_ok=True)
                print(f"moving {source_dir} to {trash_dest}")
                if not dry_run:
                    try:
                        shutil.move(source_dir, trash_dest)
                    except Exception as e:
                        print(e)
                        breakpoint()
            else:
                if clear_all_except_config_and_best_checkpoint:
                    if backup_all_ckpts:
                        print(f"copying {source_dir} to {trash_dest}")
                        if not dry_run:
                            try:
                                shutil.copytree(source_dir, trash_dest, copy_function=shutil.copy2)
                            except Exception as e:
                                print(e)
                                breakpoint()
                    all_files = glob.glob(os.path.join(source_dir, "**"), recursive=True)
                    best_checkpoint = get_best_checkpoint(
                        os.path.join(source_dir, "checkpoints"),
                        monitor_mode=clear_all_except_config_and_best_checkpoint_monitor_mode,
                    )
                    for file in all_files:
                        if (
                            not file.endswith(".yaml")
                            and os.path.basename(file) != os.path.basename(best_checkpoint)
                            and not os.path.isdir(file)
                        ):
                            print(f"Deleting {file}")
                            if not dry_run:
                                os.remove(file)
