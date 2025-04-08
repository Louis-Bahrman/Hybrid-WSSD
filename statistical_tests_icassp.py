#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 00:39:42 2024

@author: louis
"""
import os
import scipy
import numpy as np
import itertools


models = {
    "FSN+cRM": "FullSubNet_dry_wsj1/version_4_test_wsj1_nonoise/version_0",
    "FSN+h": "FSN_rereverb_withlog_lr4_wsj1/version_0_test_wsj1_nonoise/version_3",
    "FSN+theta": "FSN_train_polack_wsj1_newschedule/version_0_test_wsj1_nonoise/version_3",
    "FSN+rt60+sigma": "FSN_train_polack_meanearly_wsj1_newschedule/version_2_test_wsj1_nonoise/version_3",
    "FSN+rt60": "FSN_train_polack_meanearly_meansigma_wsj1_newschedule/version_2_test_wsj1_nonoise/version_3",
    "bilstm+S": "bilstm_dry_wsj1/version_1_test_wsj1_nonoise/version_0",
    "bilstm+h": "bilstm_rereverb_withlog_lr4_wsj1/version_0_test_wsj1_nonoise/version_3/",
    "bilstm+theta": "bilstm_train_polack_wsj1_newschedule/version_0_test_wsj1_nonoise/version_3",
    "bilstm+rt60+sigma": "bilstm_train_polack_meanearly_wsj1_newschedule/version_0_test_wsj1_nonoise/version_3/",
    "bilstm+rt60": "bilstm_train_polack_meanearly_meansigma_wsj1_newschedule/version_2_test_wsj1_nonoise/version_3/",
    "baseline": "speechbrain_baseline_test/version_0",
}

metrics = {
    "SISDR": "val_dry_speech_model_ScaleInvariantSignalDistortionRatio0256.npy",
    "ESTOI": "val_dry_speech_model_ShortTimeObjectiveIntelligibility0256.npy",
    "WB-PESQ": "val_dry_speech_model_PerceptualEvaluationSpeechQuality0256.npy",
    "SRMR": "val_dry_speech_model_SRMRWrapper0256.npy",
}

input_metrics = {k: v[:-4] + "_input.npy" for k, v in metrics.items()}


def load_model_metric(model, metric):
    if model.lower() in "wet input reverberant":
        return np.load(os.path.join("remote_logs", models["baseline"], "latest_results", input_metrics[metric]))
    else:
        return np.load(os.path.join("remote_logs", models[model], "latest_results", metrics[metric]))


def hypothesis_test(model_1, model_2, metric, pvalue=0.001):
    """check if 1 > 2 for given metric"""
    array_1 = load_model_metric(model_1, metric)
    array_2 = load_model_metric(model_2, metric)
    res = scipy.stats.wilcoxon(array_1, array_2, alternative="greater").pvalue
    if pvalue is None:
        return res
    return res < pvalue


# %% Check that all our methods outperform input for all metrics


def test_all_ours_better_input():
    for model in models.keys():
        for metric in metrics.keys():
            if model.lower() not in "baseline":
                print(hypothesis_test(model, "input", metric))


# %% Baseline fails for SISDR and SRMR


def test_baseline_srmr():
    print("The baseline (BiLSTM + SRMR) excels in terms of SRMR")
    for model in models.keys():
        if model.lower() not in "baseline":
            print(hypothesis_test("baseline", model, "SRMR"))
    # print(hypothesis_test("baseline", "input", "WB-PESQ"))


def test_baseline_stoi_sisdr():
    print(
        "but this performance comes at the cost of its SISDR and ESTOI results, which are degraded compared to the reverberant input."
    )
    print(hypothesis_test("input", "baseline", "ESTOI"))
    print(hypothesis_test("input", "baseline", "SISDR"))


# %%


def test_FSN_cRM_better_anything_else():
    print(
        "The best performing method FullSubNet benefits from strong supervision, both when trained on its original complex masking loss or using the oracle RIR.",
        "This is for cIRM",
    )
    for model in models.keys():
        if model.lower() not in "baseline FSN+cRM".lower():
            for metric in metrics.keys():
                print(model, metric, hypothesis_test("FSN+cRM", model, metric))


def test_FSN_h_better_anything_below():
    print(
        "The best performing method FullSubNet benefits from strong supervision, both when trained on its original complex masking loss or using the oracle RIR.",
        "For h",
    )
    for model in models.keys():
        if model.lower() not in "baseline FSN+cRM FSN+h".lower():
            for metric in metrics.keys():
                print(model, metric, hypothesis_test("FSN+cRM", model, metric))


def test_bilstm_weak_better():
    print(
        "the less-complex BiLSTM widely benefits from weak supervision, and performs better in terms of SISDR when weakly supervised by Polack's model than when it has access to the ground-truth RIR $h$"
    )
    "The original statement including WB-PESQ is not true"
    for model in ("bilstm+theta", "bilstm+rt60+sigma", "bilstm+rt60"):
        for metric in ("SISDR", "WB-PESQ"):
            print(model, metric, hypothesis_test(model, "bilstm+h", metric, None))


def test_bilstm_rt60_better():
    print(
        "Supervision by RT60 only even improves the model’s SISDR performance above its original supervision based on magnitude spectra"
    )
    print(hypothesis_test("bilstm+rt60", "bilstm+S", "SISDR", None))


def test_rt60_only_is_better_than_more():
    print(
        "Comparing reverberation-weak supervision approaches, we remark that they perform better in terms of SISDR when having no access to the acoustic parameters used to estimate the mixing time and Polack's model $\sigma$."
    )
    metric = "SISDR"
    for dereverb_model in ("FSN", "bilstm"):
        for supervision in ("theta", "rt60+sigma"):
            print(
                dereverb_model + "+" + supervision,
                metric,
                hypothesis_test(dereverb_model + "+rt60", dereverb_model + "+" + supervision, metric),
            )


def compute_best_significant_model_for_each_metric():
    for dereverb_model in ("FSN", "bilstm"):
        for metric in ("SISDR", "ESTOI", "WB-PESQ"):
            for candidate_best, comparison_1, comparison_2 in itertools.permutations(
                ("theta", "rt60+sigma", "rt60"), 3
            ):
                print(
                    f"{metric}: {dereverb_model}+{candidate_best}",
                    hypothesis_test(dereverb_model + "+" + candidate_best, dereverb_model + "+" + comparison_1, metric)
                    and hypothesis_test(
                        dereverb_model + "+" + candidate_best, dereverb_model + "+" + comparison_2, metric
                    ),
                )


# %%
if __name__ == "__main__":
    test_all_ours_better_input()
    test_baseline_srmr()
    test_baseline_stoi_sisdr()
    test_FSN_cRM_better_anything_else()
    test_FSN_h_better_anything_below()
    test_bilstm_weak_better()
    test_bilstm_rt60_better()
    test_rt60_only_is_better_than_more()
    compute_best_significant_model_for_each_metric()
