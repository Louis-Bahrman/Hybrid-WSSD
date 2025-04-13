#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 12:12:36 2024

@author: louis
"""
import torch
from torch import nn


class ComplexToRealMSELoss(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        assert reduction in ("mean", "none")
        self.reduction = reduction

    def forward(self, pred, target):
        loss_non_reduced = (pred - target).abs().square()
        if self.reduction == "mean":
            return loss_non_reduced.mean()
        return loss_non_reduced


class SumLosses(nn.Module):
    def __init__(
        self,
        loss_1: nn.Module,
        loss_2: nn.Module,
        loss_2_weight_multiplier: float = 1.0,  # if use_gradnorm==True: weight applied after normalization
        use_gradnorm: bool = False,
        gradnorm_rate: float = 0.01,
        loss_2_gradnorm_initial_weight: float = 1.0,
        backprop_only_best_channel: bool = False,
        channel_dim: int = 1,
    ):
        super().__init__()
        self.loss_1 = loss_1
        self.loss_2 = loss_2
        self.loss_2_weight_multiplier = loss_2_weight_multiplier
        self.use_gradnorm = use_gradnorm
        if self.use_gradnorm:
            self.register_buffer("loss_2_gradnorm_weight", torch.tensor(loss_2_gradnorm_initial_weight))
            self.gradnorm_rate = gradnorm_rate
        self.backprop_only_best_channel = backprop_only_best_channel
        self.channel_dim = channel_dim

    def forward(self, pred, target):
        if not self.use_gradnorm:
            return self.loss_1(pred, target) + self.loss_2_weight_multiplier * self.loss_2(pred, target)
        unnormalized_loss_1 = self.loss_1(pred, target)
        unnormalized_loss_2 = self.loss_2(pred, target)
        if unnormalized_loss_1.requires_grad and unnormalized_loss_2.requires_grad:
            # we are training, so we update norm 1 and 2
            unnormalized_loss_1_mean = unnormalized_loss_1.mean()
            unnormalized_loss_2_mean = unnormalized_loss_2.mean()
            with torch.no_grad():
                grad_loss_1_wrt_pred = torch.autograd.grad(
                    unnormalized_loss_1_mean,
                    pred,
                    retain_graph=True,
                )[0]
                grad_loss_2_wrt_pred = torch.autograd.grad(
                    unnormalized_loss_2_mean,
                    pred,
                    retain_graph=True,
                )[0]
            if self.backprop_only_best_channel:
                # select the gradnorm
                gradnorm_1 = grad_loss_1_wrt_pred.norm(
                    dim=tuple(range(self.channel_dim + 1, grad_loss_1_wrt_pred.ndim))
                )  # Shape B,C
                gradnorm_2 = grad_loss_2_wrt_pred.norm(
                    dim=tuple(range(self.channel_dim + 1, grad_loss_2_wrt_pred.ndim))
                )  # Shape B,C
            else:
                gradnorm_1 = grad_loss_1_wrt_pred.norm()  # Shape 0
                gradnorm_2 = grad_loss_2_wrt_pred.norm()  # Shape 0
            # if gradnorm_1.isfinite() and gradnorm_2.isfinite():
            weight_2 = gradnorm_1 / gradnorm_2.clip(min=1e-6)  # shape B,C if backprop_only_best_channel else 0
            loss_2_gradnorm_weight = (
                self.gradnorm_rate * weight_2 + (1 - self.gradnorm_rate) * self.loss_2_gradnorm_weight
            )  # shape B,C if backprop_only_best_channel else 0
            if not self.backprop_only_best_channel:
                self.loss_2_gradnorm_weight = loss_2_gradnorm_weight
                return (
                    unnormalized_loss_1
                    + unnormalized_loss_2 * self.loss_2_gradnorm_weight * self.loss_2_weight_multiplier
                )
            else:
                loss_per_batch_and_channel = (
                    unnormalized_loss_1.mean(axis=tuple(range(self.channel_dim + 1, grad_loss_2_wrt_pred.ndim)))
                    + unnormalized_loss_2.mean(axis=tuple(range(self.channel_dim + 1, grad_loss_2_wrt_pred.ndim)))
                    * loss_2_gradnorm_weight
                    * self.loss_2_weight_multiplier
                )
                best_channel_idx = torch.argmin(loss_per_batch_and_channel, dim=self.channel_dim)
                best_loss = loss_per_batch_and_channel[:, best_channel_idx].mean()
                self.loss_2_gradnorm_weight = loss_2_gradnorm_weight[:, best_channel_idx].mean()
                return best_loss
        loss = unnormalized_loss_1 + unnormalized_loss_2 * self.loss_2_gradnorm_weight * self.loss_2_weight_multiplier
        if self.backprop_only_best_channel:
            return (
                loss.mean(axis=tuple(range(self.channel_dim + 1, loss.ndim)))
                .min(dim=self.channel_dim, keepdim=True)
                .values.mean()
            )
        return loss


class LogLoss(nn.Module):
    def __init__(self, epsilon=1.0, base_loss: nn.Module | None = None):
        super().__init__()
        self.epsilon = epsilon
        if base_loss is None:
            base_loss = nn.MSELoss()
        self.base_loss = base_loss

    def forward(self, pred, target):
        return self.base_loss((pred.abs() + self.epsilon).log(), (target.abs() + self.epsilon).log())


class ScaleInvariant(nn.Module):
    def __init__(self, base_loss: nn.Module):
        super().__init__()
        self.base_loss = base_loss

    def forward(self, preds, target):
        preds_scaled = preds / preds.abs().square().mean(-1, keepdim=True).sqrt()
        target_scaled = target / target.abs().square().mean(-1, keepdim=True).sqrt()
        return self.base_loss(preds_scaled, target_scaled)


def test_gradnorm(backprop_only_best_channel=False):
    print(f"backprop_only_best_channel={backprop_only_best_channel}")

    class Loss1(nn.Module):
        def __init__(self, reduction="mean"):
            super().__init__()
            self.reduction = reduction

        def forward(self, t, *args):
            if self.reduction == "mean":
                return (t).mean()
            return t

    class Loss2(nn.Module):
        def __init__(self, reduction="mean"):
            super().__init__()
            self.reduction = reduction

        def forward(self, t, *args):
            if self.reduction == "mean":
                return (2 * t).mean()
            return 2 * t

    loss_1 = Loss1(reduction="none" if backprop_only_best_channel else "mean")
    loss_2 = Loss2(reduction="none" if backprop_only_best_channel else "mean")
    sumloss = SumLosses(
        loss_1,
        loss_2,
        use_gradnorm=True,
        gradnorm_rate=0.5,
        loss_2_weight_multiplier=6,
        backprop_only_best_channel=backprop_only_best_channel,
    )

    for i in range(20):
        if i % 3 == 0:
            print("Val step")
            t = torch.arange(3.0, requires_grad=False)[None, None, None, :].expand(8, 2, 3, -1)
        else:
            print("Train step")
            t = torch.arange(3.0, requires_grad=True)[None, None, None, :].expand(8, 2, 3, -1)
        l1 = loss_1(t)
        l2 = loss_2(t)
        # print(f"L1 {l1}")
        # print(f"L2 {l2}")
        print(f"Gradnormalized loss: {sumloss(t, torch.zeros_like(t).detach())}")
        print(f"Gradnorm L2 weight: {sumloss.loss_2_gradnorm_weight}")
        print()


if __name__ == "__main__":
    test_gradnorm(backprop_only_best_channel=False)
    test_gradnorm(backprop_only_best_channel=True)
