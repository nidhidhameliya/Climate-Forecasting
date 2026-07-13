import torch
import json
import math


def rmse(pred, target):
    return torch.sqrt(torch.mean((pred - target) ** 2))


def extreme_rmse(pred, target, threshold):

    mask = target > threshold

    if mask.sum() == 0:
        return torch.tensor(0.0)

    return torch.sqrt(torch.mean((pred[mask] - target[mask]) ** 2))


def hit_rate(pred, target, threshold):
    pred_extreme = pred > threshold
    true_extreme = target > threshold

    hits = (pred_extreme & true_extreme).sum().float()
    total = true_extreme.sum().float()

    if total == 0:
        return torch.tensor(0.0)

    return hits / total