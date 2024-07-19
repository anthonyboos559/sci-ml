from typing import NamedTuple
import torch
import pandas as pd

    
class ModelOutputs(NamedTuple):
    encoder_act: list
    qz: torch.Tensor
    latent: torch.Tensor
    x_hat: torch.Tensor
    
class _REGISTRY_KEYS_NT(NamedTuple):
    LOSS: str = "loss"
    RECON_LOSS: str = "recon_loss"
    ADV_LOSS: str = "adv_loss"
    KL_LOSS: str = "kl_loss"
    KL_WEIGHT: str = "kl_weight"
    LOSS_MEAN: str = "loss_m"
    RECON_LOSS_MEAN: str = "recon_loss_m"
    ADV_LOSS_MEAN: str = "adv_loss_m"
    KL_LOSS_MEAN: str = "kl_loss_m"
    KL_WEIGHT: str = "kl_weight"
    LABELS: str = "labels"
    QZM: str = "qzm"
    QZV: str = "qzv"
    Z: str = "Z"
    Z_STAR: str = "Z_STAR"
    X: str = "X"
    X_HAT: str = "X_HAT"
    Y: str = "Y"
    METADATA: str = "METADATA"
    EXPERT: str = "expert"
    HUMAN: str = "human"
    MOUSE: str = "mouse"
    

REGISTRY_KEYS = _REGISTRY_KEYS_NT()