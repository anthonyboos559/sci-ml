from typing import Union
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from .base import utils
from sciml.utils.constants import REGISTRY_KEYS as RK
from .base import BaseVAEModel
from sciml.modules import MMVAE
pl.Trainer
class MMVAEModel(BaseVAEModel):
    """
    Multi-Modal Variational Autoencoder (MMVAE) model for handling expert-specific data.

    Args:
        module (MMVAE): Multi-Modal VAE module.
        **kwargs: Additional keyword arguments for the base VAE model.

    Attributes:
        automatic_optimization (bool): Flag to control automatic optimization. Set to False for manual optimization.
    """
    
    def __init__(self, module: MMVAE, **kwargs):
        super().__init__(module, **kwargs)
        self.automatic_optimization = False  # Disable automatic optimization for manual control
        self.zstars_container = {RK.HUMAN: [], RK.MOUSE: []}
        self.metadata_container = {RK.HUMAN: [], RK.MOUSE: []}
        print(self.module, flush=True)
        
    def training_step(self, batch, batch_idx):
        """
        Training step for the model.

        Args:
            batch (dict): Batch of data containing inputs and target labels.
            batch_idx (int): Index of the batch.

        Returns:
            None
        """
        expert_id = batch[RK.EXPERT_ID]
        # Retrieve optimizers
        shared_opt, human_opt, mouse_opt = self.optimizers()

        # Select expert-specific optimizer based on the expert ID in the batch
        expert_opt = human_opt if expert_id == RK.HUMAN else mouse_opt
        
        # Zero the gradients for the shared and expert-specific optimizers
        shared_opt.zero_grad()
        expert_opt.zero_grad()
        
        # Perform forward pass and compute the loss
        model_inputs, model_outputs, loss = self(batch, module_input_kwargs={'target': expert_id}, loss_kwargs={'kl_weight': self.kl_annealing_fn.kl_weight}) 

        # Perform manual backpropagation
        self.manual_backward(loss[RK.LOSS])

        # Clip gradients for stability
        self.clip_gradients(shared_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        
        # Update the weights
        shared_opt.step()
        expert_opt.step()
        self.kl_annealing_fn.step()
        
        # Log the loss
        self.auto_log(loss, tags=[self.stage_name, expert_id])
        
    def validation_step(self, batch):
        """
        Validation step for the model.

        Args:
            batch (dict): Batch of data containing inputs and target labels.

        Returns:
            None
        """
        # Perform forward pass and compute the loss with cross-generation loss
        model_inputs, _, loss = self(batch, loss_kwargs={'use_cross_gen_loss': True, 'kl_weight': self.kl_annealing_fn.kl_weight})
        
        # Log the loss if not in sanity checking phase
        if not self.trainer.sanity_checking:
            self.auto_log(loss, tags=[self.stage_name, batch[RK.EXPERT_ID]])
        
    # Alias for validation_step method to reuse for testing
    test_step = validation_step

    def predict_step(self, batch):
        x = batch[RK.X]
        metadata = batch[RK.METADATA]
        exper_id = batch[RK.EXPERT_ID]

        x = self.model.experts[exper_id].encode(x)
        _, z = self.model.vae.encode(x)
        self.zstars_container[exper_id].append(z)
        self.metadata_container[exper_id].append(metadata)

    def on_predict_epoch_end(self):

        npz = torch.cat(self.zstars_container[RK.HUMAN]).numpy()
        metadata = pd.concat(self.metadata_container[RK.HUMAN], axis=0)
        
        np.save(f"{self.logger.log_dir}/{RK.HUMAN}_z_values.npz", npz)
        metadata.to_pickle(f"{self.logger.log_dir}/{RK.HUMAN}_metadata.pkl")

        npz = torch.cat(self.zstars_container[RK.MOUSE]).numpy()
        metadata = pd.concat(self.metadata_container[RK.MOUSE], axis=0)
        
        np.save(f"{self.logger.log_dir}/{RK.MOUSE}_z_values.npz", npz)
        metadata.to_pickle(f"{self.logger.log_dir}/{RK.MOUSE}_metadata.pkl")