from typing import Union
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils
from sciml.utils.constants import REGISTRY_KEYS as RK, ModelOutputs

class MMVAEModel(pl.LightningModule):

    def __init__(
        self,
        mmvae: nn.Module,
        predict_keys = [RK.X_HAT, RK.Z],
        kl_weight=1.0,
        batch_size: int = 128,
        plot_z_embeddings: bool = False,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['mmvae'], logger=True)
        self.mmvae = mmvae
        
        # Register kl_weight as buffer
        self.register_buffer('kl_weight', torch.tensor(self.hparams.kl_weight, requires_grad=False))
        
        # container for z embeddings/metadata/global_step
        self.z_val = {RK.HUMAN: [], RK.MOUSE: []}
        self.z_val_metadata = {RK.HUMAN: [], RK.MOUSE: []}
        self.validation_epoch_end = -1
        self.automatic_optimization = False
    
    @property
    def plot_z_embeddings(self):
        return self.hparams.plot_z_embeddings and not self.trainer.sanity_checking
        
    def forward(self, x):
        return self.mmvae(x)
            
    def criterion(self, x, forward_outputs: ModelOutputs):
        # Mean Square Error of input batch to reconstructed batch
        recon_loss = F.mse_loss(forward_outputs.x_hat, x, reduction='none')
        # Kl Divergence from posterior distribution to normal distribution
        kl_loss = utils.kl_divergence(forward_outputs.qzm, forward_outputs.qzv)
        loss = recon_loss.sum() + self.kl_weight * kl_loss.sum()
        return {
            RK.KL_LOSS: kl_loss.mean(),
            RK.RECON_LOSS: recon_loss.sum(dim=1).mean(),
            RK.LOSS: loss
        }

    def cross_gen_loss(self, batch_dict):
        init_expert = batch_dict[RK.EXPERT]
        cross_expert = RK.MOUSE if init_expert == RK.HUMAN else RK.HUMAN
        loss_out = self.mmvae.cross_generate(batch_dict)
        loss_out = self.mmvae.cross_generate({RK.X: loss_out.x_hat, RK.METADATA: batch_dict[RK.METADATA], RK.EXPERT: cross_expert})
        cross_loss = F.mse_loss(loss_out.x_hat, batch_dict[RK.X], reduction='none')
        return {f"cross_gen_loss/{init_expert}_to_{cross_expert}": cross_loss.sum(dim=1).mean()}
    
    def configure_optimizers(self):
        return self.mmvae.configure_optimizers()
    
    def loss(self, batch_dict, return_outputs = False):
        
        # Call forward method on self
        forward_outputs = self(batch_dict)
        # Compute loss outputs based on criterion
        loss_outputs = self.criterion(batch_dict[RK.X], forward_outputs)
        # If forward outputs are needed return forward_outputs
        if return_outputs:
            return loss_outputs, forward_outputs
        else:
            return loss_outputs
    
    def training_step(self, batch_dict, batch_idx):
        shared_opt, human_opt, mouse_opt = self.optimizers()
        if batch_dict[RK.EXPERT] == RK.HUMAN:
            expert_opt = human_opt
        else:
            expert_opt = mouse_opt
        shared_opt.zero_grad()
        expert_opt.zero_grad()
        # Compute loss outputes, expecting None back as this is the train loop and
        # do not need the forward outputs
        loss_outputs = self.loss(batch_dict)
        self.manual_backward(loss_outputs[RK.LOSS])
        self.clip_gradients(shared_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        shared_opt.step()
        expert_opt.step()
        # Tag loss output keys with 'train_'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'train', batch_dict[RK.EXPERT], sep='/')
        # Log loss outputs for every step and epoch to logger
        self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

    def on_train_epoch_end(self) -> None:
        self.trainer.train_dataloader.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch_dict, batch_idx):
        # Compute loss_outputs returning the forward_outputes if needed for plotting z embeddings
        if self.plot_z_embeddings:
            loss_outputs, forward_outputs = self.loss(batch_dict, return_outputs=self.plot_z_embeddings)
        else:
            loss_outputs = self.loss(batch_dict, return_outputs=False)
        # Tag loss output keys with 'val_'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'val', batch_dict[RK.EXPERT], sep='/')
        
        # Prevent logging sanity_checking steps to logger
        # Not needed for validation and also throws error if batch_size not configured correctly
        if not self.trainer.sanity_checking:
            self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

        cross_loss_outputs = self.cross_gen_loss(batch_dict)
        if not self.trainer.sanity_checking:
            self.log_dict(cross_loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

        # If plotting z embeddings
        if self.plot_z_embeddings:
            # Accumlate the Z tensors and associated metadata
            self.z_val.append(forward_outputs[RK.Z].detach().cpu())
            self.z_val_metadata.append(batch_dict.get(RK.METADATA))
    
    def on_validation_epoch_end(self):
        self.trainer.val_dataloaders.reset()
        if self.plot_z_embeddings:
            # Record for global step
            self.validation_epoch_end += 1
            headers = list(self.z_val_metadata[0].keys())
            # Concatenate Z tensors
            embeddings = torch.cat(self.z_val, dim=0)
            # Concatenate metadata
            metadata = np.concatenate(self.z_val_metadata, axis=0)
            # get tensorboard SummaryWriter instance from trainer.logger
            writer = self.trainer.logger.experiment
            # Record z embeddings and metadata
            writer.add_embedding(
                mat=embeddings, 
                metadata=metadata.tolist(), 
                global_step=self.validation_epoch_end, 
                metadata_header=headers)
            # Empty the Z tensors and metadata containers
            self.z_val = {RK.HUMAN: [], RK.MOUSE: []}
            self.z_val_metadata = {RK.HUMAN: [], RK.MOUSE: []}
        return super().on_validation_epoch_end()
            
    def test_step(self, batch_dict, batch_idx):
        # Compute loss_outputs
        loss_outputs, model_outputs = self.loss(batch_dict, return_outputs=True)
        z_star = model_outputs.z_star.detach().cpu()
        self.z_val[batch_dict[RK.EXPERT]].append(z_star.numpy())
        self.z_val_metadata[batch_dict[RK.EXPERT]].append(batch_dict.get(RK.METADATA))
        # Tag loss output keys with 'test'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'test', batch_dict[RK.EXPERT], sep='/')
        self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
        cross_loss_outputs = self.cross_gen_loss(batch_dict)
        cross_loss_outputs = utils.tag_mm_loss_outputs(cross_loss_outputs, 'test', batch_dict[RK.EXPERT], sep='/')
        self.log_dict(cross_loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
    
    def on_test_epoch_end(self):
        np.savez('/mnt/projects/debruinz_project/summer_census_data/z_star_test_data/human_z_star.npz', np.concatenate(self.z_val[RK.HUMAN]))
        np.savez('/mnt/projects/debruinz_project/summer_census_data/z_star_test_data/mouse_z_star.npz', np.concatenate(self.z_val[RK.MOUSE]))
        import pandas as pd
        metadata = pd.concat(self.z_val_metadata[RK.HUMAN])
        metadata.to_pickle('/mnt/projects/debruinz_project/summer_census_data/z_star_test_data/human_z_star_metadata.pkl')
        metadata = pd.concat(self.z_val_metadata[RK.MOUSE])
        metadata.to_pickle('/mnt/projects/debruinz_project/summer_census_data/z_star_test_data/mouse_z_star_metadata.pkl')
        self.z_val = {RK.HUMAN: [], RK.MOUSE: []}
        self.z_val_metadata = {RK.HUMAN: [], RK.MOUSE: []}
        return super().on_test_epoch_end()

    def predict_step(self, batch_dict, batch_idx):
        # Run forward pass on model
        forward_outputs = self(batch_dict)
        # Return the tensors from the keys specified in hparams
        return { 
            key: value for key, value in forward_outputs 
            if key in self.hparams.predict_keys
        }
        
    def get_latent_representations(
        self,
        adata,
        batch_size
    ):
        from sciml.data.server import AnnDataDataset, collate_fn
        from torch.utils.data import DataLoader
        
        from lightning.pytorch.trainer import Trainer
        
        dataset = AnnDataDataset(adata)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

        zs = []
        self.eval()
        with torch.no_grad():
            for batch_dict in dataloader:
                for key in batch_dict.keys():
                    if isinstance(batch_dict[key], torch.Tensor):
                        batch_dict[key] = batch_dict[key].to('cuda')
                    
                predict_outputs = self.predict_step(batch_dict, None)
                zs.append(predict_outputs[RK.Z])
        
        return torch.cat(zs).numpy()