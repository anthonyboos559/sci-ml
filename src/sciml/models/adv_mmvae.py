from typing import Union
from lightning.pytorch.utilities.types import EVAL_DATALOADERS
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from . import utils
from sciml.utils.constants import REGISTRY_KEYS as RK
from sciml.modules.adv_mmvae_module import AdvMMVAE
from torch.distributions import Normal, kl_divergence


class AdvMMVAEModel(pl.LightningModule):

    def __init__(
        self,
        mmvae: AdvMMVAE,
        predict_keys = [RK.X_HAT, RK.Z],
        kl_weight=1.0,
        adv_weight=0.0,
        batch_size: int = 128,
        plot_z_embeddings: bool = False,
        disc_warmup = 0
    ):
        super().__init__()
        self.mmvae = mmvae
        print(self.mmvae) 
        self.save_hyperparameters(logger=True)

        # Register kl_weight as buffer
        self.register_buffer('kl_weight', torch.tensor(self.hparams.kl_weight, requires_grad=False))
        
        # container for z embeddings/metadata/global_step
        self.z_val = []
        self.z_val_metadata = []
        self.validation_epoch_end = -1
        self.automatic_optimization = False
        self.adv_weight = adv_weight
        self.disc_warmup = disc_warmup
    
    @property
    def plot_z_embeddings(self):
        return self.hparams.plot_z_embeddings and not self.trainer.sanity_checking
        
    def forward(self, x):
        return self.mmvae(x)
    
            
    def criterion(self, x, forward_outputs, trick_label, predictions):
        # gen_weight = self.adv_weight if self.disc_warmup >= self.current_epoch else 0
        gen_weight=self.adv_weight
        recon_loss = F.mse_loss(forward_outputs.x_hat, x, reduction='sum')
        recon_loss_mean = F.mse_loss(forward_outputs.x_hat, x, reduction='mean')
        # Kl Divergence from posterior distribution to normal distribution

        pz = Normal(torch.zeros_like(forward_outputs.latent), torch.ones_like(forward_outputs.latent))
        kl_loss = torch.distributions.kl_divergence(forward_outputs.qz, pz).sum(dim=-1).mean()

        if x.layout == torch.sparse_csr:
            x = x.to_dense()

        if trick_label == None:
            loss = recon_loss + self.kl_weight * kl_loss
            loss_mean = recon_loss_mean + self.kl_weight * kl_loss
            # return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }, {RK.LOSS_MEAN: loss_mean, RK.RECON_LOSS_MEAN : recon_loss_mean, RK.KL_LOSS_MEAN: kl_loss}
            return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }
        else:
            adv_loss = 0
            adv_loss_mean = 0
            for pred in predictions:
                adv_loss += F.binary_cross_entropy(pred, trick_label, reduction='sum')
                adv_loss_mean += F.binary_cross_entropy(pred, trick_label, reduction='mean')
            loss = recon_loss + self.kl_weight * kl_loss + gen_weight * adv_loss
            loss_mean = recon_loss_mean + self.kl_weight * kl_loss + gen_weight * adv_loss_mean
            # return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss }, {RK.LOSS_MEAN: loss_mean, RK.RECON_LOSS_MEAN : recon_loss_mean, RK.KL_LOSS_MEAN: kl_loss}
            return { RK.KL_LOSS: kl_loss, RK.RECON_LOSS: recon_loss, RK.LOSS: loss, RK.ADV_LOSS: adv_loss}
    
    def disc_criterion(self, truth, disc_prediction):
        loss = F.binary_cross_entropy(disc_prediction, truth, reduction='sum')
        return loss
    
    def cross_gen_loss(self, batch_dict):
        init_expert = batch_dict[RK.EXPERT]
        cross_expert = RK.MOUSE if init_expert == RK.HUMAN else RK.HUMAN
        loss_out = self.mmvae.cross_generate(batch_dict)
        cross_loss = F.mse_loss(loss_out['reversed_gen'].x_hat, batch_dict[RK.X], reduction='mean')
        return {f"cross_gen_loss/{init_expert}_to_{cross_expert}": cross_loss}
    
    def configure_optimizers(self):
        return self.mmvae.configure_optimizers()
        
    def loss(self, batch_dict, trick_labels=None, return_outputs = False):
        forward_outputs = self(batch_dict)
        predictions = self.get_disc_predictions(forward_outputs)
        # loss_outputs, loss_outputs_mean = self.criterion(batch_dict[RK.X], forward_outputs, trick_labels, predictions)
        loss_outputs = self.criterion(batch_dict[RK.X], forward_outputs, trick_labels, predictions)
        # If forward outputs are needed return forward_outputs
        if return_outputs:
            return loss_outputs, forward_outputs
        else:
            return loss_outputs
    
    def get_disc_predictions(self, forward_outputs):
        predictions = []
        for index, disc in enumerate(self.mmvae.discriminators.values()):
            predictions.append(disc(forward_outputs.encoder_act[index]))
        return predictions
    
    def disc_loss(self, batch_dict, truth_label):
        forward_outputs = self(batch_dict)
        predictions = self.get_disc_predictions(forward_outputs)
        loss_dict = {}
        for index, pred in enumerate(predictions):
            loss_output = self.disc_criterion(truth_label, pred)
            loss_dict["disc"+str((index + 1))] = loss_output

        self.log_dict(loss_dict, on_step=False, on_epoch=True, logger=True,batch_size=self.hparams.batch_size)
        return loss_dict
    
    def mean_items(self, value, batch_size):
        return torch.mean(value) / batch_size
    def training_step(self, batch_dict, batch_idx):
        shared_opt, human_opt, mouse_opt, disc_1_opt, disc_2_opt = self.optimizers()
        disc_opts = [disc_1_opt, disc_2_opt]

        human_label = torch.zeros(self.hparams.batch_size, 1, device=self.device)
        mouse_label = torch.ones(self.hparams.batch_size, 1, device=self.device)

        if batch_dict[RK.EXPERT] == "human":
            expert_opt = human_opt
            truth = human_label
            trick = mouse_label
        else:
            expert_opt = mouse_opt
            truth = mouse_label
            trick = human_label

        for disc_optimizer in disc_opts:
            disc_optimizer.zero_grad()

        losses = self.disc_loss(batch_dict, truth) 
        
        for loss in losses.values():
            self.manual_backward(loss, retain_graph=True)

        for disc_optimizer in disc_opts:
            disc_optimizer.step()


        shared_opt.zero_grad()
        expert_opt.zero_grad()
        # Compute loss outputes, expecting None back as this is the train loop and
        # do not need the forward outputs
        # loss_outputs, means = self.loss(batch_dict, trick)
        loss_outputs = self.loss(batch_dict, trick)
        self.manual_backward(loss_outputs[RK.LOSS])
        self.clip_gradients(shared_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        self.clip_gradients(expert_opt, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
        shared_opt.step()
        expert_opt.step()
        # Tag loss output keys with 'train_'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'train', batch_dict[RK.EXPERT], sep='/')
        # mean_losses = {k: self.mean_items(v, self.hparams.batch_size) for k,v in loss_outputs.items()}
        # Log loss outputs for every step and epoch to logger
        means = {key: value / self.hparams.batch_size for key, value in loss_outputs.items()}
        self.log_dict(means, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

    def on_train_epoch_end(self) -> None:
        self.trainer.train_dataloader.reset()
        return super().on_train_epoch_end()

    def validation_step(self, batch_dict, batch_idx):
        # Compute loss_outputs returning the forward_outputes if needed for plotting z embeddings
        if self.plot_z_embeddings:
            loss_outputs, means, forward_outputs = self.loss(batch_dict, return_outputs=self.plot_z_embeddings)
        else:
            loss_outputs = self.loss(batch_dict, return_outputs=False)
        # Tag loss output keys with 'val_'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'val', batch_dict[RK.EXPERT], sep='/')
        
        # Prevent logging sanity_checking steps to logger
        # Not needed for validation and also throws error if batch_size not configured correctly
        if not self.trainer.sanity_checking:
            means = {key: value / self.hparams.batch_size for key, value in loss_outputs.items()}
            self.log_dict(means, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

        cross_loss_outputs = self.cross_gen_loss(batch_dict)
        cross_loss_outputs = utils.tag_mm_loss_outputs(cross_loss_outputs, 'val', batch_dict[RK.EXPERT], sep='/')

        # if not self.trainer.sanity_checking:
            # self.log_dict(cross_loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

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
            self.z_val = []
            self.z_val_metadata = []
        return super().on_validation_epoch_end()
            
    def test_step(self, batch_dict, batch_idx):
        # Compute loss_outputs
        loss_outputs = self.loss(batch_dict, return_outputs=False)
        # Tag loss output keys with 'test'
        loss_outputs = utils.tag_mm_loss_outputs(loss_outputs, 'test', batch_dict[RK.EXPERT], sep='/')
        self.log_dict(loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)
        cross_loss_outputs = self.cross_gen_loss(batch_dict)
        cross_loss_outputs = utils.tag_mm_loss_outputs(cross_loss_outputs, 'test', batch_dict[RK.EXPERT], sep='/')
        # self.log_dict(cross_loss_outputs, on_step=False, on_epoch=True, logger=True, batch_size=self.hparams.batch_size)

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
    