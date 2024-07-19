import torch
import torch.nn as nn
from .vae import VAE
        
from typing import Union
import torch
import torch.nn as nn
from .mixins.adv_vae import AdvVAEMixIn
from .mixins.init import HeWeightInitMixIn
from ._lightning import LightningSequential

    
class AdvVAE(AdvVAEMixIn, HeWeightInitMixIn, nn.Module):
    def __init__(
        self,
        encoder: LightningSequential,
        decoder: LightningSequential,
        mean: nn.Linear,
        var: nn.Linear,
        use_he_init: bool = True,
        optimizer: bool = None,
        fc_mean_lr: float = 1e-4,
        fc_var_lr: float = 1e-4,
        encoder_lr: Union[float, list[float]] = 1e-3,
        decoder_lr: Union[float, list[float]] = 1e-4,
    ):
        super().__init__()
        self.fc_mean = mean
        self.fc_var = var
        self.encoder = encoder
        self.decoder = decoder
        self.optimizer = optimizer
        self.fc_mean_lr = fc_mean_lr
        self.fc_var_lr = fc_var_lr
        self.encoder_lr = encoder_lr
        self.decoder_lr = decoder_lr
        self.latent_dim = self.fc_mean.out_features
        if use_he_init:
            self.init_weights()
            
    def configure_optimizers(self):
        
        if isinstance(self.encoder_lr, list):
            assert len(self.encoder_lr) == len(self.encoder)
            group = [{'params': getattr(self, name).parameters(), 'lr': getattr(self, f"{name}_lr")} for name in ('fc_mean', 'fc_var', 'encoder', 'decoder')]
            return torch.optim.Adam(group)
        
class BasicAdvVAE(AdvVAE):
    
    def __init__(
        self,
        encoder_layers = [60664, 1024, 512], 
        latent_dim = 256, 
        decoder_layers = [512, 1024, 60664],
        use_he_init = False,
    ):
        super().__init__(
            encoder=self.build_encoder(encoder_layers),
            decoder=self.build_decoder(latent_dim, decoder_layers),
            mean=self.build_mean(encoder_layers, latent_dim),
            var=self.build_var(encoder_layers, latent_dim),
            use_he_init=use_he_init
        )
        
    def build_encoder(self, encoder_layers):
        layers = []
        n_in = encoder_layers[0]
        i = 0
        for n_out in encoder_layers[1:]:
            if i == 0:
                layers.extend([
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, eps=0.001, momentum=0.01),
                    nn.ReLU(),
                    nn.Dropout(p=0.1, inplace=False)
                ])
            else:
                layers.extend([
                    nn.Linear(n_in, n_out),
                    nn.BatchNorm1d(n_out, eps=0.001, momentum=0.01),
                    nn.ReLU(),
                ])
            n_in = n_out
            i += 1
        return nn.Sequential(*layers)
    
    def build_decoder(self, latent_dim, decoder_layers):
        layers = []
        n_in = latent_dim
        for n_out in decoder_layers:
            layers.extend([
                nn.Linear(n_in, n_out),
                nn.ReLU(),
            ])
            n_in = n_out
        return nn.Sequential(*layers)
    
    def _build_mean_var(self, encoder_layers, latent_dim):
        return nn.Linear(encoder_layers[-1], latent_dim)
    
    build_mean = _build_mean_var
    build_var = _build_mean_var

