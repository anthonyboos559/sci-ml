import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence

from sciml.utils.constants import REGISTRY_KEYS as RK, ModelOutputs
from .vae import VAEMixIn

class AdvVAEMixIn(VAEMixIn):
    def encode(self, x):
        activations = []
        for _, layer in self.encoder.named_children():
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                activations.append(x.clone())

        x = self.before_reparameterize(x)
        return activations, self.fc_mean(x), self.fc_var(x)

    def forward(self, batch_dict):
        x = batch_dict.get(RK.X)
        metadata = batch_dict.get(RK.METADATA)
        
        encoder_acts, qzm, qzv = self.encode(x)
        qzv = torch.exp(qzv) + 1e-4
        qz = Normal(qzm, qzv.sqrt())
        latent = qz.rsample()
        x_hat = self.decode(latent)
        
        return ModelOutputs(
            encoder_act=encoder_acts,
            latent = latent,
            qz = qz, 
            x_hat = x_hat, 
        )
