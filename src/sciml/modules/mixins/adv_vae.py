import torch
import torch.nn as nn

from sciml.utils.constants import REGISTRY_KEYS as RK, ModelOutputs
from vae import VAEMixIn

class AdvMMVAEMixIn(VAEMixIn):

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
        z = self.reparameterize(qzm, qzv)
        z_star = self.after_reparameterize(z, metadata)
        x_hat = self.decode(z_star)
        
        return ModelOutputs(
            encoder_act=encoder_acts,
            qzm = qzm,
            qzv = qzv,
            z = z, 
            z_star = z_star,
            x_hat = x_hat, 
        )
