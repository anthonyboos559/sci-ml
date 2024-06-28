import torch

from sciml.utils.constants import REGISTRY_KEYS as RK

class MMVAEMixIn:
    """
    Defines mmvae forward pass.
    Expectes encoder, decoder, fc_mean, fc_var, and experts to be defined
    """
    
    def cross_generate(self, input_dict):
        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        expert_id = input_dict[RK.EXPERT]
        other_expert = RK.MOUSE if expert_id == RK.HUMAN else RK.HUMAN

        x = self.experts[expert_id].encode(x)
        vae_out = self.vae({RK.X: x, RK.METADATA: metadata})
        x_hat = self.experts[other_expert].decode(vae_out.x_hat)

        return vae_out._replace(x_hat=x_hat)
    
    def forward(self, input_dict):
        
        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        expert_id = input_dict[RK.EXPERT]

        x = self.experts[expert_id].encode(x)
        vae_out = self.vae({RK.X: x, RK.METADATA: metadata})
        x_hat = self.experts[expert_id].decode(vae_out.x_hat)

        # adv1_loss = self.adv1(vae_out.encoder_activations[0])

        return vae_out._replace(x_hat=x_hat)