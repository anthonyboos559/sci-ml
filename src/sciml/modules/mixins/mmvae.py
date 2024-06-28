import torch

from sciml.utils.constants import REGISTRY_KEYS as RK
from sciml.utils.constants import ModelOutputs

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
        shared_output = self.vae({RK.X: x, RK.METADATA: metadata})
        x_hat = self.experts[other_expert].decode(shared_output.x_hat)

        expert1_output = ModelOutputs(shared_output.encoder_act,
                                    shared_output.qzm,
                                    shared_output.qzv,
                                    shared_output.z,
                                    shared_output.z_star,
                                    x_hat
                                    )
                                    
        cross_gen_dict["initial_gen"] = expert1_output

        x = self.experts[other_expert].encode(expert1_output.x_hat)
        shared_output = self.vae({RK.X: x, RK.METADATA: metadata})
        x_hat = self.experts[expert_id].decode(shared_output.x_hat)

        expert2_output = ModelOutputs(shared_output.encoder_act,
                                    shared_output.qzm,
                                    shared_output.qzv,
                                    shared_output.z,
                                    shared_output.z_star,
                                    x_hat
                                    )
        cross_gen_dict["reversed_gen"] = expert2_output

        return cross_gen_dict
    
    def forward(self, input_dict):
        
        x = input_dict[RK.X]
        metadata = input_dict.get(RK.METADATA)
        expert_id = input_dict[RK.EXPERT]

        x = self.experts[expert_id].encode(x)
        shared_output : ModelOutputs = self.vae({RK.X: x, RK.METADATA: metadata})
        x_hat = self.experts[expert_id].decode(shared_output.x_hat)

        model_output = ModelOutputs(shared_output.encoder_act,
                                    shared_output.qzm,
                                    shared_output.qzv,
                                    shared_output.z,
                                    shared_output.z_star,
                                    x_hat
                                    )

        return model_output