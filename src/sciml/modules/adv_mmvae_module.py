import torch
import torch.nn as nn
from .mixins.mmvae import MMVAEMixIn
from sciml.utils.constants import REGISTRY_KEYS as RK

from .expert_module import Expert
from .adv_vae_module import AdvVAE
from .disc_module import Discriminator

class AdvMMVAE(MMVAEMixIn, nn.Module):
    
    def __init__(
        self,
        vae : AdvVAE,
        human_expert : Expert,
        mouse_expert : Expert,
        disc_1 : Discriminator,
        disc_2 : Discriminator
    ):
        super().__init__()
        self.vae = vae
        self.experts = nn.ModuleDict({RK.HUMAN: human_expert, RK.MOUSE: mouse_expert})
        self.discriminators = nn.ModuleDict({"disc 1": disc_1, "disc 2": disc_2})

    def configure_optimizers(self):
        return (torch.optim.Adam(self.vae.parameters()),
                torch.optim.Adam(self.experts[RK.HUMAN].parameters()),
                torch.optim.Adam(self.experts[RK.MOUSE].parameters()),

                torch.optim.Adam(self.discriminators["disc 1"].parameters()),
                torch.optim.Adam(self.discriminators["disc 2"].parameters())
                )
