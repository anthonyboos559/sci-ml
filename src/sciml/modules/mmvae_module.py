import torch
import torch.nn as nn
import torch.nn.functional as F
from sciml.utils.constants import REGISTRY_KEYS as RK

from .mixins.mmvae import MMVAEMixIn
from .basic_vae_module import BasicVAE
from .expert_module import Expert


class MMVAE(MMVAEMixIn, nn.Module):
    
    def __init__(
        self,
        vae: BasicVAE,
        human_expert: Expert,
        mouse_expert: Expert
    ):
        super().__init__()
        self.vae = vae
        self.experts = nn.ModuleDict({RK.HUMAN: human_expert, RK.MOUSE: mouse_expert})
    
    def configure_optimizers(self):
        return (torch.optim.Adam(self.vae.parameters()),
                torch.optim.Adam(self.experts[RK.HUMAN].parameters()),
                torch.optim.Adam(self.experts[RK.MOUSE].parameters()))
    
    def cross_gen_by_metadata(self, h_batch_dict, m_batch_dict):
        h_cis_gen = self(h_batch_dict)
        h_cross_gen = self.cross_generate(h_batch_dict)
        m_cis_gen = self(m_batch_dict)
        m_cross_gen = self.cross_generate(m_batch_dict)

        h_cis_loss = F.mse_loss(h_cis_gen.x_hat, h_batch_dict[RK.X], reduction='mean')
        h_cross_loss = F.mse_loss(h_cross_gen.x_hat, m_batch_dict[RK.X], reduction='mean')

        m_cis_loss = F.mse_loss(m_cis_gen.x_hat, m_batch_dict[RK.X], reduction='mean')
        m_cross_loss = F.mse_loss(m_cross_gen.x_hat, h_batch_dict[RK.X], reduction='mean')

        return {
            'human_cis': h_cis_loss.cpu(),
            'human_cross': h_cross_loss.cpu(),
            'mouse_cis': m_cis_loss.cpu(),
            'mouse_cross': m_cross_loss.cpu()
        }
        # return {
        #     'human_cis': (h_cis_loss, h_cis_gen.x_hat),
        #     'human_cross': (h_cross_loss, h_cross_gen.x_hat),
        #     'mouse_cis': (m_cis_loss, m_cis_gen.x_hat),
        #     'mouse_cross': (m_cross_loss, m_cross_gen.x_hat),
        # }