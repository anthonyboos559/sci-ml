from jsonargparse import ArgumentParser
from sciml.models import AdvMMVAEModel
import data.local._cellxgene_datapipe as cellxgene_datapipe
import data.local._multi_modal_loader as multi_modal_loader

import torch

from torch.utils.data import DataLoader

def load_model_from_checkpoint(checkpoint_path, config_path:str):
    parser = ArgumentParser()
    parser.add_argument('model', type=AdvMMVAEModel)
    config = parser.parse_path(config_path)
    config_init = parser.instantiate_classes(config)
    return AdvMMVAEModel.load_from_checkpoint(checkpoint_path, model=config_init.model)

def create_dataloader(batch_size: int):
    human_dataloader = cellxgene_datapipe.CellxgeneDataPipe(
        "/mnt/projects/debruinz_project/summer_census_data/3m_subset",
        "3m_human_counts_14.npz",
        "3m_human_metadata_14.pkl",
        batch_size
    )
    mouse_dataloader = cellxgene_datapipe.CellxgeneDataPipe(
        "/mnt/projects/debruinz_project/summer_census_data/3m_subset",
        "3m_mouse_counts_14.npz",
        "3m_mouse_metadata_14.pkl",
        batch_size
    )

    mm_loader = multi_modal_loader.MMLoader(human_dataloader, mouse_dataloader)

    return mm_loader