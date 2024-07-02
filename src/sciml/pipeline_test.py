import os, torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import Zipper
from lightning.pytorch.cli import LightningArgumentParser
from jsonargparse import ArgumentParser
from .models.mmvae import MMVAEModel
from .modules.mmvae_module import MMVAE
from .modules.basic_vae_module import BasicVAE
from .modules.expert_module import Expert
from .data.local._cross_gen_by_md_pipe import GC_MD_CellxgeneDataPipe
from .utils.constants import REGISTRY_KEYS as RK

_DATA_DIRECTORY = '/mnt/projects/debruinz_project/summer_census_data/filtered_by_metadata/'
_CKPT_PATH = '/mnt/projects/debruinz_project/tensorboard_logs/tony_boos/New_MMVAE/cross_gen_1B/checkpoints/epoch=29-step=2439840.ckpt'
_MODEL_CONFIG = '/mnt/projects/debruinz_project/tony_boos/sci-ml/configs/multi_modal_model.yaml'

class CxG_Pipeline:

    def __init__(self, ckpt_path, model_config) -> None:
        model = self._get_model_from_config(ckpt_path, model_config)
        self.model = model.mmvae.to("cuda")

    def _get_model_from_config(self, ckpt_path, model_config):
        print("Setting up parser")
        # parser = LightningArgumentParser()
        parser = ArgumentParser()
        print("Adding args")
        # parser.add_lightning_class_args(MMVAEModel, 'model')
        parser.add_argument('model', type=MMVAEModel)
        print('Parsing path')
        config = parser.parse_path(model_config)
        print("Init classes")
        config_init = parser.instantiate_classes(config)
        print(type(config_init.model))
        print(type(config_init.model.mmvae))
        return MMVAEModel.load_from_checkpoint(ckpt_path, mmvae=config_init.model.mmvae)

def check_matching_md(h_data, m_data, cols_to_check):
    h_md = h_data.get(RK.METADATA).reset_index(drop=True)
    m_md = m_data.get(RK.METADATA).reset_index(drop=True)
    return h_md[cols_to_check].equals(m_md[cols_to_check])

def configure_dataloader():
    h_dp = GC_MD_CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                   npz_mask=['human*.npz'],
                                   metadata_mask=['human*.pkl'],
                                   batch_size=128,
                                   return_dense=True,
                                   seed=42,
                                   verbose=False)
    m_dp = GC_MD_CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                   npz_mask=['mouse*.npz'],
                                   metadata_mask=['mouse*.pkl'],
                                   batch_size=128,
                                   return_dense=True,
                                   seed=42,
                                   verbose=False)
    zip_dp = Zipper(h_dp, m_dp)
    dataloader = DataLoader(dataset=zip_dp, 
                            batch_size=None,
                            timeout=30,
                            shuffle=False,
                            collate_fn=lambda x: x,
                            pin_memory=True,
                            num_workers=1)
    return dataloader

def process_outputs(md_tag, data_dict):
    pass

def main():
    dataloader = configure_dataloader()
    pipeline = CxG_Pipeline(_CKPT_PATH, _MODEL_CONFIG)
    pipeline.model.eval()
    checked_columns = ['sex', 'assay', 'cell_type', 'tissue']
    metrics = {}
    for h_data, m_data in dataloader:
        if check_matching_md(h_data, m_data, checked_columns):
            h_data.update({RK.EXPERT: RK.HUMAN})
            m_data.update({RK.EXPERT: RK.MOUSE})
            h_data[RK.X] = h_data[RK.X].to("cuda")
            m_data[RK.X] = m_data[RK.X].to("cuda")
            with torch.no_grad():
                loss_outputs = pipeline.model.cross_gen_by_metadata(h_data, m_data)
            md_vals = h_data.get(RK.METADATA).iloc[0, :][checked_columns]
            md_tag = ', '.join(f'{k}: {v}' for k, v in md_vals.items())
            metrics.update({md_tag: loss_outputs})
            print(f'Metadata: {md_tag}')
            for k, v in loss_outputs.items():
                print(f'Loss: {k}, Value: {v}')
        else:
            print('ERROR: Metadata did not match!')
            print(f'Human: {h_data.get(RK.METADATA).iloc[0, :][checked_columns]}')
            print(f'Mouse: {m_data.get(RK.METADATA).iloc[0, :][checked_columns]}')
    n_metrics = len(metrics.keys())
    fig, axs = plt.subplots(n_metrics, 1, figsize=(10, 6*n_metrics))
    for i, splot in enumerate(metrics.keys()):
        x_tags = list(metrics[splot].keys())
        vals = list(metrics[splot].values())
        axs[i].bar(x_tags, vals)
        axs[i].set_title(splot)
        axs[i].set_ylabel('Value')
        axs[i].set_xlabel('Metric')
    plt.tight_layout()
    plt.savefig(os.path.join(_DATA_DIRECTORY, 'test_cross_gen_metrics.png'))

if __name__ == "__main__":
    main()