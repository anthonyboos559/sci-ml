import os, torch, random, argparse
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, linregress
from torch.utils.data import DataLoader
from torchdata.datapipes.iter import Zipper
from jsonargparse import ArgumentParser
from models import MMVAEModel
from data.local import GC_MD_CellxgeneDataPipe
from data.local import CellxgeneDataPipe
from data.local import MMLoader
from utils.constants import REGISTRY_KEYS as RK

_DEFAULT_CKPT_PATH = '/mnt/projects/debruinz_project/tensorboard_logs/tony_boos/New_MMVAE/new_codebase/checkpoints/epoch=29-step=2439840.ckpt'
_DEFAULT_MODEL_CONFIG = '/mnt/projects/debruinz_project/tony_boos/sci-ml/configs/mmvae/model.yaml'
_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class CxG_Pipeline:

    def __init__(self, ckpt_path, model_config) -> None:
        model = self._get_model_from_config(ckpt_path, model_config)
        self.model = model.module.to(_DEVICE)
        self.model.eval()

        # self.human_input_data = []
        # self.human_input_metadata = []
        # self.human_output_data = []
        # self.human_cross_gen_data = []

        # self.mouse_input_data = []
        # self.mouse_input_metadata = []
        # self.mouse_output_data = []
        # self.mouse_cross_gen_data = []
        

    def _get_model_from_config(self, ckpt_path, model_config):
        parser = ArgumentParser()
        parser.add_argument('model', type=MMVAEModel)
        config = parser.parse_path(model_config)
        config_init = parser.instantiate_classes(config)
        return MMVAEModel.load_from_checkpoint(ckpt_path, module=config_init.model.module)
    
    def _get_filtered_metadata_dataloader(self):
        _DATA_DIRECTORY = '/mnt/projects/debruinz_project/summer_census_data/filtered_by_metadata/'
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

    def get_test_dataloader(self):
        _DATA_DIRECTORY = '/mnt/projects/debruinz_project/summer_census_data/3m_subset/'
        h_dp = CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                   npz_mask=['3m_human_counts_15.npz'],
                                   metadata_mask=['3m_human_metadata_15.pkl'],
                                   batch_size=128,
                                   return_dense=True,
                                   seed=42,
                                   verbose=False)
        m_dp = CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                    npz_mask=['3m_mouse_counts_15.npz'],
                                    metadata_mask=['3m_mouse_metadata_15.pkl'],
                                    batch_size=128,
                                    return_dense=True,
                                    seed=42,
                                    verbose=False)
        h_dl = DataLoader(dataset=h_dp, 
                                batch_size=None,
                                timeout=60,
                                shuffle=False,
                                collate_fn=lambda x: x,
                                pin_memory=True,
                                num_workers=1)
        m_dl = DataLoader(dataset=m_dp, 
                                batch_size=None,
                                timeout=60,
                                shuffle=False,
                                collate_fn=lambda x: x,
                                pin_memory=True,
                                num_workers=1)
        return MMLoader(human_dl=h_dl, mouse_dl=m_dl)

    def get_val_dataloader(self):
        _DATA_DIRECTORY = '/mnt/projects/debruinz_project/summer_census_data/3m_subset/'
        h_dp = CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                   npz_mask=['3m_human_counts_14.npz'],
                                   metadata_mask=['3m_human_metadata_14.pkl'],
                                   batch_size=128,
                                   return_dense=True,
                                   seed=42,
                                   verbose=False)
        m_dp = CellxgeneDataPipe(directory_path=_DATA_DIRECTORY,
                                    npz_mask=['3m_mouse_counts_14.npz'],
                                    metadata_mask=['3m_mouse_metadata_14.pkl'],
                                    batch_size=128,
                                    return_dense=True,
                                    seed=42,
                                    verbose=False)
        h_dl = DataLoader(dataset=h_dp, 
                                batch_size=None,
                                timeout=30,
                                shuffle=False,
                                collate_fn=lambda x: x,
                                pin_memory=True,
                                num_workers=1)
        m_dl = DataLoader(dataset=m_dp, 
                                batch_size=None,
                                timeout=30,
                                shuffle=False,
                                collate_fn=lambda x: x,
                                pin_memory=True,
                                num_workers=1)
        return MMLoader(human_dl=h_dl, mouse_dl=m_dl)
    
    def _check_matching_md(self, h_data, m_data, cols_to_check):
        h_md = h_data.get(RK.METADATA).reset_index(drop=True)
        m_md = m_data.get(RK.METADATA).reset_index(drop=True)
        return h_md[cols_to_check].equals(m_md[cols_to_check])
    
    def get_cross_gen_filtered(self, file_path):
        dataloader = self._get_filtered_metadata_dataloader()
        checked_columns = ['sex', 'assay', 'cell_type', 'tissue']
        metrics = {}
        for h_data, m_data in dataloader:
            if self._check_matching_md(h_data, m_data, checked_columns):
                h_data.update({RK.EXPERT: RK.HUMAN})
                m_data.update({RK.EXPERT: RK.MOUSE})
                h_data[RK.X] = h_data[RK.X].to(_DEVICE)
                m_data[RK.X] = m_data[RK.X].to(_DEVICE)
                with torch.no_grad():
                    loss_outputs = self.model.cross_gen_by_metadata(h_data, m_data)
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
        plt.savefig(file_path)

    def plot_scatter(self, path, file, title, x, y):
        # # Calculate distances from the 1:1 line
        # distances = np.abs(y - x)

        # # Determine colors: blue if distance <= 0.5 or any value is 0, otherwise red
        # colors = np.where((distances <= 0.5) | (y == 0) | (x == 0), 'blue', 'red')

        correlation_coef, _ = pearsonr(x, y)
        title = title + f' Pearson Correlation: {correlation_coef}'
        axis = np.arange(0, 6, 0.5)
        # Create the reconstruction plot
        plt.figure(figsize=(10, 10))
        plt.scatter(x, axis, c='blue', label='Input')
        plt.scatter(y, axis, c='red', label='Output')
        plt.xlabel('True Input')
        plt.ylabel('Reconstructed Output')
        plt.title(title)
        plt.grid(True)
        plt.savefig(os.path.join(path, file))
        plt.close()

    def get_reconstruction_scatterplots(self, output_file_path):
        human_in = []
        human_out = []
        mouse_in = []
        mouse_out = []
        
        dataloader = self.get_test_dataloader()

        for batch in dataloader:
            batch[RK.X] = batch[RK.X].to(_DEVICE)
            with torch.no_grad():
                outputs = self.model(batch)
            if batch[RK.EXPERT] == RK.HUMAN:
                inp = batch[RK.X].cpu()
                out = outputs.x_hat.cpu()
                human_in.append(inp.numpy())
                human_out.append(out.numpy())
            else:
                inp = batch[RK.X].cpu()
                out = outputs.x_hat.cpu()
                mouse_in.append(inp.numpy())
                mouse_out.append(out.numpy())

        human_in = np.concatenate(human_in)
        human_out = np.concatenate(human_out)
        mouse_in = np.concatenate(mouse_in)
        mouse_out = np.concatenate(mouse_out)
        single_sample = random.randint(0, human_in.shape[0])
        single_feature_h = random.randint(0, human_in.shape[1])
        single_feature_m = random.randint(0, mouse_in.shape[1])

        self.plot_scatter(output_file_path, 'human_single_sample.png', 'Single Human Sample', human_in[single_sample, :], human_out[single_sample, :])
        self.plot_scatter(output_file_path, 'mouse_single_sample.png', 'Single Mouse Sample', mouse_in[single_sample, :], mouse_out[single_sample, :])

        self.plot_scatter(output_file_path, 'human_single_feature.png', 'Single Human Feature', human_in[:, single_feature_h], human_out[:, single_feature_h])
        self.plot_scatter(output_file_path, 'mouse_single_feature.png', 'Single Mouse Feature', mouse_in[:, single_feature_m], mouse_out[:, single_feature_m])

        row_indices = np.random.choice(np.arange(human_in.shape[0]), size=60000, replace=False)
        column_indices = np.random.choice(np.arange(human_out.shape[1]), size=60000, replace=False)
        index_pairs = list(zip(row_indices, column_indices))
        random_in = []
        random_out = []
        for row, col in index_pairs:
            random_in.append(human_in[row, col])
            random_out.append(human_out[row, col])
        self.plot_scatter(output_file_path, 'human_random_sample.png', 'Random Human Sample', np.concatenate(random_in), np.concatenate(random_out))

        row_indices = np.random.choice(np.arange(mouse_in.shape[0]), size=60000, replace=False)
        column_indices = np.random.choice(np.arange(mouse_out.shape[1]), size=60000, replace=False)
        index_pairs = list(zip(row_indices, column_indices))
        random_in = []
        random_out = []
        for row, col in index_pairs:
            random_in.append(mouse_in[row, col])
            random_out.append(mouse_out[row, col])
        self.plot_scatter(output_file_path, 'mouse_random_sample.png', 'Random Mouse Sample', np.concatenate(random_in), np.concatenate(random_out))

    def get_zstar(self, out_dir, filename):
        dataloader = self.get_test_dataloader()
        zstars = {RK.HUMAN: [], RK.MOUSE: []}
        metadata = {RK.HUMAN: [], RK.MOUSE: []}
        for i in dataloader:
            with torch.no_grad():
                x = i[RK.X].to(_DEVICE)
                x = self.model.experts[i[RK.EXPERT_ID]].encode(x)
                _, z = self.model.vae.encode(x)
                z = z.detach().cpu()
            zstars[i[RK.EXPERT_ID]].append(z.numpy())
            metadata[i[RK.EXPERT_ID]].append(i[RK.METADATA])
        np.savez(os.path.join(out_dir, f'human_{filename}_zstar.npz'), np.concatenate(zstars[RK.HUMAN]))
        np.savez(os.path.join(out_dir, f'mouse_{filename}_zstar.npz'), np.concatenate(zstars[RK.MOUSE]))
        
        md = pd.concat(metadata[RK.HUMAN])
        md.to_pickle(os.path.join(out_dir, f'human_{filename}_metadata.pkl'))
        md = pd.concat(metadata[RK.MOUSE])
        md.to_pickle(os.path.join(out_dir, f'mouse_{filename}_metadata.pkl'))

def main(ckpt_path, model_config):
    pipeline = CxG_Pipeline(ckpt_path, model_config)
    # pipeline.get_cross_gen_filtered('/mnt/projects/debruinz_project/tony_boos/pipeline_met_test.png')
    # pipeline.get_reconstruction_scatterplots('/mnt/projects/debruinz_project/tony_boos/scatter_test/')
    pipeline.get_zstar('/mnt/projects/debruinz_project/summer_census_data/z_star_test_data/', 'sum_loss')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt_path', type=str, default=_DEFAULT_CKPT_PATH)
    parser.add_argument('--config', type=str, default=_DEFAULT_MODEL_CONFIG)

    args = parser.parse_args()

    main(args.ckpt_path, args.config)