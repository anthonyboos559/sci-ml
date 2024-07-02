from typing import Union
import numpy as np
import scipy.sparse as sp
import pandas as pd
import pickle

from sciml.utils.constants import REGISTRY_KEYS as RK

import torch
from torchdata.datapipes.iter import FileLister, IterDataPipe, Zipper
from torch.utils.data import functional_datapipe
from torch.utils.data.datapipes.datapipe import IterDataPipe
from torch.utils.data.datapipes.iter import sharding
from ._cellxgene_datapipe import LoadCSRMatrixAndMetadataDataPipe, SparseCSRMatrixBatcherDataPipe

class CG_MD_ChunkloaderDataPipe(IterDataPipe):
    
    def __init__(self, directory_path: str, npz_masks: Union[str, list[str]], metadata_masks: Union[str, list[str]], verbose: bool = False):
        super(CG_MD_ChunkloaderDataPipe, self).__init__()
        
        # Create file lister datapipe for all npz files in dataset
        self.npz_paths_dp = FileLister(
            root=directory_path, 
            masks=npz_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )

        # Create file lister datapipe for all metadata files 
        self.metadata_paths_dp = FileLister(
            root=directory_path, 
            masks=metadata_masks,
            recursive=False,
            abspath=True,
            non_deterministic=False
        )
        
        self.zipped_paths_dp = Zipper(self.npz_paths_dp, self.metadata_paths_dp)
        # Make sure each worker gets individual chunk
        self.zipped_paths_dp = self.zipped_paths_dp.sharding_filter()
        
        # Sanity check that the metadata files and npz files are correlated
        # and all files are masked correctly
        chunk_paths = []
        for npz_path, metadata_path in self.zipped_paths_dp:
            chunk_paths.append(f"\n\t{npz_path}\n\t{metadata_path}")
        if not chunk_paths:
            raise RuntimeError("No files found for masks from file lister")
        
        if verbose:
            for path in chunk_paths:
                print(path)
                
        self.verbose = verbose
                
    def __iter__(self):
        try:
            yield from self.zipped_paths_dp \
                .load_matrix_and_metadata(self.verbose)
        except Exception as e:
            print(f"Error during iteration: {e}")
            raise
        finally:
            # Ensure all resources are properly cleaned up
            pass
        
class GC_MD_CellxgeneDataPipe(IterDataPipe):
    
    def __init__(
        self,
        directory_path: str, 
        npz_mask: Union[str, list[str]], 
        metadata_mask: Union[str, list[str]], 
        batch_size: int,
        return_dense=False,
        seed: int = 42,
        verbose=False, 
    ) -> IterDataPipe: # type: ignore
        """
        Pipeline built to load Cell Census sparse csr chunks. 
        
        Args:
        - directory_path: str, 
        - npz_mask: Union[str, list[str]], 
        - metadata_mask: Union[str, list[str]], 
        - batch_size: int,
        - return_dense (bool): Converts torch.sparse_csr_tensor batch to dense
        - verbose (bool): print npz and metata file pairs, 

        Important Note: The sharding_filter is applied aftering opening files 
            to ensure no duplication of chunks between worker processes.
        """
        super().__init__()
        self.datapipe = CG_MD_ChunkloaderDataPipe(directory_path, npz_mask, metadata_mask, verbose=verbose) \
            .batch_sparse_csr_matrix(batch_size, return_dense=return_dense)
        self.seed = seed
        
    def __iter__(self):
        seed = np.random.randint(0, 100)
        np.random.seed(seed)
        torch.manual_seed(seed)
        yield from self.datapipe