# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import bisect
import logging

import numpy as np
from torch.utils.data.dataloader import default_collate

from . import ConcatDataset


class ConcatDatasetLarge(ConcatDataset):
    def __init__(self, datasets, sample_ratios=1):
        super().__init__(datasets, sample_ratios)

    @property
    def sizes(self):
        total_size = sum(len(ds.sizes) * sr if isinstance(ds.sizes, np.ndarray) 
                     else len(ds.sizes)[0] * sr for ds, sr in zip(self.datasets, self.sample_ratios))
        _dataset_sizes = np.zeros(total_size, dtype=np.uint64) 

        current_idx = 0
        for ds, sr in zip(self.datasets, self.sample_ratios):
            logging.info(f"ds.sizes: {ds.sizes}, type {type(ds.sizes)}, len: {len(ds.sizes)}")
            tiled_sizes = np.tile(ds.sizes, sr) if isinstance(ds.sizes, np.ndarray) else np.tile(ds.sizes[0], sr)
            logging.info(f"np.tile(ds.sizes, sr): {tiled_sizes}, shape: {tiled_sizes.shape}")
            _dataset_sizes[current_idx:current_idx + len(tiled_sizes)] = tiled_sizes
            current_idx += len(tiled_sizes)
            
        return _dataset_sizes