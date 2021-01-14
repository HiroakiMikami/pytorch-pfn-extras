from typing import TypeVar, Optional

import torch
from torch.utils.data import SubsetRandomSampler, Sampler
import torch.distributed as dist

import numpy as np


T_co = TypeVar('T_co', covariant=True)


class DistributedSubsetSampler(Sampler[T_co]):
    def __init__(
        self,
        dataset: torch.utils.data.dataset.Dataset,
        num_replicas: Optional[int] = None,
        rank: Optional[int] = None,
        shuffle: bool = True,
        seed: int = 0,
    ):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        n_total_samples = len(dataset)
        n_sub_samples = (n_total_samples + num_replicas - 1) // num_replicas
        b = n_total_samples * rank // num_replicas
        e = b + n_sub_samples

        if shuffle:
            all_indices = list(range(n_total_samples))
            rng = np.random.RandomState(seed)
            rng.shuffle(all_indices)
            self.indices = all_indices[b:e]
            generator = torch.Generator()
            generator.manual_seed(seed)
            self._sampler = SubsetRandomSampler(self.indices, generator=generator)
        else:
            self.indices = list(range(b, e))
            self._sampler = None

    def __iter__(self):
        if self._sampler is not None:
            return self._sampler.__iter__()
        else:
            return iter(self.indices)

    def __len__(self):
        if self._sampler is not None:
            return self._sampler.__len__()
        else:
            return len(self.indices)
