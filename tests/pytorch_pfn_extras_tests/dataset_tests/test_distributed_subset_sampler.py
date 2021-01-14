import pytorch_pfn_extras as ppe
import torch


class DummySharedDataset(ppe.dataset.SharedDataset):
    def __init__(self):
        self.data = torch.arange(100).reshape(100, 1)
        super().__init__(self.data.shape)

    def __getitem__(self, idx):
        try:
            x = super().__getitem__(idx)
        except ppe.dataset.ItemNotFoundException:
            x = self.data[idx]
            self.cache_item(idx, x)
        return x

    def __len__(self):
        return len(self.data)


def test_not_shuffle_sampler():
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    # rank=0: [0, 1, 2, 3]
    # rank=1: [3, 4, 5, 6]
    # rank=2: [6, 7, 8, 9]
    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=0, shuffle=False)
    assert list(sampler) == [0, 1, 2, 3]
    assert len(sampler) == 4

    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=1, shuffle=False)
    assert list(sampler) == [3, 4, 5, 6]
    assert len(sampler) == 4

    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=2, shuffle=False)
    assert list(sampler) == [6, 7, 8, 9]
    assert len(sampler) == 4


def test_shuffle_sampler():
    dataset = torch.utils.data.TensorDataset(torch.arange(10))
    indices = []
    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=0, shuffle=True)
    indices.extend(list(sampler))
    assert len(sampler) == 4

    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=1, shuffle=True)
    indices.extend(list(sampler))
    assert len(sampler) == 4

    sampler = ppe.dataset.DistributedSubsetSampler(dataset, num_replicas=3, rank=2, shuffle=True)
    indices.extend(list(sampler))
    assert len(sampler) == 4

    assert set(indices) == set(range(10))
