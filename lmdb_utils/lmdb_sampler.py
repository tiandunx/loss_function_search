import torch
from torch.utils.data import Sampler
import torch.distributed as dist


class LMDBDistributeSampler(Sampler):
    """Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        shuffle (optional): If true (default), sampler will shuffle the indices
    """

    def __init__(self, dataset, rank=None, shuffle=True):
        if not dist.is_available():
            raise RuntimeError("Requires distributed package to be available")
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.rank = rank
        self.epoch = 0
        self.num_samples = len(dataset)
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(self.num_samples, generator=g).tolist()
        else:
            indices = list(range(self.num_samples))
        assert len(indices) == self.num_samples
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch
