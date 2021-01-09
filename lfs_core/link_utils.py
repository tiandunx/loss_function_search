import torch.distributed as dist
import torch


def broadcast_params(model, rank):
    for _, item in model.state_dict().items():
        dist.broadcast(item, rank)
    dist.barrier()


def all_gather(tensor):
    res = [torch.ones_like(tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(res, tensor)
    return torch.stack(res)


def all_reduce(tensor):
    dist.all_reduce(tensor)
    world_size = dist.get_world_size()
    tensor.data /= world_size


def broadcast(tensor, rank=0):
    dist.broadcast(tensor, rank)
