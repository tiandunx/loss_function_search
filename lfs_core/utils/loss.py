import torch
import torch.nn.functional as F
import math


def my_loss(x, lb, p_bins, a, sm, search_type):
    batch_size = x.shape[0]
    new_x = 1.0 * x
    if search_type == 'global':
        if a[0] <= 0:
            b = 1.0 - a[0] * math.exp(sm / 3)
        else:
            b = 1.0
        gt = x[torch.arange(batch_size), lb]
        new_x[torch.arange(batch_size), lb] = gt / (a[0] * math.exp(sm / 3) * gt + b)
    elif search_type == 'local':
        for i in range(batch_size):
            for j in range(len(p_bins) - 1):
                if x[i, lb[i]].item() <= p_bins[j + 1]:
                    if a[j] <= 0:
                        b = 1.0 - a[j] * math.exp(sm / 2)
                    else:
                        b = 1.0
                    new_x[i, lb[i]] = x[i, lb[i]] / (a[j] * math.exp(sm / 2) * x[i, lb[i]] + b)
                    break
    else:
        raise Exception('Unknown search type!')
    return new_x


def loss_search(outputs, targets, p_bins, a, sm, search_type='global'):
    # assume outputs is already pass through softmax
    lb = targets.view(-1, 1)
    if lb.is_cuda:
        lb = lb.cpu()
    outputs = my_loss(outputs, lb, p_bins, a, sm, search_type)
    loss = F.nll_loss(torch.log(outputs), targets)
    return loss
