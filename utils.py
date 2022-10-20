import torch


def torch_cos_sim(x, y):
    if len(x.shape) == 1:
        x = x.unsqueeze(0)

    if len(y.shape) == 1:
        y = y.unsqueeze(0)

    x_norm = torch.nn.functional.normalize(x, p=2, dim=1)
    y_norm = torch.nn.functional.normalize(y, p=2, dim=1)
    cos_sims = torch.mm(x_norm, y_norm.transpose(0, 1))
    return cos_sims


def torch_unique(x):
    unique, inverse = torch.unique(x, sorted=True, return_inverse=True)
    perm = torch.arange(inverse.size(0), dtype=inverse.dtype, device=inverse.device)
    inverse, perm = inverse.flip([0]), perm.flip([0])
    perm = inverse.new_empty(unique.size(0)).scatter_(0, inverse, perm)
    return unique, perm


def expand(elems):
    if elems.size == 1:
        return [elems]
    else:
        return elems
