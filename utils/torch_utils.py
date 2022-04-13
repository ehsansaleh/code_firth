import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import _calculate_fan_in_and_fan_out


def torch_gen_xavier_uniform_(tensor, gain=1.0, generator=None):
    # Why this function was necessary? because torch.nn.init.xavier_uniform_ is not accepting generators yet!
    # Note: Although we define our weights in a transposed manner compared to nn.Linear, the sum of 
    # fan_in and fan_out is the same. So there is no need to modify the code.
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))
    a = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        tensor.uniform_(-a, a, generator=generator)


def conjugate_gradient(closure, b, x=None, stop_criterion=('eps', 1e-8), verbose=False):
    ref_eps_sq = 1e-20
    if x is None:
        x = torch.zeros_like(b)
    with torch.no_grad():
        if torch.sum(b ** 2) < ref_eps_sq:
            return x, torch.tensor(0.).double(), 0., 0.
    r = torch.zeros_like(b)
    r = closure(x, out=r).detach()
    r.sub_(b)
    r.neg_()

    p = r.clone().detach()
    rsold = torch.dot(r, r)
    rs_init = rsold.item()
    rsnew = rsold

    dim_ = torch.numel(b)
    Ap = torch.zeros_like(p)
    for k in range(dim_):
        Ap = closure(p, out=Ap).detach()
        ptAp = torch.dot(Ap, p)
        alpha = rsold / ptAp
        x = torch.add(x, p, alpha=alpha, out=x)  # x.add_(p, alpha)
        r = torch.sub(r, Ap, alpha=alpha, out=r)  # r.sub_(Ap, alpha)

        rsnew = torch.dot(r, r)
        if verbose:
            print(f'k: {k}, r**2: {rsnew}', flush=True)

        finished = False
        for cri_cntr in range(len(stop_criterion)//2):
            cond = stop_criterion[2*cri_cntr]
            argcond = stop_criterion[2*cri_cntr + 1]
            if cond == 'eps':
                finished = finished or rsnew < argcond ** 2
            elif cond == 'iter':
                finished = finished or k >= argcond
            elif cond == 'frac':
                finished = finished or rsnew < rs_init * argcond
            else:
                raise Exception(f'Unkown stop_criterion: {stop_criterion}')

        if rsold < ref_eps_sq and verbose:
            msg = 'CG Residual is too small. Consider using a smaller number of iterations or a better stopping ' \
                  'criterion. '
            print(msg, flush=True)

        if finished:
            xtAx = torch.dot(b, x) - torch.dot(r, x)
            return x, xtAx, rs_init, rsnew.item()

        p.mul_(rsnew / rsold).add_(r)
        rsold = rsnew
    xtAx = torch.dot(b, x) - torch.dot(r, x)
    return x, xtAx, rs_init, rsnew.item()


def update_net(net, net_init, alpha, update_dir):
    with torch.no_grad():
        for stuff in zip(sorted(net.named_parameters()),
                         sorted(net_init.named_parameters()),
                         update_dir):
            (p_name_, p_), (init_p_name_, init_p_), update_dir_p_ = stuff
            p_.set_(init_p_ + alpha * update_dir_p_)
    return net


def minkowski_norm_batched(net, norm=2):
    if norm != 2:
        raise ValueError(f'Norm {norm} not implemented')
    exp_bs_list_ = [p.shape[0] for p in net.parameters()]
    exp_bs_ = exp_bs_list_[0]
    assert all(exp_bs_ == x for x in exp_bs_list_), 'all net params must have the same leading batch dim'
    with torch.no_grad():
        ssqs_list = [p.reshape(exp_bs_, -1).square().sum(dim=1, keepdim=True) for p in net.parameters()]
        ssqs_ = torch.cat(ssqs_list, dim=1).sum(dim=1)
        return ssqs_.tolist()


class ExpBatchLinNet(nn.Module):
    def __init__(self, exp_bs, in_dim, out_dim, device, tch_dtype):
        super(ExpBatchLinNet, self).__init__()
        self.exp_bs = exp_bs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Parameter(torch.zeros((exp_bs, in_dim, out_dim), device=device, dtype=tch_dtype))
        self.fc1_bias = torch.nn.Parameter(torch.zeros((exp_bs, 1, out_dim), device=device, dtype=tch_dtype))

    def forward(self, x):
        assert x.ndim == 3, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[0] == self.exp_bs, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[2] == self.in_dim, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        x = x.matmul(self.fc1) + self.fc1_bias
        return x

    
class ExpBatchMLPNet(nn.Module):
    def __init__(self, exp_bs, in_dim, out_dim, device, tch_dtype):
        super(ExpBatchMLPNet, self).__init__()
        self.exp_bs = exp_bs
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.fc1 = torch.nn.Parameter(torch.zeros((exp_bs, in_dim, 100), device=device, dtype=tch_dtype))
        self.fc1_bias = torch.nn.Parameter(torch.zeros((exp_bs, 1, 100), device=device, dtype=tch_dtype))
        self.fc2 = torch.nn.Parameter(torch.zeros((exp_bs, 100, 50), device=device, dtype=tch_dtype))
        self.fc2_bias = torch.nn.Parameter(torch.zeros((exp_bs, 1, 50), device=device, dtype=tch_dtype))
        self.fc3 = torch.nn.Parameter(torch.zeros((exp_bs, 50, out_dim), device=device, dtype=tch_dtype))
        self.fc3_bias = torch.nn.Parameter(torch.zeros((exp_bs, 1, out_dim), device=device, dtype=tch_dtype))

    def forward(self, x):
        assert x.ndim == 3, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[0] == self.exp_bs, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        assert x.shape[2] == self.in_dim, f'{x.shape} != ({self.exp_bs}, n_samps_per_exp, {self.in_dim})'
        x = x.matmul(self.fc1) + self.fc1_bias
        x = F.relu(x)
        x = x.matmul(self.fc2) + self.fc2_bias
        x = F.relu(x)
        x = x.matmul(self.fc3) + self.fc3_bias
        return x
