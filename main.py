import os
import random
import socket
import time
import json
import io
from collections import defaultdict, OrderedDict
from itertools import product, chain
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torchvision import transforms

from utils.datasets import MiniImageNet
from utils.datasets import TieredImageNet, CifarFS, FeaturesDataset
from utils.datasets import make_backbone
from utils.torch_utils import torch_gen_xavier_uniform_
from utils.torch_utils import conjugate_gradient, minkowski_norm_batched
from utils.torch_utils import ExpBatchLinNet, ExpBatchMLPNet
from utils.torch_utils import update_net
from utils.io_utils import DataWriter, dict_hash, append_to_tar
from utils.io_utils import get_git_commit, check_cfg_dict, logger
from utils.io_utils import get_cuda_memory_per_seed as getmemperseed


def main(config_dict):
    check_cfg_dict(config_dict)

    config_id = config_dict['config_id']
    device_name = config_dict['device_name']
    data_dir = config_dict['data_dir']
    features_dir = config_dict['features_dir']
    backbones_dir = config_dict['backbones_dir']
    start_seed = config_dict['start_seed']
    num_seeds = config_dict['num_seeds']
    dataset_name_list = config_dict['dataset_name_list']
    backbone_arch_list = config_dict['backbone_arch_list']
    data_type_list = config_dict['data_type_list']
    n_shots_list = config_dict['n_shots_list']
    firth_coeff_list = config_dict['firth_coeff_list']
    entropy_coeff_list = config_dict['entropy_coeff_list']
    l2_coeff_list = config_dict['l2_coeff_list']

    learning_rate = config_dict['learning_rate']
    batch_size = config_dict['batch_size']
    n_epochs = config_dict['n_epochs']
    optim_type = config_dict['optim_type']
    n_query = config_dict['n_query']
    fix_query_set = config_dict['fix_query_set']

    permute_labels = config_dict['permute_labels']
    torch_threads = config_dict['torch_threads']
    nshots_to_clsfreqs = config_dict['nshots_to_clsfreqs']  # a dictionary
    shuffle_mb = config_dict['shuffle_mb']
    firth_prior_type_list = config_dict['firth_prior_type_list']

    # model setting
    clf_type = config_dict['clf_type']
    n_ways_list = config_dict['n_ways_list']
    dump_period = config_dict['dump_period']
    store_results = config_dict['store_results']
    results_dir = config_dict.get('results_dir', None)
    store_predictions = config_dict['store_predictions']
    store_clfweights = config_dict['store_clfweights']
    clfweights_dir = config_dict.get('clfweights_dir', None)
    assert not store_clfweights, 'The storing of classifier weights is broken and takes too long.'
    hostname = socket.gethostname()
    commit_hash = get_git_commit()

    target_cuda_memory = 6e9
    device = torch.device(device_name)
    tch_dtype = torch.float32

    data_writer = None
    if store_results:
        assert results_dir is not None, 'Please provide results_dir in the config_dict.'
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        data_writer = DataWriter(dump_period=dump_period)

    if store_clfweights:
        assert clfweights_dir is not None, 'Please provide clfweights_dir in the config_dict.'
        Path(clfweights_dir).mkdir(parents=True, exist_ok=True)

    untouched_torch_thread = torch.get_num_threads()
    if torch_threads:
        torch.set_num_threads(torch_threads)

    # Other configs that worked so well it was not worth adding them to the cfg files.
    cg_eta = .5
    pring_cg_log = False

    for setting in product(dataset_name_list, data_type_list,
                           firth_prior_type_list, backbone_arch_list, n_ways_list,
                           n_shots_list, l2_coeff_list, firth_coeff_list,
                           entropy_coeff_list):

        (dataset_name, data_type, firth_prior_type, backbone_arch, n_ways,
         n_shots, l2_coeff, firth_coeff, ent_coeff) = setting

        mem_per_seed = getmemperseed(backbone_arch, n_ways, n_shots, tch_dtype,
                                     n_query, nshots_to_clsfreqs, clf_type,
                                     optim_type, batch_size)

        exp_bs_max = int(target_cuda_memory) // mem_per_seed
        rng_seeds_list = [list(range(i, min(i+exp_bs_max, start_seed+num_seeds)))
                          for i in range(start_seed, start_seed+num_seeds, exp_bs_max)]
        assert sum(len(x) for x in rng_seeds_list) == num_seeds

        for rng_seeds in rng_seeds_list:
            exp_bs = len(rng_seeds)
            logger(f'1) Starting a New Seed Batch (exp_bs = {exp_bs})')
            assert dataset_name in ('miniimagenet', 'tieredimagenet', 'cifarfs')

            logger(f'2) Loading the Data from the Disk Storage')
            # Loading the Data:
            feat_cache_path = f'{features_dir}/{dataset_name}_{data_type}_{backbone_arch}.pth'
            ckpt_path = f'{backbones_dir}/{dataset_name}_{backbone_arch}.pth.tar'
            data_root = f'{data_dir}/{dataset_name}'
            if not os.path.exists(feat_cache_path):
                print(f'Generating {feat_cache_path}.')
                norm_img_mean = [0.485, 0.456, 0.406]
                norm_img_std = [0.229, 0.224, 0.225]
                normalize = transforms.Normalize(mean=norm_img_mean, std=norm_img_std)

                if backbone_arch.startswith('resnet'):
                    resize_pixels = 256
                    crop_pixels = 224
                elif backbone_arch in ('densenet121', 'mobilenet84'):
                    enlarge = False  # From SimpleShot
                    resize_pixels = int(84 * 256. / 224.) if enlarge else 84
                    crop_pixels = 84
                else:
                    raise ValueError(f'backbone {backbone_arch} transformation not implemented.')

                eval_tnsfm_list = [transforms.Resize(resize_pixels),
                                   transforms.CenterCrop(crop_pixels),
                                   transforms.ToTensor(),
                                   normalize]
                eval_transforms = transforms.Compose(eval_tnsfm_list)

                if dataset_name == 'miniimagenet':
                    imgset = MiniImageNet(root=data_root, data_type=data_type, transform=eval_transforms)
                elif dataset_name == 'tieredimagenet':
                    imgset = TieredImageNet(root=data_root, data_type=data_type, transform=eval_transforms)
                elif dataset_name == 'cifarfs':
                    imgset = CifarFS(root=data_root, data_type=data_type, transform=eval_transforms)
                else:
                    raise ValueError(f'Unknown dataset {dataset_name}')

                img_loader = torch.utils.data.DataLoader(imgset, batch_size=32, shuffle=False,
                                                         num_workers=12, prefetch_factor=48,
                                                         pin_memory=False)

                feature_model = make_backbone(backbone_arch=backbone_arch, ckpt_path=ckpt_path, device=device)
            else:
                img_loader = None
                feature_model = None

            featset = FeaturesDataset(feat_cache_path, feature_model=feature_model,
                                      img_loader=img_loader, device=device)

            logger(f'3) Finding num_all_classes and Supoort/Query Sample Sizes')
            all_embedding, all_labels = featset.data, featset.targets
            feat_dim = all_embedding.shape[1]

            all_labels_np = all_labels.detach().cpu().numpy()
            assert np.all(all_labels_np[:-1] <= all_labels_np[1:])

            handoff_idxs = (np.where(all_labels_np[:-1] < all_labels_np[1:])[0] + 1).tolist()
            all_cls_start_indxs = [0] + handoff_idxs
            all_cls_end_indxs = handoff_idxs + [all_labels_np.size]

            num_all_classes = len(all_cls_end_indxs)
            real_n_ways = min(n_ways, num_all_classes)

            if nshots_to_clsfreqs is None:
                suppcls_n_shots_list = [n_shots] * real_n_ways
                print('  --> Working in balanced setting', flush=True)
                do_imbalanced = False
            else:
                suppcls_n_shots_list = nshots_to_clsfreqs[str(n_shots)]
                print('  --> Working in imbalanced setting', flush=True)
                do_imbalanced = True

            with torch.no_grad():
                if firth_prior_type == 'uniform':
                    firth_prior = (torch.ones(real_n_ways, device=device, dtype=tch_dtype)/real_n_ways)
                elif firth_prior_type == 'class_freq':
                    firth_prior = torch.tensor(suppcls_n_shots_list).to(device=device, dtype=tch_dtype)
                else:
                    raise ValueError(f'Unknown prior {firth_prior_type}')
                firth_prior = firth_prior / torch.sum(firth_prior)

            assert len(suppcls_n_shots_list) == real_n_ways, f'{len(suppcls_n_shots_list)} != {real_n_ways}'

            logger(f'4.1) Constructing Random Generators')
            # function of : 1) real_n_ways, num_all_classes
            rlib_rngs = [random.Random(seed) for seed in rng_seeds]
            tch_rngs = [torch.Generator(device=device).manual_seed(int(rlib_rng.randint(0, 2**31-1)))
                        for rlib_rng in rlib_rngs]

            logger(f'4.2) Picking Random Classes and Support/Query Indexes')

            all_exp_supp_idxs, all_exp_supp_lbls = [], []
            all_exp_query_idxs, all_exp_query_lbls = [], []

            for exp_idx in range(exp_bs):
                rlib_rng = rlib_rngs[exp_idx]
                if permute_labels:
                    my_classes = rlib_rng.sample(range(num_all_classes), real_n_ways)
                else:
                    my_classes = list(range(real_n_ways))
                my_cls_start_indxs = (all_cls_start_indxs[i] for i in my_classes)
                my_cls_end_indxs = (all_cls_end_indxs[i] for i in my_classes)

                assert all(all_cls_end_indxs[i] - all_cls_start_indxs[i] + 1 >= suppcls_n_shot + n_query
                           for suppcls_n_shot, i in zip(suppcls_n_shots_list, my_classes))

                if not fix_query_set:
                    my_supp_quer_idxs = [rlib_rng.sample(range(st_idx, end_idx), suppcls_n_shot + n_query)
                                         for suppcls_n_shot, st_idx, end_idx in
                                         zip(suppcls_n_shots_list, my_cls_start_indxs, my_cls_end_indxs)]

                else:
                    my_supp_quer_idxs = [rlib_rng.sample(range(st_idx, end_idx-n_query), suppcls_n_shot)  # support
                                         + list(range(end_idx-n_query, end_idx))  # query
                                         for suppcls_n_shot, st_idx, end_idx in
                                         zip(suppcls_n_shots_list, my_cls_start_indxs, my_cls_end_indxs)]

                # There is no reliable way of copying generators in python,
                # So, we have to create two identical generators for each of
                # support and query indecis.
                my_supp_idxs_2dlist1 = (x[:cls_n_shot] for cls_n_shot, x in
                                        zip(suppcls_n_shots_list, my_supp_quer_idxs))
                my_supp_idxs_2dlist2 = (x[:cls_n_shot] for cls_n_shot, x in
                                        zip(suppcls_n_shots_list, my_supp_quer_idxs))
                my_query_idxs_2dlist1 = (x[cls_n_shot:] for cls_n_shot, x in
                                         zip(suppcls_n_shots_list, my_supp_quer_idxs))
                my_query_idxs_2dlist2 = (x[cls_n_shot:] for cls_n_shot, x in
                                         zip(suppcls_n_shots_list, my_supp_quer_idxs))

                my_supp_idxs = [idx for lbl, cls_idxs in enumerate(my_supp_idxs_2dlist1) for idx in cls_idxs]
                my_supp_lbls = [lbl for lbl, cls_idxs in enumerate(my_supp_idxs_2dlist2) for idx in cls_idxs]

                my_query_idxs = [idx for lbl, cls_idxs in enumerate(my_query_idxs_2dlist1) for idx in cls_idxs]
                my_query_lbls = [lbl for lbl, cls_idxs in enumerate(my_query_idxs_2dlist2) for idx in cls_idxs]

                all_exp_supp_idxs.extend(my_supp_idxs)
                all_exp_supp_lbls.append(my_supp_lbls)
                all_exp_query_idxs.extend(my_query_idxs)
                all_exp_query_lbls.append(my_query_lbls)

            n_total_supp = sum(suppcls_n_shots_list)
            n_total_query = n_query * real_n_ways

            logger(f'4) Slicing the Support/Query Embeddings')

            labels = torch.tensor(all_exp_supp_lbls)
            embedding = all_embedding[all_exp_supp_idxs].reshape(exp_bs, n_total_supp, feat_dim)
            labels_heldout = torch.tensor(all_exp_query_lbls)

            assert labels.shape == (exp_bs, n_total_supp), f'{labels.shape} != {(exp_bs, n_total_supp)}'
            assert embedding.shape == (exp_bs, n_total_supp, feat_dim)
            assert labels_heldout.shape == (exp_bs, n_total_query)

            # Setting default values in case not found
            logger(f'5) Creating the Classifier Networks')

            stat_freq = 50
            if clf_type == 'mlp':
                Net = ExpBatchMLPNet
            elif clf_type == 'lin':
                Net = ExpBatchLinNet
            else:
                raise Exception(f"Unknown classifier type! {clf_type}")

            # Training Loop
            print(f'  --> Firth Coefficient: {firth_coeff}', flush=True)
            print(f'  --> L2 Coefficient: {l2_coeff}', flush=True)
            print(f'  --> Entropy Coefficient: {ent_coeff}', flush=True)
            print(f'  --> Number of Shots: {n_shots}', flush=True)
            print(f'  --> Number of Classes: {n_ways}', flush=True)
            print(f'  --> Backbone Architecture: {backbone_arch}', flush=True)
            print(f'  --> Classifier Type: {clf_type}', flush=True)
            print(f'  --> Data Type: {data_type}', flush=True)
            print(f'  --> Dataset: {dataset_name}', flush=True)

            firth_prior_str = f'  --> Firth Prior: {firth_prior}'
            print(firth_prior_str.replace("\n", "\n"+" "*19), flush=True)
            print('', flush=True)

            net_kwargs = dict(exp_bs=exp_bs, in_dim=feat_dim, out_dim=real_n_ways,
                              device=device, tch_dtype=tch_dtype)
            net = Net(**net_kwargs).to(device)
            net_backup = Net(**net_kwargs).to(device)

            # Initializing
            logger(f'6) Initializing the Classifier Networks')
            for name_, param_ in net.named_parameters():
                if name_ in ('fc1', 'fc2', 'fc3'):
                    for exp_idx in range(exp_bs):
                        tch_rng = tch_rngs[exp_idx]
                        torch_gen_xavier_uniform_(param_.data[exp_idx], gain=1., generator=tch_rng)
                elif name_ in ('fc1_bias', 'fc2_bias', 'fc3_bias'):
                    param_.data.fill_(0.)
                else:
                    raise RuntimeError(f'init for {name_} layer not implemented')

            logger(f'7) Creating the Optimizer')
            lbfgs_optim_maker = None
            if optim_type == 'adam':
                optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=l2_coeff)
            elif optim_type == 'sgd':
                optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
            elif optim_type == 'lbfgs':
                def lbfgs_optim_maker(net_params):
                    return optim.LBFGS(net_params, lr=1., max_iter=20, max_eval=None,
                                       tolerance_grad=-1., tolerance_change=-1.,
                                       history_size=500, line_search_fn=None)
                optimizer = lbfgs_optim_maker(net.parameters())
            elif optim_type == 'cg':
                optimizer = None
                grad_output = tuple(torch.zeros_like(param) for name, param in sorted(net_backup.named_parameters()))
            else:
                optimizer = None

            def l2_regularizer(params):
                dim_ctr = 0.
                param_ssq_lst = []
                len_params = 0
                exp_bs_ = None
                for param in params:
                    exp_bs_ = param.shape[0]
                    param_2d = param.reshape(exp_bs_, -1)
                    dim_ = param_2d.shape[1]
                    dim_ctr = dim_ctr + dim_
                    param_ssq_lst.append(param_2d.square().sum(dim=1, keepdim=True))
                    len_params += 1
                param_ssq = torch.cat(param_ssq_lst, dim=1)
                assert param_ssq.shape == (exp_bs_, len_params)
                l2_reg = param_ssq.sum(dim=1)
                if dim_ctr == 0:
                    dim_ctr = dim_ctr + 1
                l2_reg_mean_over_dim = l2_reg / dim_ctr
                return l2_reg_mean_over_dim

            def firth_regularizer(outs, prior, logp=None):
                exp_bs_, n_samps_, n_ways_ = outs.shape
                if prior is None:
                    with torch.no_grad():
                        prior = (torch.ones(n_ways_, device=outs.device, dtype=outs.dtype)/n_ways_)
                with torch.no_grad():
                    assert (torch.is_tensor(prior)), 'prior is not tensor'
                    assert (prior.numel() == n_ways_), 'prior does not cover all classes'
                    assert np.isclose(torch.sum(prior).item(), 1.), f'prior does not sum to one: {torch.sum(prior)}'

                if logp is None:
                    logp = outs - torch.logsumexp(outs, dim=-1, keepdim=True)
                assert logp.shape == (exp_bs_, n_samps_, n_ways_)

                negce = logp @ prior.reshape(1, n_ways_, 1).expand(exp_bs_, n_ways_, 1)
                assert negce.shape == (exp_bs_, n_samps_, 1)
                ceavg = -negce.mean(dim=[1, 2])  # mean over samples
                assert ceavg.shape == (exp_bs_,)
                return ceavg

            def entropy_regularizer(outs, logp=None):
                exp_bs_, n_samps_, n_ways_ = outs.shape

                if logp is None:
                    logp = outs - torch.logsumexp(outs, dim=-1, keepdim=True)
                assert logp.shape == (exp_bs_, n_samps_, n_ways_)

                prob = torch.exp(logp)
                ent = torch.sum(logp * prob, dim=-1)
                assert ent.shape == (exp_bs_, n_samps_)
                entavg = ent.mean(dim=-1)  # mean over samples
                assert entavg.shape == (exp_bs_,)

                return entavg

            class LossCalculator:
                def __init__(self, firth_coeff_, l2_coeff_, firth_prior_, ent_coeff_):
                    self.firth_coeff_ = firth_coeff_
                    self.l2_coeff_ = l2_coeff_
                    self.firth_prior_ = firth_prior_
                    self.ent_coeff_ = ent_coeff_

                def loss_pack(self, outputs_, labels_, params):
                    prior = self.firth_prior_
                    firth_coeff_ = self.firth_coeff_
                    l2_coeff_ = self.l2_coeff_
                    ent_coeff_ = self.ent_coeff_
                    # exp_bs_, n_samps_, n_ways_ = outputs_.shape
                    logps_ = outputs_ - torch.logsumexp(outputs_, dim=-1, keepdim=True)
                    loss_ce_ = -torch.gather(logps_, dim=-1, index=labels_.unsqueeze(-1)).mean(dim=[-1, -2])
                    # after gather and before mean, the shape is (exp_bs_, n_samps_, 1)
                    loss_firth_ = firth_regularizer(outputs_, prior=prior, logp=logps_)
                    loss_ent_ = entropy_regularizer(outputs_, logp=logps_)
                    loss_l2_ = l2_regularizer(params)
                    loss_total_ = loss_ce_ + firth_coeff_ * loss_firth_ + l2_coeff_ * loss_l2_ + ent_coeff_ * loss_ent_
                    loss_dict = dict(total=loss_total_, ce=loss_ce_, l2=loss_l2_, firth=loss_firth_, ent=loss_ent_)
                    return loss_dict

            loss_calc = LossCalculator(firth_coeff, l2_coeff, firth_prior, ent_coeff)

            logger(f'8) Sending the Support/Query Embeddings/Labels to {device_name}')
            embedding = embedding.to(device)
            labels = labels.to(device)

            loss_hist = []
            stat_ctr, print_col_ctr = 0, 0

            st_time = time.time()
            logger(f'9) Starting the Optimization')

            for epoch in range(n_epochs):
                running_losses = defaultdict(lambda: 0.0)

                if shuffle_mb:
                    shuff_idxs = list(chain.from_iterable(rlib_rng.sample(range(exp_idx*n_total_supp,
                                                                                (exp_idx+1)*n_total_supp),
                                                                          n_total_supp)
                                                          for exp_idx, rlib_rng in enumerate(rlib_rngs)))
                    with torch.no_grad():
                        lbls_t = labels.reshape(exp_bs * n_total_supp)[shuff_idxs].reshape(exp_bs, n_total_supp)
                        emb_t_2d = embedding.reshape(exp_bs * n_total_supp, feat_dim)[shuff_idxs]
                        emb_t = emb_t_2d.reshape(exp_bs, n_total_supp, feat_dim)
                else:
                    lbls_t, emb_t = labels, embedding

                assert emb_t.shape[1] == lbls_t.shape[1] == n_total_supp

                mb_idx = 0
                running_ctr = 0
                nan_break = False
                epoch_done = False
                cg_break = False
                cg_alpha = 0.
                while not epoch_done:
                    net_backup.load_state_dict(net.state_dict())
                    if optim_type in ('adam', 'sgd'):
                        # Even if we do not have batch_size items left,
                        # python's slicing will take whatever's left.
                        mb_inputs = emb_t[:, mb_idx: mb_idx + batch_size, :]
                        mb_labels = lbls_t[:, mb_idx: mb_idx + batch_size]
                        mb_labels = mb_labels.reshape(exp_bs, -1)

                        # zero the parameter gradients
                        optimizer.zero_grad()

                        # forward + backward + optimize
                        mb_outputs = net(mb_inputs)
                        loss_pack = loss_calc.loss_pack(mb_outputs, mb_labels, net.parameters())
                        loss_total = loss_pack['total']
                        loss_ce = loss_pack['ce']
                        loss_firth = loss_pack['firth']
                        loss_l2 = loss_pack['l2']
                        loss_ent = loss_pack['ent']

                        loss_total = loss_total * float(mb_labels.shape[1]) / batch_size

                        loss_total.sum().backward()
                        optimizer.step()
                    elif optim_type == 'lbfgs':
                        assert len('exp_bs is not ready') == 0, 'exp_bs not implemented yet'
                        mb_inputs = emb_t
                        mb_labels = lbls_t
                        mb_labels = mb_labels.reshape(-1, 1)

                        def closure(backward=True):
                            if backward:
                                optimizer.zero_grad()
                            outputs_ = net(mb_inputs)
                            loss_pack_ = loss_calc.loss_pack(outputs_, mb_labels, net.parameters())
                            loss_total_ = loss_pack_['total']
                            if backward:
                                loss_total_.backward()
                                return loss_total_
                            else:
                                return loss_pack_

                        optimizer.step(closure)

                        with torch.no_grad():
                            loss_pack = closure(backward=False)
                            loss_total = loss_pack['total']
                            loss_ce = loss_pack['ce']
                            loss_firth = loss_pack['firth']
                            loss_l2 = loss_pack['l2']
                        if torch.isnan(loss_total):
                            net.load_state_dict(net_backup.state_dict())
                            del optimizer
                            optimizer = lbfgs_optim_maker(net.parameters())
                    elif optim_type == 'cg':
                        assert len('exp_bs is not ready') == 0, 'exp_bs not implemented yet'
                        mb_inputs = emb_t.to(device_name)
                        mb_labels = lbls_t.to(device_name)
                        mb_labels = mb_labels.reshape(-1, 1)

                        mb_outputs = net(mb_inputs)
                        loss_total = loss_calc.loss_pack(mb_outputs, mb_labels, net.parameters())['total']
                        params_tuple = tuple(p for name, p in sorted(net.named_parameters()))
                        grad_output = torch.autograd.grad(outputs=loss_total, inputs=params_tuple,
                                                          grad_outputs=None, retain_graph=None,
                                                          create_graph=False, only_inputs=True,
                                                          allow_unused=True)

                        grad_flat = _flatten_dense_tensors(grad_output)
                        params_flat = _flatten_dense_tensors(params_tuple)

                        def vhp_loss_func(params_flat_):
                            params = _unflatten_dense_tensors(params_flat_, grad_output)
                            outputs_ = mb_inputs @ params[1].t() + params[0]
                            loss_total_ = loss_calc.loss_pack(outputs_, mb_labels, params)['total']
                            return loss_total_

                        def hvp(v, out=None):
                            vv = torch.autograd.functional.vhp(vhp_loss_func, inputs=params_flat,
                                                               v=v, create_graph=False, strict=False)[1]
                            return vv.t() + cg_damping * v

                        stop_criterion = ('iter', 10)  # 'eps', 1e-08
                        cg_damping = 1e-10
                        cg_output = conjugate_gradient(hvp, b=grad_flat, x=None,
                                                       stop_criterion=stop_criterion,
                                                       verbose=False)
                        H_inv_g_flat, xtHx, rs_init, rs_new = cg_output
                        neg_H_inv_g_flat = -H_inv_g_flat
                        neg_H_inv_g = _unflatten_dense_tensors(neg_H_inv_g_flat, grad_output)

                        # Although CG gives us the gradient-update dot product,
                        # it may not be nemerically accurate. Thus, we compute it again.
                        with torch.no_grad():
                            neg_gt_H_inv_g = torch.sum(grad_flat * neg_H_inv_g_flat)

                        assert cg_alpha == 0.
                        if pring_cg_log:
                            print('\n---- Begin: CG Log -----', flush=True)
                            print('  CG: alpha=%.3f --> loss=%.3f' % (0., loss_total.item()), flush=True)

                        for alpha in (2. ** -np.arange(10)):
                            net = update_net(net, net_backup, alpha, neg_H_inv_g)
                            with torch.no_grad():
                                mb_outputs_alpha = net(mb_inputs)
                                loss_total_alpha = loss_calc.loss_pack(mb_outputs_alpha, mb_labels,
                                                                       net.parameters())['total']
                                max_good_loss = loss_total + cg_eta * cg_alpha * neg_gt_H_inv_g
                                is_good_alpha = (loss_total_alpha < max_good_loss)
                                l_str = '  CG: alpha=%.3f --> loss=%.3f' % (alpha, loss_total_alpha.item())

                            if pring_cg_log:
                                print(l_str, flush=True)
                            if is_good_alpha:
                                cg_alpha = alpha
                                break
                        if pring_cg_log:
                            print('---- End  : CG Log -----', flush=True)

                        if cg_alpha == 0.:
                            cg_break = True
                            if pring_cg_log:
                                print('\nWarning: no appropriate alpha was found in the CG line search. '
                                      'Terminating the optimization\n', flush=True)

                        net = update_net(net, net_backup, cg_alpha, neg_H_inv_g)
                        with torch.no_grad():
                            mb_outputs = net(mb_inputs)
                            loss_pack = loss_calc.loss_pack(mb_outputs, mb_labels, net.parameters())
                            loss_total = loss_pack['total']
                            loss_ce = loss_pack['ce']
                            loss_firth = loss_pack['firth']
                            loss_l2 = loss_pack['l2']
                            loss_ent = loss_pack['ent']

                    # print statistics
                    with torch.no_grad():
                        assert all(loss_tensor.shape == (exp_bs,) for loss_name, loss_tensor in loss_pack.items())
                        running_losses['total'] += loss_total.mean().detach().cpu().item()
                        running_losses['ce'] += loss_ce.mean().detach().cpu().item()
                        running_losses['firth'] += loss_firth.mean().detach().cpu().item()
                        running_losses['l2'] += loss_l2.mean().detach().cpu().item()
                        running_losses['ent'] += loss_ent.mean().detach().cpu().item()
                        loss_hist.append([loss_total.tolist(), loss_firth.tolist(), loss_ce.tolist(), loss_l2.tolist()])

                    row_cols = 3
                    if ((stat_ctr+1) % stat_freq == 0) or (optim_type in ('lbfgs', 'cg')):
                        print_msg = ('[%03d]: T:%.3f CE:%.3f F:%.3f L2:%.3f E:%.3f' %
                                     (epoch + 1, running_losses['total'] / (running_ctr+1),
                                      running_losses['ce'] / (running_ctr+1),
                                      running_losses['firth'] / (running_ctr+1),
                                      running_losses['l2'] / (running_ctr+1),
                                      running_losses['ent'] / (running_ctr+1)))
                        if optim_type == 'cg':
                            print_msg = print_msg + f' CG_a:%.3f' % cg_alpha
                            row_cols = 3
                        print(print_msg, end=',   ' if (print_col_ctr+1) % row_cols != 0 else ',\n',
                              flush=True)
                        print_col_ctr += 1
                    stat_ctr += 1
                    mb_idx = mb_idx + batch_size

                    if optim_type in ('lbfgs', 'cg'):
                        epoch_done = True
                    else:
                        epoch_done = not(mb_idx < n_total_supp)

                    running_ctr += 1
                    nan_break = np.any([torch.isnan(p).any().cpu().numpy() for p in net.parameters()])
                    assert not nan_break
                    if nan_break:
                        net.load_state_dict(net_backup.state_dict())
                        break
                    if cg_break:
                        break

                if nan_break:
                    break
                if cg_break:
                    break

            if optim_type in ('lbfgs',):
                del optimizer
            spent_time = time.time() - st_time
            print('\n\n  --> Optimization Time for All Epochs Combined: %.2f Sec.' % spent_time, flush=True)

            # Evaluation
            logger(f'10) Evaluating Test Accuracy')
            assert labels_heldout.shape == (exp_bs, n_total_query)
            all_exp_query_idxs_np = np.array(all_exp_query_idxs).reshape(exp_bs, n_total_query)
            labels_heldout = labels_heldout.to(device)
            net = net.to(device)
            weight_magnitude = minkowski_norm_batched(net, norm=2)

            # Evaluating Test Accuracy
            with torch.no_grad():
                mb_idx = 0
                outputs_heldout_list = []
                while mb_idx < n_total_query:
                    curr_idxs = all_exp_query_idxs_np[:, mb_idx: mb_idx + batch_size].reshape(-1)
                    mb_inputs = all_embedding[curr_idxs].reshape(exp_bs, -1, feat_dim).to(device)
                    outputs_heldout_list.append(net(mb_inputs))
                    mb_idx += batch_size

                outputs_heldout = torch.cat(outputs_heldout_list, dim=1).reshape(exp_bs, n_total_query, real_n_ways)
                assert outputs_heldout.shape == (exp_bs, n_total_query, real_n_ways)
                _, predicted_heldout = torch.max(outputs_heldout, dim=-1)
                assert predicted_heldout.shape == (exp_bs, n_total_query)
                if store_predictions:
                    predicted_heldout_np = predicted_heldout.detach().cpu().numpy()
                    labels_heldout_np = labels_heldout.detach().cpu().numpy()
                c = (predicted_heldout == labels_heldout)
                heldout_acc = c.double().mean(dim=-1)
                heldout_acc_all_exps = heldout_acc.mean().item()
                heldout_acc_se_all_exps = 2 * heldout_acc.std().item() / np.sqrt(heldout_acc.numel())
                heldout_acc = heldout_acc.tolist()

                # Evaluating Test Loss
                heldout_loss_pack = loss_calc.loss_pack(outputs_heldout, labels_heldout, net.parameters())
                with torch.no_grad():
                    assert all(loss_tensor.shape == (exp_bs,) for loss_name, loss_tensor in heldout_loss_pack.items())
                    heldout_loss_total = heldout_loss_pack['total'].detach().cpu().tolist()
                    heldout_loss_ce = heldout_loss_pack['ce'].detach().cpu().tolist()
                    heldout_loss_ent = heldout_loss_pack['ent'].detach().cpu().tolist()
                    heldout_loss_firth = heldout_loss_pack['firth'].detach().cpu().tolist()
                    heldout_loss_reg = heldout_loss_pack['l2'].detach().cpu().tolist()

            logger(f'11) Evaluating Train Accuracy')
            # Evaluating Train Accuracy
            with torch.no_grad():
                outputs = net(embedding)
                assert outputs.shape == (exp_bs, n_total_supp, real_n_ways)
                _, predicted = torch.max(outputs, dim=-1)
                c = (predicted == labels)
                acc = c.double().mean(dim=-1)
                acc_all_exps = acc.mean().item()

            print_msg = '  --> Final Train Accuracy:   %.3f' % (acc_all_exps*100) + '%' + '\n'
            hacc_pair = (heldout_acc_all_exps * 100, heldout_acc_se_all_exps * 100)
            print_msg = print_msg + '  --> Final  Test Accuracy:   %.3f +/- %.3f' % hacc_pair + '%'
            print(print_msg, flush=True)
            print('------------', flush=True)

            loss_hist_np = np.array(loss_hist)
            assert loss_hist_np.shape[-1] == exp_bs
            assert loss_hist_np.ndim == 3
            loss_hist_np_t = np.transpose(loss_hist_np, axes=(2, 0, 1))
            assert loss_hist_np_t.shape[0] == exp_bs

            logger(f'12) Storing the Results')
            for exp_idx in range(exp_bs):
                # Things to store in a csv file
                row_dict = OrderedDict(firth_coeff=firth_coeff,
                                       l2_coeff=l2_coeff,
                                       ent_coeff=ent_coeff,
                                       n_shots=n_shots,
                                       param_l2=weight_magnitude[exp_idx],
                                       do_imbalanced=do_imbalanced,
                                       permute_labels=permute_labels,
                                       rng_seed=rng_seeds[exp_idx],
                                       firth_prior_type=firth_prior_type,
                                       clf_type=clf_type,
                                       backbone_arch=backbone_arch,
                                       data_type=data_type,
                                       learning_rate=learning_rate,
                                       batch_size=batch_size,
                                       n_epochs=n_epochs,
                                       n_ways=n_ways,
                                       optim_type=optim_type,
                                       n_query=n_query,
                                       hostname=hostname,
                                       shuffle_mb=shuffle_mb,
                                       fix_query_set=fix_query_set,
                                       dataset_name=dataset_name,
                                       commit_hash=commit_hash,
                                       test_acc=heldout_acc[exp_idx],
                                       total_loss_test=heldout_loss_total[exp_idx],
                                       ce_loss_test=heldout_loss_ce[exp_idx],
                                       regul_loss_test=heldout_loss_reg[exp_idx],
                                       firth_loss_test=heldout_loss_firth[exp_idx],
                                       ent_loss_test=heldout_loss_ent[exp_idx])
                rand_postfix = dict_hash(row_dict)[:16]
                pt_filename = f'{config_id}_{rand_postfix}.pt'
                row_dict['pt_filename'] = pt_filename
                if store_predictions:
                    assert real_n_ways <= 255
                    row_dict['query_predictions'] = predicted_heldout_np[exp_idx].astype(np.uint8)
                    row_dict['query_labels'] = labels_heldout_np[exp_idx].astype(np.uint8)

                if store_results:
                    extension = 'h5' if store_predictions else 'csv'
                    csv_path = f'{results_dir}/{config_id}.{extension}'
                    data_writer.add(row_dict, csv_path)

                # Storing the trained classifier weights
                if store_clfweights:
                    tar_path = f'{clfweights_dir}/{config_id}.tar'
                    exp_net_sdict = OrderedDict()
                    with torch.no_grad():
                        for name, param in net.state_dict().items():
                            exp_net_sdict[name] = param[exp_idx].detach().cpu()
                    save_dict = {**row_dict,
                                 'net': exp_net_sdict,
                                 'training_loss_total': loss_hist_np_t[exp_idx, :, 0],
                                 'training_loss_fim': loss_hist_np_t[exp_idx, :, 1],
                                 'training_loss_cross': loss_hist_np_t[exp_idx, :, 2],
                                 'training_loss_l2reg': loss_hist_np_t[exp_idx, :, 3]}
                    file_like_obj = io.BytesIO()
                    torch.save(save_dict, file_like_obj)
                    file_like_obj.seek(0)
                    append_to_tar(tar_path, pt_filename, file_like_obj)

            logger(f'13) Finished Storing the Results!')

    if store_results:
        # We need to make a final dump before exiting to make sure all data is stored
        data_writer.dump()

    torch.set_num_threads(untouched_torch_thread)


#  %% Loading Configs and Running the Function
if __name__ == '__main__':
    use_argparse = True
    if use_argparse:
        import argparse
        my_parser = argparse.ArgumentParser()
        my_parser.add_argument('--configid', action='store', type=str, required=True)
        my_parser.add_argument('--device', action='store', type=str, required=True)
        args = my_parser.parse_args()
        args_configid = args.configid
        args_device_name = args.device
    else:
        args_configid = 'configs/myconf/myconf.json'
        args_device_name = 'cuda:0'

    PROJPATH = os.getcwd()
    if '/' in args_configid:
        args_configid_split = args_configid.split('/')
        my_config_id = args_configid_split[-1]
        config_tree = '/'.join(args_configid_split[:-1])
    else:
        my_config_id = args_configid
        config_tree = ''

    cfg_dir = f'{PROJPATH}/configs'
    os.makedirs(cfg_dir, exist_ok=True)

    cfg_path = f'{PROJPATH}/configs/{config_tree}/{my_config_id}.json'
    print(f'Reading Configuration from {cfg_path}', flush=True)

    with open(cfg_path) as f:
        proced_config_dict = json.load(f)

    proced_config_dict['config_id'] = my_config_id
    proced_config_dict['results_dir'] = f'{PROJPATH}/results/{config_tree}'
    proced_config_dict['clfweights_dir'] = f'{PROJPATH}/storage/{config_tree}'
    proced_config_dict['data_dir'] = f'{PROJPATH}/datasets'
    proced_config_dict['features_dir'] = f'{PROJPATH}/features'
    proced_config_dict['backbones_dir'] = f'{PROJPATH}/backbones'
    proced_config_dict['device_name'] = args_device_name

    main(proced_config_dict)
