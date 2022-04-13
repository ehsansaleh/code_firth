import os
import io
import json
import time
import hashlib
import tarfile
import subprocess
import h5py
from datetime import datetime
from collections import OrderedDict
from typing import Dict, Any
import torch
import numpy as np
import pandas as pd


def dict_hash(dictionary: Dict[str, Any]) -> str:
    """MD5 hash of a dictionary."""
    dhash = hashlib.md5()
    # We need to sort arguments so {'a': 1, 'b': 2} is
    # the same as {'b': 2, 'a': 1}
    encoded = json.dumps(dictionary, sort_keys=True).encode()
    dhash.update(encoded)
    return dhash.hexdigest()


class DataWriter:
    def __init__(self, dump_period=10):
        self.dump_period = dump_period
        self.file_path = None
        self.data = None

    @property
    def data_len(self):
        dlen = 0
        if self.data is not None:
            dlen = len(list(self.data.values())[0])
        return dlen

    def set_path(self, file_path):
        if file_path is not None:
            if file_path != self.file_path:
                self.dump()
        self.file_path = file_path

    @property
    def file_ext(self):
        assert self.file_path is not None
        if self.file_path.endswith('.csv'):
            return 'csv'
        elif self.file_path.endswith('.h5'):
            return 'h5'
        else:
            raise ValueError(f'Unknown extension for {self.file_path}')

    def add(self, row_dict, file_path):
        self.set_path(file_path)
        assert isinstance(row_dict, OrderedDict)
        if self.data is None:
            self.data = OrderedDict()
        else:
            msg_assert = 'input keys and my columns are different:\n'
            msg_assert = msg_assert + f'  input keys: {set(row_dict.keys())}\n'
            msg_assert = msg_assert + f'  my columns: {set(self.data.keys())}\n'
            assert list(self.data.keys()) == list(row_dict.keys()), msg_assert

        # Checking if any of the numpy arrays have changed shape
        for i, (key, val) in enumerate(row_dict.items()):
            if key not in self.data:
                self.data[key] = []
            if (self.file_ext == 'h5') and self.is_attachment(val):
                if len(self.data[key]) > 0:
                    if val.shape != self.data[key][0].shape:
                        self.dump()

        for i, (key, val) in enumerate(row_dict.items()):
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(val)

        if (self.data_len % self.dump_period == 0) and (self.data_len > 0):
            self.dump()

    def is_attachment(self, var):
        use_np_protocol = isinstance(var, np.ndarray)
        if use_np_protocol:
            use_np_protocol = var.size > 1
        return use_np_protocol

    def dump(self):
        if self.data_len == 0:
            return None

        assert self.file_path is not None

        if self.file_ext == 'csv':
            data_df = pd.DataFrame(self.data)
            columns = list(self.data.keys())
            # Appending the latest row to file_path
            if not os.path.exists(self.file_path):
                data_df.to_csv(self.file_path, mode='w', header=True, index=False, columns=columns)
            else:
                # First, check if we have the same columns
                old_cols = pd.read_csv(self.file_path, nrows=1).columns.tolist()
                old_cols_set = set(old_cols)
                my_cols_set = set(columns)
                msg_assert = 'file columns and my columns are different:\n'
                msg_assert = msg_assert + f'  file cols: {old_cols_set}\n'
                msg_assert = msg_assert + f'  my columns: {my_cols_set}\n'
                assert old_cols_set == my_cols_set, msg_assert
                data_df.to_csv(self.file_path, mode='a', header=False, index=False, columns=old_cols)
        elif self.file_ext == 'h5':
            np_data = {}
            pd_data = {}
            for key, valslist in self.data.items():
                if self.is_attachment(valslist[0]):
                    np_data[key] = np.stack(valslist, axis=0)
                else:
                    pd_data[key] = valslist

            # Writing the main table
            data_df = pd.DataFrame(pd_data)
            nextpart = 0
            if os.path.exists(self.file_path):
                with pd.HDFStore(self.file_path) as hdf:
                    nextpart = len(tuple(x for x in hdf.keys() if x.startswith('/main/part')))
                    assert nextpart not in hdf.keys()
            data_df.to_hdf(self.file_path, key=f'/main/part{nextpart}',
                           mode='a', index=False, append=False)

            # Writing the numpy attachments
            hdf_obj = h5py.File(self.file_path, mode='a', driver='core')
            for key, np_arr in np_data.items():
                hdf_obj.create_dataset(f'/attachments/part{nextpart}/{key}',
                                       shape=np_arr.shape, dtype=np_arr.dtype, data=np_arr,
                                       compression="gzip", compression_opts=9)
            hdf_obj.close()
        else:
            raise ValueError(f'file extension not implemented for {self.file_path}.')

        for key in self.data.keys():
            self.data[key] = []


def append_to_tar(tar_path, file_name, file_like_obj):
    archive = tarfile.open(tar_path, "a")
    info = tarfile.TarInfo(name=file_name)
    file_like_obj.seek(0, io.SEEK_END)
    info.size = file_like_obj.tell()
    info.mtime = time.time()
    file_like_obj.seek(0, io.SEEK_SET)
    archive.addfile(info, file_like_obj)
    archive.close()
    file_like_obj.close()


def get_git_commit():
    try:
        commit_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode()
    except Exception:
        commit_hash = 'no git repo found'
    return commit_hash


def check_cfg_dict(config_dict):
    msg_cfg_id = 'I need a config_id, mainly for naming the files that I store. \n' \
                 'Example: \n' \
                 '  "config_id" : "test"'
    config_id = config_dict.get('config_id', 'test')
    if 'config_id' not in config_dict:
        print('-'*20 + '\nConfig Error: ' + msg_cfg_id + '\n' + '-'*20, flush=True)

    store_results = config_dict.get('store_results', True)
    msg_res_dir = 'You asked me to store the csv results by leaving store_results=True.\n' + \
                  'Either \n' + \
                  '  (1) disable this by setting "store_results" : False in the config_dict, or\n' + \
                  '  (2) Specify "results_dir" in the config_dict.\n\n' + \
                  'Example 1: \n' + \
                  f'  "store_results" : True,\n' + \
                  f'  "results_dir" : "./results/{config_id},"\n\n' + \
                  'Example 2: \n' + \
                  f'  "store_results" : False,'
    if store_results and ('results_dir' not in config_dict):
        print('-'*20 + '\nConfig Error: ' + msg_res_dir + '\n' + '-'*20, flush=True)

    store_trained_models = config_dict.get('store_clfweights', False)
    msg_sto_dir = 'You asked me to store the trained models by leaving store_clfweights=True.\n' + \
                  'Either \n' + \
                  '  (1) disable this by setting "store_clfweights" : False in the config_dict, or\n' + \
                  '  (2) Specify "clfweights_dir" in the config_dict.\n\n' + \
                  'Example 1: \n' + \
                  f'  "store_clfweights" : True,\n' + \
                  f'  "clfweights_dir" : "./storage/{config_id},"\n\n' + \
                  'Example 2: \n' + \
                  f'  "store_clfweights" : False,'
    if store_trained_models and ('clfweights_dir' not in config_dict):
        print('-'*20 + '\nConfig Error: ' + msg_sto_dir + '\n' + '-'*20, flush=True)

    msg_data_dir = 'I need the features_dir to be specified in the config_dict.\n' \
                   'Example: \n' \
                   '  "features_dir": "./features/"'
    if 'features_dir' not in config_dict:
        print('-'*20 + '\nConfig Error: ' + msg_data_dir + '\n' + '-'*20, flush=True)

    assert 'config_id' in config_dict
    if store_results:
        assert 'results_dir' in config_dict
    if store_trained_models:
        assert 'clfweights_dir' in config_dict
    assert 'features_dir' in config_dict


def get_cuda_memory_per_seed(backbone_arch, n_ways, n_shots, tch_dtype, n_query,
                             nshots_to_clsfreqs, clf_type, optim_type, batch_size):
    n_featdim = dict(resnet10=512, resnet18=512, resnet34=512,
                     resnet50=2048, resnet101=2048,
                     densenet121=1024, mobilenet84=1024)[backbone_arch]
    bytes_per_el = torch.tensor(1, dtype=tch_dtype).element_size()

    if nshots_to_clsfreqs is not None:
        assert str(n_shots) in nshots_to_clsfreqs
        n_total_supp = sum(nshots_to_clsfreqs[str(n_shots)])
    else:
        n_total_supp = n_shots * n_ways
    n_total_query = n_query * n_ways

    expbs_ = 1
    embedding_numel = expbs_ * n_total_supp * n_featdim
    # Since we're doing the query estimation batch-wise, we don't have to worry about this
    # embedding_query_numel = expbs_ * n_total_query * n_featdim
    embedding_query_numel = expbs_ * batch_size * n_featdim
    labels_numel = expbs_ * n_total_supp
    labels_query_numel = expbs_ * n_total_query

    if clf_type == 'lin':
        net_numel = expbs_ * (n_featdim + 1) * n_ways
    elif clf_type == 'mlp':
        net_numel = expbs_ * ((n_featdim + 1) * 100 + 101 * 50 + 51 * n_ways)
    else:
        raise ValueError(f'clf_type {clf_type} number of parameters not implemented')

    if optim_type == 'sgd':
        net_copies = 3  # two for net and net_backup, and one for the gradient!
    elif optim_type == 'adam':
        net_copies = 5  # 3 for sgd, and two for the adaptive moments!
    else:
        raise ValueError(f'optim_type {optim_type} number of parameters not implemented')

    total_numel = embedding_numel + embedding_query_numel + labels_numel
    total_numel += labels_query_numel + net_copies * net_numel
    total_bytes_per_seed = total_numel * bytes_per_el
    return total_bytes_per_seed


def logger(*args, **kwargs):
    now = datetime.now()
    dt_string = now.strftime("%H:%M:%S")
    print(f'[{dt_string}] ', end='')
    print(*args, **kwargs)
