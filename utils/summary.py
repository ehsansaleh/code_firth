import os
import pandas as pd
import numpy as np
from itertools import product
from pandas.errors import EmptyDataError
from sklearn.metrics import precision_recall_fscore_support
from os.path import abspath, dirname
import h5py
import tables.atom
import pickle5
from utils.cfg import naming_scheme as proj_naming_scheme
from utils.cfg import main_acc as proj_main_acc
from utils.cfg import firth_reg_col, crn_cols, do_supplement_cols, scale_percent
from utils.cfg import prop_cols as proj_prop_cols

try:
    from mpi4py import MPI
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()
    use_mpi = True
except ImportError:
    mpi_rank, mpi_size, mpi_comm, use_mpi = 0, 1, None, False

tables.atom.pickle = pickle5


def add_missing_dflt_vals(df, def_dict):
    for colname, defval in def_dict.items():
        if colname in df.columns:
            df[colname].fillna(defval, inplace=True)
        else:
            df[colname] = defval
    return df


def default_str_maker(flt_mean, flt_ci=None):
    if flt_ci is not None:
        pm = 'PM'  # '+/-'
        out_str = f'%+0.2f {pm} %.2f' % (scale_percent*flt_mean, scale_percent*flt_ci)
    else:
        out_str = f'%+0.2f' % scale_percent*flt_mean
    return out_str + '%'


def gen_table(summary_df, row_tree, col_tree, y_col, y_col_ci=None, str_maker=None):
    if str_maker is None:
        str_maker = default_str_maker
    x_cols = row_tree + col_tree
    my_summary_df = summary_df.copy(deep=False)

    row_tree_uniques = [my_summary_df[col].unique().tolist() for col in row_tree]
    col_tree_uniques = [my_summary_df[col].unique().tolist() for col in col_tree]
    x_col_uniques = row_tree_uniques + col_tree_uniques
    np_ndarr = np.full(tuple(len(a) for a in x_col_uniques), np.nan)
    np_ndarr_raveled = np.empty_like(np_ndarr.reshape(-1), dtype=object)
    for i, x_tup in enumerate(product(*x_col_uniques)):
        df = my_summary_df
        for x_col, x_val in zip(x_cols, x_tup):
            df = df[df[x_col] == x_val]
        assert len(df) == 1, f'The combination {x_cols}={x_tup} has {len(df)} rows in it instead of 1: \n{df}'

        entry = df[y_col].values.item()
        if y_col_ci is not None:
            entry_ci = df[y_col_ci].values.item()
        else:
            entry_ci = None
        np_ndarr_raveled[i] = str_maker(entry, entry_ci)

    np_ndarr = np_ndarr_raveled.reshape(*np_ndarr.shape)

    nrows = np.prod(tuple(len(a) for a in row_tree_uniques))
    ncols = np.prod(tuple(len(a) for a in col_tree_uniques))
    out_df = pd.DataFrame(np_ndarr.reshape(nrows, ncols),
                          columns=pd.MultiIndex.from_product(col_tree_uniques),
                          index=pd.MultiIndex.from_product(row_tree_uniques))
    return out_df


# getting the files in each folder
def get_csvh5files(fldr_mini, results_dir):
    fldr_mini_files = []
    for fldr in fldr_mini:
        for file in os.listdir(f'{results_dir}/{fldr}'):
            if file.endswith(".csv") or file.endswith(".h5"):
                fldr_mini_files.append(f'{results_dir}/{fldr}/{file}')
    return fldr_mini_files


def read_csvh5(filename):
    if filename.endswith('.csv'):
        try:
            df = pd.read_csv(filename, sep=',')
        except EmptyDataError:
            if mpi_rank == 0:
                print(f'WARNING: {filename} seems to be empty. I will ignore it and move on', flush=True)
            df = None
    elif filename.endswith('.h5'):
        if mpi_rank == 0:
            print(f'    -> Reading {filename}', flush=True)

        with h5py.File(filename, mode='r') as hdf_obj:
            parts = sorted(hdf_obj['main'].keys(), key=lambda x: int(x.split('part')[-1]))

        with pd.HDFStore(filename, mode='r') as hdftbl:
            raw_dfs = [pd.read_hdf(hdftbl, key=f'/main/{part}') for part in parts]

        raw_np_dicts = []
        with h5py.File(filename, mode='r') as hdf_obj:
            for part in parts:
                raw_np_dicts.append({key: val[()] for key, val in
                                     hdf_obj['attachments'][part].items()})
        supplemented_df_list = []
        if mpi_rank == 0:
            print(f'    -> Supplementing {filename}', flush=True)
        for raw_df, raw_np_dict in zip(raw_dfs, raw_np_dicts):
            if do_supplement_cols:
                df_supplemented = supplement_columns(raw_df, raw_np_dict)
            else:
                df_supplemented = raw_df
            supplemented_df_list.append(df_supplemented)

        df = pd.concat(supplemented_df_list, axis=0, ignore_index=True)
    else:
        raise ValueError('extension not implemented.')

    return df


def vec2freq(np_vec):
    """
    This function replaces each entry in the input with its frequency in the entire array.
    Example:
      input  = np.array([1, 1, 2, 5, 4, 5, 5])
      output = np.array([2, 2, 1, 3, 1, 3, 3])
    """
    assert np_vec.ndim == 1
    # extract unique elements and the indices to reconstruct the array
    unq, idx = np.unique(np_vec, return_inverse=True)
    # calculate the weighted frequencies of these indices
    freqs_idx = np.bincount(idx, weights=None)
    # reconstruct the array of frequencies of the elements
    frequencies = freqs_idx[idx]

    return frequencies


def supplement_columns(df, attachments_dict=None):
    n_rows = df.shape[0]
    new_cols = [f'{a}_{b}_{c}' for (a, b, c) in product(['precision', 'recall', 'fscore'],
                                                        ['bal', 'imbal'],
                                                        ['micro', 'macro'])]
    new_cols = new_cols + ['acc_bal', 'acc_imbal']

    new_cols_dict = {col: np.full(n_rows, -1., dtype=np.float32) for col in new_cols}
    attachments_dict = attachments_dict or dict()
    is_data_avail = (('query_labels' in attachments_dict) and
                     ('query_predictions' in attachments_dict))

    if is_data_avail:

        y_true_all = attachments_dict['query_labels']
        y_pred_all = attachments_dict['query_predictions']
        shot_dict = {"8": np.array([2, 2, 2, 2, 4, 4, 4, 4, 8, 8, 8, 8, 16, 16, 16, 16]),
                     "15": np.array([1, 1, 5, 5, 9, 9, 13, 13, 17, 17, 21, 21, 25, 25, 29, 29])}

        for i, (index, row) in enumerate(df.iterrows()):
            do_imbalanced = row['do_imbalanced']
            y_true = y_true_all[i]
            y_pred = y_pred_all[i]

            if do_imbalanced:
                training_shots = shot_dict[str(row['n_shots'])]
            else:
                training_shots = np.ones(y_true.max()+1) * row['n_shots']

            balancing_weights = 1. / vec2freq(y_true)
            imbalanced_weights = balancing_weights * training_shots[y_true]

            balancing_weights = balancing_weights / balancing_weights.sum()
            imbalanced_weights = imbalanced_weights / imbalanced_weights.sum()

            acc_bal = np.sum((y_true == y_pred) * balancing_weights)
            acc_imbal = np.sum((y_true == y_pred) * imbalanced_weights)
            new_cols_dict[f'acc_bal'][i] = acc_bal
            new_cols_dict[f'acc_imbal'][i] = acc_imbal

            for averaging in ('micro', 'macro')[:0]:
                pkg = precision_recall_fscore_support(y_true, y_pred, sample_weight=balancing_weights,
                                                      average=averaging, zero_division=0)
                precision_bal, recall_bal, fscore_bal, support_bal = pkg

                pkg = precision_recall_fscore_support(y_true, y_pred, sample_weight=imbalanced_weights,
                                                      average=averaging, zero_division=0)
                precision_imbal, recall_imbal, fscore_imbal, support_imbal = pkg

                new_cols_dict[f'precision_bal_{averaging}'][i] = precision_bal
                new_cols_dict[f'recall_bal_{averaging}'][i] = recall_bal
                new_cols_dict[f'fscore_bal_{averaging}'][i] = fscore_bal
                new_cols_dict[f'precision_imbal_{averaging}'][i] = precision_imbal
                new_cols_dict[f'recall_imbal_{averaging}'][i] = recall_imbal
                new_cols_dict[f'fscore_imbal_{averaging}'][i] = fscore_imbal

    for key, arr in new_cols_dict.items():
        df[key] = arr

    return df


def beststat_summarizer(df, prop_cols=None, y_col=None, reg_column=None,
                        reg_sources=None, sync_rng=True, naming_scheme=None, cond=None):
    if reg_sources is None:
        reg_sources = ['val']
    if prop_cols is None:
        prop_cols = proj_prop_cols
    if naming_scheme is None:
        naming_scheme = proj_naming_scheme
    assert naming_scheme in ('s2m2rf', 'firth')
    if y_col is None:
        y_col = proj_main_acc

    if naming_scheme == 's2m2rf':
        assert 'split' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'seed' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'iter' in df.columns, 'Are you sure you are using the right naming scheme?'
    elif naming_scheme == 'firth':
        assert 'data_type' in df.columns, 'Are you sure you are using the right naming scheme?'
        assert 'rng_seed' in df.columns, 'Are you sure you are using the right naming scheme?'

    if reg_column is None:
        reg_column = firth_reg_col

    if naming_scheme == 's2m2rf':
        df_val = df[df['split'] == 'val']
        df_test = df[df['split'] == 'novel']
    elif naming_scheme == 'firth':
        df_val = df[df['data_type'] == 'val']
        df_test = df[df['data_type'] == 'novel']

    if not sync_rng:
        stats_val = df_val.groupby(reg_column)[y_col].agg(['mean', 'count', 'std'])
        df_val_mean = df_val.groupby([reg_column]).mean()
        df_val_mean[f'{y_col}_ci'] = 1. * stats_val['std'] / np.sqrt(stats_val['count'])
        df_val_mean.reset_index(inplace=True)

        stats_test = df_test.groupby([reg_column])[y_col].agg(['mean', 'count', 'std'])
        df_test_mean = df_test.groupby([reg_column]).mean()
        df_test_mean[f'{y_col}_ci'] = 1. * stats_test['std'] / np.sqrt(stats_test['count'])
        df_test_mean.reset_index(inplace=True)
    else:
        msg_ = 'A prop col may be missing in narrowing down the data. its not safe to keep going.'
        if len(df_val.groupby(prop_cols)) != 1:
            trash_dir = f'{dirname(dirname(abspath(__file__)))}/trash'
            os.makedirs(trash_dir, exist_ok=True)
            fp = f'{trash_dir}/df_val_dbg_rank{mpi_rank}.csv'
            df_val.to_csv(fp)
            msg_ += f'\nHere is the df_val to look at: {fp}'
            msg_ += f'\nHere is the conditioning: {cond}'
        assert len(df_val.groupby(prop_cols)) == 1, msg_
        out_df_list = []
        for name, df_grp in df_val.groupby(crn_cols):
            if not (df_grp[reg_column] == 0.).any():
                continue
            base_acc = df_grp.loc[df_grp[reg_column] == 0., y_col]
            if len(base_acc) > 1:
                a = df_grp.loc[df_grp[reg_column] == 0., :]
                assert np.allclose(base_acc.values, base_acc.values[0]), a

            base_acc_item = base_acc.values[0].item()
            df_grp[f'delta_{y_col}'] = df_grp[y_col] - base_acc_item
            out_df_list.append(df_grp)
        out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
        df_val_mean = out_df.groupby(prop_cols + [reg_column]).mean()
        for ci_col in [y_col, f'delta_{y_col}']:
            stats_val = out_df.groupby(prop_cols + [reg_column])[ci_col].agg(['mean', 'count', 'std'])
            df_val_mean[f'{ci_col}_ci'] = 1.96 * stats_val['std'] / np.sqrt(stats_val['count'])
        df_val_mean.reset_index(inplace=True)

        if len(df_test.groupby(prop_cols)) != 1:
            trash_dir = f'{dirname(dirname(abspath(__file__)))}/trash'
            os.makedirs(trash_dir, exist_ok=True)
            fp = f'{trash_dir}/df_test_dbg_rank{mpi_rank}.csv'
            df_test.to_csv(fp)
            msg_ += f'\nHere is the df_val to look at: {fp}'
            msg_ += f'\nHere is the conditioning: {cond}'
        assert len(df_test.groupby(prop_cols)) == 1, msg_
        out_df_list = []
        for name, df_grp in df_test.groupby(crn_cols):
            if not (df_grp[reg_column] == 0.).any():
                continue
            base_acc = df_grp.loc[df_grp[reg_column] == 0., y_col]
            if len(base_acc) > 1:
                assert np.allclose(base_acc.values, base_acc.values[0]), df_grp.loc[df_grp[reg_column] == 0., :]
            base_acc_item = base_acc.values[0].item()
            df_grp[f'delta_{y_col}'] = df_grp[y_col] - base_acc_item
            out_df_list.append(df_grp)
        out_df = pd.concat(out_df_list, axis=0, ignore_index=True)
        df_test_mean = out_df.groupby(prop_cols + [reg_column]).mean()
        for ci_col in [y_col, f'delta_{y_col}']:
            stats_val = out_df.groupby(prop_cols + [reg_column])[ci_col].agg(['mean', 'count', 'std'])
            df_test_mean[f'{ci_col}_ci'] = 1.96 * stats_val['std'] / np.sqrt(stats_val['count'])
        df_test_mean.reset_index(inplace=True)

    out_dict = {'val': df_val_mean, 'test': df_test_mean}
    for reg_src in reg_sources:
        if reg_src == 'val':
            best_val_firth_coeff = df_val_mean.loc[df_val_mean[y_col].idxmax()][reg_column]
        elif reg_src == 'test':
            best_val_firth_coeff = df_test_mean.loc[df_test_mean[y_col].idxmax()][reg_column]
        else:
            raise Exception(f'Unknown reg_source: {reg_src}')

        base_y = df_test_mean[df_test_mean[reg_column] == 0.][y_col].values.item()
        base_y_ci = df_test_mean[df_test_mean[reg_column] == 0.][f'{y_col}_ci'].values.item()
        row_df = df_test_mean[df_test_mean[reg_column] == best_val_firth_coeff]
        row_df[f'delta_{y_col}'] = row_df[y_col] - base_y
        if not sync_rng:
            row_df[f'delta_{y_col}_ci'] = 2 * 1.96 * row_df[f'{y_col}_ci']

        ##########
        base_test_row = df_test_mean[df_test_mean[reg_column] == 0.0]
        msg_ = f'found many rows in base_test_row. The conditioning is incomplete:\n{base_test_row}'
        assert len(base_test_row) == 1, msg_
        row_df[f'base_{y_col}'] = base_y
        row_df[f'base_{y_col}_ci'] = base_y_ci

        row_df.reset_index(drop=True)
        out_dict[f'{reg_src}2test'] = row_df

    return out_dict
