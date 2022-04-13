import pandas as pd
from itertools import product
from pathlib import Path
from utils.summary import gen_table
from utils.cfg import y_list as proj_y_list
from utils.cfg import smry_tbls_dir, paper_tbls_dir, reg_sources
from utils.cfg import table_sep_cols, row_tree, col_tree
from utils.cfg import naming_scheme, scale_percent, prop_cols

y_list = proj_y_list
no_ci = False
for reg_src in reg_sources:
    tables_dict = dict()
    for y_col, y_col_ci, y_name in y_list:
        my_outdir = f'{paper_tbls_dir}'
        Path(my_outdir).mkdir(parents=True, exist_ok=True)
        # Generating the Summarized DataFrame
        summary_df = pd.read_csv(f'{smry_tbls_dir}/{reg_src}2test.csv')
        summary_df = summary_df[summary_df['experiment'] == 'firth_1layer']
        # Creating the latex tables
        tables_dict[y_col] = dict()

        if len(table_sep_cols):
            grps = summary_df.groupby(table_sep_cols)
        else:
            grps = [('', summary_df)]

        if no_ci:
            def str_maker(flt_mean, flt_ci=None):
                return f'%0.2f' % (scale_percent*flt_mean) + '%'
        else:
            def str_maker(flt_mean, flt_ci=None):
                if flt_ci is not None:
                    pm = 'PM'
                    out_str = f'%0.2f {pm} %.2f' % (scale_percent*flt_mean, scale_percent*flt_ci)
                else:
                    out_str = f'%0.2f' % (scale_percent*flt_mean)
                return out_str + '%'

        for tbl_id, summary_table in grps:
            tables_dict.setdefault(tbl_id, dict())
            if naming_scheme == 's2m2rf':
                assert 'n_shot' in summary_table.columns, 'Are you sure you are using the right naming scheme?'
                summary_table = summary_table.sort_values(by=['test_n_way'])
                shots_col = 'n_shot'
                ways_col = 'test_n_way'
            elif naming_scheme == 'firth':
                assert 'n_shots' in summary_table.columns, 'Are you sure you are using the right naming scheme?'
                summary_table = summary_table.sort_values(by=['n_shots'])
                shots_col = 'n_shots'
                ways_col = 'n_ways'
            else:
                raise Exception('Not Implemented Error')

            summary_table[shots_col] = [f'{x}-shot' + ('s' if x > 1 else '')[:0] for x in summary_table[shots_col]]
            if ways_col is not None:
                summary_table[ways_col] = [f'{x}-way' for x in summary_table[ways_col]]

            tbl = gen_table(summary_table, row_tree, col_tree, y_col, y_col_ci=y_col_ci, str_maker=str_maker)
            tbl = tbl.rename(columns={"16-way": y_name})
            tables_dict[tbl_id][y_col] = tbl
    
    
    for tbl_id, summary_table in grps:
        print(f'  * Creating the latex/csv tables for {tbl_id}.')
        tbl = pd.concat(list(tables_dict[tbl_id].values()), axis=1)
        ltx_tbl_str = tbl.to_latex(multicolumn=True, escape=True, multicolumn_format='c|', column_format='|c'*4 + '|')
        ltx_tbl_str = ltx_tbl_str.replace('PM', '$\\pm$')
        ltx_tbl_str = ltx_tbl_str.replace('\\\n', 'midrule\n')

        if isinstance(tbl_id, (list, tuple)):
            list_tbl_id = tbl_id
        else:
            list_tbl_id = [tbl_id]

        tbl_name = '_'.join([str(x) for x in list_tbl_id])
        if reg_src != 'val':
            tbl_name = tbl_name + f'_{reg_src}2test'
        if no_ci:
            tbl_name = tbl_name + '_noci'
        tbl_name = f'{tbl_name}'
        
        ltx_save_path = f'{my_outdir}/{tbl_name}.tex'
        csv_save_path = f'{my_outdir}/{tbl_name}.csv'
        with open(ltx_save_path, 'w') as f_ptr:
            f_ptr.write(ltx_tbl_str)
        print(f'  *   --> Latex table saved at {ltx_save_path}.')
        tbl.to_csv(csv_save_path)
        print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-'*80)

        
# Entropy regularization vs. Firth bias reduction table
print(f'  * Creating the latex/csv tables for entropy vs. firth comparison on 3-layer classifiers.')
summary_df = pd.read_csv(f'{smry_tbls_dir}/{reg_sources[0]}2test.csv')
exps = ['entropy', 'firth_3layer']
df_entfirth = summary_df[summary_df['experiment'].isin(exps)]
match_cols = list(set(prop_cols).difference({'experiment'}))
y_col = 'delta_test_acc'
y_cols = [y_col, f'{y_col}_ci']

unncsray_cols = set(df_entfirth.columns)
unncsray_cols = unncsray_cols.difference(set(match_cols + y_cols + ['experiment']))
unncsray_cols = list(unncsray_cols)
df_entfirth = df_entfirth.drop(columns=unncsray_cols)

row_list = []
for colvals, df_grp in df_entfirth.groupby(match_cols):
    if len(df_grp['experiment']) != len(exps):
        continue
    if set(df_grp['experiment']) != set(exps):
        continue
    row_dict = {col:val for col, val in zip(match_cols, colvals)}
    for exp, exp_df in df_grp.groupby('experiment'):
        for col in set(exp_df.columns).difference(set(match_cols + ['experiment'])):
            row_dict[f'{exp}/{col}'] = exp_df[col].tolist()[0]
    row_list.append(row_dict)
poced_df = pd.DataFrame(row_list)

static_cols = [col for col in match_cols 
               if len(poced_df[col].unique()) == 1]
non_static_cols = list(set(poced_df.columns).difference(set(static_cols)))
small_df = poced_df.drop(columns=static_cols)
for exp in exps:
    mean_colname, ci_colname = f'{exp}/{y_col}', f'{exp}/{y_col}_ci'
    means_list = small_df[mean_colname].tolist()
    cis_list = small_df[ci_colname].tolist()
    small_df[exp] = [f'{m*scale_percent:.2f} +/- {e*scale_percent:.2f}' + '%'
                     for m, e in zip(means_list, cis_list)]
    small_df = small_df.drop(columns=[mean_colname, ci_colname])

col_renaming = {'n_shots': 'Number of Shots',
                'entropy': 'Confidence Penalty Improvements',
                'firth_3layer': 'Firth Improvements'}
tbl = small_df.rename(columns=col_renaming)

tbl_name = 'entropy_vs_firth'
ltx_tbl_str = tbl.to_latex(multicolumn=True, escape=True, index=False,
                           multicolumn_format='c|', column_format='|c'*3+'|')
ltx_save_path = f'{my_outdir}/{tbl_name}.tex'
csv_save_path = f'{my_outdir}/{tbl_name}.csv'
with open(ltx_save_path, 'w') as f_ptr:
    f_ptr.write(ltx_tbl_str)
print(f'  *   --> Latex table saved at {ltx_save_path}.')
tbl.to_csv(csv_save_path)
print(f'  *   --> CSV table saved at {csv_save_path}.\n  ' + '-'*80)
