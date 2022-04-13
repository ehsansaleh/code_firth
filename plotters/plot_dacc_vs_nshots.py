import os
import pandas as pd
from pathlib import Path
from utils.cfg import smry_tbls_dir, expname_regcol
from utils.cfg import main_acc, paper_plots_dir, scale_percent
from utils.plot_utils import plot_stat_combined

pd.options.mode.chained_assignment = None

summ_file_dir = f'{smry_tbls_dir}/val2test.csv'
df_summ = pd.read_csv(summ_file_dir, sep=',')
ds_abrv_dict = {'miniimagenet': 'mini', 'tieredimagenet': 'tiered', 'cifarfs': 'cifarfs'}

def hue_sorting_key(backbone_arch):
    """
    This function is just for making sure we first go through 
    resnet10, resnet18, resnet34, resnet50, and then resnet101
    in order. This is especially important for ordering in figure
    legends, since by default, we have the resnet10, resnet101, 
    resnet18, resnet34, and resnet50 ordering due to backbone_arch 
    being a string.
    """
    out = backbone_arch
    if isinstance(backbone_arch, str):
        arch_no_str = backbone_arch.lower().split('resnet')[-1]
        if arch_no_str.isnumeric():
            # 'AAAAAAAA' is to make sure "resnet" entries are first!
            out = f'AAAAAAAA{int(arch_no_str):03d}'
    return out

exp_isrel_list = [('firth_1layer', False),
                  ('l2_1layer', False),
                  ('firth_3layer', False),
                  ('l2_3layer', False),
                  ('firth_tieredcifar', False),
                  ('l2_tieredcifar', False),
                  ('firth_5_10way', False),
                  ('firth_1layer', True),
                  ('l2_1layer', True),
                  ('firth_3layer', True),
                  ('l2_3layer', True)]

for (experiment, dataset_name, n_ways), df_exp in df_summ.groupby(['experiment', 'dataset_name', 'n_ways']):
    for is_relative in [False, True]:
        if (experiment, is_relative) not in exp_isrel_list:
            continue
        dacc_type = "relative" if is_relative else "absolute"
        print(f'  * Plotting the {dacc_type} accuracy improvement vs. number of shots \n' + 
              f'    for the {experiment} experiment, {dataset_name} dataset, ' +
              f'and {n_ways}-way classification.')
        save_dir = f'{paper_plots_dir}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)

        ds_abrv = ds_abrv_dict[dataset_name]
        experiment_abrv = experiment
        experiment_abrv = experiment_abrv.replace('_tieredcifar', '')
        experiment_abrv = experiment_abrv.replace('_5_10way', '')
        figname = f'dacc_vs_nshots_{experiment_abrv}_{ds_abrv}'
        if n_ways != 16:
            figname = figname + f'_{n_ways}ways'
        if is_relative:
            figname = f'rel{figname}'

        save_dict = {'do_save': True, 'save_dir': save_dir, 'figname': figname}
        cond_dict = {'experiment': experiment, 'dataset_name': dataset_name}
        
        fmlregnames = {'firth_coeff': 'Firth Bias Reduction', 
                       'l2_coeff':'L2 Regularization',
                       'ent_coeff': 'Entropy Regularization'}
        rename_dict = {exp_: fmlregnames[regcol_] for exp_, regcol_ in expname_regcol.items()}

        if is_relative:
            ylabel = r'$\Delta(ACC)/ACC$%'
        else:
            ylabel = r'$\Delta(ACC)$%'

        dsname_frml_dict = {'miniimagenet': 'mini-ImageNet',
                            'tieredimagenet': 'tiered-ImageNet',
                            'cifarfs': 'CIFAR-FS'}

        dsname_frml = dsname_frml_dict.get(dataset_name, dataset_name)
        if experiment == 'firth_5_10way':
            fig_title = f'{n_ways}-way'
        else:
            fig_title = f'{rename_dict.get(experiment,experiment)} ({dsname_frml})'

        fig_dict = {'ylabel': ylabel, 'xlabel': 'Number of Shots',
                    'title': fig_title, 'x_relabel': True}

        my_df_summ = df_exp.copy()
        if is_relative:
            y_col = f'rel_delta_{main_acc}'
            yerr_col = f'rel_delta_{main_acc}_ci'
            my_df_summ[y_col] = my_df_summ[f'delta_{main_acc}'] / my_df_summ[f'base_{main_acc}']
            my_df_summ[yerr_col] = my_df_summ[f'delta_{main_acc}_ci'] / my_df_summ[f'base_{main_acc}']
        else:
            y_col = f'delta_{main_acc}'
            yerr_col = f'delta_{main_acc}_ci'

        my_df_summ[y_col] = scale_percent * my_df_summ[y_col]
        my_df_summ[yerr_col] = scale_percent * my_df_summ[yerr_col]
        
        legend_namer = lambda bbarch: bbarch.capitalize().replace('net', 'Net')
        fig, axes = plot_stat_combined(my_df_summ, y=y_col, x='n_shots',
                                       yerr_col=yerr_col, hue_name='backbone_arch',
                                       fig_dict=fig_dict, cond_dict=cond_dict,
                                       legend_namer=legend_namer,
                                       hue_sorting_key=hue_sorting_key)

        ax = axes[0, 0]
        if is_relative:
            if experiment in ('firth_1layer', 'l2_1layer'):
                ax.set_ylim([-1.5, 35.0])
            elif experiment in ('firth_3layer', 'l2_3layer'):
                ax.set_ylim([-1.5, 18.5])
        else:
            if experiment in ('firth_1layer', 'l2_1layer', 'firth_5_10way'):
                ax.set_ylim([-0.15, 3.1])
            elif experiment in ('firth_3layer', 'l2_3layer'):
                ax.set_ylim([-0.15, 1.7])
            elif experiment in ('firth_tieredcifar', 'l2_tieredcifar'):
                ax.set_ylim([-0.2, 8.0])

        # save settings:
        if save_dict['do_save']:
            save_dir = save_dict.get('save_dir', './')
            figname = save_dict.get('figname', 'template')
            dpi = save_dict.get('dpi', 300)
            os.makedirs(save_dir, exist_ok=True)
            fig.tight_layout()
            save_path = f'{save_dir}/{figname}.pdf'
            fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
            print(f'  *   --> Figure saved at {save_path}.\n  ' + '-'*80)
