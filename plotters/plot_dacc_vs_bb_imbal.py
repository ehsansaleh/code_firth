import os
import pandas as pd
from pathlib import Path
from utils.cfg import smry_tbls_dir
from utils.cfg import main_acc, paper_plots_dir, scale_percent
from utils.plot_utils import plot_stat_combined

pd.options.mode.chained_assignment = None

for clf_type in ["mlp", 'lin']:
    summ_file_dir = f'{smry_tbls_dir}/val2test.csv'
    df_summ = pd.read_csv(summ_file_dir, sep=',')
    n_layers = {'mlp': 3, 'lin': 1}[clf_type]

    # Conditioning on imbalanced settings
    if clf_type == 'mlp':
        exp_conds = ['imbal_3layer']
    elif clf_type == 'lin':
        exp_conds = ['imbal_1layer']
    else:
        raise Exception(f'I do not know clf_type={clf_type}')
    df_summ = df_summ[df_summ['experiment'].isin(exp_conds)]

    # specific shot plots
    for nshot, df_exp in df_summ.groupby('n_shots'):
        rename_dict = {8: '7.5 Average Shots', 15: '15 Average Shots'}
        print(f'  * Plotting the accuracy improvement vs. the backbone architecure \n' + 
              f'    with imbalanced datasets and {rename_dict[nshot]} ' + 
              f'and {n_layers}-layer classifiers.')
        save_dir = f'{paper_plots_dir}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        a = f'_{main_acc}' if main_acc != 'test_acc' else ''
        figname = f'dacc_vs_arch_{nshot}shots_{n_layers}layer{a}'
        save_dict = {'do_save': True, 'save_dir': save_dir, 'figname': figname}
        cond_dict = {'n_shots': nshot}

        fig_dict = {'ylabel': r'$\Delta(ACC)$%', 'xlabel': 'Backbone Architecture',
                    'title': f'{rename_dict.get(nshot,nshot)}', 'x_relabel': False}
        y_col = f'delta_{main_acc}'
        yerr_col = f'delta_{main_acc}_ci'

        my_df_summ = df_exp.copy()
        my_df_summ[y_col] = scale_percent * my_df_summ[y_col]
        my_df_summ[yerr_col] = scale_percent * my_df_summ[yerr_col]
        my_df_summ['arch_no'] = [int(x_.lower().split('resnet')[-1]) for x_ in my_df_summ['backbone_arch']]
        my_df_summ['arch_abrv'] = ['RN' + x_.lower().split('resnet')[-1] for x_ in my_df_summ['backbone_arch']]

        arch_order = ['RN10', 'RN18', 'RN34', 'RN50', 'RN101']
        x_sorter_key = lambda a: arch_order.index(a) if a in arch_order else len(arch_order)
        legend_renaming = {'uniform': 'Uniform Prior', 'class_freq': 'Non-uniform Prior'}

        fig, axes = plot_stat_combined(my_df_summ, y=y_col, x='arch_abrv',
                                       yerr_col=yerr_col, hue_name='firth_prior_type',
                                       fig_dict=fig_dict, cond_dict=cond_dict,
                                       plot_type='bar', plt_kwargs=dict(x_sorter_key=x_sorter_key),
                                       legend_namer=lambda exp: legend_renaming.get(exp, exp))

        ax = axes[0, 0]
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        if clf_type == "mlp":
            ax.set_ylim([0., 1.1])
        elif clf_type == "lin":
            pass

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
