import os
import pandas as pd
from pathlib import Path
from utils.cfg import smry_tbls_dir
from utils.cfg import main_acc, paper_plots_dir, scale_percent
from utils.plot_utils import plot_stat_combined

pd.options.mode.chained_assignment = None
fig_axes = None
summ_file_dir = f'{smry_tbls_dir}/test.csv'
df_summ = pd.read_csv(summ_file_dir, sep=',')

# Conditioning on imbalanced settings
df_summ = df_summ[df_summ['experiment'].isin(['firth_1layer', 'firth_3layer'])]
df_summ = df_summ[df_summ['firth_coeff'] == 0]
print(f'  * Plotting the base accuracy values for different backbone architectures and 1- or 3-layer classifiers.')
for ax_idx, (arch, df_exp) in enumerate(df_summ.groupby('backbone_arch')):
    save_dir = f'{paper_plots_dir}'
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    save_dict = {'do_save': True, 'save_dir': save_dir,
                 'figname': f'baseacc_vs_nways'}
    cond_dict = {'backbone_arch': arch}

    title_rename_dict = {f'resnet{x}':f'ResNet{x}' for x in [10, 18, 34, 50, 101]}
    ax_order = [f'resnet{x}' for x in [10, 18, 34, 50, 101]]
    ax_idx = ax_order.index(arch.lower()) if arch.lower() in ax_order else ax_idx
    nrows, ncols = 2, 3
    ax_row = ax_idx // ncols
    ax_col = ax_idx % ncols
    fig_dict = {'ylabel': r'$ACC$%', 'xlabel': 'Number of Shots',
                'title': f'{title_rename_dict.get(arch,arch)} Backbone',
                'x_relabel': True, 'width': 2., 'ratio': 2.5,
                'nrows': nrows, 'ncols': ncols, 'ax_row': ax_row,
                'ax_col': ax_col}

    y_col = f'{main_acc}'
    yerr_col = f'{main_acc}_ci'
    my_df_summ = df_summ.copy()
    my_df_summ[y_col] = scale_percent * my_df_summ[y_col]
    my_df_summ[yerr_col] = scale_percent * my_df_summ[yerr_col]
    my_df_summ['arch_no'] = [int(x_.lower().split('resnet')[-1]) for x_ in my_df_summ['backbone_arch']]
    clf_renaming = {'lin': '1-Layer', 'mlp': '3-Layer'}
    legend_namer = lambda clf_type: f'{clf_renaming.get(clf_type, clf_type)} Classifier'
    fig_axes = plot_stat_combined(my_df_summ, y=y_col, x='n_shots', yerr_col=yerr_col,
                                  hue_name='clf_type', fig_dict=fig_dict, cond_dict=cond_dict,
                                  fig_axes=fig_axes, legend_namer=legend_namer)
    fig, axes = fig_axes
    ax = axes[ax_row, ax_col]
    ax.set_ylim([7.0, 10.35])

axes[1, 2].remove()
if save_dict['do_save']:
    save_dir = save_dict.get('save_dir', './')
    figname = save_dict.get('figname', 'template')
    dpi = save_dict.get('dpi', 300)
    os.makedirs(save_dir, exist_ok=True)
    fig.tight_layout()
    save_path = f'{save_dir}/{figname}.pdf'
    fig.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    print(f'  *   --> Figure saved at {save_path}.\n  ' + '-'*80)
