import os
import pandas as pd
from pathlib import Path
from matplotlib.ticker import FormatStrFormatter
from utils.cfg import smry_tbls_dir
from utils.cfg import main_acc, paper_plots_dir, scale_percent
from utils.plot_utils import plot_stat_combined

pd.options.mode.chained_assignment = None
summ_file_dir = f'{smry_tbls_dir}/val2test.csv'
df_summ = pd.read_csv(summ_file_dir, sep=',')
ds_abrv_dict = {'miniimagenet': 'mini', 'tieredimagenet': 'tiered', 'cifarfs': 'cifarfs'}

def hue_sorting_key(nshots_str):
    """
    This function is just for making sure we first go through 
    1-Shot, 5-Shot, and then 10-Shot in order. This is especially 
    important for ordering in figure legends, since by default, we 
    have the 1-Shot, 10-Shot, and then 5-Shot ordering due to 
    nshots_str being a string.
    """
    out = nshots_str 
    if isinstance(nshots_str, str):
        a = str(nshots_str).lower().replace('-shot', '')
        if a.isnumeric():
            out = f'{int(a):03d}'
    return out

groupping_vars = ['experiment', 'dataset_name', 'backbone_arch']
for (experiment, dataset_name, backbone_arch), df_exp in df_summ.groupby(groupping_vars):
    if experiment not in ['firth_densemobile']:
        continue
    for is_relative in [False]:
        dacc_type = "relative" if is_relative else "absolute"
        print(f'  * Plotting the {dacc_type} accuracy improvement vs. number of ways \n' + 
              f'    for the {experiment} experiment, {dataset_name} dataset, ' +
              f'and the {backbone_arch} backbone architecture.')
        x_name = 'n_ways'
        fig_width, fig_ratio = 2.5, 3

        arch_abrv = backbone_arch.lower().replace('resnet', 'rn')
        arch_abrv = arch_abrv.replace('densenet', 'dn')
        arch_abrv = arch_abrv.replace('mobilenet', 'mn')

        save_dir = f'{paper_plots_dir}'
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        ds_abrv = ds_abrv_dict[dataset_name]
        figname = f'dacc_vs_nways_{arch_abrv}_{ds_abrv}'
        if is_relative:
            figname = f'rel{figname}'
        save_dict = {'do_save': True, 'save_dir': save_dir, 'figname': figname}
        cond_dict = {'experiment': experiment, 'dataset_name': dataset_name}

        if is_relative:
            ylabel = r'$\Delta(ACC)/ACC$%'
        else:
            ylabel = r'$\Delta(ACC)$%'

        dsname_frml_dict = {'miniimagenet': 'mini-ImageNet',
                            'tieredimagenet': 'tiered-ImageNet',
                            'cifarfs': 'CIFAR-FS'}
        fig_title = ''
        if backbone_arch == 'densenet121':
            fig_title = "DenseNet"
        elif backbone_arch == 'mobilenet84':
            fig_title = "MobileNet"
        else:
            raise ValueError(f'figname for {backbone_arch} not implemented')
        fig_dict = {'ylabel': ylabel, 'xlabel': 'Number of Classes',
                    'width': fig_width, 'ratio': fig_ratio, 'title': fig_title,
                    'x_relabel': True}

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
        my_df_summ['formal_shots'] = [f'{x_}-Shot' for x_ in my_df_summ['n_shots']]

        fig, axes = plot_stat_combined(my_df_summ, y=y_col, x=x_name,
                                       yerr_col=yerr_col, hue_name='formal_shots',
                                       fig_dict=fig_dict, cond_dict=cond_dict,
                                       hue_sorting_key=hue_sorting_key,
                                       legend_kwargs=dict(loc='upper right',
                                                          bbox_to_anchor=(1.0, 0.7)))

        ax = axes[0, 0]

        use_large_yscale = True
        if is_relative:
            ax.set_ylim([-0.5, 7.0])
        else:
            ax.set_ylim([-0.15, 3.0])
        axes[0, 0].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

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
