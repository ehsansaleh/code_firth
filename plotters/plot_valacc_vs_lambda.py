import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from utils.cfg import PROJPATH, paper_plots_dir, fig_format, draw_trnsprnt_figs

pd.options.mode.chained_assignment = None
reg_column = 'firth_coeff'
baseline_acc = False
save_figs = True
print_figtitle = True
assert reg_column in ['l2_coeff', 'firth_coeff'], "invalid regularization column is provided"

# setting
for (indir, nlayer) in [(f'{PROJPATH}/results/04_firth_3layer', 3),
                        (f'{PROJPATH}/results/01_firth_1layer', 1)]:
    print(f'  * Plotting the validation accuracy vs. the firth coefficient for {nlayer}-layer classifiers.')
    figpath = f'{paper_plots_dir}/valacc_vs_lambda_{nlayer}layer.{fig_format}'
    if os.path.exists(figpath):
        print(f'    --> Figure {os.path.basename(figpath)} already exists.\n  ' + '-'*80)
        continue
    if not os.path.exists(indir):
        print(f'    --> The results dir {indir} does not exist. Skipping plotting.\n  ' + '-'*80)
        continue

    df_full = pd.concat([pd.read_csv(f'{indir}/{x}') for x in os.listdir(indir)
                         if x.endswith('.csv') and not ('scratch' in x)],
                        axis=0, ignore_index=True)

    df = df_full.copy(deep=True)
    # conditioning on the regularization we're interested in
    if reg_column == 'firth_coeff':
        df = df[df['l2_coeff'] == 0]
    elif reg_column == 'l2_reg_coeff':
        df = df[df['firth_coeff'] == 0]

    sns_pal = sns.color_palette("bright", 12)
    # *** Validation Acc vs. Coefficients ***
    nrows, ncols = 2, 3
    figsize = (4.3*ncols, 2.6*nrows+1.5)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=144)
    axes = np.array(axes).reshape(-1)

    for i, fixed_var_val in enumerate(list(set(df_full['n_shots']))[::-1]):
        ax = axes[i]
        fixed_var = 'n_shots'
        fixed_var_val = fixed_var_val
        x_var = 'backbone_arch'
        myrow, mycol = i // ncols, i % ncols

        for i_arch, arch in enumerate(df_full[x_var].unique()):
            df = df_full.copy(deep=True)
            y_col = 'test_acc'
            # Let's get rid of the l2 regularization experiments
            if reg_column == 'firth_coeff':
                df = df[df['l2_coeff'] == 0]
            elif reg_column == 'l2_coeff':
                df = df[df['firth_coeff'] == 0]
            df = df[(df[fixed_var] == fixed_var_val) & (df[x_var] == arch)]
            df_val = df[df['data_type'] == 'val']
            df_test = df[df['data_type'] == 'novel']

            stats_val = df_val.groupby([reg_column])[y_col].agg(['mean', 'count', 'std'])
            df_val_mean = df_val.groupby([reg_column]).mean()
            df_val_mean[f'{y_col}_ci'] = 1. * stats_val['std'] / np.sqrt(stats_val['count'])
            df_val_mean.reset_index(inplace=True)
            df_val_mean[x_var] = arch

            ax.errorbar(df_val_mean[reg_column], df_val_mean[y_col]*100.,
                        yerr=df_val_mean[f'{y_col}_ci']*100, fmt='o-',
                        c=sns_pal[i_arch], label=arch.replace('ResNet', 'ResNet'))
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(decimals=1))
            ax.set_xscale('symlog', linthresh=0.001 if reg_column == "firth_coeff" else 1.)

        reg_type = "Firth" if reg_column == "firth_coeff" else "L2"
        if myrow == nrows-1:
            ax.set_xlabel(f'{reg_type} Regularization Coefficient')

        if mycol == 0:
            ax.set_ylabel('Validation Accuracy')

        ax.set_title(f'{fixed_var_val}-Shot')

    plt.legend(loc='upper center', bbox_to_anchor=(-0.8, -0.18), ncol=6)
    if print_figtitle:
        fig.suptitle(f'{nlayer}-Layer Logistic Classifier')

    if save_figs:
        fig.savefig(figpath, dpi=300, bbox_inches='tight', transparent=draw_trnsprnt_figs)
        print(f'  *   --> Figure saved at {figpath}.\n  ' + '-'*80)
