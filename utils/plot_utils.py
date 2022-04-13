import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class ColorGiver:
    def __init__(self):
        sns_pal = sns.color_palette("bright", 12)
        self.pal = sns_pal
        self.c_blue, self.c_orange, self.c_green = sns_pal[0:3]
        self.c_red, self.c_purple, self.c_brown = sns_pal[3:6]
        self.c_pink, self.c_gray, self.c_yellow = sns_pal[6:9]
        self.c_cyan = sns_pal[10]

        self.ptr = 0
        self.memory = dict()

    def __getitem__(self, hue_var):
        color = None
        c_dict = {'resnet10': self.c_blue, 'resnet18': self.c_orange, 
                  'resnet34': self.c_green, 'resnet50': self.c_red,
                  'resnet101': self.c_purple, 'densenet121': self.c_cyan,
                  'mobilenet84': self.c_brown,
                  'uniform': self.c_blue, 'class_freq': self.c_red,
                  '1-Shot': self.c_blue, '10-Shot': self.c_red,
                  '5-Shot': self.c_purple,
                  
                  8: self.c_orange, 15: self.c_green,
                  'Firth+L2': self.c_purple, 'Firth': self.c_blue, 'L2': self.c_red,
                  'mlp': self.c_red, 'lin': self.c_green}

        color = c_dict.get(hue_var, None)
        #print(hue_var)

        if color is None:
            self.memory[hue_var] = self.pal[self.ptr]
            self.ptr = (self.ptr + 1) % len(self.pal)
        return color


def plot_stat_combined(df_input, y, x, hue_name, yerr_col=None, xerr_col=None,
                       cond_dict=None, fig_dict=None, save_dict=None,
                       plot_type='line', legend_namer=None, fig_axes=None,
                       plt_kwargs=None, legend_kwargs=None, hue_sorting_key=None):
    save_dict = save_dict or dict()
    fig_dict = fig_dict or dict()
    cond_dict = cond_dict or dict()
    legend_namer = legend_namer or (lambda hue_val: hue_val)
    plt_kwargs = plt_kwargs or dict()
    legend_kwargs = legend_kwargs or dict()

    # figure settings:
    nrows = fig_dict.get('nrows', 1)
    ncols = fig_dict.get('ncols', 1)
    width = fig_dict.get('width', 3)
    ratio = fig_dict.get('ratio', 1.25)
    ax_row_idx = fig_dict.get('ax_row', 0)
    ax_col_idx = fig_dict.get('ax_col', 0)

    if fig_axes is None:
        figsize = fig_dict.get('figsize', (nrows*width*ratio, ncols*width))
        fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=144)
        axes = np.array(axes).reshape(nrows, ncols)
        ax = axes[ax_row_idx, ax_col_idx]
    else:
        fig, axes = fig_axes
        ax = axes[ax_row_idx, ax_col_idx]

    sort_key = None
    x_ticks_bar = None
    x_range_bar = None
    if plot_type == 'bar':
        x_ticks_bar = df_input[x].unique().tolist()
        sort_key = plt_kwargs.get('x_sorter_key', sort_key)
        plt_kwargs = {x: v for x, v in plt_kwargs.items() if x != 'x_sorter_key'}
        x_ticks_bar = sorted(x_ticks_bar, key=sort_key)
        x_range_bar = np.arange(len(x_ticks_bar))  # the label locations

    legends = []
    hue_iterator = sorted(list(set(df_input[hue_name])), key=hue_sorting_key)
    for i_, hue_val in enumerate(hue_iterator):
        cond_dict[hue_name] = hue_val
        # conditioning on dataset, shots, ...
        df = df_input.copy(deep=True)
        for var, val in cond_dict.items():
            df = df[df[var] == val]

        if sort_key is not None:
            df['sort_order'] = [sort_key(x) for x in df[x]]
            df.sort_values(by=['sort_order'], inplace=True)
        else:
            df.sort_values(by=[x], inplace=True)

        if yerr_col in df.columns:
            yerr = df[yerr_col]
        else:
            yerr = None

        if (not xerr_col) and (xerr_col in df.columns):
            xerr = df[xerr_col]
        else:
            xerr = None
        if plot_type == 'line':
            ax.errorbar(x=df[x], y=df[y], yerr=yerr, xerr=xerr, marker='o',
                        color=color_dict[hue_val], **plt_kwargs)
        elif plot_type == 'bar':
            bar_width = plt_kwargs.get('bar_width', 0.35)  # the width of the bars
            plt_kwargs = {x: v for x, v in plt_kwargs.items() if x != 'bar_width'}
            numhues = len(hue_iterator)
            if numhues > 1:
                shifts = np.arange(numhues)
                shifts = shifts - np.mean(shifts)
                shifts = bar_width * shifts / (shifts.max() - shifts.min())
                each_bar_width = 1.00 * (shifts[1] - shifts[0])
            else:
                shifts = [0]
                each_bar_width = bar_width

            ax.bar(x_range_bar - shifts[i_], df[y], each_bar_width, yerr=yerr,
                   label=hue_val, color=color_dict[hue_val], **plt_kwargs)
            ax.set_xticks(x_range_bar)
            ax.set_xticklabels(x_ticks_bar)
        legends.append(legend_namer(hue_val))

    ax.legend(legends, **legend_kwargs)
    ax.set_ylabel(fig_dict.get('ylabel', ''))
    ax.set_xlabel(fig_dict.get('xlabel', ''))
    ax.set_title(fig_dict.get('title', ''))

    if fig_dict.get('x_relabel', False):
        xtick_labels = df[x].tolist()
        ax.set_xticks(xtick_labels)

    if save_dict.get('do_save', False):
        save_dir = save_dict.get('save_dir', './')
        figname = save_dict.get('figname', 'template')
        dpi = save_dict.get('dpi', 300)
        os.makedirs(save_dir, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f'{save_dir}/{figname}.pdf', dpi=dpi)

    return fig, axes


color_dict = ColorGiver()
