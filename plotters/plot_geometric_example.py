import os
import sys
import numpy as np
from collections import defaultdict
from numpy.random import default_rng
import matplotlib.pyplot as plt
from utils.cfg import paper_plots_dir

print(f"  * Plotting the geometric experiment's MLE bias reduction figures.")
fig1_path = f"{paper_plots_dir}/avgmle_vs_nsamples_geom.pdf"
fig2_path = f"{paper_plots_dir}/logmlebias_vs_lognsamples_geom.pdf"

if os.path.exists(fig1_path) and os.path.exists(fig2_path):
    print(f'    --> Figures {fig1_path} already exists.')
    print(f'    --> Figures {fig2_path} already exists.')
    print(f'    --> Skipping plotting them.\n  ' + '-'*80)
    sys.exit(0)


def estim_geom(y_, type_='mle', firth_coeff_=1):
    if type_ == 'mle':
        p_hat = 1/np.mean(y_)
    elif type_ == 'firth_mle':
        n = y_.size
        if n == 1:
            n = 1.01
        p_hat = (n-firth_coeff_)/(n*np.mean(y_) - 1)
    else:
        raise ValueError(f"{type_} not valid")
    return p_hat


seed = 12345
rng = default_rng(seed)
sample_sizes = range(1, 100)
trials = 10000
P = [0.5]  # this can be repeated with other values
firth_coeff = 1
p_dict = {}

for p in P:
    # per sample size generate data from geometric distribution for exp_count times
    firth_mle_d = defaultdict(list)
    mle_d = defaultdict(list)
    for sample_size in sample_sizes:
        for trial in range(trials):
            y = rng.geometric(p=p, size=sample_size)
            mle_estim = estim_geom(y, type_='mle',
                                   firth_coeff_=firth_coeff)
            firth_mle_estim = estim_geom(y, type_='firth_mle',
                                         firth_coeff_=firth_coeff)
            mle_d[sample_size].append(mle_estim)
            firth_mle_d[sample_size].append(firth_mle_estim)

    mle_d_sum, firth_mle_d_sum = {}, {}
    for sample_size in mle_d.keys():
        mean_ = np.mean(np.array(mle_d[sample_size]))
        mle_d_sum[sample_size] = mean_

        mean_ = np.mean(np.array(firth_mle_d[sample_size]))
        firth_mle_d_sum[sample_size] = mean_

    mle_vals_mean = mle_d_sum.values()
    firth_mle_vals_mean = firth_mle_d_sum.values()
    p_dict[p] = (mle_vals_mean, firth_mle_vals_mean, mle_d, firth_mle_d)

p = 0.5
nrows, ncols = 1, 1
figsize = (3.0*ncols, 2.2*nrows)
fig, axes = plt.subplots(nrows, ncols, figsize=figsize, dpi=200)
ax = axes
(mle_vals, firth_mle_vals, _, _) = p_dict[p]
sample_sizes_np = np.array(sample_sizes)
mle_vals_np = np.array(list(mle_vals))
firth_mle_vals_np = np.array(list(firth_mle_vals))

keep_ = np.logical_or((sample_sizes_np % 10 == 0), (sample_sizes_np < 11))
keep_ = np.logical_and(keep_, (sample_sizes_np < 11))

ax.axhline(0.5, xmin=0.05, xmax=0.95, lw=0.9, color='black', ls='--')

ax.plot(sample_sizes_np[keep_], mle_vals_np[keep_], color='b', ls='--', lw=0.5)
ax.scatter(sample_sizes_np[keep_], mle_vals_np[keep_], color='b', s=15)

ax.plot(sample_sizes_np[keep_], firth_mle_vals_np[keep_],
        color='r', ls='--', lw=0.5, zorder=9)
ax.scatter(sample_sizes_np[keep_], firth_mle_vals_np[keep_],
           color='r', s=15, zorder=5)
ax.set_xlabel(r'Sample Size ($N$)')
ax.set_ylabel(r'Mean $\hat{\beta}$ Estimate')
ax.set_ylim(0.48, 0.71)
ax.set_xlim(0.5, 10.5)

ax.scatter([1.0], [0.5], s=55, color='black', marker='*', zorder=10)

ax.annotate(r'$E[\hat{\beta}_{MLE}]$', xy=(5.3, 0.550), xytext=(7, 0.59),
            arrowprops=dict(arrowstyle="->", linewidth=.75, color='black',
                            connectionstyle="angle3,angleA=40,angleB=0"))

dd = dict(arrowstyle="<|-|>", shrinkA=0, shrinkB=0, linestyle='--',
          mutation_scale=12, color='black', linewidth=.5)
ax.annotate('', xy=(0.8, 0.501), xytext=(0.8, 0.702),
            arrowprops=dd, zorder=15)

ax.annotate(r'$bias(\hat{\beta}_{MLE}) = E[\hat{\beta}_{MLE}]-\beta^{*}$',
            xy=(0.8, 0.63), xytext=(2.2, 0.68),
            arrowprops=dict(arrowstyle="->", linewidth=.75,
                            connectionstyle="angle3,angleA=55,angleB=15",
                            relpos=(0, 0.1)))

ax.annotate(r'$E[\hat{\beta}_{Firth}]$', xy=(2.0, 0.50), xytext=(2.8, 0.525),
            arrowprops=dict(arrowstyle="->", linewidth=.75,
                            color='black', relpos=(0, 0.1),
                            connectionstyle="angle3,angleA=39,angleB=0"))

ticks = ax.get_xticks()
ax.set_xticks(list(range(1, 11)))
fig.savefig(fig1_path, bbox_inches='tight', pad_inches=0, dpi=200)
print(f'  *   --> Figure saved at {fig1_path}.')

# The log-log plot of MLE bias vs the sample-size
p = 0.5
nrows, ncols = 1, 1
figsize = (3.*ncols, 2.2*nrows)
fig, ax = plt.subplots(nrows, ncols, figsize=figsize, dpi=200)

(mle_vals, firth_mle_vals, _, _) = p_dict[p]
mle_vals, firth_mle_vals = np.array(list(mle_vals)), np.array(list(firth_mle_vals))

ax.set_ylabel(r"$\log_{10}(E[\hat{\beta}_{MLE}^{(N)}]-\beta^{*})$")
ax.set_xlabel(r"$\log_{10}(N)$")

sample_sizes_np = np.array(sample_sizes)

a = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 18,
              20, 25, 28, 32, 35, 40, 50, 60, 70, 80, 99])
b = sample_sizes_np.reshape(-1, 1) == a.reshape(1, -1)
keep_ = np.any(b, axis=1)
x = np.log10(sample_sizes)[keep_]
y = np.log10(mle_vals - p)[keep_]

str_ = r'$E[\hat{\beta}_{MLE}^{(N)}]-\beta^{*} \simeq \frac{1}{4N}$'
ax.annotate(str_, xy=(0.5, -1.10), xytext=(0.9, -.9),
            arrowprops=dict(arrowstyle="->", linewidth=.75,
                            color='black', relpos=(0, 0.5),
                            connectionstyle="angle3,angleA=45,angleB=0"))

ax.scatter(x, y, color='purple', marker='o', s=15)
ax.plot(x, -x-0.6, color='black', lw=1.0, ls='--', zorder=0)

fig.savefig(fig2_path, bbox_inches='tight', pad_inches=0, dpi=200)
print(f'  *   --> Figure saved at {fig2_path}.\n  ' + '-'*80)
