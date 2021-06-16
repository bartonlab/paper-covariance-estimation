"""
This Python file contains code for generating plots in:
    Title of paper
    Yunxiao Li, and
    John P. Barton (john.barton@ucr.edu)

Additional code to pre-process the data and pass it to these plotting
routines is contained in the Jupyter notebook `figures.ipynb`.
"""

#############  PACKAGES  #############

import numpy as np

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

############# PARAMETERS #############

# GLOBAL VARIABLES

N = 1000        # number of sequences in the population
L = 50          # sequence length
T = 700         # number of generations
MU = 1e-3       # mutation rate
num_selections = 10  # number of sets of selection coefficients that we used
num_trials = 20  # for each set of selection coefficients, we simulated for 20 times
NUM_BASIC_METHODS = 3  # SL, est and MPL methods
SAMPLE = [1000, 500, 100, 50, 10]  # sampling depth options when subsampling
RECORD = [1, 3, 5, 10]  # sampling time interval options when subsampling
LINEAR = [5, 10, 20, 50, 100, 300]  # linear shrinkage strengths that we tested
GAMMA = [5e-5, 5e-4, 5e-3, 5e-2, 5e-1, 1]  # non-linear shrinkage parameter choices that we tested
TRUNCATE = [200, 300, 400, 500, 600, 700]  # truncate options to test performance using different lengths of data
WINDOW = [0, 1, 2, 3, 4, 5, 10]  # window choices that we tested
# loss functions for non-linear shrinkage that we tested
LOSS = ['Fro | $\hat{C}-C$', 'Fro | $\hat{C}^{-1}-C^{-1}$', 'Fro | $C^{-1}\hat{C}-I$',
        'Fro | $\hat{C}^{-1}C-I$', 'Fro | $\hat{C}^{-1/2}C\hat{C}^{-1/2}-I$',
        'Nuc | $\hat{C}-C$', 'Nuc | $\hat{C}^{-1}-C^{-1}$', 'Nuc | $C^{-1}\hat{C}-I$',
        'Nuc | $\hat{C}^{-1}C-I$', 'Nuc | $\hat{C}^{-1/2}C\hat{C}^{-1/2}-I$']

# Standard color scheme

BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'

COLOR_BASIC = ['blue', 'orange', 'grey', 'grey']
MARKER_BASIC = ['o', '^', 's', 'D']
LABEL_BASIC = ['SL', 'MPL', 'est', 'est (uncalibrated)']
COLOR_LINEAR = 'red'
MARKER_LINEAR = 's'
LABEL_LINEAR = 'linear-cov'
COLOR_DCORR = 'green'
MARKER_DCORR = '^'
LABEL_DCORR = 'non-linear-corr'
LINESTYLE_LOSS = ['solid'] * 5 + ['dashed'] * 5
COLOR_LOSS = ['red', 'orange', 'blue', 'green', 'grey'] * 2
MARKER_LOSS = ['.', 's', 'p', 'v', 'd'] * 2
LABEL_LOSS = LOSS

# Plot conventions

def cm2inch(x): return float(x)/2.54
SINGLE_COLUMN   = cm2inch(8.5)
ONE_FIVE_COLUMN = cm2inch(11.4)
DOUBLE_COLUMN   = cm2inch(17.4)

GOLDR        = (1.0 + np.sqrt(5)) / 2.0
TICKLENGTH   = 3
TICKPAD      = 3
AXWIDTH      = 0.4
SUBLABELS    = ['A', 'B', 'C', 'D', 'E', 'F']

# paper style

FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZELEGEND   = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZELINE     = 0.6

DEF_FIGPROPS = {
    'transparent' : True,
    'edgecolor'   : None,
    'dpi'         : 1000
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False
}

DEF_SUBLABELPOSITION_1x2 = {
    'x'       : -0.15,
    'y'       : 1.05
}

DEF_SUBLABELPOSITION_2x2 = {
    'x'       : -0.18,
    'y'       : 1.1
}

DEF_SUBPLOTS_ADJUST_1x2 = {
    'bottom'  : 0.22,
    'wspace'  : 0.4
}

DEF_SUBPLOTS_ADJUST_2x2 = {
    'bottom'  : 0.15,
    'wspace'  : 0.4,
    'hspace'  : 0.4
}

DEF_TICKPROPS = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : True,
    'left'      : True,
    'top'       : False,
    'right'     : False
}

DEF_TICKPROPS_HEATMAP = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : False,
    'top'       : False,
    'right'     : False
}

DEF_AXPROPS = {
    'linewidth' : AXWIDTH,
    'linestyle' : '-',
    'color'     : BKCOLOR
}

DEF_HEATMAP_KWARGS = {
    'vmin'      : 0.65,
    'vmax'      : 0.9,
    'cmap'      : 'GnBu',
    'alpha'     : 0.75,
    'cbar'      : False,
    'annot'     : True,
    'annot_kws' : {"fontsize": 6}
}

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(PARAMS)

############# PLOTTING  FUNCTIONS #############

def plot_figure_estimation_example(true_cov, est_cov, save_file=None):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    w = DOUBLE_COLUMN
    goldh = w / 2.5
    fig, axes = plt.subplots(1, 3, figsize=(w, goldh))

    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = ["true", "estimate", "error"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
#     cbar_ax = fig.add_axes(rect=[.93, .1, .02, .8])  # color bar on the right
    cbar_ax = fig.add_axes(rect=[.128, .175, .773, .02])  # color bar on the bottom, rect=[left, bottom, width, height]
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    matrix_labels = [1] + [10 * i for i in range(1, 6)]
    matrix_ticks = [l - 0.5 for l in matrix_labels]
    for i, cov in enumerate(cov_list):
        plt.sca(axes[i])
        ax = plt.gca()
        plot_cbar = (i == 0)
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True),
                    square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None,
                    cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"))
        plt.title(title_list[i], fontsize=SIZELABEL)
        if i == 0:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        # plt.xticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL, rotation=0)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)

    plt.subplots_adjust(bottom=0.25)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_traj_example(traj, selection, alpha=0.75, save_file=None):
    """Plots a figure of allele frequency trajectories; Color trajectories according to their selections."""

    w = SINGLE_COLUMN
    goldh = w / 1.4
    fig = plt.figure(figsize=(w, goldh))

    for l in range(len(traj[0])):
        if selection[l] > 0:
            color = C_BEN_LT
        elif selection[l] == 0:
            color = C_NEU_LT
        else:
            color = C_DEL_LT
        plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE)

    plt.ylabel("Mutant allele frequency", fontsize=SIZELABEL)
    plt.xlabel("Generation", fontsize=SIZELABEL)
    plt.gca().tick_params(**DEF_TICKPROPS)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_ample_data(spearmanr_basic, error_basic, truncate_list=TRUNCATE, ylim=(0.5, 0.95), save_file=None):
    """Plots a figure comparing performances of the SL, MPL and est methods using complete data."""

    w = SINGLE_COLUMN
    goldh = w / 1.4
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0  # ample data
    plot_data = [np.mean(spearmanr_basic[:, :, :, p, q], axis=(1, 2)), np.mean(error_basic[:, :, :, p, q], axis=(1, 2))]
    ylabels = [r"Spearman's $\rho$", r'Error of inference']
    yticks = np.linspace(0.5, 0.9, 5)
    legend_loc = [4, 1]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        for j in range(NUM_BASIC_METHODS):
            plt.scatter(truncate_list, data[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
            plt.plot(truncate_list, data[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE)

        if i == 0:
            plt.ylim(ylim)
            ax.set_yticks(yticks)
            plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
        plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=1 if i == 1 else None)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Generations of data used"
    add_shared_xlabel(fig, xlabel, labelpad=-1)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def add_shared_xlabel(fig, xlabel, labelpad=-3, fontsize=SIZELABEL):
    """Adds a xlabel shared by subplots."""

    # add a big axes, hide frame in order to generate a shared x-axis label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    plt.xlabel(xlabel, fontsize=fontsize)
    plt.gca().xaxis.labelpad = labelpad


def plot_figure_performance_with_limited_data(spearmanr_basic, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, ylim=(0.5, 0.95), save_file=None):
    """Plots a figure comparing performances of the SL, MPL and est methods using limited (subsampled) data."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.4
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    truncate = truncate_list[truncate_index]
    plot_data = [np.mean(spearmanr_basic[truncate_index, :, :, :, 0], axis=(0, 1)),
                 np.mean(spearmanr_basic[truncate_index, :, :, 0, :], axis=(0, 1))]
    plot_x = [sample_list, record_list]
    yticks = np.linspace(0.5, 0.9, 5)

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        for j in range(NUM_BASIC_METHODS):
            plt.scatter(plot_x[i], data[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
            plt.plot(plot_x[i], data[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE)

        if i == 0:
            plt.xlabel('Number of samples drawn\nat each generation', fontsize=SIZELABEL)
            plt.xscale('log')
            plt.ylabel(r"Spearman's $\rho$", fontsize=SIZELABEL)
            # plt.xticks(ticks=sample_list, labels=[1000, r'500', r'100', r'50', r'10'])
            plt.xticks(ticks=sample_list, labels=sample_list, rotation=-90)
            plt.legend(fontsize=SIZELABEL, frameon=False)
        if i == 1:
            plt.xlabel('Time intervals between sampling\n(generation)', fontsize=SIZELABEL)
            plt.xticks(ticks=record_list, labels=record_list)
            ax.set_yticklabels([])

        at_left = (i == 0)
        ax.set_yticks(yticks)
        plt.ylim(ylim)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_linear_shrinkage(spearmanr_basic, spearmanr_linear, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, save_file=None):
    """Plots a figure showing performances with different strengths of linear shrinkage under limited sampling effects."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.9
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh), gridspec_kw={'width_ratios': [len(sample_list), len(record_list)]})

    mean_spearmanr_est = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 2], axis=(0, 1))
    mean_spearmanr_linear = np.mean(spearmanr_linear[truncate_index], axis=(0, 1))
    # We want matrices of dimension LINEAR x SAMPLE and LINEAR x RECORD.
    # mean_spearmanr_linear has dimension SAMPLE x RECORD x LINEAR hence needs to be transposed.
    matrix_sample = np.concatenate((mean_spearmanr_est[np.newaxis, :, 0], mean_spearmanr_linear[:, 0, :].T), axis=0)
    matrix_record = np.concatenate((mean_spearmanr_est[np.newaxis, 0, :], mean_spearmanr_linear[0, :, :].T), axis=0)

    plot_data = [matrix_sample, matrix_record]
    xlabels = ['Number of samples drawn\nat each generation', 'Time intervals between sampling\n(generation)']
    xticklabels = [sample_list, record_list]
    xticks = [ticks_for_heatmap(len(_)) for _ in xticklabels]
    ylabel = r'$\gamma$'
    yticklabels = ['1(est)'] + [str(gamma) for gamma in linear_list]
    yticks = ticks_for_heatmap(len(yticklabels))

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS)

        at_left = (i == 0)
        plt.xlabel(xlabels[i], fontsize=SIZELABEL)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL, rotation=0, labelpad=-3)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        plt.xticks(ticks=xticks[i], labels=xticklabels[i])
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def ticks_for_heatmap(length):
    """Returns ticks of a certain length, with each tick at position (i + 0.5)."""
    return np.arange(0.5, length + 0.5, 1)


def plot_figure_performance_with_nonlinear_shrinkage(spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=3, gamma_selected=0, save_file=None):
    """Plots a figure showing performance with non-lienar shrinkage on correlation matrix, using a particular loss and a particular gamma, under limited sampling effects."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    fig= plt.figure(figsize=(w, goldh))

    matrix = np.mean(spearmanr_dcorr[truncate_index, :, :, :, :, loss_selected, gamma_selected], axis=(0, 1))
    sns.heatmap(matrix, **DEF_HEATMAP_KWARGS)

    plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list, rotation=0)
    plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
    plt.ylabel('Number of samples drawn\nat each generation', fontsize=SIZELABEL)
    plt.xlabel(f"Time intervals between sampling\n(generation)", fontsize=SIZELABEL)
    plt.tick_params(**DEF_TICKPROPS_HEATMAP)

    plt.subplots_adjust(bottom=0.3, left=0.33, right=0.65, top=0.9)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_multiple_replicates(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_selected=1,  loss_selected=3, gamma_selected=0, plot_regularization=True, save_file=None, ylim=(0.53, 0.97)):
    """Plots a figure comparing performances of all methods (SL, est, MPL, linear shrinkage, non-linear shrinkage) using replicates individually, and combining multiple replicates for inference."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.4
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    plot_data_basic = [np.mean(spearmanr_basic[:, :, :, p, q], axis=(1, 2)),
                       np.mean(spearmanr_basic_combine[:, :, p, q], axis=(1))]
    plot_data_linear = [np.mean(spearmanr_linear[:, :, :, p, q, linear_selected], axis=(1, 2)),
                        np.mean(spearmanr_linear_combine[:, :, p, q, linear_selected], axis=(1))]
    plot_data_dcorr = [np.mean(spearmanr_dcorr[:, :, :, p, q, loss_selected, gamma_selected], axis=(1, 2)),
                       np.mean(spearmanr_dcorr_combine[:, :, p, q, loss_selected, gamma_selected], axis=(1))]
    yticks = np.linspace(0.55, 0.95, 9)
    yticklabels = ['%.2f' % (_) for _ in yticks]

    for i, data_basic in enumerate(plot_data_basic):
        if plot_regularization:
            data_linear, data_dcorr = plot_data_linear[i], plot_data_dcorr[i]
        plt.sca(axes[i])
        ax = axes[i]
        for j in range(3):
            plt.scatter(truncate_list, data_basic[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
            plt.plot(truncate_list, data_basic[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE)
        if plot_regularization:
            plt.scatter(truncate_list, data_linear, s=SMALLSIZEDOT, marker=MARKER_LINEAR, color=COLOR_LINEAR, label=LABEL_LINEAR)
            plt.plot(truncate_list, data_linear, color=COLOR_LINEAR, linewidth=SIZELINE)
            plt.scatter(truncate_list, data_dcorr, color=COLOR_DCORR, label=LABEL_DCORR, marker=MARKER_DCORR, s=SMALLSIZEDOT)
            plt.plot(truncate_list, data_dcorr, color=COLOR_DCORR, linewidth=SIZELINE)

        at_left = (i == 0)
        plot_legend = (i == 1)
        plt.ylabel(r"Spearman's $\rho$" if at_left else "", fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        plt.ylim(ylim)
        if plot_legend:
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Generations of data used"
    add_shared_xlabel(fig, xlabel, labelpad=-1)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_selection(selections, alpha=0.7, save_file=None):
    """Plots a figure showing 10 sets of selection coefficients used in simulations."""

    w = DOUBLE_COLUMN
    goldh = w / 1.4
    nRow, nCol = 2, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    for s, selection in enumerate(selections):
        plt.sca(axes[s // nCol, s % nCol])
        ax = plt.gca()
        plt.hist([_ for _ in selection if _ > 0], color=C_BEN_LT, bins=11, range=(-0.05, 0.05), alpha=alpha, label="Beneficial")
        plt.hist([_ for _ in selection if _ == 0], color=C_NEU_LT, bins=11, range=(-0.05, 0.05), alpha=alpha, label="Neutral")
        plt.hist([_ for _ in selection if _ < 0], color=C_DEL_LT, bins=11, range=(-0.05, 0.05), alpha=alpha, label="Deleterious")

        at_top_middle = (s == 2)
        at_top = (s < 5)
        at_left = (s == 0 or s == 5)
        if at_top_middle:
            plt.legend(fontsize=SIZELEGEND)
        if at_top:
            plt.ylim(0, 42)
            ax.set_xticklabels([])
        else:
            plt.ylim(0, 20)
            ax.set_yticks([0, 5, 10, 15, 20])
        if not at_left:
            ax.set_yticklabels([])
        plt.title(f'Set {s + 1}', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)

    xlabel = "Selection coefficients"
    add_shared_xlabel(fig, xlabel, labelpad=-1)

    plt.subplots_adjust(hspace=0.3)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_different_time_window(spearmanr_basic_window, spearmanr_linear_window, spearmanr_dcorr_window, window_list=WINDOW, sample_list=SAMPLE, record_list=RECORD, ylim=(0.63, 0.95), save_file=None):
    """Plots a figure showing performances with different choices of time window."""

    w = DOUBLE_COLUMN
    goldh = w / 1.4
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    max_p, max_q = len(sample_list) - 1, len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        # SL and MPL
        for j in range(2):
            plt.plot(range(len(window_list)), [np.mean(spearmanr_basic_window[0, :, :, p, q, j])] * len(window_list),
                     linewidth=SIZELINE, linestyle='dashed', color=COLOR_BASIC[j], label=LABEL_BASIC[j])
        # est
        j = 2
        plt.plot(range(len(window_list)), np.mean(spearmanr_basic_window[:, :, :, p, q, j], axis=(1, 2)), linewidth=SIZELINE,
                 marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        plt.plot(range(len(window_list)), np.mean(spearmanr_linear_window[:, :, :, p, q], axis=(1, 2)), linewidth=SIZELINE,
                 marker=MARKER_LINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_LINEAR, label=LABEL_LINEAR)

        plt.plot(range(len(window_list)), np.mean(spearmanr_dcorr_window[:, :, :,p, q], axis=(1, 2)), linewidth=SIZELINE,
                 marker=MARKER_DCORR, markersize=SMALLSIZEDOT - 2, color=COLOR_DCORR, label=LABEL_DCORR)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=np.arange(len(window_list)), labels=window_list if at_bottom else [])
        plt.xlabel('window' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        if not at_left:
            ax.set_yticklabels([])
        plt.ylabel(r"Spearman's $\rho$" if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=True)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_without_calibration(spearmanr_basic, error_basic, spearmanr_cov, error_cov, truncate_list=TRUNCATE, save_file=None):
    """Plots a figure showing performances with and without calibration when estimating covariance."""

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.4
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    p, q = 0, 0
    # select out est and est(uncalibrated) from all basic methods, and average over all simulations
    plot_data = [spearmanr_basic[:, :, :, p, q, 2:4], error_basic[:, :, :, p, q, 2:4], spearmanr_cov, error_cov]
    plot_data = [np.mean(_, axis=(1, 2)) for _ in plot_data]
    ylabels = [r"Spearman's $\rho$", r"Error of inferred selections", r"Spearman's $\rho$", r"Error of estimated covariance"]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()
        for j in range(2, 4):
            plt.plot(truncate_list, data[:, j - 2],
                     color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 3,
                     linewidth=SIZELINE, linestyle='dashed' if j == 3 else 'solid')

        at_bottom = (i == 2 or i == 3)
        plt.xticks(ticks=truncate_list, labels=truncate_list if at_bottom else [])
        plt.xlabel("Generations of data used" if at_bottom else "", fontsize=SIZELABEL)
        plt.ylabel(ylabels[i], fontsize=SIZELABEL)
        if i == 2:
            yticks = np.arange(0.93, 0.96, 0.01)
            plt.yticks(ticks=yticks, labels=yticks)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f'))
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_different_loss_gamma(spearmanr_dcorr, truncate_index=5, ylim=(0.78, 0.88), alpha=0.5, sample_list=SAMPLE, record_list=RECORD, loss_list=LOSS, gamma_list=GAMMA, save_file=None):
    """Plots a figure comparing performances of non-linear shrinkage on correlation matrix using different loss functions & gamma."""

    w = DOUBLE_COLUMN
    goldh = w / 1.4
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    max_p, max_q = len(sample_list) - 1, len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]
    xticklabels=[r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'0.2']
    xlabel = r'$\gamma \: \: (5 \times) $'
    yticks = np.arange(ylim[0], ylim[1] + 0.02, 0.02)
    yticklabels = ['%.2f' % (_) for _ in yticks]

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()
        for l in range(len(loss_list)):
            plt.plot(np.log(gamma_list), np.mean(spearmanr_dcorr[truncate_index, :, :, p, q, l, :], axis=(0, 1)),
                     color=COLOR_LOSS[l], linestyle=LINESTYLE_LOSS[l], linewidth=SIZELINE,
                     marker=MARKER_LOSS[l], markersize=SMALLSIZEDOT - 3,
                     label=LABEL_LOSS[l] if (i == 0 and l < 5 or i == 1 and l >= 5) else None, alpha=0.5)

        at_top = (i == 0 or i == 1)
        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=np.log(gamma_list), labels=xticklabels if at_bottom else [])
        plt.xlabel(xlabel if at_bottom else "", fontsize=SIZELABEL)
        plt.ylim(ylim)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [])
        plt.ylabel(r"Spearman's $\rho$" if at_left else "", fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if at_top:
            plt.legend(fontsize=SIZELEGEND, frameon=False, ncol=1)
        plt.title(f'sample={sample_list[p]}, interval={record_list[q]}', fontsize=SIZELABEL)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
