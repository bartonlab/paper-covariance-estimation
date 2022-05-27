"""
This Python file contains code for generating plots in:
    Estimating genetic linkage and selection from allele frequency trajectories
    Yunxiao Li, and
    John P. Barton (john.barton@ucr.edu)

Additional code to pre-process the data and pass it to these plotting
routines is contained in the Jupyter notebook `figures.ipynb`.
"""

#############  PACKAGES  #############

import numpy as np

from copy import deepcopy

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from scipy import stats

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
LINEAR = [1, 5, 10, 20, 50, 100, 300]  # linear shrinkage strengths that we tested
GAMMA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # nonlinear shrinkage parameter choices that we tested
TRUNCATE = [200, 300, 400, 500, 600, 700]  # truncate options to test performance using different lengths of data
WINDOW = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 160]  # window choices that we tested
# loss functions for nonlinear shrinkage that we tested
LOSS = ['Fro | $\hat{R}-R$', 'Fro | $\hat{R}^{-1}-R^{-1}$', 'Fro | $R^{-1}\hat{R}-I$',
        'Fro | $\hat{R}^{-1}R-I$', 'Fro | $\hat{R}^{-1/2}R\hat{R}^{-1/2}-I$',
        'Nuc | $\hat{R}-R$', 'Nuc | $\hat{R}^{-1}-R^{-1}$', 'Nuc | $R^{-1}\hat{R}-I$',
        'Nuc | $\hat{R}^{-1}R-I$', 'Nuc | $\hat{R}^{-1/2}R\hat{R}^{-1/2}-I$']

# Standard color scheme

BKCOLOR  = '#252525'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'

GREY_TEXT = '\033[38;5;07m'
CMAP = matplotlib.cm.get_cmap('GnBu')
CMAP_COLORS = [CMAP(i) for i in np.arange(0, 1.1, 0.1)]
TEXT_COLOR_CODES = [93, 92, 91, 90, 89, 88, 52]
TEXT_COLORS = [u"\u001b[38;5;" + str(code) + "m" for code in TEXT_COLOR_CODES]

COLOR_BASIC = ['blue', 'orange', 'grey', 'grey']
MARKER_BASIC = ['o', '*', 'v', '^']
LABEL_BASIC = ['SL', 'MPL', 'Est', 'Est (Unnormalized)']
COLOR_LINEAR = 'red'
MARKER_LINEAR = '<'
LABEL_LINEAR = 'Linear-cov'
COLOR_NONLINEAR = 'green'
MARKER_NONLINEAR = '>'
LABEL_NONLINEAR = 'Nonlinear-corr'
LINESTYLE_LOSS = ['solid'] * 5 + ['dashed'] * 5
COLOR_LOSS = ['red', 'orange', 'blue', 'green', 'grey'] * 2
MARKER_LOSS = ['.', 's', 'p', 'v', 'd'] * 2
LABEL_LOSS = LOSS
COLOR_VARIANCE = '#F16913'
DIC_PERF_COV_KEYS = ['spearmanr_cov_calibrated', 'error_cov_calibrated',
                     'spearmanr_cov_uncalibrated', 'error_cov_uncalibrated',
                     'spearmanr_cov_calibrated_linear', 'error_cov_calibrated_linear',
                     'spearmanr_cov_uncalibrated_linear', 'error_cov_uncalibrated_linear',
                     'spearmanr_cov_calibrated_nonlinear', 'error_cov_calibrated_nonlinear',
                     'spearmanr_cov_uncalibrated_nonlinear', 'error_cov_uncalibrated_nonlinear']

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
    'dpi'         : 1000,
    # 'bbox_inches' : 'tight',
    'pad_inches'  : 0.05,
}

DEF_SUBLABELPROPS = {
    'family'  : FONTFAMILY,
    'size'    : SIZESUBLABEL+1,
    'weight'  : 'bold',
    'ha'      : 'center',
    'va'      : 'center',
    'color'   : 'k',
    'clip_on' : False,
}

DEF_SUBLABELPOSITION_1x2 = {
    'x'       : -0.15,
    'y'       : 1.05,
}

DEF_SUBLABELPOSITION_2x1 = {
    'x'       : -0.175,
    'y'       : 1.05,
}

DEF_SUBLABELPOSITION_2x2 = {
    'x'       : -0.07,
    'y'       : 1.07,
}

DEF_SUBPLOTS_ADJUST_1x2 = {
    'wspace'  : 0.4,
}

DEF_SUBPLOTS_ADJUST_2x2 = {
    'left'    : 0.076,
    'bottom'  : 0.067,
    'right'   : 0.998,
    'top'     : 0.949,
    'wspace'  : 0.21,
    'hspace'  : 0.3,
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
    'right'     : False,
}

DEF_TICKPROPS_TOP = {
    'length'    : TICKLENGTH,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : False,
    'left'      : True,
    'top'       : True,
    'right'     : False,
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
    'right'     : False,
    'pad'       : 2,
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

FIG4_LEFT, FIG4_BOTTOM, FIG4_RIGHT, FIG4_TOP = 0.14, 0.223, 0.965, 0.935
GOLD_RATIO = 1.4
YLABEL_SPEARMANR = "Rank correlation between true and inferred\nselection coefficients" + r" (Spearman’s $\rho$)"
YLABEL_SPEARMANR_COVARIANCE = "Rank correlation between true and inferred\ncovariances" + r" (Spearman’s $\rho$)"

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42}
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update(PARAMS)

############# HELPER  FUNCTIONS #############

def add_shared_label(fig, xlabel=None, ylabel=None, xlabelpad=-3, ylabelpad=-3, fontsize=SIZELABEL):
    """Adds a xlabel shared by subplots."""

    # add a big axes, hide frame in order to generate a shared x-axis label
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axes
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.grid(False)
    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=fontsize)
        plt.gca().xaxis.labelpad = xlabelpad
    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=fontsize)
        plt.gca().yaxis.labelpad = ylabelpad


def ticks_for_heatmap(length):
    """Returns ticks of a certain length, with each tick at position (i + 0.5)."""
    return np.arange(0.5, length + 0.5, 1)


def get_color_by_value_percentage(value, vmin=0, vmax=1, colors=CMAP_COLORS):
    if value <= vmin:
        index = 0
    elif value >= vmax:
        index = -1
    else:
        index = int((value - vmin) / (vmax - vmin) * len(colors))
    return colors[index]


############# PLOTTING  FUNCTIONS #############

############# MAIN FIGURES #############

def plot_figure_traj_example(traj, selection, alpha=0.85, save_file=None):
    """Plots a figure of allele frequency trajectories; Color trajectories according to their selections."""

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig = plt.figure(figsize=(w, goldh / 2))
    traj_list = []
    for l in range(len(traj[0])):
        traj_list.append(list(traj[:, l]))
    sorted_pair = sorted(zip(selection, traj_list), key=lambda x: np.mean(x[1]), reverse=True)
    selection = [i for i, _ in sorted_pair]
    traj = np.array([i for _, i in sorted_pair]).T
    ax = plt.gca()
    first_ben = first_neu = first_del = True
    for l in range(len(traj[0])):
        label = None
        if selection[l] > 0:
            color = C_BEN_LT
            if first_ben:
                label = 'Beneficial'
                first_ben = False
        elif selection[l] == 0:
            color = C_NEU_LT
            if first_neu:
                label = 'Neutral'
                first_neu = False
        else:
            color = C_DEL_LT
            if first_del:
                label = 'Deleterious'
                first_del = False
        if label is not None:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, label=label)
        else:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE)

    plt.ylabel("Mutant allele frequency    ", fontsize=SIZELABEL)
    plt.xlabel("Generation", fontsize=SIZELABEL)
    plt.ylim(-0.02, 1.12)
    plt.xlim(-10, 710)
    plt.yticks(np.linspace(0, 1, 6))
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # plt.legend(loc=0, frameon=False, fontsize=SIZELEGEND)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.07), fontsize=SIZELEGEND, frameon=False, ncol=3)
    # ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=SIZELEGEND)
    plt.subplots_adjust(0.11, 0.25, 0.99, 0.98)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_cov_example(true_cov, est_cov, save_file=None):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    w = DOUBLE_COLUMN
    goldh = w / 2.5
    fig, axes = plt.subplots(1, 3, figsize=(w, goldh))

    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = [r"True covariance matrix, $C$", r"Estimated covariance matrix, $\hat{C}$", "Error of estimated covariance matrix, $\hat{C}-C$"]
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
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"))
        plt.title(title_list[i], fontsize=SIZELABEL)
        if i == 0:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
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


def plot_figure_traj_cov_example(traj, selection, true_cov, est_cov, alpha=0.85, save_file=None):
    w = SINGLE_COLUMN
    # goldh = w / GOLD_RATIO
    fig = plt.figure(figsize=(w, w * 4 / 3))
    nRow = 17
    nCol = nRow * 2
    num_divider, divider_rowspan = 1, 1
    ax2_ax1_rowspan_ratio = 3 / 2
    ax1_rowspan = int((nRow - num_divider * divider_rowspan) / (2 * ax2_ax1_rowspan_ratio + 1))
    ax2_rowspan = int(ax1_rowspan * ax2_ax1_rowspan_ratio)
    ax2_colspan = nCol//2 - 1
    ax1 = plt.subplot2grid((nRow, nCol), (0, 0), rowspan=ax1_rowspan, colspan=nCol)
    ax2 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, 0), rowspan=ax2_rowspan, colspan=ax2_colspan)
    ax3 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + divider_rowspan, nCol//2 + 1), rowspan=ax2_rowspan, colspan=ax2_colspan)
    ax4 = plt.subplot2grid((nRow, nCol), (ax1_rowspan + ax2_rowspan + divider_rowspan, 0), rowspan=ax2_rowspan, colspan=ax2_colspan)

    heatmaps = [ax2, ax3, ax4]
    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = [r"True covariance matrix, $C$", r"Estimated covariance matrix, $\hat{C}$", "Error of estimated covariance matrix, $\hat{C}-C$"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    # cbar_ax = fig.add_axes(rect=[.91, .56, .015, .4])
    cbar_ax = fig.add_axes(rect=[.59, .0285, .02, 0.3])
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    matrix_labels = [1] + [10 * i for i in range(1, 6)]
    matrix_ticks = [l - 0.5 for l in matrix_labels]

    plt.sca(ax1)
    ax = plt.gca()
    traj_list = []
    for l in range(len(traj[0])):
        traj_list.append(list(traj[:, l]))
    sorted_pair = sorted(zip(selection, traj_list), key=lambda x: np.mean(x[1]), reverse=True)
    selection = [i for i, _ in sorted_pair]
    traj = np.array([i for _, i in sorted_pair]).T
    first_ben = first_neu = first_del = True
    for l in range(len(traj[0])):
        label = None
        if selection[l] > 0:
            color = C_BEN_LT
            if first_ben:
                label = 'Beneficial'
                first_ben = False
        elif selection[l] == 0:
            color = C_NEU_LT
            if first_neu:
                label = 'Neutral'
                first_neu = False
        else:
            color = C_DEL_LT
            if first_del:
                label = 'Deleterious'
                first_del = False
        if label is not None:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, label=label)
        else:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE)

    plt.ylabel("Mutant allele\nfrequency", fontsize=SIZELABEL)
    plt.xlabel("Generation", fontsize=SIZELABEL)
    plt.ylim(-0.02, 1.12)
    plt.xlim(-10, 710)
    plt.yticks(np.linspace(0, 1, 6))
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # plt.legend(loc=0, frameon=False, fontsize=SIZELEGEND)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=SIZELEGEND, frameon=False, ncol=3)
    plt.text(x=-0.1, y=1, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    for i, cov in enumerate(cov_list):
        plt.sca(heatmaps[i])
        ax = plt.gca()
        plot_cbar = (i == 2)
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="vertical"))
        # plt.title(title_list[i], fontsize=SIZELABEL)
        plt.xlabel(title_list[i], fontsize=SIZELABEL, labelpad=3 if i == 0 else 2)
        if i == 0 or i == 2:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        # plt.xticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL, rotation=0)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.text(x=-0.06 if i == 1 else -0.23, y=0.975, s=SUBLABELS[i + 1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(0.135, 0.01, 0.988, 0.985)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_estimation(true_cov, est_cov, error, error_uncalibrated, error_window, window_list=WINDOW, plot_hist_for_windows=False, windows_for_hist=[0, 5, 20, 40], alpha=0.7, save_file=None):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    w = DOUBLE_COLUMN
    goldh = w / 1.5
    fig = plt.figure(figsize=(w, goldh))
    hist_ylim = (0, 0.16)
    hist_yticks = np.linspace(0, 0.15, 6)

    nCol = 48
    heatmaps = [plt.subplot2grid((2, nCol), (0, 0), colspan=nCol//3), plt.subplot2grid((2, nCol), (0, nCol//3), colspan=nCol//3), plt.subplot2grid((2, nCol), (0, nCol//3 * 2), colspan=nCol//3)]

    ax2 = plt.subplot2grid((2, nCol), (1, 2), colspan=nCol//2 - 4)
    ax3 = plt.subplot2grid((2, nCol), (1, nCol//2 + 4), colspan=nCol//2 - 4)

    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = [r"True covariance matrix, $C$", r"Estimated covariance matrix, $\hat{C}$", "Error of estimated covariance matrix, $\hat{C}-C$"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    cbar_ax = fig.add_axes(rect=[.91, .56, .015, .4])  # color bar on the right
    # cbar_ax = fig.add_axes(rect=[.128, .175, .773, .02])  # color bar on the bottom, rect=[left, bottom, width, height]
    cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    matrix_labels = [1] + [10 * i for i in range(1, 6)]
    matrix_ticks = [l - 0.5 for l in matrix_labels]
    for i, cov in enumerate(cov_list):
        plt.sca(heatmaps[i])
        ax = plt.gca()
        plot_cbar = (i == 0)
        sns.heatmap(cov, center=0, vmin=vmin - 5, vmax=vmax + 5, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="vertical"))
        plt.title(title_list[i], fontsize=SIZELABEL)
        if i == 0:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        # plt.xticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL, rotation=0)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            cbar_ax.tick_params(labelsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        if i == 0:
            plt.text(x=-0.18, y=1.05, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(ax2)
    ax = plt.gca()
    abs_range = (-15, 15)
    bins = 40
    plt.hist(error, weights=np.ones(len(error)) / len(error), range=abs_range, bins=bins, alpha=alpha, label="Error of normalized estimates")
    plt.hist(error_uncalibrated, weights=np.ones(len(error_uncalibrated)) / len(error_uncalibrated), range=abs_range, bins=bins, alpha=alpha, label="Error of unnormalized estimates")
    # plt.title("Error of estimation", fontsize=SIZELABEL)
    plt.legend(fontsize=SIZELABEL, frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    plt.ylim(hist_ylim)
    plt.yticks(hist_yticks)
    plt.ylabel("Frequency", fontsize=SIZELABEL)
    plt.xticks(fontsize=SIZELABEL)
    plt.xlabel("Error of estimated covariances, " + r'$\hat{C}_{ij} - C_{ij}$', fontsize=SIZELABEL)
    plt.text(x=-0.208, y=1.05, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.sca(ax3)
    ax = plt.gca()
    abs_range = (-15, 15)
    bins = 40
    if plot_hist_for_windows:
        for i, window in enumerate(window_list):
            if window in window_selected:
                plt.hist(error_window[i], weights=np.ones(len(error_window[i])) / len(error_window[i]), range=abs_range, bins=bins, alpha=alpha, label=f'window={window}')
        plt.title("Error of estimating covariances", fontsize=SIZELABEL)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
        plt.yticks(ticks=[], labels=[])
        plt.xticks(fontsize=SIZELABEL)
        plt.legend(fontsize=SIZELABEL, frameon=False)
        plt.ylim(hist_ylim)
    else:
        xticks = np.arange(0, len(window_list))
        plt.plot(xticks, [np.mean(np.absolute(error_window_i)) for error_window_i in error_window], marker='.', markersize=SMALLSIZEDOT - 2, linewidth=SIZELINE)
        plt.yticks(fontsize=SIZELABEL)
        plt.ylabel("MAE of estimating covariances", fontsize=SIZELABEL)
        # plt.ylim(1.3, 3.7)
        # plt.yticks(np.linspace(1.5, 3.5, 5))
        plt.ylim(1.55, 3.35)
        plt.yticks(np.linspace(1.7, 3.2, 6))
        plt.xticks(ticks=xticks, labels=window_list, fontsize=SIZELABEL)
        plt.xlabel("Window", fontsize=SIZELABEL)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.subplots_adjust(0.05, 0.08, 0.9, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_estimation_normalization_window(true_cov, est_cov, est_cov_uncalibrated, error_window, error_window_uncalibrated=None, window_list=WINDOW, alpha=0.7, single_column=False, save_file=None):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    w = SINGLE_COLUMN if single_column else DOUBLE_COLUMN
    fig = plt.figure(figsize=(w, w / 3))
    hist_ylim = (0, 0.16)
    hist_yticks = np.linspace(0, 0.15, 6)

    nCol = 48
    ax2 = plt.subplot2grid((1, nCol), (0, 2), colspan=nCol//2 - 4)
    ax3 = plt.subplot2grid((1, nCol), (0, nCol//2 + 4), colspan=nCol//2 - 4)

    plt.sca(ax2)
    ax = plt.gca()
    abs_range = (-15, 15)
    bins = 40
    error = np.ndarray.flatten(est_cov - true_cov)
    error_uncalibrated = np.ndarray.flatten(est_cov_uncalibrated - true_cov)
    plt.hist(error, weights=np.ones(len(error)) / len(error), range=abs_range, bins=bins, alpha=alpha, label="Error of normalized estimates")
    plt.hist(error_uncalibrated, weights=np.ones(len(error_uncalibrated)) / len(error_uncalibrated), range=abs_range, bins=bins, alpha=alpha, label="Error of unnormalized estimates")
    # plt.title("Error of estimation", fontsize=SIZELABEL)
    plt.legend(fontsize=SIZELABEL, frameon=False)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1, decimals=0))
    plt.ylim(hist_ylim)
    plt.yticks(hist_yticks)
    plt.ylabel("Frequency", fontsize=SIZELABEL)
    plt.xticks(fontsize=SIZELABEL)
    plt.xlabel("Error of estimated covariances, " + r'$\hat{C}_{ij} - C_{ij}$', fontsize=SIZELABEL)
    plt.text(x=-0.14, y=1, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.sca(ax3)
    ax = plt.gca()
    abs_range = (-15, 15)
    bins = 40
    xticks = np.arange(0, len(window_list))
    plt.plot(xticks, [np.mean(_) for _ in error_window], marker='.', markersize=SMALLSIZEDOT - 2, linewidth=SIZELINE)
    if error_window_uncalibrated is not None:
        plt.plot(xticks, [np.mean(_) for _ in error_window_uncalibrated], marker='.', markersize=SMALLSIZEDOT - 2, linewidth=SIZELINE)
    plt.yticks(fontsize=SIZELABEL)
    plt.ylabel("MAE of estimating covariances", fontsize=SIZELABEL)
    plt.ylim(1.55, 3.35)
    # plt.ylim(1.55, 5)
    plt.yticks(np.linspace(1.7, 3.2, 6))
    plt.xticks(ticks=xticks, labels=window_list, fontsize=SIZELABEL)
    plt.xlabel("Window", fontsize=SIZELABEL)
    plt.text(x=-0.14, y=1, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.subplots_adjust(0.05, 0.152, 0.9, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_ample_data(spearmanr_basic, error_basic, truncate_list=TRUNCATE, plotUnnormalized=False, arrange_vertically=False, save_file=None):
    """Plots a figure comparing performances of the SL, MPL and est methods using complete data."""

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    if arrange_vertically:
        fig, axes = plt.subplots(2, 1, figsize=(w, goldh))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0  # ample data
    plot_data = [np.mean(spearmanr_basic[:, :, :, p, q], axis=(1, 2)), np.mean(error_basic[:, :, :, p, q], axis=(1, 2))]
    ylabels = [YLABEL_SPEARMANR, 'MAE of inferred selection coefficients']
    # yticks = np.linspace(ylim[0], ylim[1] - 0.05, int(10 * (ylim[1] - ylim[0])) + 1)
    yticks = np.linspace(0.5, 0.9, 5)
    legend_loc = [4, 1]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        num_basic_methods = NUM_BASIC_METHODS if not plotUnnormalized else 4
        for j in range(num_basic_methods):
            linestyle = 'solid' if j < 3 else 'dashed'
            plt.scatter(truncate_list, data[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
            plt.plot(truncate_list, data[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE, linestyle=linestyle)

        if i == 0:
            plt.ylim(0.45, 0.95)
            ax.set_yticks(np.linspace(0.5, 0.9, 5))
            plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
            if arrange_vertically:
                ax.set_xticklabels([])
        elif i == 1:
            plt.ylim(0.002, 0.022)
            ax.set_yticks(np.linspace(0.004, 0.02, 5))
            # plt.ylim(0.0045, 0.02)
            # ax.set_yticks(np.linspace(0.006, 0.018, 5))
            # plt.ylim(0.0055, 0.0205)
            # ax.set_yticks(np.linspace(0.007, 0.019, 5))
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
            ax.tick_params(axis='y', which='major', pad=1.5)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if arrange_vertically:
            plt.text(**DEF_SUBLABELPOSITION_2x1, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Generations of data used"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    if arrange_vertically:
        plt.subplots_adjust(hspace=0.2)
    else:
        plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_limited_data(spearmanr_basic, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, ylim=(0.45, 0.95), save_file=None):
    """Plots a figure comparing performances of the SL, MPL and est methods using limited (subsampled) data."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    truncate = truncate_list[truncate_index]
    plot_data = [np.mean(spearmanr_basic[truncate_index, :, :, :, 0], axis=(0, 1)),
                 np.mean(spearmanr_basic[truncate_index, :, :, 0, :], axis=(0, 1))]
    plot_x = [sample_list, record_list]
    yticks = np.linspace(0.5, 0.9, 5)
    yticklabels = ['%.1f' % (_) for _ in yticks]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        for j in range(NUM_BASIC_METHODS):
            plt.scatter(plot_x[i], data[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
            plt.plot(plot_x[i], data[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE)
        at_left = (i == 0)
        if at_left:
            plt.xlabel('Number of samples drawn\nat each generation', fontsize=SIZELABEL)
            plt.xscale('log')
            plt.ylabel(YLABEL_SPEARMANR, fontsize=SIZELABEL)
            # plt.xticks(ticks=sample_list, labels=[1000, r'500', r'100', r'50', r'10'])
            plt.xticks(ticks=sample_list, labels=sample_list, rotation=-90)
            plt.legend(fontsize=SIZELABEL, frameon=False)
        else:
            plt.xlabel('Time intervals between sampling\n' + r'$\Delta g$' + ' (generation)', fontsize=SIZELABEL)
            plt.xticks(ticks=record_list, labels=record_list)

        plt.ylim(ylim)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(FIG4_LEFT, FIG4_BOTTOM, FIG4_RIGHT, FIG4_TOP, **DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_linear_shrinkage(spearmanr_basic, spearmanr_linear, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, save_file=None):
    """Plots a figure showing performances with different strengths of linear shrinkage under limited sampling effects."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 1.8
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh), gridspec_kw={'width_ratios': [len(sample_list), len(record_list)]})

    mean_spearmanr_est = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 2], axis=(0, 1))
    mean_spearmanr_linear = np.mean(spearmanr_linear[truncate_index], axis=(0, 1))
    # We want matrices of dimension LINEAR x SAMPLE and LINEAR x RECORD.
    # mean_spearmanr_linear has dimension SAMPLE x RECORD x LINEAR hence needs to be transposed.
    matrix_sample = np.concatenate((mean_spearmanr_est[np.newaxis, :, 0], mean_spearmanr_linear[:, 0, :].T), axis=0)
    matrix_record = np.concatenate((mean_spearmanr_est[np.newaxis, 0, :], mean_spearmanr_linear[0, :, :].T), axis=0)

    plot_data = [matrix_sample, matrix_record]
    xlabels = ['Number of samples drawn\nat each generation', 'Time intervals between sampling\n' + r'$\Delta g$' + ' (generation)']
    xticklabels = [sample_list, record_list]
    xticks = [ticks_for_heatmap(len(_)) for _ in xticklabels]
    ylabel = r'$\frac{\gamma}{\Delta g}$'
    # ylabel = r'$\gamma \: / \: \Delta g$'
    yticklabels = ['Est'] + [str(gamma) for gamma in linear_list]
    yticks = ticks_for_heatmap(len(yticklabels))

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS)

        at_left = (i == 0)
        plt.xlabel(xlabels[i], fontsize=SIZELABEL)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL * GOLD_RATIO if i == 0 else SIZELABEL, rotation=0, labelpad=5)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        plt.xticks(ticks=xticks[i], labels=xticklabels[i])
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.text(x=-0.12 if i == 0 else -0.1, y=1.05, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(0.11, 0.215, 0.945, 0.93, wspace=0.222)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


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
    plt.xlabel('Time intervals between sampling\n' + r'$\Delta g$' + ' (generation)', fontsize=SIZELABEL)
    plt.tick_params(**DEF_TICKPROPS_HEATMAP)

    plt.subplots_adjust(bottom=0.3, left=0.33, right=0.65, top=0.9)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_both_regularizations(spearmanr_linear, spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=2, loss_selected=3, gamma_selected=0, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    matrix_linear = np.mean(spearmanr_linear[truncate_index, :, :, :, :, linear_selected], axis=(0, 1))
    matrix_dcorr = np.mean(spearmanr_dcorr[truncate_index, :, :, :, :, loss_selected, gamma_selected], axis=(0, 1))

    plot_data = [matrix_linear, matrix_dcorr]
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    titles = [LABEL_LINEAR, LABEL_NONLINEAR]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS)

        at_left = (i == 0)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)
    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2, left=0.18, right=0.9)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_multiple_replicates(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_selected=2,  loss_selected=3, gamma_selected=0, plot_regularization=True, save_file=None, ylim=(0.45, 0.95), w_h_ratio = 1.55, alpha=0.75):
    """Plots a figure comparing performances of all methods (SL, est, MPL, linear shrinkage, nonlinear shrinkage) using replicates individually, and combining multiple replicates for inference."""

    # In def plot_figure_performance_with_limited_data(), it generates figure where the plotting area is of size (SINGLE_COLUMN * (FIG4_RIGHT - FIG4_LEFT), SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM))
    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / w_h_ratio
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    plot_data_basic = [np.mean(spearmanr_basic[:, :, :, p, q], axis=(1, 2)),
                       np.mean(spearmanr_basic_combine[:, :, -1, p, q], axis=(1))]
    plot_data_linear = [np.mean(spearmanr_linear[:, :, :, p, q, linear_selected], axis=(1, 2)),
                        np.mean(spearmanr_linear_combine[:, :, -1, p, q, linear_selected], axis=(1))]
    plot_data_dcorr = [np.mean(spearmanr_dcorr[:, :, :, p, q, loss_selected, gamma_selected], axis=(1, 2)),
                       np.mean(spearmanr_dcorr_combine[:, :, -1, p, q, loss_selected, gamma_selected], axis=(1))]
    yticks = np.linspace(0.5, 0.9, 5)
    yticklabels = ['%.1f' % (_) for _ in yticks]

    for i, data_basic in enumerate(plot_data_basic):
        if plot_regularization:
            data_linear, data_dcorr = plot_data_linear[i], plot_data_dcorr[i]
        plt.sca(axes[i])
        ax = axes[i]
        for j in range(3):
            plt.scatter(truncate_list, data_basic[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT, alpha=alpha)
            plt.plot(truncate_list, data_basic[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE, alpha=alpha)
        if plot_regularization:
            plt.scatter(truncate_list, data_linear, s=SMALLSIZEDOT, marker=MARKER_LINEAR, color=COLOR_LINEAR, label=LABEL_LINEAR, alpha=alpha)
            plt.plot(truncate_list, data_linear, color=COLOR_LINEAR, linewidth=SIZELINE, alpha=alpha)
            plt.scatter(truncate_list, data_dcorr, color=COLOR_NONLINEAR, label=LABEL_NONLINEAR, marker=MARKER_NONLINEAR, s=SMALLSIZEDOT, alpha=alpha)
            plt.plot(truncate_list, data_dcorr, color=COLOR_NONLINEAR, linewidth=SIZELINE, alpha=alpha)

        at_left = (i == 0)
        plot_legend = (i == 1)
        plt.ylabel(YLABEL_SPEARMANR if at_left else "", fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        plt.ylim(ylim)
        if plot_legend:
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Generations of data used"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    top = 0.93
    bottom = top - (FIG4_TOP - FIG4_BOTTOM) / GOLD_RATIO * w_h_ratio
    plt.subplots_adjust(FIG4_LEFT, bottom, FIG4_RIGHT, top, **DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_different_numbers_of_replicates(spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, num_basic=4, truncate_index=None, linear_selected=2, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=3, gamma_selected=0, p=1, min_t=None, max_t=None, max_q=None, ylim=[(0.45, 0.95), (0.45, 0.95)], alpha=0.75, double_column=False, save_file=None):
    """Plots a figure showing performances combining different numbers of replicates under limited sampling effects."""

    # In def plot_figure_performance_with_limited_data(), it generates figure where the plotting area is of size (SINGLE_COLUMN * (0.97- 0.115), SINGLE_COLUMN / GOLD_RATIO * (0.935 - 0.223))
    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    height = SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM)
    left, right = FIG4_LEFT, FIG4_RIGHT
    bottom, top = 0.0705, 0.958
    wspace, hspace = 0.4, 0.25
    goldh = height * (2 + hspace) / (top - bottom)

    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if truncate_index is not None:
        spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine = (spearmanr_basic_combine[truncate_index],
                                                                                      spearmanr_linear_combine[truncate_index],
                                                                                      spearmanr_dcorr_combine[truncate_index])
    # num_basic = 4
    mean_spearmanr_basic = np.mean(spearmanr_basic_combine, axis=(1))
    mean_spearmanr_linear = np.mean(spearmanr_linear_combine[:, :, :, :, :, linear_selected], axis=(1))
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr_combine[:, :, :, :, :, loss_selected, gamma_selected], axis=(1))
    number_replicates = np.arange(1, 21)
    xticks = np.array([1, 5, 10, 15, 20])
    if double_column:
        indices = np.arange(len(number_replicates))
    else:
        # indices = xticks - 1
        indices = np.arange(len(number_replicates))[::2]

    if max_t is None:
        max_t = len(truncate_list) - 1
    if min_t is None:
        min_t = 0
    if max_q is None:
        max_q = len(record_list) - 1
    # tq_list = [(0, 0), (max_t, 0), (0, max_q), (max_t, max_q)]
    tq_list = [(min_t, 0), (min_t, max_q), (max_t, 0), (max_t, max_q)]

    yticks = np.linspace(0.5, 0.9, 5)
    yticklabels = ['%.1f' % (_) for _ in yticks]

    for i, (t, q) in enumerate(tq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        for j in range(num_basic):
            plt.scatter(number_replicates[indices], mean_spearmanr_basic[t, :, p, q, j][indices], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT, alpha=alpha)
            plt.plot(number_replicates[indices], mean_spearmanr_basic[t, :, p, q, j][indices], color=COLOR_BASIC[j], linewidth=SIZELINE, alpha=alpha)
        plt.scatter(number_replicates[indices], mean_spearmanr_linear[t, :, p, q][indices], s=SMALLSIZEDOT, marker=MARKER_LINEAR, color=COLOR_LINEAR, label=LABEL_LINEAR, alpha=alpha)
        plt.plot(number_replicates[indices], mean_spearmanr_linear[t, :, p, q][indices], color=COLOR_LINEAR, linewidth=SIZELINE, alpha=alpha)
        plt.scatter(number_replicates[indices], mean_spearmanr_dcorr[t, :, p, q][indices], color=COLOR_NONLINEAR, label=LABEL_NONLINEAR, marker=MARKER_NONLINEAR, s=SMALLSIZEDOT, alpha=alpha)
        plt.plot(number_replicates[indices], mean_spearmanr_dcorr[t, :, p, q][indices], color=COLOR_NONLINEAR, linewidth=SIZELINE, alpha=alpha)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=xticks, labels=xticks if at_bottom else [])
        plt.xlabel('Number of replicates' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim[i // nCol])
        plt.ylabel(YLABEL_SPEARMANR if at_left else '', fontsize=SIZELABEL)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [], rotation=0)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 2:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        plt.text(x=-0.1, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"{truncate_list[t]} generations, " + r"$\Delta g$" + f"={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(left, bottom, right, top, wspace=wspace, hspace=hspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)

############# SUPPLEMENTARY FIGURES #############

def plot_supplementary_figure_selections(selections, ylim_top=(0, 42), ylim_bottom=(0, 20), range=(-0.05, 0.05), bins=11, alpha=0.7, save_file=None):
    """Plots a figure showing 10 sets of selection coefficients used in simulations."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    for s, selection in enumerate(selections):
        plt.sca(axes[s // nCol, s % nCol])
        ax = plt.gca()
        plt.hist([_ for _ in selection if _ > 0], color=C_BEN_LT, bins=bins, range=range, alpha=alpha, label="Beneficial")
        plt.hist([_ for _ in selection if _ == 0], color=C_NEU_LT, bins=bins, range=range, alpha=alpha, label="Neutral")
        plt.hist([_ for _ in selection if _ < 0], color=C_DEL_LT, bins=bins, range=range, alpha=alpha, label="Deleterious")

        at_top_middle = (s == 2)
        at_top = (s < 5)
        at_left = (s == 0 or s == 5)
        if at_top_middle:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        if at_top:
            plt.ylim(ylim_top)
            ax.set_xticklabels([])
        else:
            plt.ylim(ylim_bottom)
            if ylim_bottom == (0, 20):
                ax.set_yticks([0, 5, 10, 15, 20])
        if at_left:
            plt.ylabel('Count', fontsize=SIZELABEL)
        else:
            ax.set_yticklabels([])
        plt.title(f'Set {s + 1}', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)

    xlabel = "Selection coefficients"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.05, 0.06, 0.995, 0.967, hspace=0.2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_biplot_cov(true_cov, est_cov, linear_cov, nonlinear_cov, s=1, alpha=0.5, xticks_top=None, plot_xticks_top_on_top=False, xticks_bottom=None, figsize=None, save_file=None):
    w = DOUBLE_COLUMN #SLIDE_WIDTH
    nRow, nCol = 2, 3
    goldh = w * (nRow / nCol)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))
    L = len(true_cov)
    plot_data = [(true_cov, est_cov), (true_cov, linear_cov), (true_cov, nonlinear_cov)]
    inv_true_cov = np.linalg.inv(true_cov + np.identity(L))
    plot_data += [(inv_true_cov, np.linalg.inv(cov + np.identity(L))) for _, cov in plot_data]
    titles = ['Estimated v.s. true covariance matrix', 'Estimated & linearly-regularized v.s. true covariance matrix', 'Estimated & nonlinearly-regularized v.s. true covariance matrix', 'Estimated v.s. true inverse covariance matrix', 'Estimated & linearly-regularized v.s. true inverse covariance matrix', 'Estimated & nonlinearly-regularized v.s. true inverse covariance matrix']
    xlabels = ['Estimated covariance matrix', 'Estimated & linearly-regularized\ncovariance matrix', 'Estimated & nonlinearly-regularized\ncovariance matrix', 'Inverse of estimated\ncovariance matrix', 'Inverse of estimated &\nlinearly-regularized covariance matrix', 'Inverse of estimated &\nnonlinearly-regularized covariance matrix']
    ylabels = ['True covariance matrix', 'Inverse of true covariance matrix']
    min_values = [np.min([min(np.min(x), np.min(y)) for x, y in plot_data[:3]]), np.min([min(np.min(x), np.min(y)) for x, y in plot_data[3:]])]
    max_values = [np.max([max(np.max(x), np.max(y)) for x, y in plot_data[:3]]), np.max([max(np.max(x), np.max(y)) for x, y in plot_data[3:]])]
    for i, (y, x) in enumerate(plot_data):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        cov_x, cov_y = [x[i, j] for i in range(L) for j in range(i)], [y[i, j] for i in range(L) for j in range(i)]
        var_x, var_y = [x[i, i] for i in range(L)], [y[i, i] for i in range(L)]
        plt.scatter(var_x, var_y, s=s, alpha=0.8, label='Diagonal', color=COLOR_VARIANCE)
        # speamanr, _ = stats.spearmanr(x, y)
        plt.scatter(cov_x, cov_y, s=s, alpha=alpha, label='Off-diagonal')
        min_value = min_values[i//nCol]
        max_value = max_values[i//nCol]
        plt.plot([min_value, max_value], [min_value, max_value], linestyle='dashed', color='grey', linewidth=SIZELINE)
        scale = 1.1
        plt.xlim((min_value * scale, max_value * scale))
        plt.ylim((min_value * scale, max_value * scale))
        # plt.title(titles[i], fontsize=SIZELABEL)
        at_bottom = (i//nCol == nRow - 1)
        at_left = (i%nCol == 0)
        at_right = (i%nCol == nCol - 1)
        at_top = (i//nCol == 0)
        if at_left and at_top:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
            plt.text(x=-0.208, y=1.08, s=SUBLABELS[i//nCol], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        if at_left and at_bottom:
            plt.text(x=-0.208, y=1.05, s=SUBLABELS[i//nCol], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        if at_left:
            plt.ylabel(ylabels[i//nCol], fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[])
        plt.xlabel(xlabels[i], fontsize=SIZELABEL)
        if plot_xticks_top_on_top and at_top:
            ax.xaxis.tick_top()
            # ax.xaxis.set_label_position('top')
        if at_top and xticks_top is not None:
            plt.xticks(ticks=xticks_top, labels=xticks_top)
        if at_bottom and xticks_bottom is not None:
            plt.xticks(ticks=xticks_bottom, labels=xticks_bottom)
        if plot_xticks_top_on_top and at_top:
            ax.tick_params(**DEF_TICKPROPS_TOP)
        else:
            ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)

    plt.subplots_adjust(0.08, 0.1, 0.98, 0.95, hspace=0.3)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_without_normalization(spearmanr_basic, error_basic, spearmanr_cov_calibrated, error_cov_calibrated, spearmanr_cov_uncalibrated, error_cov_uncalibrated, truncate_list=TRUNCATE, save_file=None):
    """Plots a figure showing performances with and without calibration when estimating covariance."""

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    spearmanr_cov = np.stack((spearmanr_cov_calibrated[:, :, :, :, :, 0], spearmanr_cov_uncalibrated[:, :, :, :, :, 0]), axis=-1)
    error_cov = np.stack((error_cov_calibrated[:, :, :, :, :, 0], error_cov_uncalibrated[:, :, :, :, :, 0]), axis=-1)

    p, q = 0, 0
    # Normalize MAE error of estimated integrated covariances by the integration length (number of generations of the data used)
    error_cov_per_length = np.array([error_cov[tr, :, :, p, q] / truncate for tr, truncate in enumerate(truncate_list)])
    # select out est and est(uncalibrated) from all basic methods, and average over all simulations
    plot_data = [spearmanr_cov[:, :, :, p, q], error_cov_per_length, spearmanr_basic[:, :, :, p, q, 2:4], error_basic[:, :, :, p, q, 2:4]]
    # plot_data = [spearmanr_basic[:, :, :, p, q, 2:4], error_basic[:, :, :, p, q, 2:4], spearmanr_cov[:, :, :, p, q], error_cov_per_length]
    plot_data = [np.mean(_, axis=(1, 2)) for _ in plot_data]
    ylabels = [YLABEL_SPEARMANR_COVARIANCE, r"MAE of estimated covariances", YLABEL_SPEARMANR, r"MAE of inferred selection coefficients"]
    ylims = [(0.915, 0.965), (0.002, 0.016), (0.61, 0.91), (0.007, 0.015)]
    yticks = [np.linspace(0.92, 0.96, 5), np.linspace(0.003, 0.015, 5), np.linspace(0.64, 0.88, 5), np.linspace(0.008, 0.014, 4)]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()
        for j in range(2, 4):
            plt.plot(truncate_list, data[:, j - 2],
                     color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 3,
                     linewidth=SIZELINE, linestyle='dashed' if j == 3 else 'solid')

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=truncate_list, labels=truncate_list if at_bottom else [])
        plt.xlabel("Generations of data used" if at_bottom else "", fontsize=SIZELABEL)
        plt.ylabel(ylabels[i], fontsize=SIZELABEL)
        plt.ylim(ylims[i])
        plt.yticks(yticks[i])
        # if i == 0:
        #     plt.ylim(0.61, 0.91)
        #     plt.yticks(np.linspace(0.64, 0.88, 5))
        # if i == 1:
        #     plt.ylim(0.007, 0.015)
        #     plt.yticks(np.linspace(0.008, 0.014, 4))
        # if i == 2:
        #     plt.ylim(0.915, 0.965)
        #     plt.yticks(np.linspace(0.92, 0.96, 5))
        # if i == 3:
        #     plt.ylim(0.002, 0.016)
        #     plt.yticks(np.linspace(0.003, 0.015, 5))
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.2f' if at_left else '%.3f'))
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
        plt.text(x=-0.09, y=1.03, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_different_time_window(spearmanr_basic_window, spearmanr_linear_window, spearmanr_dcorr_window, window_list=WINDOW, sample_list=SAMPLE, record_list=RECORD, ylim=(0.45, 0.95), ylabel=YLABEL_SPEARMANR, max_p=None, max_q=None, combine=False, uncalibrated=False, alpha=1, save_file=None):
    """Plots a figure showing performances with different choices of time window."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if max_p is None:
        max_p = len(sample_list) - 1
    if max_q is None:
        max_q = len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]

    if combine:
        avg_axis = (1)
    else:
        avg_axis = (1, 2)

    mean_spearmanr_basic = np.mean(spearmanr_basic_window, axis=avg_axis)
    mean_spearmanr_linear = np.mean(spearmanr_linear_window, axis=avg_axis)
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr_window, axis=avg_axis)

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        # SL and MPL
        for j in range(2):
            plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], alpha=alpha,
                     linewidth=SIZELINE, linestyle='dashed', color=COLOR_BASIC[j], label=LABEL_BASIC[j])
        # est
        j = 2
        plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        if uncalibrated:
            j = 3
            plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
                     marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        plt.plot(range(len(window_list)), mean_spearmanr_linear[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_LINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_LINEAR, label=LABEL_LINEAR)

        plt.plot(range(len(window_list)), mean_spearmanr_dcorr[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_NONLINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_NONLINEAR, label=LABEL_NONLINEAR)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=np.arange(len(window_list)), labels=window_list if at_bottom else [])
        plt.xlabel('Window' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        if not at_left:
            ax.set_yticklabels([])
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        if "MAE" in ylabel:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.3f'))
            plt.ylim(0.005, 0.025)
            plt.yticks(np.linspace(0.007, 0.023, 5))
            if not at_left:
                ax.set_yticklabels([])
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        if at_left:
            plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(x=-0.07, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_different_loss_gamma(spearmanr_dcorr, truncate_index=5, ylim=(0.79, 0.89), alpha=0.5, sample_list=SAMPLE, record_list=RECORD, loss_list=LOSS, gamma_list=GAMMA, ylabel=YLABEL_SPEARMANR, save_file=None):
    """Plots a figure comparing performances of nonlinear shrinkage on correlation matrix using different loss functions & gamma."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    max_p, max_q = len(sample_list) - 1, len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]
    xticklabels=[r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'1']
    xlabel = 'Regularization strength of nonlinear shrinkage, ' + r'$\eta$'
    yticks = np.arange(int(100 * ylim[0]) // 2 * 2 / 100 + 0.02, ylim[1] + 0.01, 0.02)
    yticklabels = ['%.2f' % (_) for _ in yticks]

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()
        for l in range(len(loss_list)):
            plt.plot(np.log(gamma_list), np.mean(spearmanr_dcorr[truncate_index, :, :, p, q, l, :], axis=(0, 1)), color=COLOR_LOSS[l], linestyle=LINESTYLE_LOSS[l], linewidth=SIZELINE, marker=MARKER_LOSS[l], markersize=SMALLSIZEDOT - 3, label=LABEL_LOSS[l] if (i == 0 and l < 5 or i == 1 and l >= 5) else None, alpha=0.5)

        at_top = (i == 0 or i == 1)
        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=np.log(gamma_list), labels=xticklabels if at_bottom else [])
        plt.xlabel(xlabel if at_bottom else "", fontsize=SIZELABEL)
        plt.ylim(ylim)
        plt.yticks(ticks=yticks, labels=yticklabels if at_left else [])
        plt.ylabel(ylabel if at_left else "", fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if at_top:
            plt.legend(fontsize=SIZELEGEND, frameon=False, ncol=1)
        if at_left:
            plt.text(x=-0.09, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(x=-0.07, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f'sample={sample_list[p]}, interval={record_list[q]}', fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_all_methods(error_basic, error_linear, error_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=2, loss_selected=3, gamma_selected=0, vmin=0, vmax=0.02, figsize=None, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))

    matrix_SL = np.mean(error_basic[truncate_index, :, :, :, :, 0], axis=(0, 1))
    matrix_MPL = np.mean(error_basic[truncate_index, :, :, :, :, 1], axis=(0, 1))
    matrix_est = np.mean(error_basic[truncate_index, :, :, :, :, 2], axis=(0, 1))
    matrix_est_uncalibrated = np.mean(error_basic[truncate_index, :, :, :, :, 3], axis=(0, 1))
    matrix_linear = np.mean(error_linear[truncate_index, :, :, :, :, linear_selected], axis=(0, 1))
    matrix_dcorr = np.mean(error_dcorr[truncate_index, :, :, :, :, loss_selected, gamma_selected], axis=(0, 1))

    plot_data = [matrix_SL, matrix_MPL, matrix_est, matrix_est_uncalibrated, matrix_linear, matrix_dcorr]
    titles = LABEL_BASIC + [LABEL_LINEAR, LABEL_NONLINEAR]
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'

    for i, data in enumerate(plot_data):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        sns.heatmap(data, cmap='GnBu', alpha=0.75, cbar=False, annot=True, annot_kws={"fontsize": 6}, vmin=vmin, vmax=vmax)

        at_left = (i % nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list if at_bottom else [])
        # plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(x=-0.08, y=1.07, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, ylabel=ylabel, ylabelpad=-1, xlabelpad=-1)
    plt.subplots_adjust(0.071, 0.09, 0.95, 0.95, hspace=0.295, wspace=0.25)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)

############# ADDITIONAL FIGURES #############

def plot_additional_figure_performance_estimation_with_ample_data(dic_perf_cov, keys=DIC_PERF_COV_KEYS, truncate_list=TRUNCATE, linear_selected=2, plot_uncalibrated=True, option=0, save_file=None):
    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 1, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    spearmanr_keys = [keys[i] for i in np.arange(0, 12, 2)]
    error_keys = [keys[i] for i in np.arange(1, 13, 2)]
    labels = ['normalized', 'unnormalized', 'normalized_linear', 'unnormalized_linear', 'normalized_nonlinear', 'unnormalized_nonlinear']
    colors = [COLOR_BASIC[2], COLOR_BASIC[3], COLOR_LINEAR, COLOR_LINEAR, COLOR_NONLINEAR, COLOR_NONLINEAR]
    markers = [MARKER_BASIC[2], MARKER_BASIC[3], MARKER_LINEAR, MARKER_LINEAR, MARKER_NONLINEAR, MARKER_NONLINEAR]

    ylabels = [YLABEL_SPEARMANR_COVARIANCE, r"MAE of estimated covariances"]
    plot_dic = {key: np.mean(dic_perf_cov[key], axis=(1, 2)) for key in keys}
    for key in keys:
        if '_linear' in key:
            plot_dic[key] = plot_dic[key][:, linear_selected]
        if 'error' in key:
            plot_dic[key] = np.array([error / truncate_list[i] for i, error in enumerate(plot_dic[key])])

    for i, selected_keys in enumerate([spearmanr_keys, error_keys]):
        plt.sca(axes[i])
        ax = plt.gca()
        for j, key in enumerate(selected_keys):
            if j % 2 == 0:
                plt.plot(truncate_list, plot_dic[key][:, option],
                         color=colors[j], label=labels[j], marker=markers[j], markersize=SMALLSIZEDOT - 3,
                         linewidth=SIZELINE, linestyle='solid')
            elif plot_uncalibrated:
                plt.plot(truncate_list, plot_dic[key][:, option],
                         color=colors[j], label=labels[j], marker=markers[j], markersize=SMALLSIZEDOT - 3,
                         linewidth=SIZELINE, linestyle='dashed')
        plt.xlabel("Generations of data used", fontsize=SIZELABEL)
        plt.ylabel(ylabels[i], fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
        plt.text(x=-0.09, y=1.03, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_estimation_all_methods(dic_perf_cov, keys=DIC_PERF_COV_KEYS, plot_spearmanr=True, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=2, loss_selected=3, gamma_selected=0, plot_uncalibrated=True, option=0, vmin=0.5, vmax=1, figsize=None, save_file=None):
    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))

    if plot_spearmanr:
        selected_keys = [keys[i] for i in [0, 4, 8, 2, 6, 10]]
    else:
        selected_keys = [keys[i] for i in [1, 5, 9, 3, 7, 11]]


    plot_dic = {key: np.mean(dic_perf_cov[key], axis=(0, 1)) for key in keys}
    for key in keys:
        if '_linear' in key:
            plot_dic[key] = plot_dic[key][:, :, linear_selected, option]
        else:
            plot_dic[key] = plot_dic[key][:, :, option]
        if 'error' in key:
            plot_dic[key] = plot_dic[key] / truncate_list[truncate_index]

    titles = ['Normalized', 'Normalized_linear', 'Normalized_nonlinear',
              'Unnormalized', 'Unnormalized_linear', 'Unnormalized_nonlinear']
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'

    for i, key in enumerate(selected_keys):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        sns.heatmap(plot_dic[key], cmap='GnBu', alpha=0.75, cbar=False, annot=True, annot_kws={"fontsize": 6}, vmin=vmin, vmax=vmax)

        at_left = (i % nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list if at_bottom else [])
        # plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(x=-0.08, y=1.07, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, ylabel=ylabel, ylabelpad=-1, xlabelpad=-1)
    plt.subplots_adjust(0.071, 0.09, 0.95, 0.95, hspace=0.295, wspace=0.25)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_under_four_cases(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, gamma_selected=0, ylim=(0.63, 0.95), ylabel=r"Spearman's $\rho$", max_p=None, max_q=None, combine=False, uncalibrated=False, alpha=1, save_file=None):
    """Plots a figure showing performances with different choices of time window."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if max_p is None:
        max_p = len(sample_list) - 1
    if max_q is None:
        max_q = len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]

    if combine:
        avg_axis = (1)
    else:
        avg_axis = (1, 2)

    mean_spearmanr_basic = np.mean(spearmanr_basic, axis=avg_axis)
    mean_spearmanr_linear = np.mean(spearmanr_linear, axis=avg_axis)
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr, axis=avg_axis)

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        # SL and MPL
        for j in range(2):
            plt.plot(truncate_list, mean_spearmanr_basic[:, p, q, j], alpha=alpha,
                     linewidth=SIZELINE, linestyle='dashed', color=COLOR_BASIC[j], label=LABEL_BASIC[j])
        # est
        j = 2
        plt.plot(truncate_list, mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        if uncalibrated:
            j = 3
            plt.plot(truncate_list, mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
                     marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        plt.plot(truncate_list, mean_spearmanr_linear[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_LINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_LINEAR, label=LABEL_LINEAR)

        plt.plot(truncate_list, mean_spearmanr_dcorr[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_NONLINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_NONLINEAR, label=LABEL_NONLINEAR)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=truncate_list, labels=truncate_list if at_bottom else [])
        plt.xlabel('Generations of data used' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        if not at_left:
            ax.set_yticklabels([])
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_all_methods(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=2, loss_selected=3, gamma_selected=0, figsize=None, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))

    matrix_SL = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 0], axis=(0, 1))
    matrix_MPL = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 1], axis=(0, 1))
    matrix_est = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 2], axis=(0, 1))
    matrix_est_uncalibrated = np.mean(spearmanr_basic[truncate_index, :, :, :, :, 3], axis=(0, 1))
    matrix_linear = np.mean(spearmanr_linear[truncate_index, :, :, :, :, linear_selected], axis=(0, 1))
    matrix_dcorr = np.mean(spearmanr_dcorr[truncate_index, :, :, :, :, loss_selected, gamma_selected], axis=(0, 1))

    plot_data = [matrix_SL, matrix_MPL, matrix_est, matrix_est_uncalibrated, matrix_linear, matrix_dcorr]
    titles = LABEL_BASIC + [LABEL_LINEAR, LABEL_NONLINEAR]
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'

    for i, data in enumerate(plot_data):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS)

        at_left = (i % nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list if at_bottom else [])
        # plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, ylabel=ylabel, ylabelpad=-1, xlabelpad=-1)
    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2, left=0.18, right=0.9)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_combining_replicates_under_limited_sampling(spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, truncate_index=5, linear_selected=2, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=3, gamma_selected=0, nRow=2, nCol=3, w_h_ratio=1, double_column=True, save_file=None):
    """Plots a figure showing performances combining multiple replicates under limited sampling effects."""

    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    goldh = w / w_h_ratio
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    if truncate_index is not None:
        spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine = (spearmanr_basic_combine[truncate_index],
                                                                                      spearmanr_linear_combine[truncate_index],
                                                                                      spearmanr_dcorr_combine[truncate_index])

    mean_spearmanr_SL = np.mean(spearmanr_basic_combine[:, :, :, 0], axis=(0))
    mean_spearmanr_MPL = np.mean(spearmanr_basic_combine[:, :, :, 1], axis=(0))
    mean_spearmanr_est = np.mean(spearmanr_basic_combine[:, :, :, 2], axis=(0))
    mean_spearmanr_est_uncalibrated = np.mean(spearmanr_basic_combine[:, :, :, 3], axis=(0))
    mean_spearmanr_linear = np.mean(spearmanr_linear_combine[:, :, :, linear_selected], axis=(0))
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr_combine[:, :, :, loss_selected, gamma_selected], axis=(0))

    plot_data = [mean_spearmanr_SL, mean_spearmanr_MPL, mean_spearmanr_est, mean_spearmanr_est_uncalibrated, mean_spearmanr_linear, mean_spearmanr_dcorr]
    titles = LABEL_BASIC + [LABEL_LINEAR, LABEL_NONLINEAR]
    kwargs = deepcopy(DEF_HEATMAP_KWARGS)
    kwargs['vmin'] = np.min([np.min(data) for data in plot_data])
    kwargs['vmax'] = np.max([np.max(data) for data in plot_data])

    for i, data in enumerate(plot_data):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        sns.heatmap(data, **kwargs)

        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list, rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
        at_left = (i % nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        if at_left:
            plt.ylabel('Number of samples drawn\nat each generation', fontsize=SIZELABEL)
        if at_bottom:
            plt.xlabel('Time intervals between sampling\n' + r'$\Delta g$' + ' (generation)', fontsize=SIZELABEL)
        plt.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL)

    plt.subplots_adjust(bottom=0.3)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_combining_different_numbers_of_replicates(spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, num_basic=4, truncate_index=None, linear_selected=2, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=3, gamma_selected=0, min_p=None, max_p=None, max_q=None, ylim=(0.45, 0.95), alpha=0.75, double_column=True, w_h_ratio=GOLD_RATIO, save_file=None):
    """Plots a figure showing performances combining different numbers of replicates under limited sampling effects."""

    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    goldh = w / w_h_ratio
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if truncate_index is not None:
        spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine = (spearmanr_basic_combine[truncate_index],
                                                                                      spearmanr_linear_combine[truncate_index],
                                                                                      spearmanr_dcorr_combine[truncate_index])
    # num_basic = 4
    mean_spearmanr_basic = np.mean(spearmanr_basic_combine, axis=(0))
    mean_spearmanr_linear = np.mean(spearmanr_linear_combine[:, :, :, :, linear_selected], axis=(0))
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr_combine[:, :, :, :, loss_selected, gamma_selected], axis=(0))
    number_replicates = np.arange(1, 21)
    xticks = [1, 5, 10, 15, 20]

    if max_p is None:
        max_p = len(sample_list) - 1
    if min_p is None:
        min_p = 0
    if max_q is None:
        max_q = len(record_list) - 1
    pq_list = [(min_p, 0), (max_p, 0), (min_p, max_q), (max_p, max_q)]

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        for j in range(num_basic):
            plt.scatter(number_replicates, mean_spearmanr_basic[:, p, q, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT, alpha=alpha)
            plt.plot(number_replicates, mean_spearmanr_basic[:, p, q, j], color=COLOR_BASIC[j], linewidth=SIZELINE, alpha=alpha)
        plt.scatter(number_replicates, mean_spearmanr_linear[:, p, q], s=SMALLSIZEDOT, marker=MARKER_LINEAR, color=COLOR_LINEAR, label=LABEL_LINEAR, alpha=alpha)
        plt.plot(number_replicates, mean_spearmanr_linear[:, p, q], color=COLOR_LINEAR, linewidth=SIZELINE, alpha=alpha)
        plt.scatter(number_replicates, mean_spearmanr_dcorr[:, p, q], color=COLOR_NONLINEAR, label=LABEL_NONLINEAR, marker=MARKER_NONLINEAR, s=SMALLSIZEDOT, alpha=alpha)
        plt.plot(number_replicates, mean_spearmanr_dcorr[:, p, q], color=COLOR_NONLINEAR, linewidth=SIZELINE, alpha=alpha)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=xticks, labels=xticks if at_bottom else [])
        plt.xlabel('Number of replicates' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        if not at_left:
            ax.set_yticklabels([])
        plt.ylabel(r"Spearman's $\rho$" if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_error_int_cov_with_different_time_window(absolute_int_cov_window, error_int_cov_window, error_int_cov_uncalibrated_window, window_list=WINDOW, sample_list=SAMPLE, record_list=RECORD, ylim=(0.63, 0.95), ylabel=r"MAE of int-cov", max_p=None, max_q=None, combine=False, uncalibrated=False, alpha=1, save_file=None):
    """Plots an additional figure showing MAE of int-cov with different choices of time window."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if max_p is None:
        max_p = len(sample_list) - 1
    if max_q is None:
        max_q = len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]

    if combine:
        avg_axis = (1)
    else:
        avg_axis = (1, 2)

    mean_absolute_int_cov = np.mean(absolute_int_cov_window, axis=avg_axis)
    mean_error_int_cov = np.mean(error_int_cov_window, axis=avg_axis)
    mean_error_int_cov_uncalibrated = np.mean(error_int_cov_uncalibrated_window, axis=avg_axis)

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        # mean absolute value of int-cov
        j = 1
        plt.plot(range(len(window_list)), mean_absolute_int_cov[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label='mean abs int-cov')

        # est
        j = 2
        plt.plot(range(len(window_list)), mean_error_int_cov[:, p, q], linewidth=SIZELINE, alpha=alpha,
                 marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        if uncalibrated:
            j = 3
            plt.plot(range(len(window_list)), mean_error_int_cov_uncalibrated[:, p, q], linewidth=SIZELINE, alpha=alpha, marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=np.arange(len(window_list)), labels=window_list if at_bottom else [])
        plt.xlabel('Window' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        if not at_left:
            ax.set_yticklabels([])
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_biplot(data1, data2, label1="", label2="", plot_only_covariance=True, plot_inverse_covariance=False, data1_is_true_selection=False, p_list=np.arange(0, len(SAMPLE)), q_list=np.arange(0, len(RECORD)), s=0.5, alpha=0.5, vmin=0, vmax=1, suptitle=None, figsize=None, save_file=None):
    w = DOUBLE_COLUMN #SLIDE_WIDTH
    nRow, nCol = len(p_list), len(q_list)
    goldh = w * (nRow / nCol)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))
    L = len(data2[0, 0])
    for r, p in enumerate(p_list):
        for c, q in enumerate(q_list):
            plt.sca(axes[r, c])
            ax = plt.gca()
            if len(data2[0, 0].shape) > 1:
                # covariance
                if plot_inverse_covariance:
                    data_x, data_y = np.linalg.inv(data1[p, q] + np.identity(L)), np.linalg.inv(data2[p, q] + np.identity(L))
                else:
                    data_x, data_y = data1[p, q], data2[p, q]

                cov_x, cov_y = [data_x[i, j] for i in range(L) for j in range(i)], [data_y[i, j] for i in range(L) for j in range(i)]
                if plot_only_covariance:
                    x, y = cov_x, cov_y
                else:
                    x, y = np.ndarray.flatten(data_x), np.ndarray.flatten(data_y)
                    var_x, var_y = [data_x[i, i] for i in range(L)], [data_y[i, i] for i in range(L)]
                    plt.scatter(var_x, var_y, s=s, alpha=0.8, label='Diagonal', color=COLOR_VARIANCE)
                speamanr, _ = stats.spearmanr(x, y)
                plt.scatter(cov_x, cov_y, s=s, alpha=alpha, label='Off-diagonal', color=get_color_by_value_percentage(speamanr, vmin=vmin, vmax=vmax))
            else:
                # selection
                if data1_is_true_selection:
                    x, y = data1, data2[p, q]
                else:
                    x, y = data1[p, q], data2[p, q]
                speamanr, _ = stats.spearmanr(x, y)
                plt.scatter(x, y, s=s, alpha=alpha, label='Selection', color=get_color_by_value_percentage(speamanr, vmin=vmin, vmax=vmax))

            min_value = min(np.min(x), np.min(y))
            max_value = max(np.max(x), np.max(y))
            plt.plot([min_value, max_value], [min_value, max_value], linestyle='dashed', color='grey', linewidth=SIZELINE)
            at_bottom = (r == nRow - 1)
            at_left = (c == 0)
            at_right = (c == nCol - 1)
            at_top = (r == 0)
            record_title = f'record = {RECORD[q]}\n\n' if at_top else ""
            plt.title(record_title + f"Spearmanr's " + r'$\rho$' + '=%.2f' % speamanr, fontsize=SIZELABEL)
            if at_left and at_top:
                plt.legend(fontsize=SIZELEGEND, frameon=False)
            if at_left:
                plt.ylabel(label2, fontsize=SIZELABEL)
            else:
                plt.yticks(ticks=[], labels=[])
            if at_bottom:
                plt.xlabel(label1, fontsize=SIZELABEL)
            else:
                plt.xticks(ticks=[], labels=[])
            if at_right:
                ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
                ax2.set_ylabel(f'sample = %4d' % SAMPLE[p], rotation=0, fontsize=SIZELABEL, labelpad=30)
                ax2.set_yticks([])
            ax.tick_params(**DEF_TICKPROPS)
            plt.setp(ax.spines.values(), **DEF_AXPROPS)

    # plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=SIZESUBLABEL, y=0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
