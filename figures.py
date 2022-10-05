"""
This Python file contains code for generating plots in:
    Estimating genetic linkage and selection from allele frequency trajectories
    Yunxiao Li, and
    John P. Barton (john.barton@ucr.edu)

Additional code to pre-process the data and pass it to these plotting
routines is contained in the Jupyter notebook `figures.ipynb`.
"""

#############  PACKAGES  #############
import sys
import numpy as np

from copy import deepcopy

import matplotlib
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import seaborn as sns

from scipy import stats

sys.path.append('./src')
import mplot as mp
import simulation_setup as SS
import data_parser as DP
import estimate_covariance as EC

############# PARAMETERS #############

# GLOBAL VARIABLES

N = 1000        # number of sequences in the population
L = 50          # sequence length
T = 700         # number of generations
MU = 1e-3       # mutation rate
NUM_SELECTIONS = 10  # number of sets of selection coefficients that we used
NUM_TRIALS = 20  # for each set of selection coefficients, we simulated for 20 times
NUM_BASIC_METHODS = 3  # SL, est and MPL methods
SAMPLE = [1000, 500, 100, 50, 10]  # sampling depth options when subsampling
RECORD = [1, 3, 5, 10]  # sampling time interval options when subsampling
LINEAR = [1, 5, 10, 20, 50, 100, 300]  # linear shrinkage strengths that we tested
GAMMA = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]  # nonlinear shrinkage parameter choices that we tested
TRUNCATE = [200, 300, 400, 500, 600, 700]  # truncate options to test performance using different lengths of data
WINDOW = [0, 1, 2, 3, 4, 5, 10, 20, 40, 80, 160]  # window choices that we tested
# loss functions for nonlinear shrinkage that we tested
LOSS = ['Fro | $\it{\hat{R}-R}$', 'Fro | $\it{\hat{R}^{-1}-R^{-1}}$', 'Fro | $\it{R^{-1}\hat{R}-I}$',
        'Fro | $\it{\hat{R}^{-1}R-I}$', 'Fro | $\it{\hat{R}^{-1/2}R\hat{R}^{-1/2}-I}$',
        'Nuc | $\it{\hat{R}-R}$', 'Nuc | $\it{\hat{R}^{-1}-R^{-1}}$', 'Nuc | $\it{R^{-1}\hat{R}-I}$',
        'Nuc | $\it{\hat{R}^{-1}R-I}$', 'Nuc | $\it{\hat{R}^{-1/2}R\hat{R}^{-1/2}-I}$']
SHRINKING_FACTORS = np.arange(0, 1, 0.01)

LINEAR_SELECTED = 2
LOSS_SELECTED = 3
GAMMA_SELECTED = 0
SHRINK_SELECTED = 50  # Shrinking factor = 0.5
DCORR_SHRINK_SELECTED = 50  # Shrinking factor = 0.5

# Standard color scheme

BKCOLOR  = '#252525'
GREY_COLOR_RGB = (0.5, 0.5, 0.5)
GREY_COLOR_HEX = '#808080'
LCOLOR   = '#969696'
C_BEN    = '#EB4025' #'#F16913'
C_BEN_LT = '#F08F78' #'#fdd0a2'
C_NEU    =  LCOLOR   #'#E8E8E8' # LCOLOR
C_NEU_LT = '#E8E8E8' #'#F0F0F0' #'#d9d9d9'
C_DEL    = '#3E8DCF' #'#604A7B'
C_DEL_LT = '#78B4E7' #'#dadaeb'
C_MINT   = '#98FB98'
C_MOSS   = '#8A9A5B'
C_FOREST = '#0B6623'

GREY_TEXT = '\033[38;5;07m'
CMAP = mpl.cm.get_cmap('GnBu')
CMAP_COLORS = [CMAP(i) for i in np.arange(0, 1.1, 0.1)]
TEXT_COLOR_CODES = [93, 92, 91, 90, 89, 88, 52]
TEXT_COLORS = [u"\u001b[38;5;" + str(code) + "m" for code in TEXT_COLOR_CODES]

# COLOR_PALETTES = {
#     key: [rgb_to_hex(_) for _ in sns.color_palette(key)] for key in ['deep', 'tab10', 'bright']
# }

COLOR_PALETTES = {
    key: sns.color_palette(key) for key in ['deep', 'tab10', 'bright', 'pastel']
}
COLORS = COLOR_PALETTES['bright']
COLORS_DEEP = COLOR_PALETTES['deep']
COLORS_TAB = COLOR_PALETTES['tab10']
COLORS_PASTEL = COLOR_PALETTES['pastel']

COLOR_BASIC = [COLORS[0], COLORS[1], 'grey', 'grey']  # [COLORS[0], COLORS[1], COLORS[7], COLORS[7]]  # ['blue', 'orange', 'grey', 'grey']
MARKER_BASIC = ['o', 'd', 'v', '^']
LABEL_EST = 'Est'
LABEL_BASIC = ['SL', 'MPL', LABEL_EST, 'Est (Unnormalized)']
COLOR_LINEAR = COLORS[3]  # 'red'
MARKER_LINEAR = '<'
LABEL_LINEAR = 'Linear' # 'Linear ' + r'$\it{(\gamma=10\Delta g)}$'  # 'Linear-cov'
COLOR_NONLINEAR = COLORS[2]  # 'green'
MARKER_NONLINEAR = '>'
LABEL_NONLINEAR = 'Nonlinear'  # 'Nonlinear-corr'

LABEL_SHRINK = 'Est-norm ' + r'$\it{(f=0.5)}$'
MARKER_SHRINK  = 'P'
COLOR_SHRINK = COLORS[8]  # 'yellow'
LABEL_NONLINEAR_SHRINK = 'Nonlinear-norm ' + r'$\it{(f=0.5)}$'
LABEL_NONLINEAR_SHRINK_TWOLINE = 'Nonlinear-norm\n' + r'$\it{(f=0.5)}$'
MARKER_NONLINEAR_SHRINK  = 'X'
COLOR_NONLINEAR_SHRINK = COLORS[5]  # 'brown'

COLOR_ALTERNATIVE = COLORS[4]  # 'purple'

LABELS_OTHER_METHODS = [LABEL_LINEAR, LABEL_NONLINEAR, LABEL_SHRINK, LABEL_NONLINEAR_SHRINK]
COLORS_OTHER_METHODS = [COLOR_LINEAR, COLOR_NONLINEAR, COLOR_SHRINK, COLOR_NONLINEAR_SHRINK]
MARKERS_OTHER_METHODS = [MARKER_LINEAR, MARKER_NONLINEAR, MARKER_SHRINK, MARKER_NONLINEAR_SHRINK]

LABELS_ALL_METHODS = LABEL_BASIC + LABELS_OTHER_METHODS
COLORS_ALL_METHODS = COLOR_BASIC + COLORS_OTHER_METHODS
MARKERS_ALL_METHODS = MARKER_BASIC + MARKERS_OTHER_METHODS

LABELS_ALL_METHODS_NORM = LABEL_BASIC[:3] + LABELS_OTHER_METHODS
COLORS_ALL_METHODS_NORM = COLOR_BASIC[:3] + COLORS_OTHER_METHODS
MARKERS_ALL_METHODS_NORM = MARKER_BASIC[:3] + MARKERS_OTHER_METHODS

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
SUBLABELS    = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

# paper style

FONTFAMILY   = 'Arial'
SIZESUBLABEL = 8
SIZELABEL    = 6
SIZELEGEND   = 6
SIZETICK     = 6
SMALLSIZEDOT = 6.
SIZEDOT      = 10.
SIZELINE     = 0.6
EDGECOLORS   = 'none'
HEATMAP_LINEWIDTH = 0

DEF_FIGPROPS = {
    'transparent' : True,
    'edgecolor'   : None,
    'dpi'         : 300, # 300,  # 1000
    # 'bbox_inches' : 'tight',
    'pad_inches'  : 0.05,
    # 'backend'     : 'PDF',
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

DEF_LABELPROPS = {
    'family' : FONTFAMILY,
    'size'   : SIZELABEL,
    'color'  : BKCOLOR
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

DEF_TICKPROPS_CBAR = {
    'length'    : TICKLENGTH * 0.7,
    'width'     : AXWIDTH/2,
    'pad'       : TICKPAD/2,
    'axis'      : 'both',
    'which'     : 'both',
    'direction' : 'out',
    'colors'    : BKCOLOR,
    'labelsize' : SIZETICK,
    'bottom'    : True,
    'left'      : False,
    'top'       : False,
    'right'     : True,
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

DEF_HEATMAP_KWARGS_ERROR = {
    'vmin'      : 0,
    'vmax'      : 0.02,
    'cmap'      : 'GnBu',
    'alpha'     : 0.75,
    'cbar'      : False,
    'annot'     : True,
    'annot_kws' : {"fontsize": 5}
}

DEF_HEATMAP_KWARGS_SHRINK_PERFORMANCE_GAIN = {
    'vmin'      : -0.015,
    'vmax'      : 0.28,
    'center'    : 0,
    'cmap'      : sns.diverging_palette(145, 300, as_cmap=True),
    'alpha'     : 0.75,
    'cbar'      : False,
    'annot'     : True,
    'annot_kws' : {"fontsize": 6},
    'fmt'       : '.2f',
}

DEF_HEATMAP_KWARGS_STD_COV_VAR_RATIO = {
    'cmap'      : 'GnBu',
    'alpha'     : 0.75,
    'cbar'      : False,
    'annot'     : True,
    'annot_kws' : {"fontsize": 6}
}

FIG4_LEFT, FIG4_BOTTOM, FIG4_RIGHT, FIG4_TOP = 0.14, 0.223, 0.965, 0.935
GOLD_RATIO = 1.4

XLIM_GENERATION = (-10, 710)

YLIM_SPEARMANR = (0.59, 1.01)
YTICKS_SPEARMANR = np.linspace(0.6, 1.0, 5)
YLIM_MAE = (0.0056, 0.0224)
YTICKS_MAE = np.linspace(0.006, 0.022, 5)
# YLIM_MAE = (0.0045, 0.022)
# YTICKS_MAE = np.linspace(0.005, 0.02, 4)
# YLIM_SPEARMANR = (0.55, 0.95)
# YTICKS_SPEARMANR = np.linspace(0.6, 0.9, 4)
# YLIM_MAE = (0.006, 0.022)
# YTICKS_MAE = np.linspace(0.008, 0.02, 4)

YLIM_SPEARMANR_COMBINE = (0.49, 1.01)
YTICKS_SPEARMANR_COMBINE = np.linspace(0.5, 1.0, 6)

YLIM_SELECTION = (-0.055, 0.055)
YTICKS_SELECTION = np.linspace(-0.04, 0.04, 5)

SPEARMANR = r"Spearman’s $\it{\rho}$"
PEARSONR = r"Pearson’s $\it{r}$"
DELTA_G = r"$\it{\Delta g}$"
XLABEL_WINDOW = "Window, " + r"$\it{\delta t}$"
XLABEL_FACTOR = "Factor, " + r"$\it{f}$"
YLABEL_SPEARMANR = f"Rank correlation between true and inferred\nselection coefficients ({SPEARMANR})"
YLABEL_SPEARMANR_COVARIANCE = f"Rank correlation between true and inferred\ncovariances ({SPEARMANR})"
YLABEL_SPEARMANR_THREE = f"Rank correlation between true and\ninferred selection coefficients\n({SPEARMANR})"
YLABEL_SPEARMANR_COVARIANCE_THREE = f"Rank correlation between true\nand inferred covariances\n({SPEARMANR})"
YLABEL_MAE = 'MAE of inferred selection coefficients'

PARAMS = {'text.usetex': False, 'mathtext.fontset': 'stixsans', 'mathtext.default': 'regular', 'pdf.fonttype': 42, 'ps.fonttype': 42, 'scatter.edgecolors': 'none'}
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


def rgb_to_hex(rgb):
    rgb = tuple([round(256 * _) for _ in rgb])
    return '#%02x%02x%02x' % (rgb)


def compute_dx_dx_across_generations(traj, gens):
    dx = EC.computeDx(traj)
    T, L = traj.shape
    dx_dx_across_gens = np.zeros((len(gens), len(gens), L, L))
    for s, g1 in enumerate(gens):
        for t, g2 in enumerate(gens):
            if t >= s:
                for l1 in range(L):
                    for l2 in range(l1, L):
                        dx_dx_across_gens[s, t, l1, l2] = dx[g1, l1] * dx[g2, l2]
                        dx_dx_across_gens[s, t, l2, l1] = dx_dx_across_gens[s, t, l1, l2]

    return dx_dx_across_gens


def compute_dx_dx_across_generations_2(traj, tStart, tEnd, tau):

    dx = EC.computeDx(traj)
    T, L = traj.shape
    dx_dx_across_gens = np.zeros((tEnd - tStart, L, L))
    if tEnd - 1 + tau >= len(dx):
        print('t+tau out of bounds')
        return
    for t in range(tStart, tEnd):
        for l1 in range(L):
            for l2 in range(l1 + 1, L):
                dx_dx_across_gens[t - tStart, l1, l2] = dx[t, l1] * dx[t + tau, l2]
    return dx_dx_across_gens


def get_color_by_value_percentage(value, vmin=0, vmax=1, colors=CMAP_COLORS):
    if value <= vmin:
        index = 0
    elif value >= vmax:
        index = -1
    else:
        index = int((value - vmin) / (vmax - vmin) * len(colors))
    return colors[index]


def plot_line_from_slope_intercept(slope, intercept, color='grey', linewidth=SIZELINE):
    """Plot a line from slope and intercept"""
    axes = plt.gca()
    x_vals = np.array(axes.get_xlim())
    y_vals = intercept + slope * x_vals
    plt.plot(x_vals, y_vals, '--', color=color, linewidth=linewidth)


def plot_scatter_and_line(xs, ys, color=None, s=SIZEDOT, label=None, edgecolors=EDGECOLORS, marker='.', alpha=1, linewidth=SIZELINE):
    plt.scatter(xs, ys, s=s, marker=marker, color=color, edgecolors=edgecolors, label=label, alpha=alpha)
    plt.plot(xs, ys, color=color, linewidth=linewidth, alpha=alpha)


def set_ticks_labels_axes(xlim=None, ylim=None, xticks=None, xticklabels=None, yticks=None, yticklabels=None, xlabel=None, ylabel=None, ylabelpad=None):

    ax = plt.gca()

    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if xticks is not None and xticklabels is not None:
        plt.xticks(ticks=xticks, labels=xticklabels, fontsize=SIZELABEL, rotation=0)
    else:
        plt.xticks(fontsize=SIZELABEL)

    if yticks is not None and yticklabels is not None:
        plt.yticks(ticks=yticks, labels=yticklabels, fontsize=SIZELABEL)
    else:
        plt.yticks(fontsize=SIZELABEL)

    if xlabel is not None:
        plt.xlabel(xlabel, fontsize=SIZELABEL)

    if ylabel is not None:
        plt.ylabel(ylabel, fontsize=SIZELABEL, labelpad=ylabelpad)

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


def plot_comparison(xs, ys, xlabel=None, ylabel=None, ylabelpad=None, use_pearsonr=False, use_MAE=False, annot=False, plot_cov=False, label=None, annot_texts=None, alpha=0.6, ylim=None, yticks=None, xticks=None, xticklabels=None, yticklabels=None, plot_title=True, title_pad=4):
    L = len(xs)
    if plot_cov:
        diagonal_xs, diagonal_ys = EC.get_diagonal_terms(xs), EC.get_diagonal_terms(ys)
        off_diagonal_xs, off_diagonal_ys = EC.get_off_diagonal_terms(xs), EC.get_off_diagonal_terms(ys)
        plt.scatter(diagonal_xs, diagonal_ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label='diagonal')
        plt.scatter(off_diagonal_xs, off_diagonal_ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label='off-diagonal')
        if annot:
            index_to_pair = {}
            index = 0
            for i in range(L):
                for j in range(i):
                    index_to_pair[index] = (i, j)
                    index += 1
            for i, (x, y) in enumerate(zip(off_diagonal_xs, off_diagonal_ys)):
                if abs(y - x) > 5 and y != 0:
                    plt.text(x, y, index_to_pair[i], alpha=alpha, fontsize=10)
    else:
        plt.scatter(xs, ys, marker='o', edgecolors='none', s=SIZEDOT, alpha=alpha, label=label)
        if annot:
            for i, (x, y) in enumerate(zip(xs, ys)):
                plt.text(x, y, i, alpha=alpha, fontsize=10)
    if annot_texts is not None:
        for x, y, text in zip(xs, ys, annot_texts):
            plt.annotate(x, y, text)
    set_ticks_labels_axes(xlim=ylim, ylim=ylim, xticks=xticks, yticks=yticks, xticklabels=xticklabels, yticklabels=yticklabels, xlabel=xlabel, ylabel=ylabel, ylabelpad=ylabelpad)
    if ylim is not None:
        plt.plot(ylim, ylim, color=GREY_COLOR_RGB, alpha=alpha, linestyle='dashed')
    plt.gca().set_aspect('equal', adjustable='box')
    # plt.gca().set_aspect('equal', adjustable='datalim')
    if plot_cov:
        plt.legend()
        if plot_title:
            pearsonr_title = ("Diagonal Pearson's " + r'$r$' + ' = %.2f' + '\n' + "Off-diagonal Pearson's " + r'$r$' + ' = %.2f') % (stats.pearsonr(diagonal_xs, diagonal_ys)[0], stats.pearsonr(off_diagonal_xs, off_diagonal_ys)[0])
            spearmanr_title = ("Diagonal Spearman's " + r'$\rho$' + ' = %.2f' + '\n' + "Off-diagonal Spearman's " + r'$\rho$' + ' = %.2f') % (stats.spearmanr(diagonal_xs, diagonal_ys)[0], stats.spearmanr(off_diagonal_xs, off_diagonal_ys)[0])
            if use_pearsonr:
                plt.title(f'{spearmanr_title}\n{pearsonr_title}', fontsize=SIZELABEL, pad=title_pad)
            else:
                plt.title(spearmanr_title, fontsize=SIZELABEL, pad=title_pad)
    elif plot_title:
        spearmanr_title = "Spearman's " + r'$\rho$' + ' = %.2f' % stats.spearmanr(xs, ys)[0]
        pearsonr_title = "Pearson's " + r'$r$' + ' = %.2f' % stats.pearsonr(xs, ys)[0]
        MAE_title = "MAE = %.3f" % np.mean(np.absolute(np.array(xs) - np.array(ys)))
        if use_pearsonr:
            plt.title(f'{spearmanr_title}\n{pearsonr_title}', fontsize=SIZELABEL, pad=title_pad)
        elif use_MAE:
            plt.title(MAE_title, fontsize=SIZELABEL, pad=title_pad)
        else:
            plt.title(spearmanr_title, fontsize=SIZELABEL, pad=title_pad)


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
        if selection[l] > 0.03:
            color = C_BEN_LT
            if first_ben:
                label = 'Beneficial'
                first_ben = False
        elif selection[l] < -0.03:
            color = C_DEL_LT
            if first_del:
                label = 'Deleterious'
                first_del = False
        else:
            color = C_NEU_LT
            if first_neu:
                label = 'Neutral'
                first_neu = False
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


def plot_figure_traj_cov_example(traj, selection, true_cov, est_cov, alpha=0.85, xlim=XLIM_GENERATION, ylim=(-0.02, 1.12), matrix_labels=None, vpad=5, title_pad=3, cbar_ticks=None, rasterized=True, save_file=None):
    w = SINGLE_COLUMN
    # goldh = w / GOLD_RATIO
    ratio = 1.4
    fig = plt.figure(figsize=(w, w * ratio))

    global_left, global_right, global_bottom, global_top = 0.14, 0.98, 0.05, 0.98
    wspace = 0.15
    heatmap_width = (global_right - global_left) / (2 + wspace)
    heatmap_height = heatmap_width / ratio
    row1_bottom = 0.76
    row2_top = global_bottom + (global_right - global_left) / ratio
    row2_bottom = row2_top - heatmap_height
    row3_top = global_bottom + heatmap_height

    row1_box = dict(left=global_left, right=global_right, bottom=row1_bottom, top=global_top)
    row2_box = dict(left=global_left, right=global_right, bottom=row2_bottom, top=row2_top)
    row3_box = dict(left=global_left, right=global_right, bottom=global_bottom, top=row3_top)

    gs1 = gridspec.GridSpec(1, 1, **row1_box)
    gs2 = gridspec.GridSpec(1, 2, wspace=wspace, **row2_box)
    gs3 = gridspec.GridSpec(1, 2, wspace=wspace, **row3_box)

    ax1 = plt.subplot(gs1[0, 0])
    ax2, ax3 = plt.subplot(gs2[0, 0]), plt.subplot(gs2[0, 1])
    ax4, ax5 = plt.subplot(gs3[0, 0]), plt.subplot(gs3[0, 1])

    heatmaps = [ax2, ax3, ax4]
    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = [r"True covariance matrix, $\it{C}$", r"Estimated covariance matrix, $\it{\hat{C}}$", " " * 58 + "Error of estimated covariance matrix, $\it{\hat{C}-C}$"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    cbar_y, cbar_length, cbar_width = 0.03, heatmap_width, 0.01
    cbar_ax = fig.add_axes(rect=[global_left, cbar_y, cbar_length, cbar_width])
    # cbar_ax = fig.add_axes(rect=[.59, .0285, .02, 0.3])
    cbar_ax5 = fig.add_axes(rect=[global_left + heatmap_width * (1 + wspace), cbar_y, cbar_length, cbar_width])
    if cbar_ticks is None:
        cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
        cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    if matrix_labels is None:
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
        elif selection[l] < -0:
            color = C_DEL_LT
            if first_del:
                label = 'Deleterious'
                first_del = False
        else:
            color = C_NEU_LT
            if first_neu:
                label = 'Neutral'
                first_neu = False
        if label is not None:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, label=label, rasterized=rasterized)
        else:
            plt.plot(range(len(traj)), traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, rasterized=rasterized)

    plt.ylabel("Mutant allele\nfrequency", fontsize=SIZELABEL)
    plt.xlabel("Generation", fontsize=SIZELABEL)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.yticks(np.linspace(0, 1, 6))
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # plt.legend(loc=0, frameon=False, fontsize=SIZELEGEND)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=SIZELEGEND, frameon=False, ncol=3)
    plt.text(x=-0.1, y=1, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    sublabel_y = 0.96

    for i, cov in enumerate(cov_list):
        plt.sca(heatmaps[i])
        ax = plt.gca()
        plot_cbar = (i == 2)
        sns.heatmap(cov, center=0, vmin=vmin - vpad, vmax=vmax + vpad, cmap=sns.diverging_palette(208, 12, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"), rasterized=rasterized)
        plt.title(title_list[i], fontsize=SIZELABEL, pad=title_pad)
        # plt.xlabel(title_list[i], fontsize=SIZELABEL, labelpad=3 if i == 0 else 2)
        if i == 0 or i == 2:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        # plt.xticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL, rotation=0)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            # cbar_ax.tick_params(labelsize=SIZELABEL)
            cbar_ax.tick_params(**DEF_TICKPROPS_CBAR)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.text(x=-0.06 if i == 1 else -0.23, y=sublabel_y, s=SUBLABELS[i + 1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(ax5)
    i = 3
    cov = est_cov - true_cov
    vmin, vmax = np.min(cov), np.max(cov)
    vpad = (vmax - vmin) // 20
    cbar_ticks = np.arange(vmin//3*3, vmax//3*3 + 3, 3)
    cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    ax = plt.gca()
    sns.heatmap(cov, center=0, vmin=vmin - vpad, vmax=vmax + vpad, cmap=sns.diverging_palette(120, 300, as_cmap=True), square=True, cbar=True, cbar_ax=cbar_ax5, cbar_kws=dict(ticks=cbar_ticks, orientation="horizontal"), rasterized=rasterized)
    plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
    plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
    cbar_ax5.tick_params(**DEF_TICKPROPS_CBAR)
    ax.tick_params(**DEF_TICKPROPS_HEATMAP)
    plt.text(x=-0.06, y=sublabel_y, s=SUBLABELS[i + 1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # plt.subplots_adjust(0.01, 0.01, 0.988, 0.985)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_traj_cov_example_2(traj, selection, true_cov, est_cov, times=np.arange(0, 701, 1), alpha=0.85, xlim=XLIM_GENERATION, ylim=(-0.02, 1.12), matrix_labels=None, vpad=5, title_pad=3, cbar_ticks=None, rasterized=True, save_file=None):
    w = SINGLE_COLUMN
    # goldh = w / GOLD_RATIO
    ratio = 1.2
    fig = plt.figure(figsize=(w, w * ratio))

    global_left, global_right, global_bottom, global_top = 0.14, 0.9, 0.01, 0.98
    wspace = 0.15
    heatmap_width = (global_right - global_left) / (2 + wspace)
    heatmap_height = heatmap_width / ratio
    wspace = 0.3
    row1_bottom = 0.76
    row2_top = global_bottom + (global_right - global_left) / ratio
    row2_bottom = row2_top - heatmap_height
    row3_top = global_bottom + heatmap_height

    row1_box = dict(left=global_left, right=global_right, bottom=row1_bottom, top=global_top)
    row2_box = dict(left=global_left, right=global_right, bottom=row2_bottom, top=row2_top)
    row3_box = dict(left=global_left, right=global_right, bottom=global_bottom, top=row3_top)

    gs1 = gridspec.GridSpec(1, 1, **row1_box)
    gs2 = gridspec.GridSpec(1, 2, wspace=wspace, **row2_box)
    gs3 = gridspec.GridSpec(1, 2, wspace=wspace, **row3_box)

    ax1 = plt.subplot(gs1[0, 0])
    ax2, ax3 = plt.subplot(gs2[0, 0]), plt.subplot(gs2[0, 1])
    ax4, ax5 = plt.subplot(gs3[0, 0]), plt.subplot(gs3[0, 1])

    heatmaps = [ax2, ax3, ax4]
    cov_list = [true_cov, est_cov, est_cov - true_cov]
    title_list = [r"True covariance matrix, $\it{C}$", r"Estimated covariance matrix, $\it{\hat{C}}$", " " * 58 + "Error of estimated covariance matrix, $\it{\hat{C}-C}$"]
    vmin = min(np.min(cov_list[0]), np.min(cov_list[1]))
    vmax = max(np.max(cov_list[0]), np.max(cov_list[1]))
    # cbar_y, cbar_length, cbar_width = 0.03, heatmap_width, 0.01
    # cbar_ax = fig.add_axes(rect=[global_left, cbar_y, cbar_length, cbar_width])

    cbar_y, cbar_length, cbar_width = 0.03, 0.012, heatmap_height * 0.94
    cbar_ax = fig.add_axes(rect=[global_left + heatmap_width - 0.008, row2_bottom + 0.0075, cbar_length, cbar_width])

    # cbar_ax = fig.add_axes(rect=[.59, .0285, .02, 0.3])
    # cbar_y, cbar_length, cbar_width = 0.03, heatmap_width, 0.01
    cbar_y, cbar_length, cbar_width = global_bottom + 0.0075, 0.01, heatmap_height * 0.94
    cbar_ax5 = fig.add_axes(rect=[global_right + 0.0135, cbar_y, cbar_length, cbar_width])

    if matrix_labels is None:
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
        elif selection[l] < -0:
            color = C_DEL_LT
            if first_del:
                label = 'Deleterious'
                first_del = False
        else:
            color = C_NEU_LT
            if first_neu:
                label = 'Neutral'
                first_neu = False
        if label is not None:
            plt.plot(times, traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, label=label, rasterized=rasterized)
        else:
            plt.plot(times, traj[:, l], color=color, alpha=alpha, linewidth=SIZELINE, rasterized=rasterized)

    plt.ylabel("Mutant allele\nfrequency", fontsize=SIZELABEL)
    plt.xlabel("Generation", fontsize=SIZELABEL)
    plt.ylim(ylim)
    plt.xlim(xlim)
    plt.yticks(np.linspace(0, 1, 6))
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # plt.legend(loc=0, frameon=False, fontsize=SIZELEGEND)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.06), fontsize=SIZELEGEND, frameon=False, ncol=3)
    plt.text(x=-0.1, y=1, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    sublabel_y = 0.96


    # if cbar_ticks is None:
    #     cbar_ticks = np.arange(int(vmin/5)*5, int(vmax/5)*5, 50)
    #     cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    cbar_ticks = [-40, 0, 40, 80, 120]
    for i, cov in enumerate(cov_list):
        plt.sca(heatmaps[i])
        ax = plt.gca()
        plot_cbar = (i == 0)
        sns.heatmap(cov, center=0, vmin=vmin - vpad, vmax=vmax + vpad, cmap=sns.diverging_palette(208, 12, as_cmap=True), square=True, cbar=plot_cbar,
                    cbar_ax=cbar_ax if plot_cbar else None,
                    cbar_kws=dict(ticks=cbar_ticks, orientation="vertical"),
                    rasterized=rasterized)
        plt.title(title_list[i], fontsize=SIZELABEL, pad=title_pad)
        # plt.xlabel(title_list[i], fontsize=SIZELABEL, labelpad=3 if i == 0 else 2)
        if i == 0 or i == 2:
            plt.yticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL)
            plt.ylabel('Locus index', fontsize=SIZELABEL)
        else:
            plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        # plt.xticks(ticks=matrix_ticks, labels=matrix_labels, fontsize=SIZELABEL, rotation=0)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        if plot_cbar:
            # cbar_ax.tick_params(labelsize=SIZELABEL)
            cbar_ax.tick_params(**DEF_TICKPROPS_CBAR)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.text(x=-0.06 if i == 1 else -0.23, y=sublabel_y, s=SUBLABELS[i + 1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(ax5)
    i = 3
    cov = est_cov - true_cov
    vmin, vmax = np.min(cov), np.max(cov)
    vpad = (vmax - vmin) // 20
    # cbar_ticks = np.arange(vmin//3*3, vmax//3*3 + 3, 3)
    # cbar_ticks -= cbar_ticks[np.argmin(np.abs(cbar_ticks))]
    cbar_ticks = [-12, -8, -4, 0, 4, 8]
    ax = plt.gca()
    sns.heatmap(cov, center=0, vmin=vmin - vpad, vmax=vmax + vpad, cmap=sns.diverging_palette(120, 300, as_cmap=True), square=True, cbar=True, cbar_ax=cbar_ax5, cbar_kws=dict(ticks=cbar_ticks, orientation="vertical"), rasterized=rasterized)
    plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
    plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
    cbar_ax5.tick_params(**DEF_TICKPROPS_CBAR)
    ax.tick_params(**DEF_TICKPROPS_HEATMAP)
    plt.text(x=-0.06, y=sublabel_y, s=SUBLABELS[i + 1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # plt.subplots_adjust(0.01, 0.01, 0.988, 0.985)
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
        plt.xlabel(XLABEL_WINDOW, fontsize=SIZELABEL)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.subplots_adjust(0.05, 0.08, 0.9, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_estimation_normalization_window(true_cov, est_cov, est_cov_uncalibrated, error_window, error_window_uncalibrated=None, window_list=WINDOW, alpha=0.7, hist_ylim=(0, 0.195), hist_yticks = np.linspace(0, 0.18, 5), ylim=(1.6, 2.5), yticks=np.linspace(1.6, 2.4, 5), single_column=False, save_file=None):
    """Plots a figure comparing true and estimated integrated covariance matrix, as well as their difference."""

    w = SINGLE_COLUMN if single_column else DOUBLE_COLUMN
    fig = plt.figure(figsize=(w, w / 3))

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
    # plt.xlabel("Error of estimated covariances, " + r'$\hat{C}_{ij} - C_{ij}$', fontsize=SIZELABEL)
    # plt.xlabel("Error of estimated covariances, " + r'$\hat{C}_{ij} - C_{ij}$', fontsize=SIZELABEL)
    # plt.xlabel("Error of estimated covariances " + '$\it{what}$', fontsize=SIZELABEL)
    plt.xlabel("Error of estimated covariances, $\it{\hat{C}-C}$", fontsize=SIZELABEL)
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
    plt.ylim(ylim)
    plt.yticks(yticks)
    plt.xticks(ticks=xticks, labels=window_list, fontsize=SIZELABEL)
    plt.xlabel(XLABEL_WINDOW, fontsize=SIZELABEL)
    plt.text(x=-0.14, y=1, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    ax.tick_params(**DEF_TICKPROPS)

    plt.subplots_adjust(0.03, 0.152, 0.99, 0.96)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_ample_data(performances, truncate_list=TRUNCATE, plotUnnormalized=False, arrange_vertically=False, save_file=None, ylim_spearmanr=YLIM_SPEARMANR, yticks_spearmanr=YTICKS_SPEARMANR, ylim_MAE=YLIM_MAE, yticks_MAE=YTICKS_MAE, ylabelpad=2):
    """Plots a figure comparing performances of the SL, MPL and est methods using complete data."""

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    if arrange_vertically:
        fig, axes = plt.subplots(2, 1, figsize=(w, goldh))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0  # ample data
    plot_data = [np.mean(spearmanr_basic[:, :, :, p, q], axis=(1, 2)), np.mean(error_basic[:, :, :, p, q], axis=(1, 2))]
    ylabels = [YLABEL_SPEARMANR, YLABEL_MAE]
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
            plt.ylim(ylim_spearmanr)
            ax.set_yticks(yticks_spearmanr)
            plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=ylabelpad)
            if arrange_vertically:
                ax.set_xticklabels([])
        elif i == 1:
            plt.ylim(ylim_MAE)
            ax.set_yticks(yticks_MAE)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=ylabelpad)
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


def plot_figure_performance_with_limited_data(performances, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, ylim=YLIM_SPEARMANR, yticks=YTICKS_SPEARMANR, save_file=None):
    """Plots a figure comparing performances of the SL, MPL and est methods using limited (subsampled) data."""

    spearmanr_basic = performances['spearmanr_basic']

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    truncate = truncate_list[truncate_index]
    plot_data = [np.mean(spearmanr_basic[truncate_index, :, :, :, 0], axis=(0, 1)),
                 np.mean(spearmanr_basic[truncate_index, :, :, 0, :], axis=(0, 1))]
    plot_x = [sample_list, record_list]
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
            plt.xlabel('Time intervals between sampling\n' + DELTA_G + ' (generation)', fontsize=SIZELABEL)
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


def plot_figure_performance_with_linear_shrinkage(performances, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, save_file=None):
    """Plots a figure showing performances with different strengths of linear shrinkage under limited sampling effects."""

    spearmanr_basic, spearmanr_linear = performances['spearmanr_basic'], performances['spearmanr_linear']

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
    xlabels = ['Number of samples drawn\nat each generation', 'Time intervals between sampling\n' + DELTA_G + ' (generation)']
    xticklabels = [sample_list, record_list]
    xticks = [ticks_for_heatmap(len(_)) for _ in xticklabels]
    ylabel = r'$\it{\frac{\gamma}{\Delta g}}$'
    # ylabel = r'$\gamma \: / \: \Delta g$'
    yticklabels = ['Est'] + [str(gamma) for gamma in linear_list]
    yticks = ticks_for_heatmap(len(yticklabels))

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS, linewidths=HEATMAP_LINEWIDTH)

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


def plot_figure_performance_with_nonlinear_shrinkage(spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None):
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


def plot_figure_performance_with_both_regularizations(spearmanr_linear, spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None):
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


def plot_figure_optimal_factor_vs_var_cov_std_ratio(performances, optimal_by_MAE=False, use_true_as_base=False, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, alpha=0.5):

    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']

    ratios = [std_cov[tr] / std_var[tr], std_cov_est[tr] / std_var_est[tr], std_cov_nonlinear[tr] / std_var_nonlinear[tr],
              std_var[tr] / std_cov[tr], std_var_est[tr] / std_cov_est[tr], std_var_nonlinear[tr] / std_cov_nonlinear[tr]]
    ratios = [np.mean(_, axis=(0, 1)) for _ in ratios]

    spearmanrs = [spearmanr_shrink[tr, :, :, :, :], spearmanr_dcorr_shrink[tr, :, :, :, :]]
    spearmanrs = [np.mean(_, axis=(0, 1)) for _ in spearmanrs]

    errors = [error_shrink[tr, :, :, :, :], error_dcorr_shrink[tr, :, :, :, :]]
    errors = [np.mean(_, axis=(0, 1)) for _ in errors]

    optimal_factor_shrink = np.zeros((len(SAMPLE), len(RECORD)))
    optimal_factor_dcorr_shrink = np.zeros((len(SAMPLE), len(RECORD)))

    for p in range(len(SAMPLE)):
        for q in range(len(RECORD)):
            i = np.argmin(errors[0][p, q]) if optimal_by_MAE else np.argmax(spearmanrs[0][p, q])
            optimal_factor_shrink[p, q] = SHRINKING_FACTORS[i]

            i = np.argmin(errors[1][p, q]) if optimal_by_MAE else np.argmax(spearmanrs[1][p, q])
            optimal_factor_dcorr_shrink[p, q] = SHRINKING_FACTORS[i]

    S = len(SHRINKING_FACTORS)

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    ys_list = [optimal_factor_shrink, optimal_factor_dcorr_shrink]
    if use_true_as_base:
        xs_list = [ratios[4] / ratios[3], ratios[5] / ratios[3]]
    else:
        xs_list = [ratios[4], ratios[5]]
    ylabel = 'Optimal factor, ' + r'$\it{f^{*}}$'
    if use_true_as_base:
        # xlabels = ['est_var_cov_ratio /\ntrue_var_cov_ratio', 'nonlinear_var_cov_ratio /\ntrue_var_cov_ratio']
        xlabels = [r'$\it{\sigma(True\_cov) \: / \: \sigma(Est\_cov)}$', r'$\it{\frac{\sigma(Nonlinear\_var)}{\sigma(Nonlinear\_cov)}  \: /  \: \frac{\sigma(True\_var)}{\sigma(True\_cov)}}$']
    else:
        xlabels = ['std_var / std_cov', 'std_var / std_cov']
    titles = ['Est-norm', 'Nonlinear-norm']
    for i, (xs, ys, xlabel) in enumerate(zip(xs_list, ys_list, xlabels)):
        xs, ys = np.ndarray.flatten(xs), np.ndarray.flatten(ys)
        plt.sca(axes[i])
        ax = plt.gca()
        plt.scatter(xs, ys, alpha=alpha)
        regression = stats.linregress(xs, ys)
        slope, intercept = regression[0], regression[1]
        plot_line_from_slope_intercept(slope, intercept)
        plt.plot()
        if i == 0:
            plt.ylabel(ylabel, fontsize=SIZELABEL, labelpad=2)
        plt.xlabel(xlabel, fontsize=SIZESUBLABEL, labelpad=2)
        plt.title(titles[i], fontsize=SIZELABEL)
        # text = f'{SPEARMANR} = %.2f\n' % stats.spearmanr(xs, ys)[0] + f'{PEARSONR} = %.2f\n' % stats.pearsonr(xs, ys)[0] + '\nLinear regression:\n' + r'$\it{y}$=%.2f+%.2f$\it{x}$' % (intercept, slope)
        text = 'Linear regression:\n   ' + r'$\it{y}$=%.2f+%.2f$\it{x}$' % (intercept, slope) + f'\n{PEARSONR} = %.2f\n' % stats.pearsonr(xs, ys)[0]
        plt.text(0.015, 0.75, text, fontsize=SIZELABEL, transform=ax.transAxes)
        # plt.title(titles[i] + '\nSpearmanr=%.2f' % (stats.spearmanr(xs, ys)[0]) + '\nPearsonr=%.2f' % (stats.pearsonr(xs, ys)[0]) + '\nSlope=%.2f' % (slope) + '\nIntercept=%.2f' % (intercept), fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_optimal_factor_vs_var_cov_std_ratio_by_bins(performances, bins=np.linspace(0.4, 1.2, 11), bins_dcorr=np.linspace(0.8, 2.2, 11), histogram=False, quantile=False, num_quantiles=15, optimal_by_MAE=False, use_true_as_base=False, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, p=0, q=0, alpha=1, ylim=(0.25, 1.05), yticks=np.arange(0.3, 1.1, 0.1)):

    spearmanr_est, error_est = performances['spearmanr_basic'][tr, :, :, :, :, 2], performances['error_basic'][tr, :, :, :, :, 2]
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'][tr], performances['error_shrink'][tr]
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'][tr, :, :, :, :, loss_selected, gamma_selected], performances['error_dcorr'][tr, :, :, :, :, loss_selected, gamma_selected]
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'][tr], performances['error_dcorr_shrink'][tr]
    # Concatenate with the results without multiplyin a factor, as results of factor=1
    spearmanr_shrink = np.concatenate((spearmanr_shrink, np.expand_dims(spearmanr_est, axis=-1)), axis=4)
    spearmanr_dcorr_shrink = np.concatenate((spearmanr_dcorr_shrink, np.expand_dims(spearmanr_dcorr, axis=-1)), axis=4)

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'][tr], performances['std_cov'][tr], performances['std_var_est'][tr], performances['std_cov_est'][tr], performances['std_var_nonlinear'][tr], performances['std_cov_nonlinear'][tr]

    ratios_est = (std_var_est / std_cov_est) / (std_var / std_cov)
    ratios_nonlinear = (std_var_nonlinear / std_cov_nonlinear) / (std_var / std_cov)

    if histogram:
        _, bins = np.histogram(ratios_est, bins='auto')
        _, bins_dcorr = np.histogram(ratios_nonlinear, bins='auto')
        bins = bins[1:-1]
        bins_dcorr = bins_dcorr[1:-1]
    elif quantile:
        quantiles = np.linspace(0, 1, num_quantiles + 1)
        bins = stats.mstats.mquantiles(np.ndarray.flatten(ratios_est), quantiles)
        bins_dcorr = stats.mstats.mquantiles(np.ndarray.flatten(ratios_nonlinear), quantiles)

    bins_indices_est = np.digitize(ratios_est, bins)
    spearmanr_shrink_by_bins = [[] for _ in range(len(bins) + 1)]
    ratios_shrink_by_bins = [[] for _ in range(len(bins) + 1)]

    bins_indices_nonlinear = np.digitize(ratios_nonlinear, bins_dcorr)
    spearmanr_dcorr_shrink_by_bins = [[] for _ in range(len(bins_dcorr) + 1)]
    ratios_dcorr_shrink_by_bins = [[] for _ in range(len(bins_dcorr) + 1)]

    # print(spearmanr_shrink.shape)
    # print(np.max(bins_indices_est), len(spearmanr_shrink_by_bins))
    count_OOB_est = 0
    count_OOB_nonlinear = 0
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            for p in range(len(SAMPLE)):
                for q in range(len(RECORD)):
                    i = bins_indices_est[s, n, p, q]
                    if i >= len(bins):
                        count_OOB_est += 1
                    spearmanr_shrink_by_bins[i].append(spearmanr_shrink[s, n, p, q])
                    ratios_shrink_by_bins[i].append(ratios_est[s, n, p, q])

                    i = bins_indices_nonlinear[s, n, p, q]
                    if i >= len(bins):
                        count_OOB_nonlinear += 1
                    spearmanr_dcorr_shrink_by_bins[i].append(spearmanr_dcorr_shrink[s, n, p, q])
                    ratios_dcorr_shrink_by_bins[i].append(ratios_nonlinear[s, n, p, q])

    # print(count_OOB_est, count_OOB_nonlinear)

    optimal_factor_shrink = np.zeros(len(bins) + 1)
    optimal_factor_dcorr_shrink = np.zeros(len(bins_dcorr) + 1)
    mean_ratio_shrink = np.zeros(len(bins) + 1)
    mean_ratio_dcorr_shrink = np.zeros(len(bins_dcorr) + 1)

    shrinking_factors = list(SHRINKING_FACTORS) + [1]

    for i in range(len(bins) + 1):
        # print(len(ratios_shrink_by_bins[i]))
        mean = np.mean(np.array(spearmanr_shrink_by_bins[i]), axis=(0))
        optimal_factor_shrink[i] = shrinking_factors[np.argmax(mean)]
        mean_ratio_shrink[i] = np.mean(ratios_shrink_by_bins[i])

    for i in range(len(bins_dcorr) + 1):
        # print(len(ratios_dcorr_shrink_by_bins[i]))
        mean = np.mean(np.array(spearmanr_dcorr_shrink_by_bins[i]), axis=(0))
        optimal_factor_dcorr_shrink[i] = shrinking_factors[np.argmax(mean)]
        mean_ratio_dcorr_shrink[i] = np.mean(ratios_dcorr_shrink_by_bins[i])

    S = len(SHRINKING_FACTORS)

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    indices_shrink = [i for i in range(len(ratios_shrink_by_bins)) if len(ratios_shrink_by_bins[i]) > 10]
    indices_dcorr_shrink = [i for i in range(len(ratios_dcorr_shrink_by_bins)) if len(ratios_dcorr_shrink_by_bins[i]) > 10]
    # print(num_quantiles, len(indices_shrink), len(indices_dcorr_shrink))
    ys_list = [optimal_factor_shrink[indices_shrink], optimal_factor_dcorr_shrink[indices_dcorr_shrink]]
    xs_list = [mean_ratio_shrink[indices_shrink], mean_ratio_dcorr_shrink[indices_dcorr_shrink]]
    ylabel = 'Optimal factor, ' + r'$\it{f^{*}}$'
    if use_true_as_base:
        xlabels = [r'$\it{\sigma(True\_cov) \: / \: \sigma(Est\_cov)}$', r'$\it{\frac{\sigma(Nonlinear\_var)}{\sigma(Nonlinear\_cov)}  \: /  \: \frac{\sigma(True\_var)}{\sigma(True\_cov)}}$']
    else:
        xlabels = ['std_var / std_cov', 'std_var / std_cov']
    titles = ['Est-norm', 'Nonlinear-norm']
    for i, (xs, ys, xlabel) in enumerate(zip(xs_list, ys_list, xlabels)):
        xs, ys = np.ndarray.flatten(xs), np.ndarray.flatten(ys)
        plt.sca(axes[i])
        ax = plt.gca()
        ys_smaller_than_one = ys[ys < 1]
        # print(len(ys), len(ys_smaller_than_one))
        if i == 0:
            plt.scatter(xs[ys < 1], ys[ys < 1], s=SIZEDOT, alpha=alpha)
            plt.scatter(xs[ys == 1], ys[ys == 1], s=SIZEDOT, alpha=alpha)
        else:
            plt.scatter(xs[ys < 1], ys[ys < 1], s=SIZEDOT, alpha=alpha, label=r'$\it{f^{*} < 1}$')
            plt.scatter(xs[ys == 1], ys[ys == 1], s=SIZEDOT, alpha=alpha, label=r'$\it{f^{*} = 1}$')
        regression = stats.linregress(xs[ys < 1], ys[ys < 1])
        slope, intercept = regression[0], regression[1]
        plot_line_from_slope_intercept(slope, intercept, color='grey', linewidth=SIZELINE)
        plt.plot()
        if i == 0:
            plt.ylabel(ylabel, fontsize=SIZELABEL, labelpad=2)
            plt.yticks(ticks=yticks, labels=['%.1f' % _ for _ in yticks])
        else:
            plt.legend(fontsize=SIZELABEL, loc='center right', frameon=True)
            plt.yticks(ticks=[], labels=[])
        plt.ylim(ylim)
        plt.xlabel(xlabel, fontsize=SIZESUBLABEL, labelpad=2)
        plt.title(titles[i], fontsize=SIZELABEL)
        # text = f'{SPEARMANR} = %.2f\n' % stats.spearmanr(xs, ys)[0] + f'{PEARSONR} = %.2f\n' % stats.pearsonr(xs, ys)[0] + '\nLinear regression:\n' + r'$\it{y}$=%.2f+%.2f$\it{x}$' % (intercept, slope)
        text = 'Linear regression:\n   ' + r'$\it{y}$=%.2f+%.2f$\it{x}$' % (intercept, slope) + f'\n{PEARSONR} = %.2f\n' % stats.pearsonr(xs, ys)[0]
        # plt.text(0.015, 0.75, text, fontsize=SIZELABEL, transform=ax.transAxes)
        if i == 0:
            plt.text(0.15, 0.7, text, fontsize=SIZELABEL, transform=ax.transAxes)
        else:
            plt.text(0.15, 0, text, fontsize=SIZELABEL, transform=ax.transAxes)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_multiple_replicates(performances, performances_combine, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_selected=LINEAR_SELECTED,  loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, save_file=None, ylim=YLIM_SPEARMANR, yticks=YTICKS_SPEARMANR, w_h_ratio=1.55, alpha=0.75, suptitle=None, double_column=False):
    """Plots a figure comparing performances of all methods (SL, est, MPL, linear shrinkage, nonlinear shrinkage) using replicates individually, and combining multiple replicates for inference."""

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_shrink, spearmanr_dcorr_shrink = performances['spearmanr_basic'], performances['spearmanr_linear'], performances['spearmanr_dcorr'], performances['spearmanr_shrink'], performances['spearmanr_dcorr_shrink']

    spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, spearmanr_shrink_combine, spearmanr_dcorr_shrink_combine = performances_combine['spearmanr_basic_combine'], performances_combine['spearmanr_linear_combine'], performances_combine['spearmanr_dcorr_combine'], performances_combine['spearmanr_shrink_combine'], performances_combine['spearmanr_dcorr_shrink_combine']

    # In figure_performance_with_limited_data(), it generates figure where the plotting area is of size (SINGLE_COLUMN * (FIG4_RIGHT - FIG4_LEFT), SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM))
    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / w_h_ratio
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    p, q = 0, 0
    xlabel = "Generations of data used"
    ylabel = YLABEL_SPEARMANR
    xs = truncate_list
    ys_lists = [
        [np.mean(spearmanr_basic[:, :, :, p, q, 0], axis=(1, 2)),
         np.mean(spearmanr_basic[:, :, :, p, q, 1], axis=(1, 2)),
         np.mean(spearmanr_basic[:, :, :, p, q, 2], axis=(1, 2)),
         np.mean(spearmanr_linear[:, :, :, p, q, linear_selected], axis=(1, 2)),
         np.mean(spearmanr_dcorr[:, :, :, p, q, loss_selected, gamma_selected], axis=(1, 2)),
         np.mean(spearmanr_shrink[:, :, :, p, q, shrink_selected], axis=(1, 2)),
         np.mean(spearmanr_dcorr_shrink[:, :, :, p, q, dcorr_shrink_selected], axis=(1, 2)),
         ],
        [np.mean(spearmanr_basic_combine[:, :, -1, p, q, 0], axis=(1)),
         np.mean(spearmanr_basic_combine[:, :, -1, p, q, 1], axis=(1)),
         np.mean(spearmanr_basic_combine[:, :, -1, p, q, 2], axis=(1)),
         np.mean(spearmanr_linear_combine[:, :, -1, p, q, linear_selected], axis=(1)),
         np.mean(spearmanr_dcorr_combine[:, :, -1, p, q, loss_selected, gamma_selected], axis=(1)),
         np.mean(spearmanr_shrink_combine[:, :, -1, p, q, shrink_selected], axis=(1)),
         np.mean(spearmanr_dcorr_shrink_combine[:, :, -1, p, q, dcorr_shrink_selected], axis=(1)),
         ],
    ]
    labels = LABELS_ALL_METHODS_NORM
    if not double_column:
        labels[-1] = LABEL_NONLINEAR_SHRINK_TWOLINE
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM

    for i, ys_list in enumerate(ys_lists):
        plt.sca(axes[i])
        ax = plt.gca()
        for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
            plot_scatter_and_line(xs, ys, color=color, s=SIZEDOT, label=label, edgecolors=EDGECOLORS, marker=marker, alpha=alpha, linewidth=SIZELINE)

        at_left = (i == 0)
        plot_legend = (i == 1)
        plt.ylabel(ylabel if at_left else "", fontsize=SIZELABEL)
        if double_column:
            plt.xlabel(xlabel, fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.yticks(ticks=yticks, labels=yticks if at_left else [], rotation=0)
        plt.ylim(ylim)
        if plot_legend:
            if double_column:
                plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
            else:
                plt.legend(loc='center', bbox_to_anchor=(0.47, 0.3), frameon=False, fontsize=SIZELEGEND)
        if double_column:
            plt.text(x=-0.15, y=1.015, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    if not double_column:
        add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    top = 0.93
    bottom = top - (FIG4_TOP - FIG4_BOTTOM) / GOLD_RATIO * w_h_ratio
    plt.subplots_adjust(FIG4_LEFT, bottom, FIG4_RIGHT, top, **DEF_SUBPLOTS_ADJUST_1x2)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=SIZESUBLABEL, y=1.05)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_multiple_replicates_2x2(performances, performances_combine, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, p=0, q=0, linear_selected=LINEAR_SELECTED,  loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, save_file=None, ylim=YLIM_SPEARMANR, yticks=YTICKS_SPEARMANR, ylim_bottom=(0.0025, 0.0225), yticks_bottom=[0.005, 0.01, 0.015, 0.02], w_h_ratio=1, alpha=0.75, suptitle=None, double_column=False):
    """Plots a figure comparing performances of all methods (SL, est, MPL, linear shrinkage, nonlinear shrinkage) using replicates individually, and combining multiple replicates for inference."""

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr, spearmanr_shrink, spearmanr_dcorr_shrink = performances['spearmanr_basic'], performances['spearmanr_linear'], performances['spearmanr_dcorr'], performances['spearmanr_shrink'], performances['spearmanr_dcorr_shrink']

    spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, spearmanr_shrink_combine, spearmanr_dcorr_shrink_combine = performances_combine['spearmanr_basic_combine'], performances_combine['spearmanr_linear_combine'], performances_combine['spearmanr_dcorr_combine'], performances_combine['spearmanr_shrink_combine'], performances_combine['spearmanr_dcorr_shrink_combine']

    error_basic, error_linear, error_dcorr, error_shrink, error_dcorr_shrink = performances['error_basic'], performances['error_linear'], performances['error_dcorr'], performances['error_shrink'], performances['error_dcorr_shrink']

    error_basic_combine, error_linear_combine, error_dcorr_combine, error_shrink_combine, error_dcorr_shrink_combine = performances_combine['error_basic_combine'], performances_combine['error_linear_combine'], performances_combine['error_dcorr_combine'], performances_combine['error_shrink_combine'], performances_combine['error_dcorr_shrink_combine']

    # In figure_performance_with_limited_data(), it generates figure where the plotting area is of size (SINGLE_COLUMN * (FIG4_RIGHT - FIG4_LEFT), SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM))
    # w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN #SLIDE_WIDTH
    # goldh = w / w_h_ratio

    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    height = SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM)
    left, right = FIG4_LEFT, FIG4_RIGHT
    bottom, top = 0.0705, 0.958
    wspace, hspace = 0.4, 0.25
    sublabel_x, sublabel_y = -0.1, 1.072
    goldh = height * (2 + hspace) / (top - bottom)

    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    xlabel = "Generations of data used"
    ylabel = YLABEL_SPEARMANR_THREE if double_column else YLABEL_SPEARMANR
    ylabel_bottom = YLABEL_MAE
    xs = truncate_list
    ys_lists = [
        [np.mean(spearmanr_basic[:, :, :, p, q, 0], axis=(1, 2)),
         np.mean(spearmanr_basic[:, :, :, p, q, 1], axis=(1, 2)),
         np.mean(spearmanr_basic[:, :, :, p, q, 2], axis=(1, 2)),
         np.mean(spearmanr_linear[:, :, :, p, q, linear_selected], axis=(1, 2)),
         np.mean(spearmanr_dcorr[:, :, :, p, q, loss_selected, gamma_selected], axis=(1, 2)),
         # np.mean(spearmanr_shrink[:, :, :, p, q, shrink_selected], axis=(1, 2)),
         # np.mean(spearmanr_dcorr_shrink[:, :, :, p, q, dcorr_shrink_selected], axis=(1, 2)),
         ],
        [np.mean(spearmanr_basic_combine[:, :, -1, p, q, 0], axis=(1)),
         np.mean(spearmanr_basic_combine[:, :, -1, p, q, 1], axis=(1)),
         np.mean(spearmanr_basic_combine[:, :, -1, p, q, 2], axis=(1)),
         np.mean(spearmanr_linear_combine[:, :, -1, p, q, linear_selected], axis=(1)),
         np.mean(spearmanr_dcorr_combine[:, :, -1, p, q, loss_selected, gamma_selected], axis=(1)),
         # np.mean(spearmanr_shrink_combine[:, :, -1, p, q, shrink_selected], axis=(1)),
         # np.mean(spearmanr_dcorr_shrink_combine[:, :, -1, p, q, dcorr_shrink_selected], axis=(1)),
         ],
        [np.mean(error_basic[:, :, :, p, q, 0], axis=(1, 2)),
         np.mean(error_basic[:, :, :, p, q, 1], axis=(1, 2)),
         np.mean(error_basic[:, :, :, p, q, 2], axis=(1, 2)),
         np.mean(error_linear[:, :, :, p, q, linear_selected], axis=(1, 2)),
         np.mean(error_dcorr[:, :, :, p, q, loss_selected, gamma_selected], axis=(1, 2)),
         # np.mean(error_shrink[:, :, :, p, q, shrink_selected], axis=(1, 2)),
         # np.mean(error_dcorr_shrink[:, :, :, p, q, dcorr_shrink_selected], axis=(1, 2)),
         ],
        [np.mean(error_basic_combine[:, :, -1, p, q, 0], axis=(1)),
         np.mean(error_basic_combine[:, :, -1, p, q, 1], axis=(1)),
         np.mean(error_basic_combine[:, :, -1, p, q, 2], axis=(1)),
         np.mean(error_linear_combine[:, :, -1, p, q, linear_selected], axis=(1)),
         np.mean(error_dcorr_combine[:, :, -1, p, q, loss_selected, gamma_selected], axis=(1)),
         # np.mean(error_shrink_combine[:, :, -1, p, q, shrink_selected], axis=(1)),
         # np.mean(error_dcorr_shrink_combine[:, :, -1, p, q, dcorr_shrink_selected], axis=(1)),
         ],

    ]
    labels = LABELS_ALL_METHODS_NORM
    if not double_column:
        labels[-1] = LABEL_NONLINEAR_SHRINK_TWOLINE
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM
    titles = ['Using single replicate', 'Combining 20 replicates']

    for i, ys_list in enumerate(ys_lists):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
            plot_scatter_and_line(xs, ys, color=color, s=SIZEDOT, label=label, edgecolors=EDGECOLORS, marker=marker, alpha=alpha, linewidth=SIZELINE)

        at_left = (i % nCol == 0)
        at_top = (i // nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        plot_legend = (i == 1)
        if double_column and at_bottom:
            plt.xlabel(xlabel, fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if at_top:
            plt.ylabel(ylabel if at_left else "", fontsize=SIZELABEL)
            plt.yticks(ticks=yticks, labels=yticks if at_left else [], rotation=0)
            plt.ylim(ylim)
            plt.title(titles[i % nCol], fontsize=SIZELABEL)
        else:
            plt.ylabel(ylabel_bottom if at_left else "", fontsize=SIZELABEL)
            plt.yticks(ticks=yticks_bottom, labels=['%.3f'%_ for _ in yticks_bottom] if at_left else [], rotation=0)
            plt.ylim(ylim_bottom)
        if plot_legend:
            if double_column:
                plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
            else:
                plt.legend(loc='center', bbox_to_anchor=(0.67, 0.23), frameon=False, fontsize=SIZELEGEND)
        if double_column:
            plt.text(x=-0.15, y=1.015, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    if not double_column:
        add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    # top = 0.93
    # bottom = top - (FIG4_TOP - FIG4_BOTTOM) / GOLD_RATIO * w_h_ratio
    # plt.subplots_adjust(FIG4_LEFT, bottom, FIG4_RIGHT, top, **DEF_SUBPLOTS_ADJUST_1x2)
    if double_column:
        plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    else:
        plt.subplots_adjust(left, bottom, right, top, wspace=wspace, hspace=hspace)
    if suptitle is not None:
        plt.suptitle(suptitle, fontsize=SIZESUBLABEL, y=1.05)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_combining_different_numbers_of_replicates(performances_combine, num_basic=3, truncate_index=None, linear_selected=LINEAR_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, p=1, min_t=None, max_t=None, max_q=None, ylim=YLIM_SPEARMANR_COMBINE, yticks=YTICKS_SPEARMANR_COMBINE, alpha=0.75, double_column=False, save_file=None):
    """Plots a figure showing performances combining different numbers of replicates under limited sampling effects."""

    spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, spearmanr_shrink_combine, spearmanr_dcorr_shrink_combine = performances_combine['spearmanr_basic_combine'], performances_combine['spearmanr_linear_combine'], performances_combine['spearmanr_dcorr_combine'], performances_combine['spearmanr_shrink_combine'], performances_combine['spearmanr_dcorr_shrink_combine']

    # In figure_performance_with_limited_data(), it generates figure where the plotting area is of size (SINGLE_COLUMN * (0.97- 0.115), SINGLE_COLUMN / GOLD_RATIO * (0.935 - 0.223))
    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    height = SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM)
    left, right = FIG4_LEFT, FIG4_RIGHT
    bottom, top = 0.0705, 0.958
    wspace, hspace = 0.4, 0.25
    sublabel_x, sublabel_y = -0.1, 1.072
    goldh = height * (2 + hspace) / (top - bottom)

    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    if truncate_index is not None:
        spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine = (spearmanr_basic_combine[truncate_index], spearmanr_linear_combine[truncate_index], spearmanr_dcorr_combine[truncate_index])

    # num_basic = 4
    mean_spearmanr_basic = np.mean(spearmanr_basic_combine, axis=(1))
    mean_spearmanr_linear = np.mean(spearmanr_linear_combine[:, :, :, :, :, linear_selected], axis=(1))
    mean_spearmanr_shrink = np.mean(spearmanr_shrink_combine[:, :, :, :, :, shrink_selected], axis=(1))
    mean_spearmanr_dcorr = np.mean(spearmanr_dcorr_combine[:, :, :, :, :, loss_selected, gamma_selected], axis=(1))
    mean_spearmanr_dcorr_shrink = np.mean(spearmanr_dcorr_shrink_combine[:, :, :, :, :, dcorr_shrink_selected], axis=(1))
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

    xs = number_replicates[indices]
    other_methods_list = [mean_spearmanr_linear, mean_spearmanr_dcorr,
                          # mean_spearmanr_shrink, mean_spearmanr_dcorr_shrink
                          ]
    yss_list = [np.concatenate((
        mean_spearmanr_basic[t, :, p, q, :num_basic][indices].T,
        [_[t, :, p, q][indices] for _ in other_methods_list]), axis=0)
               for (t, q) in tq_list]
    yss_list = np.array(yss_list)
    colors = COLORS_ALL_METHODS_NORM
    labels = LABELS_ALL_METHODS_NORM
    if not double_column:
        labels[-1] = LABEL_NONLINEAR_SHRINK_TWOLINE
    markers = MARKERS_ALL_METHODS_NORM

    for i, ((t, q), yss) in enumerate(zip(tq_list, yss_list)):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        for j, (ys, color, label, marker) in enumerate(zip(yss, colors, labels, markers)):
            plot_scatter_and_line(xs, ys, color=color, label=label, marker=marker, alpha=alpha)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=xticks, labels=xticks if at_bottom else [])
        plt.xlabel('Number of replicates' if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        plt.ylabel(YLABEL_SPEARMANR if at_left else '', fontsize=SIZELABEL)
        plt.yticks(ticks=yticks, labels=yticks if at_left else [], rotation=0)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 2:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        if double_column:
            plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"{truncate_list[t]} generations, " + DELTA_G + f"={record_list[q]}", fontsize=SIZELABEL)

    if double_column:
        plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    else:
        plt.subplots_adjust(left, bottom, right, top, wspace=wspace, hspace=hspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_real_data(data, selection_list, fitness_list, labels=None, ref_index=0, f_est=0.5, f_nonlinear=0.5, double_column=False, colors=COLORS_TAB, alpha=0.75, plot_evoracle=True, plot_nonlinear=True, plot_title=False, use_pearsonr=False, use_MAE=True, title_pad=2, save_file=None):

    if labels is None:
        # labels = ['MPL', f'Est-norm, f={f_est}',  f'Nonlinear-norm, f={f_nonlinear}', 'SL', 'Evoracle']
        labels = ['MPL', f'Est',  f'Nonlinear', 'SL', 'Evoracle']

    traj, times = data['traj'], data['times']

    if double_column:
        pass
    else:
        w       = SINGLE_COLUMN
        h_w_ratio = 1.78
        goldh   = w * h_w_ratio
        fig     = plt.figure(figsize=(w, goldh))

        global_top   = 0.95
        traj_bottom  = 0.75
        traj_perf_hspace = 0.05
        perf_top = traj_bottom - traj_perf_hspace
        global_left  = 0.07
        global_right = 0.995 if double_column else 0.94

        global_bottom = 0.08
        ddy = 0.027 if plot_title else 0.02  # hspace between perf subplots
        dy = (perf_top - global_bottom - 2 * ddy) / 3  # height of each perf subplot
        wspace = (global_right - global_left - dy * h_w_ratio * 2) / (dy * h_w_ratio)
        # print(wspace)

        boxes = [dict(left=global_left,
                      right=global_right,
                      bottom=traj_bottom,
                      top=global_top)] + [dict(left=global_left,
                      right=global_right,
                      bottom=perf_top-((i+1)*dy)-(i*ddy),
                      top=perf_top-(i*dy)-(i*ddy)) for i in range(len(selection_list) - 1)]
        gridspecs = [gridspec.GridSpec(1, 1, **boxes[0])] + [gridspec.GridSpec(1, 2, wspace=wspace, **box) for box in boxes[1:]]
        axes = [plt.subplot(gridspecs[0][0, 0])]
        for _ in gridspecs[1:]:
            axes += [plt.subplot(_[0, 0]), plt.subplot(_[0, 1])]

    sublabel_x, sublabel_y = -0.5, 1.05
    ylabelpad = 4

    plt.sca(axes[0])
    ax = plt.gca()
    for l in range(len(traj[0])):
        plt.plot(times, traj[:, l], color=colors[l%len(colors)], linewidth=SIZELINE, alpha=alpha)
    plt.ylabel('Mutant allele frequency', fontsize=SIZELABEL)
    plt.xlabel('Time (h)', fontsize=SIZELABEL)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    width_ratio = (global_right - global_left) / (dy * h_w_ratio)
    plt.text(x=sublabel_x / width_ratio, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # plt.text(x=-0.2, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    ax_index = 1
    min, max = 0, 0.04
    ylim = (-0.008, 0.072)
    xticks = np.linspace(0, 0.06, 4)
    yticks = np.linspace(0, 0.06, 4)
    xticklabels = ['%.2f'%_ for _ in xticks]
    yticklabels = ['%.2f'%_ for _ in yticks]
    xlabel = 'Selection inferred with\ntrue covariance'
    # ylabels = ['', 'Selection inferred with\n' + LABEL_SHRINK, 'Selection inferred with\n' + LABEL_NONLINEAR_SHRINK, 'Selection inferred when\nignoring linkage', 'Selection inferred\nwith Evoracle']
    label_reg = LABEL_NONLINEAR if plot_nonlinear else LABEL_LINEAR
    if plot_evoracle:
        ylabels = ['', 'Selection inferred\nwith Evoracle', 'Selection inferred\nwith ' + label_reg, 'Selection inferred when\nignoring linkage']
    else:
        ylabels = ['', 'Selection inferred\nwith ' + LABEL_EST, 'Selection inferred\nwith ' + label_reg, 'Selection inferred when\nignoring linkage']
    for i, selection in enumerate(selection_list):
        if i == ref_index:
            continue
        plt.sca(axes[ax_index])
        ax = plt.gca()
        at_bottom = (ax_index > (len(selection_list) - 2) * 2)
        plot_comparison(selection_list[ref_index], selection, ylim=ylim, xticks=xticks, yticks=yticks, xticklabels=xticklabels if at_bottom else [], yticklabels=yticklabels, ylabel=ylabels[i], ylabelpad=ylabelpad if i < 3 else None, xlabel=xlabel if at_bottom else None, use_pearsonr=use_pearsonr, use_MAE=use_MAE, plot_title=plot_title, title_pad=title_pad)
        ax_index += 2
        plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[ax_index//2], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    ax_index = 2
    min, max = 1, 1.5
    ylim = (0.88, 1.72)
    xticks = np.linspace(1, 1.6, 4)
    yticks = np.linspace(1, 1.6, 4)
    xticklabels = ['%.2f'%_ for _ in xticks]
    yticklabels = ['%.2f'%_ for _ in yticks]
    xlabel = 'Fitness inferred with\ntrue covariance'
    # ylabels = ['', 'Fitness inferred with\n' + LABEL_SHRINK, 'Fitness inferred with\n' + LABEL_NONLINEAR_SHRINK, 'Fitness inferred when\nignoring linkage', 'Fitness inferred\nwith Evoracle']
    if plot_evoracle:
        ylabels = ['', 'Selection inferred\nwith Evoracle', 'Selection inferred\nwith ' + label_reg, 'Selection inferred when\nignoring linkage']
    else:
        ylabels = ['', 'Selection inferred\nwith ' + LABEL_EST, 'Selection inferred\nwith ' + label_reg, 'Selection inferred when\nignoring linkage']
    for i, fitness in enumerate(fitness_list):
        if i == ref_index:
            continue
        plt.sca(axes[ax_index])
        at_bottom = (ax_index > (len(selection_list) - 2) * 2)
        plot_comparison(fitness_list[ref_index], fitness, ylim=ylim, xticks=xticks, yticks=yticks, xticklabels=xticklabels if at_bottom else [], yticklabels=yticklabels, ylabel=ylabels[i], ylabelpad=ylabelpad if i < 3 else None, xlabel=xlabel if at_bottom else None, use_pearsonr=use_pearsonr, use_MAE=use_MAE, plot_title=plot_title, title_pad=title_pad)
        ax_index += 2

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



############# SUPPLEMENTARY FIGURES #############

def plot_supplementary_figure_selections(selections, ylim_top=(0, 15), ylim_bottom=(0, 15), range=(-0.055, 0.055), bins=12, alpha=0.7, save_file=None):
    """Plots a figure showing 10 sets of selection coefficients used in simulations."""

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 5
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    for s, selection in enumerate(selections):
        row, col = s % nRow, s // nRow
        plt.sca(axes[row, col])
        ax = plt.gca()
        plt.hist([_ for _ in selection if _ > 0], color=C_BEN_LT, bins=bins, range=range, alpha=alpha, label="Beneficial")
        plt.hist([_ for _ in selection if _ == 0], color=C_NEU_LT, bins=bins, range=range, alpha=alpha, label="Neutral")
        plt.hist([_ for _ in selection if _ < 0], color=C_DEL_LT, bins=bins, range=range, alpha=alpha, label="Deleterious")

        at_top_middle = (s == 4)
        at_top = (row == 0)
        at_left = (col == 0)
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


def compute_variance_of_cov_in_a_window(s=4, n=0, sample=1000, record=1):

    subsample = SS.load_subsample(s, n, sample, record, covAtEachTime=True)
    selection = SS.load_selection(s)

    traj, cov = subsample['traj'], subsample['covAtEachTime']
    T, L = traj.shape

    windows = WINDOW[1:]
    variance_of_cov_in_a_window = [[] for win in windows]

    for w, win in enumerate(windows):
        print(win, end=', ')
        for i in range(L):
            for j in range(i + 1):
                stds = [np.std(cov[t - win:t + win, i, j]) for t in range(win, T - win)]
                variance_of_cov_in_a_window[w].append(np.mean(stds))

    # variance_of_cov_in_a_window = [np.mean(_) for _ in variance_of_cov_in_a_window]

    return windows, variance_of_cov_in_a_window


def plot_supplementary_figure_how_covariance_change(windows, variance_of_cov_in_a_window, s=4, n=0, sample=1000, record=1, highlight_pairs=[(0, 1), (14, 30), (24, 44)], highlight_colors=[C_FOREST, C_BEN, C_DEL], alpha=0.5, xticks=np.arange(0, 701, 100), ylabelpad=6, xlim=XLIM_GENERATION, suggest_pairs=False, rasterized=True, save_file=None):

    subsample = SS.load_subsample(s, n, sample, record, covAtEachTime=True)
    selection = SS.load_selection(s)

    traj, cov = subsample['traj'], subsample['covAtEachTime']

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 1
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    windows = np.array(windows)
    xs = 2 * windows + 1
    variance_of_cov_in_a_window = np.array([np.array(_) for _ in variance_of_cov_in_a_window])

    xlabel = 'Window width, ' + r'$2\it{\delta t} + 1$'

    plt.sca(axes[0])
    ax = plt.gca()
    for i in range(len(variance_of_cov_in_a_window[0])):
        plt.plot(xs, variance_of_cov_in_a_window[:, i], linewidth=SIZELINE, linestyle='solid', alpha=alpha)
    plt.xticks(ticks=[], labels=[])
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)

    plt.sca(axes[1])
    ax = plt.gca()
    mean_variance_of_cov_in_a_window = np.mean(variance_of_cov_in_a_window, axis=1)
    mean_magnitude_of_cov = np.mean(np.absolute(cov))
    plt.plot(xs, mean_variance_of_cov_in_a_window, marker='.', linewidth=SIZELINE * 3, linestyle='solid')
    plt.plot(xs, [mean_magnitude_of_cov] * len(xs), marker='.', linewidth=SIZELINE * 3, linestyle='dashed')
    plt.xlabel(xlabel, fontsize=SIZELABEL)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)

    plt.show()

    pass


def plot_supplementary_figure_biplot_cov(dic_cov, p=2, q=3, markersize=1, alpha=0.5, xlim_top=(-80, 230), xticks_top=[-50, 0, 50, 100, 150, 200], plot_xticks_top_on_top=False, xlim_bottom=(-0.04, 0.11), xticks_bottom=[-0.02, 0.02, 0.06, 0.1], figsize=None, rasterized=True, save_file=None):

    true_cov, est_cov, linear_cov, nonlinear_cov = dic_cov['int_cov'][p, q], dic_cov['int_dcov'][p, q], dic_cov['int_dcov_linear'][p, q], dic_cov['int_dcov_dcorr_reg'][p, q]

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
        at_bottom = (i//nCol == nRow - 1)
        at_left = (i%nCol == 0)
        at_right = (i%nCol == nCol - 1)
        at_top = (i//nCol == 0)

        cov_x, cov_y = [x[i, j] for i in range(L) for j in range(i)], [y[i, j] for i in range(L) for j in range(i)]
        var_x, var_y = [x[i, i] for i in range(L)], [y[i, i] for i in range(L)]
        plt.scatter(var_x, var_y, s=markersize, alpha=0.8, label='Diagonal', color=COLOR_VARIANCE, rasterized=rasterized)
        plt.scatter(cov_x, cov_y, s=markersize, alpha=alpha, label='Off-diagonal', rasterized=rasterized)
        # min_value = min_values[i//nCol]
        # max_value = max_values[i//nCol]
        # plt.plot([min_value, max_value], [min_value, max_value], linestyle='dashed', color='grey', linewidth=SIZELINE)
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

        if at_top:
            xlim, xticks = xlim_top, xticks_top
        else:
            xlim, xticks = xlim_bottom, xticks_bottom
        plt.xlim(xlim)
        plt.ylim(xlim)
        plt.xticks(ticks=xticks, labels=xticks)
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle='dashed', color='grey', linewidth=SIZELINE)

        if plot_xticks_top_on_top and at_top:
            ax.tick_params(**DEF_TICKPROPS_TOP)
        else:
            ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)

    plt.subplots_adjust(0.08, 0.1, 0.98, 0.95, hspace=0.3)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_without_normalization(performances, truncate_list=TRUNCATE, save_file=None):
    """Plots a figure showing performances with and without calibration when estimating covariance."""

    spearmanr_basic, error_basic, spearmanr_cov_calibrated, error_cov_calibrated, spearmanr_cov_uncalibrated, error_cov_uncalibrated = performances['spearmanr_basic'], performances['error_basic'], performances['spearmanr_cov_calibrated'], performances['error_cov_calibrated'], performances['spearmanr_cov_uncalibrated'], performances['error_cov_uncalibrated']

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
    ylims = [(0.85, 0.93), (0, 0.02), (0.65, 0.95), (0.007, 0.017)]
    yticks = [np.linspace(0.86, 0.92, 4), np.linspace(0.002, 0.018, 5), np.linspace(0.7, 0.9, 3), np.linspace(0.008, 0.016, 5)]

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


def plot_supplementary_figure_performance_with_different_time_window(performances_window, window_list=WINDOW, sample_list=SAMPLE, record_list=RECORD, ylim=YLIM_SPEARMANR_COMBINE, yticks=YTICKS_SPEARMANR_COMBINE, ylabel=YLABEL_SPEARMANR, xlabel=XLABEL_WINDOW, max_p=None, max_q=None, combine=False, uncalibrated=False, alpha=1, save_file=None):
    """Plots a figure showing performances with different choices of time window."""

    spearmanr_basic_window, spearmanr_linear_window, spearmanr_dcorr_window, spearmanr_shrink_window, spearmanr_dcorr_shrink_window = performances_window['spearmanr_basic_window'], performances_window['spearmanr_linear_window'], performances_window['spearmanr_dcorr_window'], performances_window['spearmanr_shrink_window'], performances_window['spearmanr_dcorr_shrink_window']

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
    mean_spearmanr_shrink = np.mean(spearmanr_shrink_window, axis=avg_axis)
    mean_spearmanr_dcorr_shrink = np.mean(spearmanr_dcorr_shrink_window, axis=avg_axis)

    xs = np.arange(len(window_list))
    xticklabels = window_list
    mean_spearmanr_list =  [
        mean_spearmanr_basic[:, :, :, 2],  # Est
        mean_spearmanr_linear, mean_spearmanr_dcorr,
        # mean_spearmanr_shrink, mean_spearmanr_dcorr_shrink
    ]
    ys_lists = [
        [_[:, p, q] for _ in mean_spearmanr_list] for p, q in pq_list
    ]
    labels = LABELS_ALL_METHODS_NORM[2:]
    # labels[-1] = LABEL_NONLINEAR_SHRINK_TWOLINE
    colors = COLORS_ALL_METHODS_NORM[2:]
    markers = MARKERS_ALL_METHODS_NORM[2:]

    for i, (p, q) in enumerate(pq_list):
        plt.sca(axes[i // nCol, i % nCol])
        ax = plt.gca()

        ys_list = ys_lists[i]

        # SL and MPL
        for j in range(2):
            plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], alpha=alpha,
                     linewidth=SIZELINE, linestyle='dashed', color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
            plot_scatter_and_line(xs, ys, color=color, s=SIZEDOT + 10, label=label, edgecolors=EDGECOLORS, marker=marker, alpha=alpha, linewidth=SIZELINE)


        # est
        # j = 2
        # plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
        #          marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        # if uncalibrated:
        #     j = 3
        #     plt.plot(range(len(window_list)), mean_spearmanr_basic[:, p, q, j], linewidth=SIZELINE, alpha=alpha,
        #              marker=MARKER_BASIC[j], markersize=SMALLSIZEDOT - 2, color=COLOR_BASIC[j], label=LABEL_BASIC[j])

        # plt.plot(range(len(window_list)), mean_spearmanr_linear[:, p, q], linewidth=SIZELINE, alpha=alpha,
        #          marker=MARKER_LINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_LINEAR, label=LABEL_LINEAR)
        #
        # plt.plot(range(len(window_list)), mean_spearmanr_dcorr[:, p, q], linewidth=SIZELINE, alpha=alpha,
        #          marker=MARKER_NONLINEAR, markersize=SMALLSIZEDOT - 2, color=COLOR_NONLINEAR, label=LABEL_NONLINEAR)

        at_bottom = (i == 2 or i == 3)
        at_left = (i == 0 or i == 2)
        plt.xticks(ticks=xs, labels=xticklabels if at_bottom else [])
        plt.xlabel(xlabel if at_bottom else '', fontsize=SIZELABEL)
        plt.ylim(ylim)
        plt.yticks(ticks=yticks, labels=yticks)
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


def plot_supplementary_figure_performance_example(selections, s=4, n=0, p=0, q=0, xlim=YLIM_SELECTION, xticks=YTICKS_SELECTION, markersize=1, alpha=0.8, hspace=0.2, wspace=0.2, figsize=None, save_file=None):

    selections_basic, selections_linear, selections_shrink, selections_dcorr, selections_dcorr_shrink = selections['selections_basic'], selections['selections_linear'], selections['selections_shrink'], selections['selections_dcorr'], selections['selections_dcorr_shrink']

    selection_true = SS.load_selection(s)
    xs = selection_true

    selections_plot = [selections_basic, selections_linear, selections_dcorr,
                       # selections_shrink, selections_dcorr_shrink
                       ]
    selections_plot = [_[s, n, p, q] for _ in selections_plot]
    plot_data = list(selections_plot[0]) + selections_plot[1:]
    titles = LABELS_ALL_METHODS
    colors = COLORS_ALL_METHODS
    xlabel = 'True selection coefficients'
    ylabel = 'Inferred selection coefficients'
    xticklabels = [('%.2f' % _) for _ in xticks]

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    # nRow, nCol = 2, 4
    nRow, nCol = 2, 3
    goldh = w * (nRow / nCol)
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))

    for i, (ys, title, color) in enumerate(zip(plot_data, titles, colors)):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        at_bottom = (i//nCol == nRow - 1)
        at_left = (i%nCol == 0)
        at_right = (i%nCol == nCol - 1)
        at_top = (i//nCol == 0)

        plt.scatter(xs, ys, s=markersize, alpha=alpha, color=color)
        plt.xlim(xlim)
        plt.ylim(xlim)
        plt.xticks(ticks=xticks, labels=xticklabels if at_bottom else [])
        plt.yticks(ticks=xticks, labels=xticklabels if at_left else [])
        if at_left:
            plt.ylabel(ylabel, fontsize=SIZELABEL)
        if at_bottom:
            plt.xlabel(xlabel, fontsize=SIZELABEL)
        plt.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], linestyle='dashed', color='grey', linewidth=SIZELINE)

        text = f'MAE = %.3f\n' % DP.MAE(xs, ys) + f'{SPEARMANR} = %.2f\n' % stats.spearmanr(xs, ys)[0] + f'{PEARSONR} = %.2f\n' % stats.pearsonr(xs, ys)[0]
        plt.text(0.06, 0.67, text, fontsize=SIZELABEL, transform=ax.transAxes)

        plt.title(f"{title}", fontsize=SIZELABEL)
        ax.set_aspect('equal', adjustable='box')
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(0.08, 0.1, 0.98, 0.95, hspace=hspace, wspace=wspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_performance_with_different_loss_gamma(performances, truncate_index=5, ylim=(0.83, 0.929), alpha=0.5, sample_list=SAMPLE, record_list=RECORD, loss_list=LOSS, gamma_list=GAMMA, ylabel=YLABEL_SPEARMANR, save_file=None):
    """Plots a figure comparing performances of nonlinear shrinkage on correlation matrix using different loss functions & gamma."""

    spearmanr_dcorr = performances['spearmanr_dcorr']

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    max_p, max_q = len(sample_list) - 1, len(record_list) - 1
    pq_list = [(0, 0), (max_p, 0), (0, max_q), (max_p, max_q)]
    xticklabels=[r'$10^{-5}$', r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'1']
    xlabel = 'Regularization strength of nonlinear shrinkage, ' + r'$\it{\eta}$'
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


def plot_supplementary_figure_performance_all_methods(performances, metrics='error', truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, vmin=0, vmax=0.02, figsize=None, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2  # w / 2.25
    # nRow, nCol = 2, 4
    nRow, nCol = 2, 3
    fig, axes = plt.subplots(nRow, nCol, figsize=figsize if figsize is not None else (w, goldh))

    metrics_basic, metrics_linear, metrics_dcorr, metrics_shrink, metrics_dcorr_shrink = performances[f'{metrics}_basic'], performances[f'{metrics}_linear'], performances[f'{metrics}_dcorr'], performances[f'{metrics}_shrink'], performances[f'{metrics}_dcorr_shrink']

    matrix_SL = np.mean(metrics_basic[truncate_index, :, :, :, :, 0], axis=(0, 1))
    matrix_MPL = np.mean(metrics_basic[truncate_index, :, :, :, :, 1], axis=(0, 1))
    matrix_est = np.mean(metrics_basic[truncate_index, :, :, :, :, 2], axis=(0, 1))
    matrix_est_uncalibrated = np.mean(metrics_basic[truncate_index, :, :, :, :, 3], axis=(0, 1))
    matrix_linear = np.mean(metrics_linear[truncate_index, :, :, :, :, linear_selected], axis=(0, 1))
    matrix_dcorr = np.mean(metrics_dcorr[truncate_index, :, :, :, :, loss_selected, gamma_selected], axis=(0, 1))
    matrix_shrink = np.mean(metrics_shrink[truncate_index, :, :, :, :, shrink_selected], axis=(0, 1))
    matrix_dcorr_shrink = np.mean(metrics_dcorr_shrink[truncate_index, :, :, :, :, dcorr_shrink_selected], axis=(0, 1))

    plot_data = [matrix_SL, matrix_MPL, matrix_est,
                 matrix_est_uncalibrated, matrix_linear, matrix_dcorr,
                 # matrix_shrink, matrix_dcorr_shrink
                 ]
    # titles = LABEL_BASIC + [LABEL_LINEAR, LABEL_NONLINEAR]
    titles = LABELS_ALL_METHODS
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + DELTA_G + ' (generation)'

    for i, data in enumerate(plot_data):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        sns.heatmap(data, cmap='GnBu', alpha=0.75, cbar=False, annot=True, annot_kws={"fontsize": 6}, vmin=vmin, vmax=vmax, linewidths=HEATMAP_LINEWIDTH)

        at_left = (i % nCol == 0)
        at_bottom = (i // nCol == nRow - 1)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list if at_bottom else [])
        # plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(x=-0.08, y=1.07, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, ylabel=ylabel, ylabelpad=-1, xlabelpad=-1)
    plt.subplots_adjust(0.071, 0.09, 0.95, 0.95, hspace=0.28, wspace=0.25)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def set_ticks_spines(ax, yspine_position=0.025, xspine_position=-0.05):

    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # ax.spines['left'].set_position(('data', yspine_position))
    # ax.spines['bottom'].set_position(('data', xspine_position))
    ax.spines['left'].set_position(('axes', yspine_position))
    ax.spines['bottom'].set_position(('axes', xspine_position))


def plot_supplementary_figure_performance_alternative_methods(MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection, two_columns=True, plot_legend=True, annot=False, save_file=None):

    method_list = ['MPL', 'SL'] + LABELS_ALL_METHODS[2:6] + ['Evoracle', 'haploSep']

    w       = DOUBLE_COLUMN
    goldh   = 0.60 * w
    fig     = plt.figure(figsize=(w, goldh))

    box_top   = 0.95
    box_left  = 0.07
    box_right = 0.995 if two_columns else 0.94
    wspace    = 0.3

    global_bottom = 0.17
    ddy = 0.12
    dy = (box_top - global_bottom - ddy) / 2

    # for i in range(len(spearmanr_cov[1])):
    #     spearmanr_cov[1][i] = 0
    metrics_list = [MAE_cov, spearmanr_cov, MAE_selection, spearmanr_selection]

    if two_columns:
        boxes = [dict(left=box_left,
                      right=box_right,
                      bottom=box_top-((i+1)*dy)-(i*ddy),
                      top=box_top-(i*dy)-(i*ddy)) for i in range(len(metrics_list)//2)]
        gridspecs = [gridspec.GridSpec(1, 2, wspace=wspace, **box) for box in boxes]
        axes = []
        for _ in gridspecs:
            axes += [plt.subplot(_[0, 1]), plt.subplot(_[0, 0])]
    else:
        boxes = [dict(left=box_left,
                      right=box_right,
                      bottom=box_top-((i+1)*dy)-(i*ddy),
                      top=box_top-(i*dy)-(i*ddy)) for i in range(len(metrics_list))]
        gridspecs = [gridspec.GridSpec(1, 1, wspace=wspace, **box) for box in boxes]
        axes = [plt.subplot(gridspec[0, 0]) for gridspec in gridspecs]

    ylim_list = [[0, 10], [0, 1.1], [0, 0.04], [0, 1.1]]
    yticks_list = [[0, 5, 10], [0, 0.5, 1], [0, 0.02, 0.04], [0, 0.5, 1]]
    yticklabels_list = [['0', '5', r'$\geq 10$'], ['0', '0.5', '1'], ['0', '0.02', r'$\geq 0.04$'], ['0', '0.5', '1']]
    ceil_list = [10, None, 0.04, None]
    floor_list = [None, None, None, None]
    ylabel_list = ['MAE of inferred\ncovariances', YLABEL_SPEARMANR_COVARIANCE_THREE, 'MAE of inferred\nselection coefficients', YLABEL_SPEARMANR_THREE]

    sublabels = ['B', 'A', 'D', 'C']
    sublabel_x, sublabel_y = -0.15, 1.05
    xs = np.arange(0, len(method_list))
    xlim = [-0.75, 9.4] if two_columns else [-0.8, 9.4]
    colors    = COLORS_PASTEL
    fc        = colors[1]  # '#ff6666'  #'#EB4025'
    ffc       = colors[1]  # '#ff6666'  #'#EB4025'
    hc        = colors[0]  # '#FFB511'
    nc        = '#E8E8E8'
    hfc       = colors[0]  # '#ffcd5e'
    nfc       = '#f0f0f0'
    c_alter   = colors[4]
    methods   = method_list
    # xticklabels = method_list
    xticklabels = [str(i) for i in range(len(methods))] if two_columns else METHODS

    colorlist   = [   fc,    hc,    nc,      nc,      nc,         nc,    c_alter, c_alter]
    fclist      = [  ffc,   hfc,   nfc,     nfc,     nfc,        nfc,    c_alter, c_alter]
    eclist      = [BKCOLOR for k in range(len(methods))]

    hist_props = dict(lw=SIZELINE/2, width=0.5, align='center', orientation='vertical',
                      edgecolor=[BKCOLOR for i in range(len(methods))])

    for row, metrics in enumerate(metrics_list):
        ylim = ylim_list[row]
        yticks, yticklabels, ylabel = yticks_list[row], yticklabels_list[row], ylabel_list[row]
        floor, ceil = floor_list[row], ceil_list[row]

        ys = [metrics[i] for i, method in enumerate(method_list)]
        y_avgs = np.mean(ys, axis=1)
        ax = axes[row]

        if row == 1:
            scatter_indices = np.array([_ for _ in range(len(ys)) if _ not in [0, 1]])  # Spearmanr of covariances does not apply to the SL/MPL method
            # scatter_indices = np.arange(len(ys))
            plt.sca(ax)
            na_x, na_y = 0.7, 0.15
            plt.plot([1, 1], [0.01, na_y - 0.03], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            na_x, na_y = -0.3, 0.15
            plt.plot([0, 0], [0.01, na_y - 0.03], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            y_avgs[0] = None
            y_avgs[1] = None
        elif row == 0:
            scatter_indices = np.array([_ for _ in range(len(ys)) if _ not in [0, 1]])
            plt.sca(ax)

            na_x, na_y = 0.7, 1.5
            plt.plot([1, 1], [0.1, na_y - 0.3], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)

            na_x, na_y = -0.3, 1.5
            plt.plot([0, 0], [0.1, na_y - 0.3], linewidth=SIZELINE, color=BKCOLOR)
            plt.text(na_x, na_y, 'NA', fontsize=SIZESUBLABEL)
            y_avgs[0] = None
            y_avgs[1] = None
        else:
            scatter_indices = np.arange(0, len(ys))

        pprops = {
                   # 'colors':      [[colorlist[_] for _ in scatter_indices]],
                   'colors':      [colorlist],
                   'xlim':        deepcopy(xlim),
                   'ylim':        ylim,
                   'xticks':      xs,
                   'xticklabels': [] if two_columns and row < 4 else xticklabels,
                   'yticks':      [],
                   'theme':       'open',
                   'hide':        ['left','right'] }

        pprops['yticks'] = yticks
        pprops['yticklabels'] = yticklabels
        pprops['ylabel'] = ylabel
        pprops['hide']   = []

        mp.plot(type='bar', ax=ax, x=[xs], y=[y_avgs], plotprops=hist_props, **pprops)
        if annot and (row == 3 or row == 2):
            for x, y in zip(xs, y_avgs):
                plt.sca(ax)
                annotation = '%.2f' % y if row == 3 else ('%.3f' % y)
                plt.text(x - 0.3, min(y * 1.2, ceil * 1.2) if ceil is not None else y * 1.2, annotation, fontsize=SIZESUBLABEL)

        del pprops['colors']

        ys = np.array(ys)
        if floor is not None:
            ys[ys <= floor] = floor
        if ceil is not None:
            ys[ys >= ceil] = ceil

        pprops['facecolor'] = ['None' for _c1 in range(len(xs))]
        pprops['edgecolor'] = [eclist[_] for _ in scatter_indices]
        # size_of_point = 2.
        size_of_point = 1.5 if two_columns else 4.
        sprops = dict(lw=AXWIDTH, s=size_of_point, marker='o', alpha=1.0)

        temp_x = [[xs[_c1] + np.random.normal(0, 0.04) for _c2 in range(len(ys[_c1]))] for _c1 in scatter_indices]
        mp.scatter(ax=ax, x=temp_x, y=ys[scatter_indices], plotprops=sprops, **pprops)

        pprops['facecolor'] = [fclist[_] for _ in scatter_indices]
        sprops = dict(lw=0, s=size_of_point, marker='o', alpha=1)
        mp.scatter(ax=ax, x=temp_x, y=ys[scatter_indices], plotprops=sprops, **pprops)
        plt.sca(ax)
        plt.ylabel(ylabel, labelpad=-6.1 if row == 2 else 0)
        set_ticks_spines(ax)
        if two_columns and row >= 2:
            plt.xticks(ticks=xs, labels=xticklabels, rotation=0)
        plt.text(x=sublabel_x, y=sublabel_y, s=sublabels[row], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    if two_columns and plot_legend:
        legend_dy =  1.3
        legend_l  =  2  # num of lines
        legend_x  = -12.97
        legend_y  = -17.5
        legend_dx =  5
        legend_d  = -0.5
        for k, method in enumerate(methods):
            axes[0].text(legend_x + legend_d + (k//legend_l * legend_dx), legend_y - (k%legend_l * legend_dy), k, ha='center', va='center', **DEF_LABELPROPS)
            axes[0].text(legend_x + (k//legend_l * legend_dx), legend_y - (k%legend_l * legend_dy), method, ha='left', va='center', **DEF_LABELPROPS)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



############# ADDITIONAL FIGURES #############

def plot_supplementary_figure_time_series_covariance(s=4, n=0, sample=1000, record=1, highlight_pairs=[(0, 1), (14, 30), (24, 44)], highlight_colors=[C_FOREST, C_BEN, C_DEL], alpha=0.5, xticks=np.arange(0, 701, 100), ylabelpad=6, xlim=XLIM_GENERATION, suggest_pairs=False, rasterized=True, save_file=None):

    subsample = SS.load_subsample(s, n, sample, record, covAtEachTime=True)
    selection = SS.load_selection(s)

    traj, cov = subsample['traj'], subsample['covAtEachTime']
    T, L = traj.shape
    for t in range(T):
        for i in range(L):
            cov[t, i, i] = 0

    if suggest_pairs:
        mean_cov = np.mean(cov, axis=0)
        min_index = np.argmin(mean_cov)
        min_i, min_j = min_index // L, min_index % L
        max_index = np.argmax(mean_cov)
        max_i, max_j = max_index // L, max_index % L
        print(f'Loci {min_i}, {min_j} has most negative covariance.')
        print(f'Loci {max_i}, {max_j} has most positive covariance.')

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 1
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    sublabel_x, sublabel_y = -0.06, 1.02

    plt.sca(axes[0])
    ax = plt.gca()
    for l in range(L):
        color_ = C_NEU
        alpha_ = alpha
        linestyle = 'solid'
        linewidth = SIZELINE
        zorder = 0
        label = None
        for pair, color in zip(highlight_pairs, highlight_colors):
            if l == pair[0]:
                color_ = color
                alpha_ = 1
                linewidth = SIZELINE * 2
                linestyle = 'solid'
                zorder = 10
                # if pair not in label_included:
                    # label = f'({pair[0]}, {pair[1]})'
                label = f'{l}, s = %.2f' % selection[l]
                    # label_included.add(pair)
                break
            if l == pair[1]:
                color_ = color
                alpha_ = 1
                linewidth = SIZELINE * 2
                linestyle = 'dashed'
                zorder = 10
                # if pair not in label_included:
                label = f'{l}, s = %.2f' % selection[l]
                break
        plt.plot(range(T), traj[:, l], color=color_, alpha=alpha_, linewidth=linewidth, linestyle=linestyle, label=label, rasterized=rasterized)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.ylabel('Mutant allele frequency', fontsize=SIZELABEL, labelpad=ylabelpad)
    plt.xticks(ticks=xticks, labels=[])
    plt.ylim(-0.02, 1.1)
    plt.xlim(xlim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=SIZELEGEND, frameon=False, ncol=6)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[1])
    ax = plt.gca()
    for i in range(L):
        for j in range(i):
            color_ = C_NEU
            alpha_ = alpha
            linewidth = SIZELINE
            zorder = 0
            label = None
            for pair, color in zip(highlight_pairs, highlight_colors):
                if (i in pair) and (j in pair):
                    color_ = color
                    alpha_ = 1
                    linewidth = SIZELINE * 2
                    zorder = 10
                    label = f'({pair[0]}, {pair[1]})'
                    break
            plt.plot(range(T), cov[:, i, j], color=color_, alpha=alpha_, linewidth=linewidth, zorder=zorder, label=label, rasterized=rasterized)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.ylabel('Mutant covariance', fontsize=SIZELABEL, labelpad=ylabelpad-3)
    plt.xlabel('Generation', fontsize=SIZELABEL)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=SIZELEGEND, frameon=False, ncol=6)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.subplots_adjust(0.05, 0.06, 0.995, 0.967, hspace=0.1)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_dxdx(s=4, n=0, sample=1000, record=1, window=20, highlight_pairs=[(0, 1), (14, 30), (24, 44)], highlight_colors=[C_FOREST, C_BEN, C_DEL], alpha=0.5, xticks=np.arange(0, 701, 100), ylabelpad=6, xlim=XLIM_GENERATION, suggest_pairs=False, rasterized=True, save_file=None):

    subsample = SS.load_subsample(s, n, sample, record, covAtEachTime=True)
    selection = SS.load_selection(s)

    traj, cov, times = subsample['traj'], subsample['covAtEachTime'], subsample['times']
    T, L = traj.shape
    for t in range(T):
        for i in range(L):
            cov[t, i, i] = 0

    dx = EC.computeDx(traj)
    dxdx = EC.computeDxdx(dx)
    est_cov, _ = EC.estimateIntegratedCovarianceMatrix(traj, times, window, population=1000, calibrate=False, return_est_at_each_time=True, interpolate=False)

    if suggest_pairs:
        mean_cov = np.mean(cov, axis=0)
        min_index = np.argmin(mean_cov)
        min_i, min_j = min_index // L, min_index % L
        max_index = np.argmax(mean_cov)
        max_i, max_j = max_index // L, max_index % L
        print(f'Loci {min_i}, {min_j} has most negative covariance.')
        print(f'Loci {max_i}, {max_j} has most positive covariance.')

    w = DOUBLE_COLUMN
    nRow, nCol = 6, 1
    goldh = w * 0.33 * nRow
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    sublabel_x, sublabel_y = -0.06, 1.02

    plt.sca(axes[0])
    ax = plt.gca()
    for l in range(L):
        color_ = C_NEU
        alpha_ = alpha
        linestyle = 'solid'
        linewidth = SIZELINE
        zorder = 0
        label = None
        for pair, color in zip(highlight_pairs, highlight_colors):
            if l == pair[0]:
                color_ = color
                alpha_ = 1
                linewidth = SIZELINE * 2
                linestyle = 'solid'
                zorder = 10
                # if pair not in label_included:
                    # label = f'({pair[0]}, {pair[1]})'
                label = f'{l}, s = %.2f' % selection[l]
                    # label_included.add(pair)
                break
            if l == pair[1]:
                color_ = color
                alpha_ = 1
                linewidth = SIZELINE * 2
                linestyle = 'dashed'
                zorder = 10
                # if pair not in label_included:
                label = f'{l}, s = %.2f' % selection[l]
                break
        plt.plot(range(T), traj[:, l], color=color_, alpha=alpha_, linewidth=linewidth, linestyle=linestyle, label=label, rasterized=rasterized)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.ylabel('Mutant allele frequency', fontsize=SIZELABEL, labelpad=ylabelpad)
    plt.xticks(ticks=xticks, labels=[])
    plt.ylim(-0.02, 1.1)
    plt.xlim(xlim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=SIZELEGEND, frameon=False, ncol=6)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[0], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    plt.sca(axes[1])
    ax = plt.gca()
    for i in range(L):
        for j in range(i):
            color_ = C_NEU
            alpha_ = alpha
            linewidth = SIZELINE
            zorder = 0
            label = None
            for pair, color in zip(highlight_pairs, highlight_colors):
                if (i in pair) and (j in pair):
                    color_ = color
                    alpha_ = 1
                    linewidth = SIZELINE * 2
                    zorder = 10
                    label = f'({pair[0]}, {pair[1]})'
                    break
            plt.plot(range(T), cov[:, i, j], color=color_, alpha=alpha_, linewidth=linewidth, zorder=zorder, label=label, rasterized=rasterized)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.ylabel('Mutant covariance', fontsize=SIZELABEL, labelpad=ylabelpad-3)
    plt.xlabel('Generation', fontsize=SIZELABEL)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.01), fontsize=SIZELEGEND, frameon=False, ncol=6)
    plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[1], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    for n, (pair, color) in enumerate(zip(highlight_pairs, highlight_colors)):
        i, j = pair
        plt.sca(axes[2 + n])
        ax = plt.gca()
        plt.plot(range(len(est_cov)), est_cov[:, i, j], color=color, label=r'$\Delta x_i(t) \Delta x_i(t)$')


    plt.sca(axes[2])
    ax = plt.gca()
    for pair, color in zip(highlight_pairs, highlight_colors):
        i, j = pair
        plt.plot(range(T - 1), dxdx[:, i, j], color=color)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


    plt.sca(axes[3])
    ax = plt.gca()
    for pair, color in zip(highlight_pairs, highlight_colors):
        i, j = pair
        plt.plot(range(len(est_cov)), est_cov[:, i, j], color=color)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


    plt.sca(axes[4])
    ax = plt.gca()
    for pair, color in zip(highlight_pairs, highlight_colors):
        i, j = pair
        plt.plot(range(T - 1), dxdx[:, i, j], color=color)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


    plt.sca(axes[5])
    ax = plt.gca()
    for pair, color in zip(highlight_pairs, highlight_colors):
        i, j = pair
        plt.plot(range(len(est_cov)), est_cov[:, i, j], color=color)
    plt.xticks(ticks=xticks, labels=xticks)
    plt.xlim(xlim)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)


    plt.subplots_adjust(0.05, 0.06, 0.995, 0.967, hspace=0.1)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_dxdx_across_generations(dx_dx_across_gens_list, gens, s=4, sample=1000, record=1, vmin=-0.00035, vmax=0.0002, alpha=0.5, rasterized=True, save_file=None):

    # dx_dx_across_gens_list = []
    # for n in range(5):
    #     print(n)
    #     subsample = SS.load_subsample(s, n, sample, record)
    #     traj = subsample['traj']
    #     dx_dx_across_gens = compute_dx_dx_across_generations(traj, gens)
    #     dx_dx_across_gens_list.append(dx_dx_across_gens_list)
    # # mean_dx_dx_across_gens = np.mean(dx_dx_across_gens, axis=(2, 3))
    #
    # dx_dx_across_gens_list = np.array(dx_dx_across_gens_list)

    w = DOUBLE_COLUMN
    nRow, nCol = 2, 2
    goldh = w
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    cbar_ax = fig.add_axes(rect=[0.91, .06, .02, .84])

    ij_list = [(0, 1), (14, 30), (24, 44), (35, 39)]

    matrix_ticks = np.arange(len(gens)) + .5
    delta = r'$<\Delta x_i(s) \Delta x_j(t)>$'
    xlabel = 't (generation)'
    ylabel = 's (generation)'

    for n, (i, j) in enumerate(ij_list):
        plt.sca(axes[n // nCol, n % nCol])
        ax = plt.gca()
        plot_cbar = (n == 0)
        at_left = (n % nCol == 0)
        at_bottom = (n // nCol == nRow - 1)
        sns.heatmap(np.mean(dx_dx_across_gens_list[:, :, :, i, j], axis=0), center=0, vmin=vmin, vmax=vmax, cmap=sns.diverging_palette(145, 300, as_cmap=True), square=True, cbar=plot_cbar, cbar_ax=cbar_ax if plot_cbar else None, cbar_kws=dict(orientation="vertical"))
        if at_bottom:
            plt.xticks(ticks=matrix_ticks, labels=gens, fontsize=SIZELABEL)
            plt.xlabel(xlabel, fontsize=SIZELABEL * 1.5)
        else:
            plt.xticks(ticks=matrix_ticks, labels=[], fontsize=SIZELABEL)
        if at_left:
            plt.yticks(ticks=matrix_ticks, labels=gens, fontsize=SIZELABEL)
            plt.ylabel(ylabel, fontsize=SIZELABEL * 1.5)
        else:
            plt.yticks(ticks=matrix_ticks, labels=[], fontsize=SIZELABEL)

        plt.title(f'{delta}, i={i}, j={j}', fontsize=SIZELABEL * 1.5)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if plot_cbar:
            cbar_ax.tick_params(**DEF_TICKPROPS_CBAR)

    plt.subplots_adjust(0.06, 0.06, 0.86, 0.9, hspace=0.15, wspace=0.15)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_dxdx_across_generations_2(dx_dx_across_gens_list, taus, s=4, sample=1000, record=1, vmin=-0.00035, vmax=0.0002, alpha=0.3, use_log_scale=True, rasterized=True, save_file=None):

    # np.arange(-0.004, 0.004, 0.001)

    w = DOUBLE_COLUMN
    nRow, nCol = 1, 1
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    delta = r'$\Delta x_i(t) \Delta x_j(t+\tau)$' + ', ' + r'$t \in [10, 60]$'
    tau_string = r'$\tau$'
    xlabel = f'{delta}'
    ylabel = 'Frequency'
    bins = list(np.linspace(-0.004, 0.004, 20))
    # bins = None

    for i, tau in enumerate(taus):
        data = []
        D1, D2, _, _ = dx_dx_across_gens_list[i].shape
        for n in range(D1):
            for t in range(D2):
                for l1 in range(L):
                    for l2 in range(l1 + 1, L):
                        data.append(dx_dx_across_gens_list[i, n, t, l1, l2])
        data = np.array(data)
        weights = np.ones(len(data)) / len(data)
        plt.hist(data, weights=weights, bins=bins, color=COLORS_TAB[i], alpha=alpha, label=f'{tau_string}={tau}')

    ax = plt.gca()
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    plt.xlabel(xlabel, fontsize=SIZELABEL)
    plt.ylabel(ylabel, fontsize=SIZELABEL)
    plt.legend(fontsize=SIZELABEL)
    if use_log_scale:
        plt.yscale('log', nonposy='clip')


    plt.subplots_adjust(0.08, 0.08, 0.95, 0.95, hspace=0.15, wspace=0.15)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_estimation_with_ample_data(dic_perf_cov, keys=DIC_PERF_COV_KEYS, truncate_list=TRUNCATE, linear_selected=LINEAR_SELECTED, plot_uncalibrated=True, option=0, save_file=None):
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


def plot_additional_figure_performance_estimation_all_methods(dic_perf_cov, keys=DIC_PERF_COV_KEYS, plot_spearmanr=True, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, plot_uncalibrated=True, option=0, vmin=0.5, vmax=1, figsize=None, save_file=None):
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


def plot_additional_figure_performance_under_four_cases(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, gamma_selected=GAMMA_SELECTED, ylim=(0.63, 0.95), ylabel=r"Spearman's $\rho$", max_p=None, max_q=None, combine=False, uncalibrated=False, alpha=1, save_file=None):
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


def plot_additional_figure_performance_all_methods(spearmanr_basic, spearmanr_linear, spearmanr_dcorr, truncate_index=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, figsize=None, save_file=None):
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


def plot_additional_figure_performance_combining_replicates_under_limited_sampling(spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, truncate_index=5, linear_selected=LINEAR_SELECTED, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, nRow=2, nCol=3, w_h_ratio=1, double_column=True, save_file=None):
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


def plot_additional_figure_performance_combining_different_numbers_of_replicates(spearmanr_basic_combine, spearmanr_linear_combine, spearmanr_dcorr_combine, num_basic=4, truncate_index=None, linear_selected=LINEAR_SELECTED, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, min_p=None, max_p=None, max_q=None, ylim=(0.45, 0.95), alpha=0.75, double_column=True, w_h_ratio=GOLD_RATIO, save_file=None):
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
        plt.xlabel(XLABEL_WINDOW if at_bottom else '', fontsize=SIZELABEL)
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


def plot_additional_figure_performance_for_different_initial_distributions(performances, tr=5, sample_list=SAMPLE, record_list=RECORD, pq_list=[(0, 0), (0, 3), (4, 0), (4, 3)], linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, double_column=True, ylim=(0.775, 0.975), xlim=(4, 26), yticks=np.arange(0.8, 0.975, 0.05), save_file=None):

    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 2, 2
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr = performances['spearmanr_basic'], performances['spearmanr_linear'], performances['spearmanr_dcorr']

    ys_lists = []
    for p, q in pq_list:
        ys_list = [np.mean(spearmanr_basic[tr, :, :, p, q, 0], axis=(0)),
              np.mean(spearmanr_basic[tr, :, :, p, q, 1], axis=(0)),
              np.mean(spearmanr_basic[tr, :, :, p, q, 2], axis=(0)),
              # np.mean(spearmanr_basic[tr, :, :, p, q, 3], axis=(0)),
              np.mean(spearmanr_linear[tr, :, :, p, q, linear_selected], axis=(0)),
              np.mean(spearmanr_dcorr[tr, :, :, p, q, loss_selected, gamma_selected], axis=(0)),]
        ys_lists.append(ys_list)

    xs = np.arange(5, 25)
    # labels = ['SL', 'MPL', 'est', 'est_unnormalized', 'linear', 'nonlinear']
    labels = ['SL', 'MPL', 'est', 'linear', 'nonlinear']
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM
    xlabel = 'Number of founder haplotypes'
    xticks = np.arange(5, 30, 5)
    ylabel = YLABEL_SPEARMANR
    yticklabels = ['%.2f' % _ for _ in yticks]

    for i, (ys_list, (p, q)) in enumerate(zip(ys_lists, pq_list)):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        at_left = (i%nCol == 0)
        at_bottom = (i//nCol == nRow - 1)
        for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
            plt.scatter(xs, ys, color=color, label=label, marker=marker, s=SMALLSIZEDOT)
            plt.plot(xs, ys, color=color, linewidth=SIZELINE, linestyle='solid' if j != 3 else 'dashed')
            plt.ylim(ylim)
            plt.xlim(xlim)
            if at_bottom:
                plt.xlabel(xlabel, fontsize=SIZELABEL)
                plt.xticks(ticks=xticks, labels=xticks)
            else:
                plt.xticks(ticks=xticks, labels=[])
            if at_left:
                plt.ylabel(ylabel, fontsize=SIZELABEL)
                plt.yticks(ticks=yticks, labels=yticklabels)
            else:
                plt.yticks(ticks=yticks, labels=[])

        if i == 0:
            plt.legend(fontsize=SIZELEGEND)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    # plt.subplots_adjust(left=0.076, bottom=0.067, right=0.99, top=)
    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_for_different_initial_distributions_3(performances, tr=5, sample_list=SAMPLE, record_list=RECORD, p=0, q=0, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, double_column=True, ylim=(0.795, 1.005), xlim=(4, 26), yticks=np.arange(0.8, 1.025, 0.05), save_file=None):

    w = DOUBLE_COLUMN if double_column else SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    nRow, nCol = 1, 1
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr = performances['spearmanr_basic'], performances['spearmanr_linear'], performances['spearmanr_dcorr']

    ys_list = [np.mean(spearmanr_basic[tr, :, :, p, q, 0], axis=(0)),
          np.mean(spearmanr_basic[tr, :, :, p, q, 1], axis=(0)),
          np.mean(spearmanr_basic[tr, :, :, p, q, 2], axis=(0)),
          # np.mean(spearmanr_basic[tr, :, :, p, q, 3], axis=(0)),
          np.mean(spearmanr_linear[tr, :, :, p, q, linear_selected], axis=(0)),
          np.mean(spearmanr_dcorr[tr, :, :, p, q, loss_selected, gamma_selected], axis=(0)),]

    xs = np.arange(5, 25)
    # labels = ['SL', 'MPL', 'est', 'est_unnormalized', 'linear', 'nonlinear']
    labels = ['SL', 'MPL', 'est', 'linear', 'nonlinear']
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM
    xlabel = 'Number of founder haplotypes'
    xticks = np.arange(5, 30, 5)
    ylabel = YLABEL_SPEARMANR
    yticklabels = ['%.2f' % _ for _ in yticks]

    ax = plt.gca()
    for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
        plt.scatter(xs, ys, color=color, label=label, marker=marker, s=SMALLSIZEDOT)
        plt.plot(xs, ys, color=color, linewidth=SIZELINE, linestyle='solid')
        plt.ylim(ylim)
        plt.xlim(xlim)
        plt.xlabel(xlabel, fontsize=SIZELABEL)
        plt.xticks(ticks=xticks, labels=xticks)
        plt.ylabel(ylabel, fontsize=SIZELABEL)
        plt.yticks(ticks=yticks, labels=yticklabels)

    plt.legend(fontsize=SIZELEGEND)
    ax.tick_params(**DEF_TICKPROPS)
    plt.setp(ax.spines.values(), **DEF_AXPROPS)
    # plt.text(**DEF_SUBLABELPOSITION_2x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
    # plt.title(f"sample={sample_list[p]}, interval={record_list[q]}", fontsize=SIZELABEL)

    # plt.subplots_adjust(left=0.076, bottom=0.067, right=0.99, top=)
    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_performance_for_different_initial_distributions_2(performances, tr=5, save_file=None, ylim_spearmanr=(0.795, 1.005), yticks_spearmanr=np.arange(0.8, 1.025, 0.05), ylim_MAE=(0.0055, 0.0125), yticks_MAE=np.arange(0.006, 0.0125, 0.002), ylabelpad=2, linear_selected=LINEAR_SELECTED, gamma_selected=GAMMA_SELECTED, loss_selected=LOSS_SELECTED):
    """Plots a figure comparing performances of the SL, MPL and est methods using complete data."""

    spearmanr_basic, spearmanr_linear, spearmanr_dcorr = performances['spearmanr_basic'], performances['spearmanr_linear'], performances['spearmanr_dcorr']

    error_basic, error_linear, error_dcorr = performances['error_basic'], performances['error_linear'], performances['error_dcorr']

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(2, 1, figsize=(w, goldh))

    p, q = 0, 0  # ample data
    xs = np.arange(5, 25)
    ys_lists = [
        [
            np.mean(spearmanr_basic[tr, :, :, p, q, 0], axis=(0)),
            np.mean(spearmanr_basic[tr, :, :, p, q, 1], axis=(0)),
            np.mean(spearmanr_basic[tr, :, :, p, q, 2], axis=(0)),
            np.mean(spearmanr_linear[tr, :, :, p, q, linear_selected], axis=(0)),
            np.mean(spearmanr_dcorr[tr, :, :, p, q, gamma_selected, loss_selected], axis=(0)),
        ],
        [
            np.mean(error_basic[tr, :, :, p, q, 0], axis=(0)),
            np.mean(error_basic[tr, :, :, p, q, 1], axis=(0)),
            np.mean(error_basic[tr, :, :, p, q, 2], axis=(0)),
            np.mean(error_linear[tr, :, :, p, q, linear_selected], axis=(0)),
            np.mean(error_dcorr[tr, :, :, p, q, gamma_selected, loss_selected], axis=(0)),
        ],
    ]
    ylabels = [YLABEL_SPEARMANR, YLABEL_MAE]
    legend_loc = [4, 1]
    labels = ['SL', 'MPL', 'est', 'linear', 'nonlinear']
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM

    for i, ys_list in enumerate(ys_lists):
        plt.sca(axes[i])
        ax = plt.gca()

        for j, (ys, label, color, marker) in enumerate(zip(ys_list, labels, colors, markers)):
            plot_scatter_and_line(xs, ys, color=color, s=SIZEDOT, label=label, edgecolors=EDGECOLORS, marker=marker, alpha=1, linewidth=SIZELINE)

        # for j in range(num_basic_methods):
        #     linestyle = 'solid' if j < 3 else 'dashed'
        #     plt.scatter(truncate_list, data[:, j], color=COLOR_BASIC[j], label=LABEL_BASIC[j], marker=MARKER_BASIC[j], s=SMALLSIZEDOT)
        #     plt.plot(truncate_list, data[:, j], color=COLOR_BASIC[j], linewidth=SIZELINE, linestyle=linestyle)

        if i == 0:
            plt.ylim(ylim_spearmanr)
            ax.set_yticks(yticks_spearmanr)
            # plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=ylabelpad)
            ax.set_xticklabels([])
        elif i == 1:
            plt.ylim(ylim_MAE)
            ax.set_yticks(yticks_MAE)
            plt.legend(loc=4, frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=ylabelpad)
            ax.tick_params(axis='y', which='major', pad=1.5)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(x=-0.135, y=1.05, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Number of founder haplotypes"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(hspace=0.2)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_with_shrink(performances, truncate_list=TRUNCATE, plotUnnormalized=False, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=None, save_file=None, p=0, q=0):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']


    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    spearmanrs = [spearmanr_basic[:, :, :, p, q, i] for i in range(3)] + [spearmanr_dcorr[:, :, :, p, q, loss_selected, gamma_selected], spearmanr_shrink[:, :, :, p, q], spearmanr_dcorr_shrink[:, :, :, p, q]]
    spearmanrs = [np.mean(_, axis=(1, 2)) for _ in spearmanrs]
    if shrink_selected is None:
        spearmanrs[-1] = np.max(spearmanrs[-1], axis=-1)
        spearmanrs[-2] = np.max(spearmanrs[-2], axis=-1)
    else:
        spearmanrs[-1] = spearmanrs[-1][:, shrink_selected]
        spearmanrs[-2] = spearmanrs[-2][:, shrink_selected]

    errors = [error_basic[:, :, :, p, q, i] for i in range(3)] + [error_dcorr[:, :, :, p, q, loss_selected, gamma_selected], error_shrink[:, :, :, p, q], error_dcorr_shrink[:, :, :, p, q]]
    errors = [np.mean(_, axis=(1, 2)) for _ in errors]
    if shrink_selected is None:
        errors[-1] = np.min(errors[-1], axis=-1)
        errors[-2] = np.min(errors[-2], axis=-1)
    else:
        errors[-1] = errors[-1][:, shrink_selected]
        errors[-2] = errors[-2][:, shrink_selected]

    plot_data = [spearmanrs, errors]
    if shrink_selected is None:
        labels = ['SL', 'MPL', 'Est', 'Nonlinear', 'Est-shrink (best)', 'Nonlinear-shrink\n(best)']
    else:
        factor = SHRINKING_FACTORS[shrink_selected]
        labels = ['SL', 'MPL', 'Est', 'Nonlinear', 'Est-shrink (%.1f)' % factor, 'Nonlinear-shrink\n(%.1f)' % factor]
    ylabels = [YLABEL_SPEARMANR, 'MAE of inferred selection coefficients']
    # yticks = np.linspace(ylim[0], ylim[1] - 0.05, int(10 * (ylim[1] - ylim[0])) + 1)
    yticks = np.linspace(0.5, 0.9, 5)
    colors = COLOR_BASIC[:3] + [COLOR_NONLINEAR, COLOR_SHRINK, COLOR_NONLINEAR_SHRINK]
    legend_loc = [4, 1]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        num_basic_methods = NUM_BASIC_METHODS if not plotUnnormalized else 4
        for j, (ys, color, label) in enumerate(zip(data, colors, labels)):
            linestyle = 'solid' if j < 4 else 'dashed'
            linewidth = SIZELINE if j < 4 else SIZELINE * 1.5
            # plt.scatter(truncate_list, ys, color=color, label=label, marker='.', s=SMALLSIZEDOT)
            plt.plot(truncate_list, ys, color=color, label=label, linewidth=SIZELINE, linestyle=linestyle, marker='.', markersize=2.5)

        if i == 0:
            plt.ylim(0.45, 0.99)
            ax.set_yticks(np.linspace(0.5, 0.9, 5))
            plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
        elif i == 1:
            plt.ylim(0.002, 0.022)
            ax.set_yticks(np.linspace(0.004, 0.02, 5))
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
            ax.tick_params(axis='y', which='major', pad=1.5)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Generations of data used"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.suptitle(f'Sample={SAMPLE[p]}, Record={RECORD[q]}', fontsize=SIZESUBLABEL)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_shrink_performance_with_varying_factors(performances, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, p=0, q=0):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    S = len(SHRINKING_FACTORS)

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    spearmanrs = [spearmanr_basic[tr, :, :, p, q, i] for i in range(3)] + [spearmanr_dcorr[tr, :, :, p, q, loss_selected, gamma_selected], spearmanr_shrink[tr, :, :, p, q], spearmanr_dcorr_shrink[tr, :, :, p, q]]
    spearmanrs = [np.mean(_, axis=(0, 1)) for _ in spearmanrs]
    for i in range(4):
        spearmanrs[i] = np.array([spearmanrs[i]] * S)

    errors = [error_basic[tr, :, :, p, q, i] for i in range(3)] + [error_dcorr[tr, :, :, p, q, loss_selected, gamma_selected], error_shrink[tr, :, :, p, q], error_dcorr_shrink[tr, :, :, p, q]]
    errors = [np.mean(_, axis=(0, 1)) for _ in errors]
    for i in range(4):
        errors[i] = np.array([errors[i]] * S)

    plot_data = [spearmanrs, errors]
    labels = ['SL', 'MPL', 'Est', 'Nonlinear', 'Est-shrink', 'Nonlinear-shrink']
    ylabels = [YLABEL_SPEARMANR, 'MAE of inferred selection coefficients']
    # yticks = np.linspace(ylim[0], ylim[1] - 0.05, int(10 * (ylim[1] - ylim[0])) + 1)
    yticks = np.linspace(0.5, 0.9, 5)
    colors = COLOR_BASIC[:3] + [COLOR_NONLINEAR, COLOR_SHRINK, COLOR_NONLINEAR_SHRINK]
    legend_loc = [4, 1]

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        for j, (ys, color, label) in enumerate(zip(data, colors, labels)):
            linestyle = 'solid' if j < 4 else 'dashed'
            linewidth = SIZELINE if j < 4 else SIZELINE * 1.5
            plt.plot(SHRINKING_FACTORS, ys, color=color, label=label, linewidth=SIZELINE, linestyle=linestyle, marker='.', markersize=2.5)

        xticks = SHRINKING_FACTORS[::2]
        xticklabels = ['%.1f' % _ for _ in xticks]
        plt.xticks(ticks=xticks, labels=xticklabels)

        if i == 0:
            plt.ylim(0.45, 0.99)
            ax.set_yticks(np.linspace(0.5, 0.9, 5))
            plt.legend(loc=legend_loc[i], frameon=False, fontsize=SIZELEGEND)
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
        elif i == 1:
            plt.ylim(0.002, 0.022)
            ax.set_yticks(np.linspace(0.004, 0.02, 5))
            plt.ylabel(ylabels[i], fontsize=SIZELABEL, labelpad=2)
            ax.tick_params(axis='y', which='major', pad=1.5)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    xlabel = "Shrinking factor"
    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.suptitle(f'Sample={SAMPLE[p]}, Record={RECORD[q]}', fontsize=SIZESUBLABEL)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_shrink_spearmanr_with_varying_factors_at_different_subsample_cases(performances, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, tr=5, pq_list=[(0, 0), (3, 0), (0, 3), (3, 3)], alpha=1, ylim=YLIM_SPEARMANR, yticks=YTICKS_SPEARMANR, save_file=None, double_column=False):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_linear, error_linear = performances['spearmanr_linear'][:, :, :, :, :, linear_selected], performances['error_linear'][:, :, :, :, :, linear_selected]
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'][:, :, :, :, :, loss_selected, gamma_selected], performances['error_dcorr'][:, :, :, :, :, loss_selected, gamma_selected]
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    spearmanr_est, error_est = performances['spearmanr_basic'][:, :, :, :, :, 2], performances['error_basic'][:, :, :, :, :, 2]
    # Concatenate with the results without multiplyin a factor, as results of factor=1
    spearmanr_shrink = np.concatenate((spearmanr_shrink, np.expand_dims(spearmanr_est, axis=-1)), axis=5)
    spearmanr_dcorr_shrink = np.concatenate((spearmanr_dcorr_shrink, np.expand_dims(spearmanr_dcorr, axis=-1)), axis=5)

    spearmanr_basic = np.mean(spearmanr_basic[tr], axis=(0, 1))
    spearmanr_linear = np.mean(spearmanr_linear[tr], axis=(0, 1))
    spearmanr_dcorr = np.mean(spearmanr_dcorr[tr], axis=(0, 1))
    spearmanr_shrink = np.mean(spearmanr_shrink[tr], axis=(0, 1))
    spearmanr_dcorr_shrink = np.mean(spearmanr_dcorr_shrink[tr], axis=(0, 1))

    S = len(SHRINKING_FACTORS) + 1

    xs = list(SHRINKING_FACTORS) + [1]
    yss_list = [[[spearmanr_basic[p, q, 0]] * S,
                 [spearmanr_basic[p, q, 1]] * S,
                 [spearmanr_basic[p, q, 2]] * S,
                 [spearmanr_linear[p, q]] * S,
                 [spearmanr_dcorr[p, q]] * S,
                 spearmanr_shrink[p, q],
                 spearmanr_dcorr_shrink[p, q]] for p, q in pq_list]

    labels = LABELS_ALL_METHODS_NORM
    labels[5] = 'Est-norm'
    labels[6] = 'Nonlinear-norm'
    colors = COLORS_ALL_METHODS_NORM
    markers = MARKERS_ALL_METHODS_NORM

    ylabel = YLABEL_SPEARMANR
    xlabel = XLABEL_FACTOR
    xticks = np.arange(0, 1.1, 0.2)
    xticklabels = ['%.1f' % _ for _ in xticks]
    left, right = FIG4_LEFT, FIG4_RIGHT
    bottom, top = 0.0705, 0.958
    wspace, hspace = 0.4, 0.25
    sublabel_x, sublabel_y = -0.1, 1.072

    if double_column:
        w = DOUBLE_COLUMN
        goldh = w / GOLD_RATIO
        nRow, nCol = 2, 2
        fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))
    else:
        w = SINGLE_COLUMN
        height = SINGLE_COLUMN / GOLD_RATIO * (FIG4_TOP - FIG4_BOTTOM)
        left, right = FIG4_LEFT, FIG4_RIGHT
        bottom, top = 0.0705, 0.958
        wspace, hspace = 0.4, 0.25
        sublabel_x, sublabel_y = -0.1, 1.072
        goldh = height * (2 + hspace) / (top - bottom)
        nRow, nCol = 2, 2
        fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    for i, ((p, q), yss) in enumerate(zip(pq_list, yss_list)):
        if double_column:
            plt.sca(axes[i//nCol, i%nCol])
        else:
            plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()

        for j, (ys, color, label, marker) in enumerate(zip(yss, colors, labels, markers)):
            if j < 5:
                plt.plot(xs, ys, alpha=alpha, linewidth=SIZELINE, linestyle='dashed', color=color, label=label)
            else:
                plt.plot(xs, ys, alpha=alpha, linewidth=SIZELINE * 1.5, linestyle='solid', color=color, label=label)
                # plot_scatter_and_line(xs, ys, color=color, label=label, marker=marker, alpha=alpha)

        if double_column:
            at_left = (i % nCol == 0)
            at_bottom = (i // nCol == nRow - 1)
            plt.ylim(ylim)
            if at_left:
                plt.ylabel(ylabel, fontsize=SIZELABEL)
                plt.yticks(ticks=yticks, labels=yticks)
            else:
                plt.yticks(ticks=yticks, labels=[])
            if at_bottom:
                plt.xlabel(xlabel, fontsize=SIZELABEL)
                plt.xticks(ticks=xticks, labels=xticklabels)
            else:
                plt.xticks(ticks=xticks, labels=[])
        else:
            at_left = (i % nCol == 0)
            at_bottom = (i // nCol == nRow - 1)
            plt.ylim(ylim)
            if at_left:
                plt.ylabel(ylabel, fontsize=SIZELABEL)
                plt.yticks(ticks=yticks, labels=yticks, fontsize=SIZELABEL)
            else:
                plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
            if at_bottom:
                plt.xlabel(xlabel, fontsize=SIZELABEL)

        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        if i == 0:
            plt.legend(fontsize=SIZELEGEND, frameon=False)
        if double_column:
            if at_left:
                plt.text(x=-0.09, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
            else:
                plt.text(x=-0.07, y=1.072, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        else:
            plt.text(x=sublabel_x, y=sublabel_y, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        plt.title(f"sample={SAMPLE[p]}, interval={RECORD[q]}", fontsize=SIZELABEL)

    if double_column:
        plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_2x2)
    else:
        plt.subplots_adjust(left, bottom, right, top, wspace=wspace, hspace=hspace)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_optimal_shrink_factor_vs_std_cov(performances, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, p=0, q=0, alpha=0.5):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']


    S = len(SHRINKING_FACTORS)

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    optimal_factors = []
    std_cov_var_ratios = []
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            i = np.argmax(spearmanr_shrink[tr, s, n, p, q])
            factor = SHRINKING_FACTORS[i]
            optimal_factors.append(factor)
            std_cov_var_ratios.append(std_cov_est[tr, s, n, p, q] / std_var_est[tr, s, n, p, q])

    optimal_factors_dcorr = []
    std_cov_var_ratios_dcorr = []
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            i = np.argmax(spearmanr_dcorr_shrink[tr, s, n, p, q])
            factor = SHRINKING_FACTORS[i]
            optimal_factors_dcorr.append(factor)
            std_cov_var_ratios_dcorr.append(std_cov_nonlinear[tr, s, n, p, q] / std_var_nonlinear[tr, s, n, p, q])

    ys_list = [optimal_factors, optimal_factors_dcorr]
    xs_list = [std_cov_var_ratios, std_cov_var_ratios_dcorr]
    ylabel = 'Optimal shrinking factor'
    xlabel = 'std_cov / std_var'
    titles = ['Est_shrink', 'Nonlinear_shrink']
    for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        plt.sca(axes[i])
        ax = plt.gca()
        plt.scatter(xs, ys, alpha=alpha)
        if i == 0:
            plt.ylabel(ylabel, fontsize=SIZELABEL, labelpad=2)
        # plt.xlabel(xlabel, fontsize=SIZELABEL, labelpad=2)
        plt.title(titles[i] + '\nSpearmanr=%.2f' % (stats.spearmanr(xs, ys)[0]), fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_optimal_shrink_factor_distribution(performances, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, p=0, q=0, alpha=0.5):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']


    S = len(SHRINKING_FACTORS)

    w = SINGLE_COLUMN
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    optimal_factors = []
    std_cov_var_ratios = []
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            i = np.argmax(spearmanr_shrink[tr, s, n, p, q])
            factor = SHRINKING_FACTORS[i]
            optimal_factors.append(factor)
            std_cov_var_ratios.append(std_cov_est[tr, s, n, p, q] / std_var_est[tr, s, n, p, q])

    optimal_factors_dcorr = []
    std_cov_var_ratios_dcorr = []
    for s in range(NUM_SELECTIONS):
        for n in range(NUM_TRIALS):
            i = np.argmax(spearmanr_dcorr_shrink[tr, s, n, p, q])
            factor = SHRINKING_FACTORS[i]
            optimal_factors_dcorr.append(factor)
            std_cov_var_ratios_dcorr.append(std_cov_nonlinear[tr, s, n, p, q] / std_var_nonlinear[tr, s, n, p, q])

    ys_list = [optimal_factors, optimal_factors_dcorr]
    xs_list = [std_cov_var_ratios, std_cov_var_ratios_dcorr]
    xlabel = 'Optimal shrinking factor'
    titles = ['Est_shrink', 'Nonlinear_shrink']
    for i, (xs, ys) in enumerate(zip(xs_list, ys_list)):
        plt.sca(axes[i])
        ax = plt.gca()
        plt.hist(ys)
        ax.tick_params(**DEF_TICKPROPS)
        plt.title(titles[i], fontsize=SIZELABEL)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_optimal_shrink_factor_vs_limited_sampling(performances, tr=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']


    S = len(SHRINKING_FACTORS)

    spearmanrs = [spearmanr_shrink[tr, :, :, :, :], spearmanr_dcorr_shrink[tr, :, :, :, :]]
    spearmanrs = [np.mean(_, axis=(0, 1)) for _ in spearmanrs]

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    matrix_shrink = np.zeros((len(SAMPLE), len(RECORD)))
    matrix_dcorr_shrink = np.zeros((len(SAMPLE), len(RECORD)))

    for p in range(len(SAMPLE)):
        for q in range(len(RECORD)):
            i = np.argmax(spearmanrs[0][p, q])
            matrix_shrink[p, q] = SHRINKING_FACTORS[i]

            i = np.argmax(spearmanrs[1][p, q])
            matrix_dcorr_shrink[p, q] = SHRINKING_FACTORS[i]

    plot_data = [matrix_shrink, matrix_dcorr_shrink]
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    titles = ['Optimal factors\nfor Est-shrink', 'Optimal factors\nfor Nonlinear-shrink']

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


def plot_figure_performance_gain_shrink_vs_limited_sampling(performances, tr=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    S = len(SHRINKING_FACTORS)

    spearmanrs = [spearmanr_basic[tr, :, :, :, :, 2], spearmanr_shrink[tr, :, :, :, :], spearmanr_dcorr[tr, :, :, :, :, loss_selected, gamma_selected], spearmanr_dcorr_shrink[tr, :, :, :, :]]
    spearmanrs = [np.mean(_, axis=(0, 1)) for _ in spearmanrs]

    w = SINGLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    fig, axes = plt.subplots(1, 2, figsize=(w, goldh))

    matrix_shrink = np.zeros((len(SAMPLE), len(RECORD)))
    matrix_dcorr_shrink = np.zeros((len(SAMPLE), len(RECORD)))

    for p in range(len(SAMPLE)):
        for q in range(len(RECORD)):
            matrix_shrink[p, q] = spearmanrs[1][p, q, shrink_selected] - spearmanrs[0][p, q]
            matrix_dcorr_shrink[p, q] = spearmanrs[3][p, q, dcorr_shrink_selected] - spearmanrs[2][p, q]

    plot_data = [matrix_shrink, matrix_dcorr_shrink]
    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    titles = ['Performance gain\nw.r.t. Est-shrink', 'Performance gain\nw.r.t. Nonlinear-shrink']

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, **DEF_HEATMAP_KWARGS_SHRINK_PERFORMANCE_GAIN)

        at_left = (i == 0)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)
    plt.subplots_adjust(**DEF_SUBPLOTS_ADJUST_1x2, left=0.18, right=0.9)
    plt.suptitle(f'Shrinking factor = %.1f' % SHRINKING_FACTORS[shrink_selected], fontsize=SIZESUBLABEL, y=1.1)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_supplementary_figure_std_cov_ratio_vs_limited_sampling(performances, tr=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, vmin=0.17, vmax=0.46, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']

    S = len(SHRINKING_FACTORS)

    # metrics = [std_var[tr], std_cov[tr], std_var_est[tr], std_cov_est[tr], std_var_nonlinear[tr], std_cov_nonlinear[tr]]
    ratios = [std_cov[tr] / std_var[tr], std_cov_est[tr] / std_var_est[tr], std_cov_nonlinear[tr] / std_var_nonlinear[tr]]
    ratios = [np.mean(_, axis=(0, 1)) for _ in ratios]
    plot_data = ratios

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.5
    fig, axes = plt.subplots(1, len(plot_data), figsize=(w, goldh))

    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    # COV_VAR_RATIO = r'$\sigma(cov) \: / \: \sigma(var)$'
    # titles = [f'{COV_VAR_RATIO} (True)', f'{COV_VAR_RATIO} (Est)', f'{COV_VAR_RATIO} (Nonlinear)']
    titles = ['True covariance\nmatrix, $\it{C}$', 'Estimated covariance\nmatrix, $\it{\hat{C}}$', 'Nonlinearly regularized\nestimates, $\it{\hat{C}^{*}}$']
    # titles = ['std_cov / std_var (True)', 'std_cov / std_var (Est)', 'std_cov / std_var (Nonlinear)']

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, vmin=vmin, vmax=vmax, **DEF_HEATMAP_KWARGS_STD_COV_VAR_RATIO)

        at_left = (i == 0)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(x=-0.13, y=1.13, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)
    plt.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.98, wspace=0.5)
    # plt.suptitle(f'Shrinking factor = %.1f' % SHRINKING_FACTORS[shrink_selected], fontsize=SIZESUBLABEL, y=1.1)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_mean_std_dxdx(mean_std_dx, mean_std_dxdx_dia, mean_std_dx_dx_off, sample_list=SAMPLE, record_list=RECORD, vmin=None, vmax=None, save_file=None):

    plot_data = [mean_std_dx, mean_std_dxdx_dia, mean_std_dx_dx_off, mean_std_dx_dx_off / mean_std_dxdx_dia]

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.5
    fig, axes = plt.subplots(1, len(plot_data), figsize=(w, goldh))

    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    titles = [r'Mean std. of $\Delta x$', r'Mean std. of $\Delta x_i \Delta x_i$', r'Mean std. of $\Delta x_i \Delta x_j, \: (i \neq j)$', 'ratio between C and B']

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, vmin=vmin, vmax=vmax, **DEF_HEATMAP_KWARGS_STD_COV_VAR_RATIO)

        at_left = (i == 0)
        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list)
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        plt.text(x=-0.13, y=1.13, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)
    plt.subplots_adjust(top=0.86, bottom=0.17, left=0.08, right=0.98, wspace=0.2)
    # plt.suptitle(f'Shrinking factor = %.1f' % SHRINKING_FACTORS[shrink_selected], fontsize=SIZESUBLABEL, y=1.1)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_dxdx_versus_interval(mean_std_dx, mean_std_dxdx_dia, mean_std_dx_dx_off, record_list=RECORD, save_file=None):

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / GOLD_RATIO
    fig, axes = plt.subplots(1, 1, figsize=(w, goldh))
    ax = plt.gca()

    ys_list = [mean_std_dx[0], mean_std_dxdx_dia[0], mean_std_dx_dx_off[0]]
    xs = record_list
    colors = ['grey', 'red', 'blue']
    labels = [r'Mean std. of $\Delta x$', r'Mean std. of $\Delta x_i \Delta x_i$', r'Mean std. of $\Delta x_i \Delta x_j, \: (i \neq j)$']
    for i, (ys, color, label) in enumerate(zip(ys_list, colors, labels)):
        plot_scatter_and_line(xs, ys, color=color, label=label)

    plt.legend(fontsize=SIZELEGEND)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_additional_figure_covariance_comparison(cov1, cov2, titles, vmin=None, vmax=None, rasterized=True, save_file=None):

    plot_data = [cov1, cov2, cov1 - cov2]

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 3.5
    fig, axes = plt.subplots(1, len(plot_data), figsize=(w, goldh))
    titles += ['difference']

    for i, data in enumerate(plot_data):
        plt.sca(axes[i])
        ax = plt.gca()
        sns.heatmap(data, center=0, vmin=vmin, vmax=vmax, cmap=sns.diverging_palette(120, 300, as_cmap=True), square=True, cbar=False, rasterized=rasterized)
        plt.yticks(ticks=[], labels=[], fontsize=SIZELABEL)
        plt.xticks(ticks=[], labels=[], fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        plt.title(titles[i], fontsize=SIZELABEL)

    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)



def plot_figure_optimal_shrink_factor_vs_std_cov_mean(performances, optimal_by_MAE=False, use_true_as_base=False, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, save_file=None, tr=5, p=0, q=0, alpha=0.5):

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']

    ratios = [std_cov[tr] / std_var[tr], std_cov_est[tr] / std_var_est[tr], std_cov_nonlinear[tr] / std_var_nonlinear[tr],
              std_var[tr] / std_cov[tr], std_var_est[tr] / std_cov_est[tr], std_var_nonlinear[tr] / std_cov_nonlinear[tr]]
    ratios = [np.mean(_, axis=(0, 1)) for _ in ratios]

    spearmanrs = [spearmanr_shrink[tr, :, :, :, :], spearmanr_dcorr_shrink[tr, :, :, :, :]]
    spearmanrs = [np.mean(_, axis=(0, 1)) for _ in spearmanrs]

    errors = [error_shrink[tr, :, :, :, :], error_dcorr_shrink[tr, :, :, :, :]]
    errors = [np.mean(_, axis=(0, 1)) for _ in errors]

    optimal_factor_shrink = np.zeros((len(SAMPLE), len(RECORD)))
    optimal_factor_dcorr_shrink = np.zeros((len(SAMPLE), len(RECORD)))

    for p in range(len(SAMPLE)):
        for q in range(len(RECORD)):
            i = np.argmin(errors[0][p, q]) if optimal_by_MAE else np.argmax(spearmanrs[0][p, q])
            optimal_factor_shrink[p, q] = SHRINKING_FACTORS[i]

            i = np.argmin(errors[1][p, q]) if optimal_by_MAE else np.argmax(spearmanrs[1][p, q])
            optimal_factor_dcorr_shrink[p, q] = SHRINKING_FACTORS[i]


    S = len(SHRINKING_FACTORS)

    w = DOUBLE_COLUMN
    goldh = w / GOLD_RATIO / 2
    fig, axes = plt.subplots(1, 4, figsize=(w, goldh))

    ys_list = [optimal_factor_shrink, optimal_factor_shrink, optimal_factor_dcorr_shrink, optimal_factor_dcorr_shrink]
    if use_true_as_base:
        xs_list = [ratios[1] / ratios[0], ratios[4] / ratios[3], ratios[2]  / ratios[0], ratios[5] / ratios[3]]
    else:
        xs_list = [ratios[1], ratios[4], ratios[2], ratios[5]]
    ylabel = 'Optimal factor, ' + r'\it{f^{*}}'
    if use_true_as_base:
        xlabels = ['est_cov_var_ratio /\ntrue_cov_var_ratio', 'est_var_cov_ratio /\ntrue_var_cov_ratio', 'nonlinear_cov_var_ratio /\ntrue_cov_var_ratio', 'nonlinear_var_cov_ratio /\ntrue_var_cov_ratio']
    else:
        xlabels = ['std_cov / std_var', 'std_var / std_cov', 'std_cov / std_var', 'std_var / std_cov']
    titles = ['Est_shrink', 'Est_shrink', 'Nonlinear_shrink', 'Nonlinear_shrink']
    for i, (xs, ys, xlabel) in enumerate(zip(xs_list, ys_list, xlabels)):
        xs, ys = np.ndarray.flatten(xs), np.ndarray.flatten(ys)
        plt.sca(axes[i])
        ax = plt.gca()
        plt.scatter(xs, ys, alpha=alpha)
        regression = stats.linregress(xs, ys)
        slope, intercept = regression[0], regression[1]
        plot_line_from_slope_intercept(slope, intercept)
        plt.plot()
        if i == 0:
            plt.ylabel(ylabel, fontsize=SIZELABEL, labelpad=2)
        plt.xlabel(xlabel, fontsize=SIZELABEL, labelpad=2)
        plt.title(titles[i] + '\nSpearmanr=%.2f' % (stats.spearmanr(xs, ys)[0]) + '\nPearsonr=%.2f' % (stats.pearsonr(xs, ys)[0]) + '\nSlope=%.2f' % (slope) + '\nIntercept=%.2f' % (intercept), fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS)
        plt.setp(ax.spines.values(), **DEF_AXPROPS)
        plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    # add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)

    plt.subplots_adjust(0.132, 0.128, FIG4_RIGHT, FIG4_TOP, wspace=0.43)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)


def plot_figure_performance_comparison_shrink_with_limited_sampling(performances, tr=5, truncate_list=TRUNCATE, sample_list=SAMPLE, record_list=RECORD, linear_list=LINEAR, linear_selected=LINEAR_SELECTED, loss_selected=LOSS_SELECTED, gamma_selected=GAMMA_SELECTED, shrink_selected=SHRINK_SELECTED, dcorr_shrink_selected=DCORR_SHRINK_SELECTED, save_file=None):
    """Plots a figure showing performance with linear and non-lienar shrinkages on correlation matrix, using a particular linear-strength, a particular loss, and a particular gamma, under limited sampling effects."""

    spearmanr_basic, error_basic = performances['spearmanr_basic'], performances['error_basic']
    spearmanr_dcorr, error_dcorr = performances['spearmanr_dcorr'], performances['error_dcorr']
    spearmanr_shrink, error_shrink = performances['spearmanr_shrink'], performances['error_shrink']
    spearmanr_dcorr_shrink, error_dcorr_shrink = performances['spearmanr_dcorr_shrink'], performances['error_dcorr_shrink']

    std_var, std_cov, std_var_est, std_cov_est, std_var_nonlinear, std_cov_nonlinear = performances['std_var'], performances['std_cov'], performances['std_var_est'], performances['std_cov_est'], performances['std_var_nonlinear'], performances['std_cov_nonlinear']

    ratios = [std_var[tr] / std_cov[tr], std_var_est[tr] / std_cov_est[tr], std_var_nonlinear[tr] / std_cov_nonlinear[tr]]

    S = len(SHRINKING_FACTORS)

    if shrink_selected == 'optimal':
        spearmanr_shrink_plot = np.max(spearmanr_shrink[tr], axis=-1)
        spearmanr_dcorr_shrink_plot = np.max(spearmanr_dcorr_shrink[tr], axis=-1)
        error_shrink_plot = np.min(error_shrink[tr], axis=-1)
        error_dcorr_shrink_plot = np.min(error_dcorr_shrink[tr], axis=-1)
    elif shrink_selected == 'heuristic':
        spearmanr_shrink_plot = np.zeros((NUM_SELECTIONS, NUM_TRIALS, len(SAMPLE), len(RECORD)))
        spearmanr_dcorr_shrink_plot = np.zeros((NUM_SELECTIONS, NUM_TRIALS, len(SAMPLE), len(RECORD)))
        error_shrink_plot = np.zeros((NUM_SELECTIONS, NUM_TRIALS, len(SAMPLE), len(RECORD)))
        error_dcorr_shrink_plot = np.zeros((NUM_SELECTIONS, NUM_TRIALS, len(SAMPLE), len(RECORD)))
        for s in range(NUM_SELECTIONS):
            for n in range(NUM_TRIALS):
                for p in range(len(SAMPLE)):
                    for q in range(len(RECORD)):
                        x = ratios[1][s, n, p, q] / ratios[0][s, n, p, q]
                        factor = x * 1.67 - 0.92
                        i = np.argmin(np.absolute(SHRINKING_FACTORS - factor))
                        spearmanr_shrink_plot[s, n, p, q] = spearmanr_shrink[tr, s, n, p, q, i]
                        error_shrink_plot[s, n, p, q] = error_shrink[tr, s, n, p, q, i]

                        x = ratios[2][s, n, p, q] / ratios[0][s, n, p, q]
                        factor = x * 1.02 - 0.83
                        i = np.argmin(np.absolute(SHRINKING_FACTORS - factor))
                        spearmanr_dcorr_shrink_plot[s, n, p, q] = spearmanr_dcorr_shrink[tr, s, n, p, q, i]
                        error_dcorr_shrink_plot[s, n, p, q] = error_dcorr_shrink[tr, s, n, p, q, i]
    else:
        spearmanr_shrink_plot = spearmanr_shrink[tr, :, :, :, :, shrink_selected]
        spearmanr_dcorr_shrink_plot = spearmanr_dcorr_shrink[tr, :, :, :, :, dcorr_shrink_selected]
        error_shrink_plot = error_shrink[tr, :, :, :, :, shrink_selected]
        error_dcorr_shrink_plot = error_dcorr_shrink[tr, :, :, :, :, dcorr_shrink_selected]

    plot_data = [spearmanr_basic[tr, :, :, :, :, 2],
                 spearmanr_shrink_plot,
                 spearmanr_dcorr[tr, :, :, :, :, loss_selected, gamma_selected],
                 spearmanr_dcorr_shrink_plot,
                 error_basic[tr, :, :, :, :, 2],
                 error_shrink_plot,
                 error_dcorr[tr, :, :, :, :, loss_selected, gamma_selected],
                 error_dcorr_shrink_plot,]
    plot_data = [np.mean(_, axis=(0, 1)) for _ in plot_data]

    w = DOUBLE_COLUMN #SLIDE_WIDTH
    goldh = w / 2.3
    nRow, nCol = 2, 4
    fig, axes = plt.subplots(nRow, nCol, figsize=(w, goldh))

    ylabel = 'Number of samples drawn\nat each generation'
    xlabel = 'Time intervals between sampling ' + r'$\Delta g$' + ' (generation)'
    titles = ['Est', 'Est-shrink', 'Nonlinear', 'Nonlinear-shrink']

    for i, data in enumerate(plot_data):
        plt.sca(axes[i//nCol, i%nCol])
        ax = plt.gca()
        at_left = (i%nCol == 0)
        at_top = (i//nCol == 0)
        at_bottom = (i//nCol == nRow - 1)

        if at_top:
            sns.heatmap(data, **DEF_HEATMAP_KWARGS)
        else:
            sns.heatmap(data, **DEF_HEATMAP_KWARGS_ERROR)

        plt.yticks(ticks=ticks_for_heatmap(len(sample_list)), labels=sample_list if at_left else [], rotation=0)
        plt.xticks(ticks=ticks_for_heatmap(len(record_list)), labels=record_list if at_bottom else [])
        plt.ylabel(ylabel if at_left else '', fontsize=SIZELABEL)
        ax.tick_params(**DEF_TICKPROPS_HEATMAP)
        if at_top:
            plt.title(titles[i], fontsize=SIZELABEL, pad=3)
        # plt.text(**DEF_SUBLABELPOSITION_1x2, s=SUBLABELS[i], transform=ax.transAxes, **DEF_SUBLABELPROPS)

    add_shared_label(fig, xlabel=xlabel, xlabelpad=-1)
    plt.subplots_adjust(left=0.18, right=0.9, wspace=0.1)
    if shrink_selected == 'optimal':
        suptitle = 'Shrinking factor = optimal to each case'
    elif shrink_selected == 'heuristic':
        suptitle = 'Shrinking factor = selected heuristically (knowing std of true cov)'
    else:
        suptitle = f'Shrinking factor = %.1f' % (SHRINKING_FACTORS[shrink_selected])
    plt.suptitle(suptitle, fontsize=SIZESUBLABEL, y=1.02)
    plt.show()
    if save_file:
        fig.savefig(save_file, facecolor=fig.get_facecolor(), **DEF_FIGPROPS)
