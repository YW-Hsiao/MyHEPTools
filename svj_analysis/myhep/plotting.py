"""
Program: This module is to plot the histogram and
         using numpy, pandas, and matplotlib.

Author: You-Wei Hsiao
Institute: Department of Physics, National Tsing Hua University, Hsinchu, Taiwan
Mail: hsiao.phys@gapp.nthu.edu.tw
History (v.3.0): 2022/05/06 First release, create plotting_basic function.
"""


################################################################################
#                              1. Import Packages                              #
################################################################################
# The Python Standard Library

# The Third-Party Library
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)


################################################################################
#                            2. Plot Basic Histogram                           #
################################################################################
# 2-1. Basic plotting for histogram graph
# ! There is a bug which is that weight cannot work well in this function.
# ! 1. If the datasets df_1, df_2, ... inside the dataset list [df_1, df_2, ...]
# !    are different shape, which means len(df_i) != len(df_j), then weight
# !    cannot work, since plotting_basic cannot iterate weight of different
# !    shape.
# ! 2. If the datasets df_1, df_2, ... inside the dataset list [df_1, df_2, ...]
# !    is the same sahpe, which means len(df_1) = len(df_2) = ..., then weight
# !    can work, therefore we just input a weight list [w1, w2, ...] which is
# !    the same shape as df_1, df_2, ... into function.
# * Therefore, this is a basic plotting function to draw the histogram.
def plotting_basic(obs, dataset, binning, data_color, data_label,
                   weight=None, selected=[],
                   figsize=(10, 10), suptitle=None, set_title='set_title',
                   xlabel=r'$x$', ylabel=r'$y$', yscale='linear',
                   xlim=None, ylim=None,
                   text=[], text_xy=(0.1, 0.9), savefig="figure.pdf"):
    """Plot basic histogram figure. The 'weight' parameter is restricted.

    Parameters
    ----------
    obs : str
        Observable equal to variable x.
    dataset : list
        List [df_1, df_2, ...] collects different datasets (df_1, df_2, ...)
        which may be different shape.
    binning : array_like
        Bin size.
    data_color : array_like
        Color of dataset.
    data_label : array_like
        Label of dataset.
    weight : array_like, optional
        Event weight, by default None
    selected : list, optional
        Selected event numbering, by default []
    figsize : tuple, optional
        Figure size, by default (10, 10)
    suptitle : str, optional
        Figure title, by default None
    set_title : str, optional
        Axes title, by default 'set_title'
    xlabel : regexp, optional
        x axis label, by default r'$'
    ylabel : regexp, optional
        y axis label, by default r'$'
    yscale : str, optional
        y axis scale, by default 'linear'
    xlim : tuple, optional
        x axis range (xmin, xmax), by default None
    ylim : tuple, optional
        y axis range (ymin, ymax), by default None
    text : list, optional
        Additional description, by default []
    text_xy : tuple, optional
        The location of first additional description, by default (0.1, 0.9)
    savefig : str, optional
        The path of saving figure, by default "figure.pdf"
    """
    # 1. construct figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # binning = np.linspace(0, 10, 11)

    # 2. plot
    for i, data in enumerate(dataset):
        ax.hist(data[obs], bins=binning, weights=weight,
                color=data_color[i], label=data_label[i], alpha=0.3)
        hist, bins = np.histogram(
            data[obs].to_numpy(), bins=binning, weights=weight)
        ax.step(bins[:-1], hist, where='post',
                color=data_color[i], label=data_label[i])

    # 3. customize plot
    # title and legend
    fig.suptitle(suptitle, fontsize=20)
    ax.legend(fontsize=15)
    # sub-title and x & y labels
    ax.set_title(set_title, fontsize=15)
    # xlabel = r'$x$ [GeV]'
    # ylabel = r'$\frac{d\sigma}{dx}$ [pb/10 GeV]'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    # x & y axis scales, limits, and tick
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.tick_params(axis='both', labelsize=12)
    # text
    for j, tex in enumerate(text):
        ax.text(text_xy[0], text_xy[1] - j/20, tex,
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=15)

    # 4. save and show plot
    plt.savefig(savefig)
    plt.show()


################################################################################
#                          3. Plot Advanced Histogram                          #
################################################################################
# 3-1. Advanced plotting for histogram graph
# ! This function is powerful to plot histogram with different shape datasets.
# ! I solved the shortcoming of function plotting_basic.
# * 1. When datasets df_1, df_2, ... are the same shape, then we can put a kind
# *    of weight and without list selected=[], shape of weight must thus be the
# *    same as df_... .
# * 2. When datasets df_1, df_2, ... are either the same or different shape,
# *    we can also put the either the same or different weight,
# *    however, we need to put the list selected=[] to iterate loop.
# *    a. different weights: len(df_i) = len(df_j),
# *       weight=[[weight_1=len(df_i)], [weight_2=len(df_i)], ...]
# *    b. the same or different weights: len(df_1) != len(df_2) != ...
# *       weight=[[weight_1], [weight_2], ...]
# *       selected=[[array_like_1=len(df_1)], [array_like_2=len(df_2)], ...]
def plotting(obs, dataset, binning, data_color, data_label, weight=None,
             selected=[], histtype='step', align='mid', where='post',
             figsize=(10, 10), suptitle=None, set_title='set_title',
             legend_loc='upper right', legend_bbox_to_anchor=(1, 1),
             xlabel=r'$x$', ylabel=r'$y$', yscale='linear', xlim=None, ylim=None,
             text=[], text_xy=(0.1, 0.9), savefig="figure.pdf"):
    """
    Plot advanced histogram figure. It works for multiple datasets with
    different shape and weight.

    Parameters
    ----------
    obs : str
        Observable equals to x-axis.
    dataset : list
        List [df_1, df_2, ...] of input data collects different datasets
        (df_1, df_2, ...) which may be different shape.
    binning : array_like
        Bin size.
    data_color : array_like
        Color of dataset ['red', 'blue', ...].
    data_label : array_like
        Label of dataset ['label 1', 'label 2', ...].
    weight : array_like or list, optional, by default None
        Event weight.
        An array of weights is the same shape as `dataset` and
        without list selected=[].
        Another case is that a weight list [weight 1, weight 2 ,...] collects
        multiple arrays of weights and each weight must be the same shape as
        corresponding dataset.
    selected : list, optional, by default []
        Selected event numbering. A list [selected 1, selected 2, ...] collects
        multiple arrays of selected events and it is used when multiple datasets
        need to be weighted.
    histtype : {'bar', 'barstacked', 'step', 'stepfilled'}, optional,
    by default 'step'
        The type of histogram to draw.
    align : {'left', 'mid', 'right'}, optional, by default 'mid'
        The horizontal alignment of the histogram bars.
    where : {'pre', 'post', 'mid'}, optional, by default 'post'
        Define where the steps should be placed.
    figsize : tuple (float, float), optional, by default (10, 10)
        Width, height in inches.
    suptitle : str, optional, by default None
        Add a centered suptitle to the figure.
    set_title : str, optional, by default 'set_title'
        Set a title for the Axes.
    legend_loc : str, optional, by default 'upper right'
        The location of the legend.
        The strings {'upper left', 'upper right', 'lower left', 'lower right'}
        place the legend at the corresponding corner of the axes/figure.
    legend_bbox_to_anchor : 2-tuple or 4-tuple of floats, optional,
    by default (0.5, 0.5)
        Box that is used to position the legend in conjunction with loc.
        A 2-tuple (x, y) places the corner of the legend specified by loc
        at x, y.
        A 4-tuple (x, y, width, height) is position and size of legend.
    xlabel : str, optional, by default r'$x$'
        Set the label for the x-axis.
    ylabel : str, optional, by default r'$y$'
        Set the label for the y-axis.
    yscale : str, optional, by default 'linear'
        Set the y-axis scale.
        The strings {"linear", "log", "symlog", "logit", ...} are the axis scale
        type to apply.
    xlim : tuple (float, float), optional, by default None
        Set the x-axis view limits (left, right).
    ylim : tuple (float, float), optional, by default None
        Set the y-axis view limits (bottom, top).
    text : list, optional, by default []
        Add text to the Axes. Put the text messages into
        the list ['text 1', 'text 2', ...].
    text_xy : tuple (float, float), optional, by default (0.1, 0.9)
        The position tuple (x, y) to place the text.
    savefig : str or path-like, optional, by default "figure.pdf"
        Save the current figure.

    Returns
    -------
    _hist : array
        The values of the histogram of each dataset.
    bins : array of dtype float
        Return the bin edges ``(length(hist)+1)``.
    """
    _hist = []
    # 1. construct figure
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    # binning = np.linspace(0, 10, 11)

    # 2. plot
    if len(selected) == 0:
        for i, data in enumerate(dataset):
            ax.hist(data[obs], bins=binning, weights=weight,
                    histtype=histtype, align=align,
                    color=data_color[i], label=data_label[i])
            hist, bins = np.histogram(data[obs].to_numpy(),
                                      bins=binning, weights=weight)
        #     ax.step(bins[:-1], hist, where=where,
            #     color=data_color[i], label=data_label[i],
            #     linestyle=(0, (5, 5)))
            _hist.append(hist)
    else:
        for i, data in enumerate(dataset):
            ax.hist(data[obs], bins=binning, weights=weight[i][selected[i]],
                    histtype=histtype, align=align,
                    color=data_color[i], label=data_label[i])
            hist, bins = np.histogram(data[obs].to_numpy(), bins=binning,
                                      weights=weight[i][selected[i]])
        #     ax.step(bins[:-1], hist, where=where,
            #     color=data_color[i], label=data_label[i],
            #     linestyle=(0, (5, 5)))
            _hist.append(hist)
    # basic operation
    # for i, data in enumerate(dataset):
        # ax.hist(data[obs], bins=binning, weights=weight,
            # color=data_color[i], label=data_label[i], alpha=0.3)
        # hist, bins = np.histogram(data[obs].to_numpy(), bins=binning, weights=weight)
        # ax.step(bins[:-1], hist, where='post', color=data_color[i], label=data_label[i])

    # 3. customize plot
    # figure title
    fig.suptitle(suptitle, fontsize=25)
    # the Axes legend and title
    ax.legend(loc=legend_loc, bbox_to_anchor=legend_bbox_to_anchor, fontsize=15)
    ax.set_title(set_title, fontsize=17)
    # x- & y-axis labels
    # xlabel = r'$x$ [GeV]', ylabel = r'$\frac{d\sigma}{dx}$ [pb/10 GeV]'
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    # x- & y-axis scales and view limits
    ax.set_yscale(yscale)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    # the appearance of ticks, tick labels, and gridlines of the Axes
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.tick_params(which='both', direction='in', top=True, right=True)
    ax.tick_params(which='major', length=7, width=1.2, labelsize=12)
    ax.tick_params(which='minor', length=4)
    # text
    for j, tex in enumerate(text):
        ax.text(text_xy[0], text_xy[1] - j/20, tex,
                horizontalalignment='left', verticalalignment='center',
                transform=ax.transAxes, fontsize=15)

    # 4. save and show plot
    plt.savefig(savefig)
    plt.show()

    return np.array(_hist), bins
