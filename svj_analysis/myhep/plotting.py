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


################################################################################
#                            2. Plot basic histogram                           #
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
        Observable is equal to variable x.
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
