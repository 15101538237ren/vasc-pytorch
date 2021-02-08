# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.util import mkdir

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_hist(dists, fig_fp, title, xlabel, ylabel, bins=50):
    plt_setting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    ax.hist(dists, bins=bins, edgecolor='black', alpha=0.5, linewidth=0.5)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, weight='bold', fontsize=10)
    ax.set_ylabel(ylabel, weight='bold', fontsize=10)
    mkdir(os.path.dirname(fig_fp))
    plt.savefig(fig_fp)
if __name__ == "__main__":
    spatial_dists = np.load("../data/drosophila/drosophila_spatial_dist.npy")
    sdists = spatial_dists[np.triu_indices(spatial_dists.shape[0])]
    spatial_hist_fp = "../figures/drosophila_spatial_hist.pdf"
    plot_hist(sdists, spatial_hist_fp, "Drosophila Spatial Hist", "Spatial Distance", "Freq")
