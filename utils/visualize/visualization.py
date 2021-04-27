# -*- coding:utf-8 -*-
import os
import numpy as np
import matplotlib.pyplot as plt
from utils.util import get_psedo_times, mkdir

def plt_setting(fontsz = 10):
    plt.rc('font', family='Arial')
    plt.rc('xtick', labelsize=fontsz)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=fontsz)  # fontsize of the tick labels

def plot_2d_features(reduced_reprs, stages, fig_name):
    plt_setting()
    cm = plt.get_cmap('gist_rainbow')
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    n_stage = len(set(stages))
    stages_arr = np.array(stages).astype(str)
    for sid, stage in enumerate(set(stages)):
        color = cm(1. * sid / n_stage)
        reprs = reduced_reprs[stages_arr == stage, :]
        ax.scatter(reprs[:, 1], reprs[:, 0], s=8, label=stage, color=color)
    ax.set_title(fig_name)
    ax.legend()
    fig_dir = os.path.join("figures")
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s.pdf" % fig_name)
    plt.savefig(fig_fp, dpi=300)

def plot_2d_features_pesudo_time(reduced_reprs, samples, args):
    #psedo_times = get_psedo_times(args.dataset_dir, args.dataset, samples)
    plt_setting()
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(left=0.2, bottom=0.2)
    points = ax.scatter(reduced_reprs[:, 1], reduced_reprs[:, 0], s=4) #@, c=1, cmap=plt.get_cmap('YlGn').reversed()
#    fig.colorbar(points)
    ax.set_title(args.dataset)
    fig_dir = os.path.join("figures")
    mkdir(fig_dir)
    fig_fp = os.path.join(fig_dir, "%s_pseudo_time.pdf" % args.dataset)
    plt.savefig(fig_fp, dpi=300)