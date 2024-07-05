import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns

color_maps = mpl.colormaps.get_cmap('tab20b')
COLORS = {
    'airplane': color_maps(0),  # airplane
    'ball': color_maps(4),  # ball
    'car': color_maps(1),  # car
    'cat': color_maps(8),  # cat
    'cup': color_maps(5),  # cup
    'duck': color_maps(9),  # duck
    'giraffe': color_maps(10),  # giraffe
    'helicopter': color_maps(2),  # helicopter
    'horse': color_maps(11),  # horse
    'mug': color_maps(6),  # mug
    'spoon': color_maps(7),  # spoon
    'truck': color_maps(3),  # truck
}


def plot_heatmap_2(arrs, xlabels, ylabels, title, size=(12, 10)):
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(data=arrs, annot=True, cbar=True, vmin=0.0, vmax=1.0, square=True, ax=ax)
    ax.set_xticks(np.arange(0.5, len(xlabels) + 0.5, 1), labels=xlabels, rotation=90)
    ax.set_yticks(np.arange(0.5, len(ylabels) + 0.5, 1), labels=ylabels, rotation=0)
    ax.set_title(title)
    fig.tight_layout()



def plot(lists, x, labels, xlabel, ylabel, title="", logy=False):
    assert len(lists) == len(labels)
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    for i in range(len(labels)):
        ax.plot(x, lists[i], label=labels[i], color=COLORS[labels[i]])

    ax.legend(loc="upper right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xlim(0, 115)
    if logy:
        ax.set_yscale('symlog')

