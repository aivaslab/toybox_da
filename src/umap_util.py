"""
Module to get UMAP visualization
"""

import umap.umap_ as umap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import os


color_maps = matplotlib.colormaps.get_cmap('tab20b')
COLORS = {
    0: color_maps(0),  # airplane
    1: color_maps(4),  # ball
    2: color_maps(1),  # car
    3: color_maps(8),  # cat
    4: color_maps(5),  # cup
    5: color_maps(9),  # duck
    6: color_maps(10),  # giraffe
    7: color_maps(2),  # helicopter
    8: color_maps(11),  # horse
    9: color_maps(6),  # mug
    10: color_maps(7),  # spoon
    11: color_maps(3),  # truck
}

    
def plot(embeddings, labels, markers, out_path):
    """
    Draw the scatter plot for the embeddings
    """
    classes_ordered_by_color = ['airplane', 'car', 'helicopter', 'truck', 'ball', 'cup', 'mug', 'spoon', 'cat',
                                'duck', 'giraffe', 'horse']
    cmap = colors.ListedColormap([color_maps(i) for i in range(12)])
    bounds = list(COLORS.keys()) + [12]
    norm = colors.BoundaryNorm(bounds, 12)
    
    os.makedirs(out_path, exist_ok=True)
    
    fig, ax = plt.subplots(dpi=600)
    for i in range(len(embeddings)):
        ax.scatter(embeddings[i][:, 0], embeddings[i][:, 1], marker=markers[i], s=2,
                   c=[COLORS[int(labels[i][idx])] for idx in range(len(labels[i]))])
    
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', ax=ax)
    cb.set_ticks(ticks=np.arange(12) + 0.5, labels=classes_ordered_by_color)
    fig.set_tight_layout(True)
    fig.savefig(out_path + "all.png", dpi=600, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(dpi=600)
    ax.scatter(embeddings[0][:, 0], embeddings[0][:, 1], marker=markers[0], s=2,
               c=[COLORS[int(labels[0][idx])] for idx in range(len(labels[0]))])

    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', ax=ax)
    cb.set_ticks(ticks=np.arange(12) + 0.5, labels=classes_ordered_by_color)
    fig.set_tight_layout(True)
    fig.savefig(out_path + "train.png", dpi=600, bbox_inches='tight')
    plt.close()

    fig, ax = plt.subplots(dpi=600)
    ax.scatter(embeddings[1][:, 0], embeddings[1][:, 1], marker=markers[1], s=2,
               c=[COLORS[int(labels[1][idx])] for idx in range(len(labels[1]))])

    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', ax=ax)
    cb.set_ticks(ticks=np.arange(12) + 0.5, labels=classes_ordered_by_color)
    fig.set_tight_layout(True)
    fig.savefig(out_path + "test.png", dpi=600, bbox_inches='tight')
    plt.close()
    
    
def get_umap_from_activations(source_act, target_acts, out_path, fnames):
    """Get umap embeddings from the provided activations"""
    assert isinstance(target_acts, list)
    assert len(fnames) == len(target_acts) + 1
    nbrs = [200]
    min_ds = [0.2, 0.5]
    for nbr in nbrs:
        for d in min_ds:
            umap_out_path = out_path + "/umap/umap_{}_{}_{}/".format(nbr, d, 'cosine')
            os.makedirs(umap_out_path, exist_ok=True)

            reducer = umap.UMAP(n_neighbors=nbr, min_dist=d, metric='cosine', n_components=2)
            reducer.fit(source_act)
            src_embeddings = reducer.transform(source_act)
            np.save(file=umap_out_path+fnames[0]+".npy", arr=src_embeddings)
            for i, target_act in enumerate(target_acts):
                embedding = reducer.transform(target_act)
                np.save(file=umap_out_path+fnames[i+1]+".npy", arr=embedding)


if __name__ == "__main__":
    load_path = "../temp/activations/"
    src_act = np.load(load_path + "toybox_train_activations.npy")
    trgt_acts = [np.load(load_path + "toybox_test_activations.npy"),
                 np.load(load_path + "in12_train_activations.npy"),
                 np.load(load_path + "in12_test_activations.npy")]
    umap_fnames = ["tb_train", "tb_test", "in12_train", "in12_test"]
    get_umap_from_activations(source_act=src_act, target_acts=trgt_acts, out_path=load_path, fnames=umap_fnames)
    