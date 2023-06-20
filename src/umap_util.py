"""
Module to get UMAP visualization
"""
import json
import csv
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
    
    
def get_umap_from_activations(source_act, source_idxs, target_acts, target_idxs, out_path, fnames):
    """Get umap embeddings from the provided activations"""
    print(source_idxs.shape)
    assert isinstance(target_acts, list)
    assert len(fnames) == len(target_acts) + 1
    umap_dict = {}
    # nbrs = [10, 20, 50, 100, 200, 500]
    # min_ds = [0.05, 0.1, 0.2, 0.5]

    nbrs = [20, 50, 100, ]  # 200, 500]
    min_ds = [0.05, 0.1, ]  # 0.2, 0.5]
    metrics = ['cosine', ]  # 'euclidean']
    umap_dict['n_neighbors'] = nbrs
    umap_dict['min_dist'] = min_ds
    umap_dict['metrics'] = metrics
    umap_dict["prefix"] = "umap/umap_"
    for nbr in nbrs:
        for d in min_ds:
            for metric in metrics:
                umap_out_path = out_path + "/umap_{}_{}_{}/".format(nbr, d, metric)
                os.makedirs(umap_out_path, exist_ok=True)
    
                reducer = umap.UMAP(n_neighbors=nbr, min_dist=d, metric=metric, n_components=2, n_epochs=10)
                
                reducer.fit(source_act)
                src_embeddings = reducer.transform(source_act)
                csv_file_name = umap_out_path + fnames[0].split(".")[0] + ".csv"
                print(csv_file_name)
                csv_file = open(csv_file_name, "w")
                csv_writer = csv.writer(csv_file)
                csv_writer.writerow(["idx", "x", "y"])
                for idx in range(len(src_embeddings)):
                    csv_writer.writerow([source_idxs[idx], src_embeddings[idx][0], src_embeddings[idx][1]])
                csv_file.close()
                np.save(file=umap_out_path+fnames[0]+".npy", arr=src_embeddings)
                
                for i, target_act in enumerate(target_acts):
                    embedding = reducer.transform(target_act)
                    csv_file_name = umap_out_path + fnames[i+1].split(".")[0] + ".csv"
                    print(csv_file_name)
                    csv_file = open(csv_file_name, "w")
                    csv_writer = csv.writer(csv_file)
                    csv_writer.writerow(["idx", "x", "y"])
                    for idx in range(len(embedding)):
                        csv_writer.writerow([target_idxs[i][idx], embedding[idx][0], embedding[idx][1]])
                    csv_file.close()
                    np.save(file=umap_out_path+fnames[i+1]+".npy", arr=embedding)
    
    json_path = out_path + "umap.json"
    json_file = open(json_path, "w")
    json.dump(umap_dict, json_file)
    json_file.close()


if __name__ == "__main__":
    dir_path = "../out/TB_SUP/exp_Apr_17_2023_23_15/"
    load_path = dir_path + "activations/"
    umap_path = dir_path + "umap/"
    src_indices = np.load(load_path + "toybox_train_indices.npy")
    src_act = np.load(load_path + "toybox_train_activations.npy")
    trgt_acts = [np.load(load_path + "toybox_test_activations.npy"),
                 np.load(load_path + "in12_train_activations.npy"),
                 np.load(load_path + "in12_test_activations.npy")]
    trgt_indices = [np.load(load_path + "toybox_test_indices.npy"),
                    np.load(load_path + "in12_train_indices.npy"),
                    np.load(load_path + "in12_test_indices.npy")]
    umap_fnames = ["tb_train", "tb_test", "in12_train", "in12_test"]
    get_umap_from_activations(source_act=src_act, source_idxs=src_indices, target_acts=trgt_acts,
                              target_idxs=trgt_indices, out_path=umap_path, fnames=umap_fnames)
    