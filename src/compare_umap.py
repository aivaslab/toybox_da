"""Module with code to plot umap"""
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import numpy as np
import json
import csv

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


def gen_one_plot(ax, embeddings, labels, markers):
    """
    Draw the scatter plot for the embeddings
    """
    for i in range(len(embeddings)):
        ax.scatter(embeddings[i][:, 0], embeddings[i][:, 1], marker=markers[i], s=100,
                   c=[COLORS[int(labels[i][idx])] for idx in range(len(labels[i]))])


def gen_plot(settings, out_path, fnames, markers, csv_data_lists, out_name):
    """Generate large plot"""
    assert len(fnames) == len(markers)
    classes_ordered_by_color = ['airplane', 'car', 'helicopter', 'truck', 'ball', 'cup', 'mug', 'spoon', 'cat',
                                'duck', 'giraffe', 'horse']
    matplotlib.rcParams.update({'font.size': 120})
    dpi = 100
    cmap = colors.ListedColormap([color_maps(i) for i in range(12)])
    bounds = list(COLORS.keys()) + [12]
    norm = colors.BoundaryNorm(bounds, 12)
    fig, ax = plt.subplots(len(settings), 2, dpi=dpi, figsize=(200, len(settings) * 50))
    for idx, setting in enumerate(settings):
        nbr, d = setting
        embeddings = []
        labels = []
        distance_metric = 'cosine'
        for i3, fname in enumerate(fnames):
            fpath = out_path + "_".join(["umap", str(nbr), str(d), distance_metric]) + "/" + fname
            fp = open(fpath, "r")
            emb_reader = list(csv.DictReader(fp))
            embedding = np.zeros(shape=(len(emb_reader), 2), dtype=float)
            label = np.zeros(shape=len(emb_reader), dtype=np.int32)
            for i in range(len(emb_reader)):
                embedding[i][0], embedding[i][1] = float(emb_reader[i]['x']), float(emb_reader[i]['y'])
                label[i] = csv_data_lists[i3][int(emb_reader[i]['idx'])]['Class ID']
            embeddings.append(embedding)
            labels.append(label)
            fp.close()
        gen_one_plot(ax=ax[idx][0], embeddings=embeddings, labels=labels, markers=markers)
        ax[idx][0].set_title(f"nbr: {nbr}  d: {d}  m: {distance_metric}")

        embeddings = []
        labels = []
        distance_metric = 'euclidean'
        for i3, fname in enumerate(fnames):
            fpath = out_path + "_".join(["umap", str(nbr), str(d), distance_metric]) + "/" + fname
            fp = open(fpath, "r")
            emb_reader = list(csv.DictReader(fp))
            embedding = np.zeros(shape=(len(emb_reader), 2), dtype=float)
            label = np.zeros(shape=len(emb_reader), dtype=np.int32)
            for i in range(len(emb_reader)):
                embedding[i][0], embedding[i][1] = float(emb_reader[i]['x']), float(emb_reader[i]['y'])
                label[i] = csv_data_lists[i3][int(emb_reader[i]['idx'])]['Class ID']
            embeddings.append(embedding)
            labels.append(label)
            fp.close()
        gen_one_plot(ax=ax[idx][1], embeddings=embeddings, labels=labels, markers=markers)
        ax[idx][1].set_title(f"nbr: {nbr}  d: {d}  m: {distance_metric}")
        
    cb = fig.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), orientation='vertical', ax=ax)
    cb.set_ticks(ticks=np.arange(12) + 0.5, labels=classes_ordered_by_color)
    # fig.set_tight_layout(True)
    # fig.savefig(out_path + "all.png", dpi=600, bbox_inches='tight')
    fig.savefig(out_path + out_name, dpi=dpi, bbox_inches='tight', pad_inches=1)
    
    # plt.show()
    plt.close()


if __name__ == "__main__":
    dir_path = "../../../Meeting Slides/Summer 2023/Jul-12-2023/umaps/tb_sup/umap_all_data/"
    json_path = dir_path + "umap.json"
    json_fp = open(json_path, "r")
    umap_settings = json.load(json_fp)
    print(umap_settings)
    nbrs = umap_settings['n_neighbors']
    min_ds = umap_settings['min_dist']
    
    json_fp.close()
    
    toybox_csv_data_train_path = "../data/data_12/Toybox/toybox_data_interpolated_cropped_dev.csv"
    toybox_csv_data_train_fp = open(toybox_csv_data_train_path, "r")
    toybox_csv_data_train = list(csv.DictReader(toybox_csv_data_train_fp))
    
    toybox_csv_data_test_path = "../data/data_12/Toybox/toybox_data_interpolated_cropped_test.csv"
    toybox_csv_data_test_fp = open(toybox_csv_data_test_path, "r")
    toybox_csv_data_test = list(csv.DictReader(toybox_csv_data_test_fp))
    
    in12_csv_data_train_path = "../data/data_12/IN-12/dev.csv"
    in12_csv_data_train_fp = open(in12_csv_data_train_path, "r")
    in12_csv_data_train = list(csv.DictReader(in12_csv_data_train_fp))
    
    in12_csv_data_test_path = "../data/data_12/IN-12/test.csv"
    in12_csv_data_test_fp = open(in12_csv_data_test_path, "r")
    in12_csv_data_test = list(csv.DictReader(in12_csv_data_test_fp))
    
    all_settings = []
    for nbr_ in nbrs:
        for min_d in min_ds:
            all_settings.append((nbr_, min_d))
            
    print(all_settings)
    chosen_idxs = np.random.randint(0, len(all_settings), size=5)
    chosen_settings = [(20, 0.05), (50, 0.05), (100, 0.05), (200, 0.05)]
    for idx_ in chosen_idxs:
        chosen_settings.append(all_settings[idx_])
    print(chosen_settings)
    
    gen_plot(settings=chosen_settings, out_path=dir_path,
             fnames=["tb_train.csv", "tb_test.csv"], markers=[".", "+"],
             csv_data_lists=[toybox_csv_data_train, toybox_csv_data_test],
             out_name="toybox_metric_comparison.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["tb_train.csv"], markers=["."], csv_data_lists=[toybox_csv_data_train],
    #          out_name="toybox_train_cosine.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["in12_train.csv", "in12_test.csv"], markers=[".", "+"],
    #          csv_data_lists=[in12_csv_data_train, in12_csv_data_test],
    #          out_name="in12_cosine.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["in12_train.csv"], markers=["."], csv_data_lists=[in12_csv_data_train],
    #          out_name="in12_train_cosine.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["tb_train.csv", "in12_test.csv"], markers=[".", "+"],
    #          csv_data_lists=[toybox_csv_data_train, in12_csv_data_test],
    #          out_name="train_cosine.png")
    #
    # metric = 'euclidean'
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["tb_train.csv", "tb_test.csv"], markers=[".", "+"],
    #          csv_data_lists=[toybox_csv_data_train, toybox_csv_data_test],
    #          out_name="toybox_euclidean.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["tb_train.csv"], markers=["."], csv_data_lists=[toybox_csv_data_train],
    #          out_name="toybox_train_euclidean.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["in12_train.csv", "in12_test.csv"], markers=[".", "+"],
    #          csv_data_lists=[in12_csv_data_train, in12_csv_data_test],
    #          out_name="in12_cosine.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["in12_train.csv"], markers=["."], csv_data_lists=[in12_csv_data_train],
    #          out_name="in12_train_euclidean.png")
    #
    # gen_plot(neighbors=nbrs, min_dists=min_ds, distance_metric=metric, out_path=dir_path,
    #          fnames=["tb_train.csv", "in12_test.csv"], markers=[".", "+"],
    #          csv_data_lists=[toybox_csv_data_train, in12_csv_data_test],
    #          out_name="train_euclidean.png")
