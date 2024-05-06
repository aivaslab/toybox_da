"""
Module to get UMAP visualization
"""
import argparse
import json
import csv
import umap.umap_ as umap
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import numpy as np
import os
import multiprocessing as mp


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
    
    
def umap_func(act_fnames, idx_fnames, out_path, fnames, nbr, d, metric, train_idxs):
    """umap_func for multiprocessing"""
    import os
    import time
    assert len(act_fnames) == len(train_idxs)
    start_time = time.time()
    umap_out_path = out_path + "/umap_{}_{}_{}/".format(nbr, d, metric)
    os.makedirs(umap_out_path, exist_ok=True)
    
    activations = []
    
    for idx in range(len(train_idxs)):
        if train_idxs[idx] == 1:
            activations.append(np.load(act_fnames[idx]))
    
    train_activations = np.concatenate(activations, axis=0)
    # print(type(train_activations), type(activations))

    reducer = umap.UMAP(n_neighbors=nbr, min_dist=d, metric=metric, n_components=2, init="random")
    reducer.fit(train_activations)
    train_embeddings = reducer.embedding_
    train_counted = 0
    for i in range(len(train_idxs)):
        indices = np.load(idx_fnames[i])
        len_embedding = len(indices)
        if train_idxs[i] == 1:
            embedding = train_embeddings[train_counted:train_counted+len_embedding]
            train_counted += len_embedding
        else:
            act = np.load(act_fnames[i])
            embedding = reducer.transform(act)
        
        csv_file_name = umap_out_path + fnames[i].split(".")[0] + ".csv"
        print(f"{csv_file_name}, {time.time() - start_time:.1f}, {os.getpid()}, {len(embedding)}, {len(indices)}")
        start_time = time.time()
        
        csv_file = open(csv_file_name, "w")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["idx", "x", "y"])
        for idx in range(len_embedding):
            csv_writer.writerow([indices[idx], embedding[idx][0], embedding[idx][1]])
        csv_file.close()
    
    
def get_umap_from_activations(act_fnames, idx_fnames, out_path, fnames, train_idxs, pref):
    """Get umap embeddings from the provided activations"""
    assert isinstance(act_fnames, list)
    assert len(fnames) == len(act_fnames)
    umap_dict = {}
    nbrs = [200, 300, 500]  # [20, 50, 100, 200]
    min_ds = [0.1]  # [0.01, 0.05, 0.1, 0.2]
    metrics = ['euclidean']  # ['cosine', 'euclidean']
    # nbrs = [500]
    # min_ds = [0.05]
    # metrics = ['cosine']
    umap_dict['n_neighbors'] = nbrs
    umap_dict['min_dist'] = min_ds
    umap_dict['metrics'] = metrics
    umap_dict["prefix"] = pref
    
    num_procs = 4
    umap_settings = []
    for nbr in nbrs:
        for d in min_ds:
            for metric in metrics:
                umap_settings.append((nbr, d, metric))
    
    idx = 0
    while idx < len(umap_settings):
        curr_procs = []
        while idx < len(umap_settings) and len(curr_procs) < num_procs:
            nbr, d, metric = umap_settings[idx]
            new_proc = mp.Process(target=umap_func, args=(act_fnames, idx_fnames, out_path, fnames, nbr, d, metric,
                                                          train_idxs))
            curr_procs.append(new_proc)
            idx += 1
            
        for proc in curr_procs:
            proc.start()
            
        for proc in curr_procs:
            proc.join()
            
    json_path = out_path + "umap.json"
    json_file = open(json_path, "w")
    json.dump(umap_dict, json_file)
    json_file.close()


def gen_umap_final(model_dir, train_type):
    """Code to generate the umaps for each epoch of training"""
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass
    train_indices_dict = {
        'toybox_train':   (1, 0, 0, 0),
        'toybox_only':    (1, 1, 0, 0),
        'in12_train':     (0, 0, 1, 0),
        'in12_only':      (0, 0, 1, 1),
        'train_only':     (1, 0, 1, 0),
        'all_data':       (1, 1, 1, 1)
    }

    umap_fnames = ["tb_train", "tb_test", "in12_train", "in12_test"]

    load_path = model_dir + f"activations/"
    umap_path = model_dir + f"umap/{train_type}/"
    activation_fnames = [load_path + "toybox_train_activations.npy", load_path + "toybox_test_activations.npy",
                         load_path + "in12_train_activations.npy", load_path + "in12_test_activations.npy"]

    index_fnames = [load_path + "toybox_train_indices.npy", load_path + "toybox_test_indices.npy",
                    load_path + "in12_train_indices.npy", load_path + "in12_test_indices.npy"]
    for fname in activation_fnames:
        assert os.path.isfile(fname), f"{fname} does not exist"
    for fname in index_fnames:
        assert os.path.isfile(fname), f"{fname} does not exist"

    get_umap_from_activations(act_fnames=activation_fnames, idx_fnames=index_fnames, out_path=umap_path,
                              fnames=umap_fnames, train_idxs=train_indices_dict[train_type],
                              pref=f"{train_type}/umap_")


def gen_umap_epochs(model_dir, train_type, num_epochs, save_freq):
    """Code to generate the umaps for each epoch of training"""
    try:
        mp.set_start_method('forkserver')
    except RuntimeError:
        pass
    train_indices_dict = {
        'toybox_train':   (1, 0, 0, 0),
        'toybox_only':    (1, 1, 0, 0),
        'in12_train':     (0, 0, 1, 0),
        'in12_only':      (0, 0, 1, 1),
        'train_only':     (1, 0, 1, 0),
        'all_data':       (1, 1, 1, 1)
    }

    umap_fnames = ["tb_train", "tb_test", "in12_train", "in12_test"]

    for ep in range(0, num_epochs + 1, save_freq):
        load_path = model_dir + f"activations_epoch_{ep}/"
        umap_path = model_dir + f"umap_epoch_{ep}_{train_type}/"
        activation_fnames = [load_path + "toybox_train_activations.npy", load_path + "toybox_test_activations.npy",
                             load_path + "in12_train_activations.npy", load_path + "in12_test_activations.npy"]

        index_fnames = [load_path + "toybox_train_indices.npy", load_path + "toybox_test_indices.npy",
                        load_path + "in12_train_indices.npy", load_path + "in12_test_indices.npy"]
        for fname in activation_fnames:
            assert os.path.isfile(fname)
        for fname in index_fnames:
            assert os.path.isfile(fname)

        get_umap_from_activations(act_fnames=activation_fnames, idx_fnames=index_fnames, out_path=umap_path,
                                  fnames=umap_fnames, train_idxs=train_indices_dict[train_type],
                                  pref=f"umap_epoch_{ep}_{train_type}/umap_")


def get_args():
    """Parser with arguments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dir-path", "-d", type=str, required=True, help="Path where model is stored.")
    parser.add_argument("--train-type", "-t", type=str, default="all_data",
                        choices=['toybox_train', 'toybox_only', 'in12_train', "in12_only", "train_only", "all_data"])
    return vars(parser.parse_args())


def main():
    """Main method"""
    args = get_args()
    dir_path = args['dir_path']
    load_path = dir_path + "activations/"
    umap_path = dir_path + "umap_" + args['train_type'] + "/"
    train_indices_dict = {
        'toybox_train': (1, 0, 0, 0),
        'toybox_only': (1, 1, 0, 0),
        'in12_train': (0, 0, 1, 0),
        'in12_only': (0, 0, 1, 1),
        'train_only': (1, 0, 1, 0),
        'all_data': (1, 1, 1, 1)
    }
    activation_fnames = [load_path + "toybox_train_activations.npy", load_path + "toybox_test_activations.npy",
                         load_path + "in12_train_activations.npy", load_path + "in12_test_activations.npy"]

    index_fnames = [load_path + "toybox_train_indices.npy", load_path + "toybox_test_indices.npy",
                    load_path + "in12_train_indices.npy", load_path + "in12_test_indices.npy"]
    for fname in activation_fnames:
        assert os.path.isfile(fname)
    for fname in index_fnames:
        assert os.path.isfile(fname)
    mp.set_start_method('forkserver')
    umap_fnames = ["tb_train", "tb_test", "in12_train", "in12_test"]
    get_umap_from_activations(act_fnames=activation_fnames, idx_fnames=index_fnames, out_path=umap_path,
                              fnames=umap_fnames, train_idxs=train_indices_dict[args['train_type']],
                              pref="umap_" + args['train_type'] + "/umap_")


if __name__ == "__main__":
    main()
    