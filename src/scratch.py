"""File for scratch code"""

# import json
# umap_dict = {}
# nbrs = [10, 20, 50, 100, 200]
# min_ds = [0.01, 0.05, 0.1, 0.2]
# metrics = ['cosine', 'euclidean']
# # nbrs = [500]
# # min_ds = [0.05]
# # metrics = ['cosine']
# umap_dict['n_neighbors'] = nbrs
# umap_dict['min_dist'] = min_ds
# umap_dict['metrics'] = metrics
# umap_dict["prefix"] = "umap_all_data/umap_"
# json_file = open("../out/ilsvrc_pretrained/umap_all_data/umap.json", "w")
# json.dump(umap_dict, json_file)
# json_file.close()

import torch

import utils

sup_con = utils.SupConLoss()
features = torch.rand(8, 128)
features_norm = torch.nn.functional.normalize(features)

f1, f2 = torch.split(features_norm, [4, 4], dim=0)
features_split = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)

labels = torch.tensor([0, 1, 2, 2])
loss = sup_con(features_split, labels)
print(loss)

