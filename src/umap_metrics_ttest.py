import pickle
from scipy import stats
import numpy as np

TB_CLASSES = ['airplane', 'ball', 'car', 'cat', 'cup', 'duck',
              'giraffe', 'helicopter', 'horse', 'mug', 'spoon',
              'truck']

file_path = f"../out/DUAL_SUP/TB_IN12/exp_Feb_15_2024_15_18/umap_epoch_0_all_data/umap_300_0.1_euclidean/data/ll.pkl"
fp = open(file_path, "rb")
dic = pickle.load(fp)

self_ll = []
cross_ll = []
for cl in TB_CLASSES:
    self_ll.append(dic[("tb_train", cl, "tb_train", cl)])

for cl_1 in TB_CLASSES:
    for cl_2 in TB_CLASSES:
        if cl_1 != cl_2:
            cross_ll.append(dic[("tb_train", cl_1, "tb_train", cl_2)])
print(len(self_ll), len(cross_ll))
tt = stats.ttest_ind(np.array(self_ll), np.array(cross_ll), equal_var=False, alternative='two-sided')
print(len(tt.pvalue), type(tt.pvalue))
print(tt.pvalue)
