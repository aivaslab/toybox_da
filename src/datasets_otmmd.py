"""Datasets for OT-guided MMD"""

import numpy as np
import torch.utils.data as torchdata
import collections
import copy

import datasets
import tb_in12_transforms


class Toybox_IN12_OT(torchdata.Dataset):
    def __init__(self, in12_cl_dict, hypertune=False, seed=None):
        self.in12_cl_dict = in12_cl_dict
        self.hypertune = hypertune
        self.seed = 0 if seed is None else seed

        self.tb_rng = np.random.default_rng(self.seed)
        self.tb_transform = tb_in12_transforms.get_ssl_transform(dset="toybox")
        self.toybox_dataset = datasets.ToyboxDataset(rng=self.tb_rng, num_instances=-1, num_images_per_class=1600,
                                                     hypertune=False, train=True, transform=self.tb_transform)

        self.in12_transform = tb_in12_transforms.get_ssl_transform(dset="in12")
        self.in12_dataset = datasets.DatasetIN12(train=True, transform=self.in12_transform, hypertune=False)

        self.sampling_rng = np.random.default_rng(384)

        assert len(self.toybox_dataset) == len(self.in12_dataset)

        self.batch_cl_dict = {}
        self.set_batch_cl_dict()

    def set_batch_cl_dict(self):
        for key in self.in12_cl_dict.keys():
            self.batch_cl_dict[key] = list(copy.deepcopy(self.in12_cl_dict[key]))
            self.sampling_rng.shuffle(self.batch_cl_dict[key])

    def __len__(self):
        return len(self.toybox_dataset)

    def __getitem__(self, index):
        _, tb_img, tb_lbl = self.toybox_dataset[index]
        if len(self.batch_cl_dict[tb_lbl]) == 0:
            in12_idx = self.sampling_rng.choice(self.in12_cl_dict[tb_lbl], 1)[0]
        else:
            in12_idx = self.batch_cl_dict[tb_lbl].pop()
        _, in12_img, in12_lbl = self.in12_dataset[in12_idx]
        # assert tb_lbl == in12_lbl
        return index, [tb_img, in12_img], tb_lbl
