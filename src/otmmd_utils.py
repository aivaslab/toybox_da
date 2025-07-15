
import collections
import numpy as np

import ot
import torch.utils.data as torchdata
import torch

import datasets
import networks
import tb_in12_transforms


class Perfect_CL_Dict:
    def __init__(self, num_images=1600):
        self.num_images = num_images
        self.cl_dict = None
        self.gen_cl_dict()
        print("Using Perfect_CL_Dict")

    def gen_cl_dict(self):
        self.cl_dict = {}
        for cl in range(12):
            self.cl_dict[cl] = np.arange(cl * self.num_images, (cl + 1) * self.num_images)

    def get_cl_dict(self):
        return self.cl_dict


class OT_CL_Dict:
    def __init__(self, backbone_weights):
        self.backbone_weights = backbone_weights
        tb_transform = tb_in12_transforms.get_eval_transform(dset="toybox")
        tb_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True, transform=tb_transform,
                                         hypertune=False, num_instances=-1, num_images_per_class=1600)
        self.tb_loader = torchdata.DataLoader(tb_data, batch_size=128, shuffle=False, num_workers=8, drop_last=False,
                                              pin_memory=True, persistent_workers=True, prefetch_factor=2)

        in12_transform = tb_in12_transforms.get_eval_transform(dset="in12")
        in12_data = datasets.DatasetIN12(train=True, transform=in12_transform, fraction=1.0, hypertune=False)
        self.in12_loader = torchdata.DataLoader(in12_data, batch_size=128, shuffle=False, num_workers=8,
                                                drop_last=False)

        self.network = networks.ResNet18Backbone(weights=self.backbone_weights).cuda()
        self.network.set_eval()
        self.cl_dict = None

        print("Using OT_CL_Dict")
        self.gen_cl_dict()

    def get_activations(self):
        in12_activations = torch.zeros((len(self.in12_loader.dataset), self.network.fc_size)).cuda()
        tb_activations = torch.zeros((len(self.tb_loader.dataset), self.network.fc_size)).cuda()

        with torch.no_grad():
            for _, ((idxs1, idxs2), images, labels) in enumerate(self.in12_loader):
                images, labels = images.cuda(), labels.cuda()
                activations = self.network.forward(images)
                in12_activations[idxs1] = activations

        with torch.no_grad():
            for _, ((idxs1, idxs2), images, labels) in enumerate(self.tb_loader):
                images = images.cuda()
                activations = self.network.forward(images)
                tb_activations[idxs1] = activations
        return tb_activations, in12_activations

    @staticmethod
    def get_in12_cl_dict(ot_matrix):
        mappings = torch.argmax(ot_matrix, dim=1)
        in12_cl_dict = collections.defaultdict(list)
        for cl in range(12):
            for tb_idx in range(cl * 1600, (cl + 1) * 1600):
                in12_cl_dict[cl].append(mappings[tb_idx].item())
            assert len(in12_cl_dict[cl]) == 1600
        return in12_cl_dict

    def gen_cl_dict(self):
        tb_act, in12_act = self.get_activations()
        dist_matrix = ot.dist(tb_act, in12_act)
        w1, w2 = (1.0 / tb_act.shape[0]) * torch.ones(tb_act.shape[0]).cuda(), (1.0 / in12_act.shape[0]) * torch.ones(
            in12_act.shape[0]).cuda()
        ot_matrix = ot.emd(w1, w2, dist_matrix, numItermax=1e7)
        self.cl_dict = self.get_in12_cl_dict(ot_matrix=ot_matrix)

    def get_cl_dict(self):
        return self.cl_dict
