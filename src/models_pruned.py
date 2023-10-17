"""Module for implementing the pruned models"""
import os
import numpy as np
import argparse

import torch.nn as nn
import torch
import torchvision.transforms as transforms
import torch.utils.data as torchdata

import networks_hooks
import utils
import datasets


class EvalModel:
    """This model is used for measuring accuracy on provided datasets"""
    
    def __init__(self, network):
        self.network = network
        self.network.cuda()
    
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
    
    def calc_val_loss(self, loader, loader_name):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        num_batches = 0
        total_loss = 0.0
        for _, (idxs, images, labels) in enumerate(loader):
            num_batches += 1
            images, labels = images.cuda(), labels.cuda()
            
            with torch.no_grad():
                logits = self.network.forward(images)
            loss = criterion(logits, labels)
            total_loss += loss.item()
        print("Validation Losses -- {:s}: {:.2f}".format(loader_name, total_loss / num_batches))
        return total_loss / num_batches
    
    
class PruneEvalModel:
    """Module that implements the network for pruning the backbone and evaluates on TB and IN-12"""
    def __init__(self, model_dir, pruned_neurons):
        self.model_dir = model_dir
        self.pruned_neurons = pruned_neurons
        
        assert os.path.isdir(self.model_dir)
        self.model_path = self.model_dir + "final_model.pt"
        assert os.path.isfile(self.model_path)
        
        model_dict = torch.load(self.model_path)
        print(f"Loading model weights from {self.model_path} ({model_dict['type']})")
        bb_wts = model_dict['backbone']
        cl_wts = model_dict['classifier'] if 'classifier' in model_dict.keys() else None
        
        net = networks_hooks.ResNet18SupPruned(backbone_weights=bb_wts, classifier_weights=cl_wts,
                                               pruned_neurons=self.pruned_neurons)
        net.set_eval()
        self.eval_model = EvalModel(network=net)

        tb_transform = transforms.Compose([transforms.ToPILImage(),
                                           transforms.Resize((224, 224)),
                                           transforms.ToTensor(),
                                           transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])

        tb_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True,
                                               transform=tb_transform,
                                               hypertune=False, num_instances=-1,
                                               num_images_per_class=1500,
                                               )
        self.tb_loader_train = torchdata.DataLoader(tb_data_train, batch_size=256, num_workers=4, shuffle=False)
        
        tb_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=False,
                                              transform=tb_transform, hypertune=False, num_instances=-1,
                                              num_images_per_class=-1)
        self.tb_loader_test = torchdata.DataLoader(tb_data_test, batch_size=256, num_workers=4, shuffle=False)
        
        in12_transform = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
        
        in12_data_train = datasets.DatasetIN12(train=True, transform=in12_transform, hypertune=False)
        self.in12_loader_train = torchdata.DataLoader(in12_data_train, batch_size=256, num_workers=4, shuffle=False)
        in12_data_test = datasets.DatasetIN12(train=False, transform=in12_transform, hypertune=False)
        self.in12_loader_test = torchdata.DataLoader(in12_data_test, batch_size=256, num_workers=4, shuffle=False)

    def eval(self):
        """Eval the model on the datasets"""
        tb_tr_acc = self.eval_model.eval(loader=self.tb_loader_train)
        tb_te_acc = self.eval_model.eval(loader=self.tb_loader_test)
        print("Toybox Train: {:.2f}".format(tb_tr_acc))
        print("Toybox Test: {:.2f}".format(tb_te_acc))
        in12_tr_acc = self.eval_model.eval(loader=self.in12_loader_train)
        in12_te_acc = self.eval_model.eval(loader=self.in12_loader_test)
        print("IN-12 Train: {:.2f}".format(in12_tr_acc))
        print("IN-12 Test: {:.2f}".format(in12_te_acc))
        return tb_tr_acc, tb_te_acc, in12_tr_acc, in12_te_acc
        

def test_prune_eval():
    """Test the prune eval model"""
    model_path = "../ICLR_OUT/DUAL_SUP/exp_Aug_10_2023_21_29/"
    for num_neurons_pruned in range(0, 513, 256):
        neurons_pruned = np.arange(num_neurons_pruned)
        print("-----------------------------------------------------------------")
        print("Number of neurons pruned: {:d}".format(num_neurons_pruned))
        evaluator = PruneEvalModel(model_dir=model_path, pruned_neurons=neurons_pruned)
        evaluator.eval()
        
        del evaluator
        print("-----------------------------------------------------------------")
    

if __name__ == "__main__":
    test_prune_eval()
    