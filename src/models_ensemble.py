"""Module containing code for implementing ensemble models"""
import torch
import time
import torch.nn as nn
import torch.utils.tensorboard as tb
import copy

import utils
import mmd_util


class JANEnsembleModel:
    """Module implementing the JAN architecture"""

    def __init__(self, networks):
        self.networks = networks
        for network in self.networks:
            network.cuda()

    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = [0] * (len(self.networks) + 1)
        n_correct = [0] * (len(self.networks) + 1)
        for network in self.networks:
            network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            all_logits = None
            for i, network in enumerate(self.networks):
                with torch.no_grad():
                    _, _, logits = network.forward(images)
                    logits_copy = copy.deepcopy(logits)
                    logits_copy = logits_copy.unsqueeze(0)
                    if all_logits is None:
                        all_logits = logits_copy
                    else:
                        all_logits = torch.cat((all_logits, logits_copy), dim=0)

                top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
                n_correct[i] += top[0].item() * pred.shape[0]
                n_total[i] += pred.shape[0]

            ave_logits = torch.mean(all_logits, dim=0)
            top, pred = utils.calc_accuracy(output=ave_logits, target=labels, topk=(1,))
            n_correct[-1] += top[0].item() * pred.shape[0]
            n_total[-1] += pred.shape[0]

        acc = [round(n_correct[i] / n_total[i], 2) for i in range(len(self.networks) + 1)]
        return acc
