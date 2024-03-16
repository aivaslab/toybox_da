"""File with code to test miscellaneous things"""
import torch

import utils

num_items = 8
feats = torch.rand(num_items, 5, dtype=torch.float)
labels = torch.multinomial(torch.tensor([1, 1, 1, 1], dtype=torch.float), num_items, replacement=True)
print(feats, labels)
print(utils.sup_con_loss(features=feats, labels=labels, temp=1.0))