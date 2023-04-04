"""Module implementing the MTL pretraining algorithm"""
import numpy as np
import os

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import datasets
import models
import networks
import utils

TEMP_DIR = "../temp/"
os.makedirs(TEMP_DIR, exist_ok=True)


def main():
    """Main method"""
    num_epochs = 5
    steps = 50
    b_size = 64

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.RandomApply([color_jitter], p=0.8),
                                          transforms.RandomApply([gaussian_blur], p=0.2),
                                          transforms.Resize(256),
                                          transforms.RandomResizedCrop(size=224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    data_train = datasets.DatasetIN12SSL(transform=transform_train, fraction=1.0, hypertune=True)
    loader_train = torchdata.DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=4)
    
    # print(utils.online_mean_and_sd(trgt_loader_test))
    
    net = networks.ResNet18SSL(num_classes=12)
    ssl_model = models.SSLModel(network=net, loader=loader_train)
    
    lr = 1e-2
    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, weight_decay=1e-4)
    optimizer.add_param_group({'params': net.ssl_head.parameters(), 'lr': lr})
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])
    
    for ep in range(1, num_epochs + 1):
        ssl_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs)
    
    save_dict = {
        'type': 'SSLModel',
        'backbone': net.backbone.model.state_dict(),
        'ssl_head': net.ssl_head.state_dict()
    }
    torch.save(save_dict, TEMP_DIR + "final_model_ssl_pre.pt")


if __name__ == "__main__":
    main()
