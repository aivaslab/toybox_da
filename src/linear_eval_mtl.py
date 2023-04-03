"""m"""
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


def get_train_test_acc(model, train_loader, test_loader):
    """Get train and test accuracy"""
    train_acc = model.eval(loader=train_loader)
    test_acc = model.eval(loader=test_loader)
    print("Source acc: {:.2f}   Target Acc:{:.2f}".format(train_acc, test_acc))


def main():
    """Main method"""
    num_epochs = 5
    b_size = 64
    transform_train = transforms.Compose([transforms.ToPILImage(),
                                          transforms.Resize(256),
                                          transforms.RandomResizedCrop(size=224),
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    data_train = datasets.DatasetIN12(train=True, transform=transform_train, hypertune=True)
    loader_train = torchdata.DataLoader(data_train, batch_size=b_size, shuffle=True, num_workers=4)
    
    transform_test = transforms.Compose([transforms.ToPILImage(),
                                         transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    
    data_test = datasets.DatasetIN12(train=False, transform=transform_test, hypertune=True)
    loader_test = torchdata.DataLoader(data_test, batch_size=b_size, shuffle=True, num_workers=4)
    
    load_file_name = "../temp/final_model_src_pre.pt"
    load_file = torch.load(load_file_name)

    if load_file['type'] == 'MTLModel' or load_file['type'] == 'SupModel':
        net = networks.ResNet18Sup(num_classes=12, backbone_weights=load_file['backbone'],
                                   classifier_weights=load_file['classifier_head'])
    elif load_file['type'] == 'SSLModel':
        net = networks.ResNet18Sup(num_classes=12, backbone_weights=load_file['backbone'])
    else:
        raise NotImplementedError("Model type not recognized. Load file types can only be MTLModel, SupModel or "
                                  "SSLModel")
    
    for params in net.backbone.parameters():
        params.requires_grad = False
    for params in net.classifier_head.parameters():
        params.requires_grad = True

    le_model = models.ModelFT(network=net, train_loader=loader_train, test_loader=loader_test)
    
    optimizer = torch.optim.SGD(net.classifier_head.parameters(), lr=0.1, weight_decay=1e-4)
    
    steps = len(loader_train)
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])
    
    for ep in range(1, num_epochs + 1):
        le_model.train(optimizer=optimizer, scheduler=combined_scheduler, ep=ep, ep_total=num_epochs)
        if ep % 20 == 0:
            get_train_test_acc(model=le_model, train_loader=loader_train, test_loader=loader_test)
    
    get_train_test_acc(model=le_model, train_loader=loader_train, test_loader=loader_test)


if __name__ == "__main__":
    main()
