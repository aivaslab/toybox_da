"""Module implementing the source pretraining algorithm"""
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


def get_train_test_acc(model, src_loader, trgt_loader):
    """Get train and test accuracy"""
    src_acc = model.eval(loader=src_loader)
    trgt_acc = model.eval(loader=trgt_loader)
    print("Source acc: {:.2f}   Target Acc:{:.2f}".format(src_acc, trgt_acc))


def main():
    """Main method"""
    num_epochs = 10
    steps = 20
    b_size = 64
    src_transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(256),
                                              transforms.RandomResizedCrop(size=224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])
    src_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(), train=True, transform=src_transform_train,
                                            hypertune=True, num_instances=-1, num_images_per_class=1000,
                                            )
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=4)
    
    src_transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize(224),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])
    
    src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=src_transform_test,
                                           hypertune=True)
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=True, num_workers=4)

    trgt_transform_test = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    trgt_data_test = datasets.DatasetIN12(train=True, transform=trgt_transform_test, fraction=1.0, hypertune=True)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=True, num_workers=4)
    
    # print(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # print(utils.online_mean_and_sd(trgt_loader_test))
    
    net = networks.ResNet18Sup(num_classes=12)
    pre_model = models.SupModel(network=net, source_loader=src_loader_train)
    
    lr = 0.1
    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, weight_decay=1e-4)
    optimizer.add_param_group({'params': net.classifier_head.parameters(), 'lr': lr})
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])
    
    for ep in range(1, num_epochs + 1):
        pre_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs)
        if ep % 20 == 0:
            get_train_test_acc(model=pre_model, src_loader=src_loader_test, trgt_loader=trgt_loader_test)
    
    get_train_test_acc(model=pre_model, src_loader=src_loader_test, trgt_loader=trgt_loader_test)
    
    save_dict = {
        'type': 'SupModel',
        'backbone': net.backbone.model.state_dict(),
        'classifier_head': net.classifier_head.state_dict(),
        }
    torch.save(save_dict, TEMP_DIR + "final_model_src_pre.pt")


if __name__ == "__main__":
    main()
