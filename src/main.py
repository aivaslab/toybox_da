"""Module implementing the MTL pretraining algorithm"""
import argparse
import numpy as np
import os
import datetime

import torch
import torch.utils.data as torchdata
import torch.utils.tensorboard as tb
import torchvision.transforms as transforms

import datasets
import models
import networks
import utils

TEMP_DIR = "../temp/BasicMTL/"
OUT_DIR = "../out/BasicMTL/"
os.makedirs(TEMP_DIR, exist_ok=True)


def get_train_test_acc(model, src_loader, trgt_loader, logger, writer, step):
    """Get train and test accuracy"""
    src_acc = model.eval(loader=src_loader)
    trgt_acc = model.eval(loader=trgt_loader)
    logger.info("Source Acc: {:.2f}   Target Acc:{:.2f}".format(src_acc, trgt_acc))
    writer.add_scalars(main_tag="Accuracies",
                       tag_scalar_dict={
                           'tb_test': src_acc,
                           'in12_test': trgt_acc,
                       },
                       global_step=step)


def get_parser():
    """Return parser for source pretrain experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs of training")
    parser.add_argument("-it", "--iters", default=500, type=int, help="Number of training steps per epoch")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiment on train+val set")
    parser.add_argument("-lr", "--lr", default=0.05, type=float, help="Learning rate for the experiment")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--instances", default=-1, type=int, help="Set number of toybox instances to train on")
    parser.add_argument("--images", default=1000, type=int, help="Set number of images per class to train on")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", type=str)
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from network pretrained on ILSVRC")
    return vars(parser.parse_args())


def main():
    """Main method"""
    exp_args = get_parser()
    num_epochs = exp_args['epochs']
    steps = exp_args['iters']
    b_size = exp_args['bsize']
    
    start_time = datetime.datetime.now()
    save_dir = OUT_DIR + start_time.strftime("%b_%d_%Y_%H_%M") + "/"
    tb_writer = tb.SummaryWriter(log_dir=save_dir)
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=save_dir + "log.txt")
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
    
    src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=True, transform=src_transform_test,
                                           num_images_per_class=3000, hypertune=True)
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=True, num_workers=4)

    color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)
    gaussian_blur = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
    
    trgt_transform_train = transforms.Compose([transforms.ToPILImage(),
                                               transforms.RandomApply([color_jitter], p=0.8),
                                               transforms.RandomGrayscale(p=0.2),
                                               transforms.Resize(256),
                                               transforms.RandomResizedCrop(size=224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    
    trgt_data_train = datasets.DatasetIN12SSL(transform=trgt_transform_train, fraction=1.0, hypertune=True)
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=True, num_workers=4)
    
    trgt_transform_test = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize(224),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    trgt_data_test = datasets.DatasetIN12(train=True, transform=trgt_transform_test, fraction=1.0, hypertune=True)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=True, num_workers=4)
    
    # print(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # print(utils.online_mean_and_sd(trgt_loader_test))
    
    net = networks.ResNet18MTL(num_classes=12)
    mtl_model = models.MTLModel(network=net, source_loader=src_loader_train, target_loader=trgt_loader_train,
                                logger=logger)
    
    lr = exp_args['lr']
    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=lr, weight_decay=exp_args['wd'])
    optimizer.add_param_group({'params': net.classifier_head.parameters(), 'lr': lr})
    optimizer.add_param_group({'params': net.ssl_head.parameters(), 'lr': lr})
    
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2*steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2*steps+1])

    get_train_test_acc(model=mtl_model, src_loader=src_loader_test, trgt_loader=trgt_loader_test, logger=logger,
                       writer=tb_writer, step=0)
    for ep in range(1, num_epochs + 1):
        mtl_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs, lmbda=1, writer=tb_writer)
        if ep % 20 == 0 and ep != num_epochs:
            get_train_test_acc(model=mtl_model, src_loader=src_loader_test, trgt_loader=trgt_loader_test, logger=logger,
                               writer=tb_writer, step=ep * steps)

    get_train_test_acc(model=mtl_model, src_loader=src_loader_test, trgt_loader=trgt_loader_test, logger=logger,
                       writer=tb_writer, step=num_epochs * steps)

    save_dict = {
        'type': 'MTLModel',
        'backbone': net.backbone.model.state_dict(),
        'classifier': net.classifier_head.state_dict(),
        'ssl_head': net.ssl_head.state_dict()
        }
    torch.save(save_dict, save_dir + "final_model.pt")


if __name__ == "__main__":
    main()
