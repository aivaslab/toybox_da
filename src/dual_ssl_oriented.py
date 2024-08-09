
import numpy as np
import datetime
import os

import torch.utils.data as torchdata
import torch
import torch.utils.tensorboard as tb

import datasets
import models_dual_ssl
import networks
import tb_in12_transforms
import dual_ssl_pretrain
import utils

OUT_DIR = "../out/DUAL_SSL_ORIENTED_V1/"
os.makedirs(OUT_DIR, exist_ok=True)


def show_images(loader, mean, std, aug):
    for _, (idxs, images, _) in enumerate(loader):
        print(type(idxs), len(idxs), idxs[0].shape, len(images), images[0].shape, images[1].shape)

        concat_images = torch.concat(images)
        batch_images = utils.get_images(concat_images, mean=mean, std=std, aug=aug)
        batch_images.show()
        break


def main():
    exp_args = dual_ssl_pretrain.get_parser()
    steps = exp_args['iters']
    num_epochs = exp_args['epochs']
    hypertune = not exp_args['final']
    workers = exp_args['workers']
    tb_ssl_type = exp_args['tb_ssl_type']
    in12_ssl_type = exp_args['in12_ssl_type']
    b_size = exp_args['bsize']
    no_save = exp_args['no_save']
    save_dir = exp_args['save_dir']
    decoupled_loss = exp_args['decoupled']
    tb_alpha = exp_args['tb_alpha']
    in12_alpha = exp_args['in12_alpha']
    save_freq = exp_args['save_freq']
    combined = exp_args['combined']

    tb_transform_train = tb_in12_transforms.get_ssl_transform(dset="toybox")
    tb_data_train = datasets.ToyboxDatasetSSLPaired(rng=np.random.default_rng(0), transform=tb_transform_train,
                                                    fraction=1.0, hypertune=hypertune, distort=tb_ssl_type)
    tb_loader_train = torchdata.DataLoader(tb_data_train, batch_size=b_size//4, shuffle=True,
                                           num_workers=workers, pin_memory=True, persistent_workers=True,
                                           drop_last=True)

    in12_transform_train = tb_in12_transforms.get_ssl_transform(dset="in12")

    if in12_ssl_type == 'class':
        in12_data_train = datasets.IN12CategorySSLWithLabels(transform=in12_transform_train, fraction=1.0,
                                                             hypertune=hypertune)
    else:
        in12_data_train = datasets.IN12SSLWithLabels(transform=in12_transform_train, hypertune=hypertune)

    in12_loader_train = torchdata.DataLoader(in12_data_train, batch_size=b_size//2, shuffle=True,
                                             num_workers=workers, pin_memory=True, persistent_workers=True,
                                             drop_last=True)

    # show_images(loader=tb_loader_train, mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True)
    # show_images(loader=in12_loader_train, mean=datasets.IN12_MEAN, std=datasets.IN12_STD, aug=True)

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"{tb_path} already exists.."
    tb_writer = tb.SummaryWriter(log_dir=tb_path + "tensorboard/") if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)

    # logger.info(utils.online_mean_and_sd(tb_loader_train), utils.online_mean_and_sd(in12_loader_train))

    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading weights from {load_file_path} ({load_file['type']})")

        bb_wts = load_file['backbone']
        ssl_wts = load_file['ssl_head'] if 'ssl_head' in load_file.keys() else None
        net = networks.ResNet18SSL(backbone_weights=bb_wts, ssl_weights=ssl_wts)
    else:
        net = networks.ResNet18SSL()
    ssl_model = models_dual_ssl.DualSSLOrientedModelV1(network=net,
                                                       src_loader=tb_loader_train, trgt_loader=in12_loader_train,
                                                       logger=logger, no_save=no_save, decoupled=decoupled_loss,
                                                       tb_alpha=tb_alpha, in12_alpha=in12_alpha, combined=combined,
                                                       num_images_in_batch=b_size, orient_alpha=0.1)

    optimizer = torch.optim.SGD(net.backbone.parameters(), lr=exp_args['lr'], weight_decay=exp_args['wd'],
                                momentum=0.9, nesterov=True)
    optimizer.add_param_group({'params': net.ssl_head.parameters(), 'lr': exp_args['lr']})

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])

    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))
        net.save_model(fpath=tb_path + "initial_model.pt")
        if save_freq > 0:
            net.save_model(fpath=tb_path + "model_epoch_0.pt")

    for ep in range(1, num_epochs + 1):
        ssl_model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                        ep=ep, ep_total=num_epochs, writer=tb_writer)
        if not no_save and save_freq > 0 and ep % save_freq == 0:
            net.save_model(fpath=tb_path + f"model_epoch_{ep}.pt")

    if not no_save:
        net.save_model(fpath=tb_path + "final_model.pt")
        if save_freq > 0:
            net.save_model(fpath=tb_path + f"model_epoch_{num_epochs}.pt")
        tb_writer.close()

        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['tb_transform'] = str(tb_transform_train)
        exp_args['in12_transform'] = str(in12_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))


if __name__ == '__main__':
    main()
    # x = torch.rand(size=(48, 512), dtype=torch.float, requires_grad=True)
    # y = torch.rand(size=(24, 512), dtype=torch.float, requires_grad=True)
    # loss = oriented_learning_signal(src_feats=x, trgt_feats=y)
    # print(loss.requires_grad)
