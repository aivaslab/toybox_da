"""Main methods for the supervised algorithm with combined data"""
import argparse
import numpy as np
import os
import datetime

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import models
import networks
import utils

TEMP_DIR = "../temp/DUAL_SUP/"
OUT_DIR = "../out/DUAL_SUP/"
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)


def get_train_test_acc(model, src_train_loader, src_test_loader, trgt_train_loader, trgt_test_loader,
                       writer: tb.SummaryWriter, step: int,
                       logger, no_save):
    """Get train and test accuracy"""
    src_tr_acc = model.eval(loader=src_train_loader)
    src_te_acc = model.eval(loader=src_test_loader)
    trgt_tr_acc = model.eval(loader=trgt_train_loader, scramble=True)
    trgt_te_acc = model.eval(loader=trgt_test_loader, scramble=True)
    logger.info("Source Train acc: {:.2f}   Source Test acc: {:.2f}   Target Train Acc: {:.2f}   "
                "Target Test Acc: {:.2f}".format(src_tr_acc, src_te_acc, trgt_tr_acc, trgt_te_acc))
    if not no_save:
        writer.add_scalars(main_tag="Accuracies",
                           tag_scalar_dict={
                               'tb_train': src_tr_acc,
                               'tb_test': src_te_acc,
                               'in12_train': trgt_tr_acc,
                               'in12_test': trgt_te_acc,
                           },
                           global_step=step)
    return src_tr_acc, src_te_acc, trgt_tr_acc, trgt_te_acc


def get_parser():
    """Return parser for JAN experiments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-e", "--epochs", default=50, type=int, help="Number of epochs of training")
    parser.add_argument("-it", "--iters", default=500, type=int, help="Number of training steps per epoch")
    parser.add_argument("-b", "--bsize", default=128, type=int, help="Batch size for training")
    parser.add_argument("-w", "--workers", default=4, type=int, help="Number of workers for dataloading")
    parser.add_argument("-f", "--final", default=False, action='store_true',
                        help="Use this flag to run experiments on train+val set")
    parser.add_argument("-c", "--combined-batch", default=False, action='store_true',
                        help="Use this flag to run experiments in the combined batch setting")
    parser.add_argument("-lr", "--lr", default=0.05, type=float, help="Learning rate for the experiment")
    parser.add_argument("-wd", "--wd", default=1e-5, type=float, help="Weight decay for optimizer")
    parser.add_argument("--instances", default=-1, type=int, help="Set number of toybox instances to train on")
    parser.add_argument("--images", default=1000, type=int, help="Set number of images per class to train on")
    parser.add_argument("--target-frac", "-tf", default=1.0, type=float,
                        help="Set fraction of training images to be used for target dataset")
    parser.add_argument("--seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--data-seed", default=-1, type=int, help="Seed for running experiments")
    parser.add_argument("--log", choices=["debug", "info", "warning", "error", "critical"],
                        default="info", type=str)
    parser.add_argument("--pretrained", default=False, action='store_true',
                        help="Use this flag to start from pretrained network")
    parser.add_argument("--load-path", default="", type=str,
                        help="Use this option to specify the directory from which model weights should be loaded")
    parser.add_argument("--scramble-cl", default=False, action="store_true",
                        help="Use this flag to use scrambled labels for the CE loss for target images")
    parser.add_argument("--no-save", default=False, action='store_true', help="Do not save any models")
    parser.add_argument("--save-freq", default=-1, type=int, help="Save frequence of model")
    parser.add_argument("--save-dir", default="", type=str, help="Directory to save")
    return vars(parser.parse_args())


def main():
    """Main method"""
    
    exp_args = get_parser()
    rand_rng = np.random.default_rng()
    exp_args['data_seed'] = int(rand_rng.random() * 1e7) if exp_args['data_seed'] == -1 else exp_args['data_seed']
    exp_args['seed'] = int(rand_rng.random() * 1e7) if exp_args['seed'] == -1 else exp_args['seed']
    print(f"Using seed: {exp_args['seed']} and data_seed: {exp_args['data_seed']}")
    num_epochs = exp_args['epochs']
    steps = exp_args['iters']
    b_size = exp_args['bsize']
    n_workers = exp_args['workers']
    hypertune = not exp_args['final']
    num_instances = exp_args['instances']
    num_images_per_class = exp_args['images']
    combined_batch = exp_args['combined_batch']
    target_frac = exp_args['target_frac']
    scramble_cl = exp_args['scramble_cl']
    save_freq = exp_args['save_freq']
    save_dir = exp_args['save_dir']
    no_save = exp_args['no_save']
    
    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "TB_IN12/" + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + "TB_IN12/" + save_dir + "/"
    tb_writer = tb.SummaryWriter(log_dir=tb_path + "tensorboard/") if not no_save else None
    logger = utils.create_logger(log_level_str=exp_args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)
    if not no_save:
        logger.info("Experimental details and results saved to {}".format(tb_path))

    prob = 0.2
    color_transforms = [transforms.RandomApply([transforms.ColorJitter(brightness=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(hue=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(saturation=0.2)], p=prob),
                        transforms.RandomApply([transforms.ColorJitter(contrast=0.2)], p=prob),
                        transforms.RandomEqualize(p=prob),
                        transforms.RandomPosterize(bits=4, p=prob),
                        transforms.RandomAutocontrast(p=prob)
                        ]
    src_transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256, 256)),
                                              transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0),
                                                                           interpolation=
                                                                           transforms.InterpolationMode.BICUBIC),
                                              transforms.RandomOrder(color_transforms),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD),
                                              transforms.RandomErasing(p=0.5)
                                              ])
    src_data_train = datasets.ToyboxDataset(rng=np.random.default_rng(exp_args['data_seed']), train=True,
                                            transform=src_transform_train,
                                            hypertune=hypertune, num_instances=num_instances,
                                            num_images_per_class=num_images_per_class,
                                            )
    logger.info(f"Source dataset: {src_data_train}  Size: {len(src_data_train)}")
    src_loader_train = torchdata.DataLoader(src_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                            drop_last=True)
    
    src_transform_test = transforms.Compose([transforms.ToPILImage(),
                                             transforms.Resize((224, 224)),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)])
    
    src_data_test = datasets.ToyboxDataset(rng=np.random.default_rng(), train=False, transform=src_transform_test,
                                           hypertune=hypertune)
    src_loader_test = torchdata.DataLoader(src_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)
    
    trgt_transform_train = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((256, 256)),
                                              transforms.RandomResizedCrop(size=224, scale=(0.5, 1.0),
                                                                           interpolation=
                                                                           transforms.InterpolationMode.BICUBIC),
                                              transforms.RandomOrder(color_transforms),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD),
                                              transforms.RandomErasing(p=0.5)
                                               ])
    trgt_data_train = datasets.DatasetIN12(train=True, transform=trgt_transform_train, fraction=target_frac,
                                           hypertune=hypertune)
    trgt_loader_train = torchdata.DataLoader(trgt_data_train, batch_size=b_size, shuffle=True, num_workers=n_workers,
                                             drop_last=True)
    logger.info(f"Target dataset: {trgt_data_train}  Size: {len(trgt_data_train)}")
    
    trgt_transform_test = transforms.Compose([transforms.ToPILImage(),
                                              transforms.Resize((224, 224)),
                                              transforms.ToTensor(),
                                              transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    trgt_data_test = datasets.DatasetIN12(train=False, transform=trgt_transform_test, fraction=1.0, hypertune=hypertune)
    trgt_loader_test = torchdata.DataLoader(trgt_data_test, batch_size=b_size, shuffle=False, num_workers=n_workers)
    
    # logger.debug(utils.online_mean_and_sd(src_loader_train), utils.online_mean_and_sd(src_loader_test))
    # logger.debug(utils.online_mean_and_sd(trgt_loader_test))
    utils.set_seeds(seed=exp_args['seed'], reproduce=True)
    if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
        load_file_path = exp_args['load_path'] + "final_model.pt"
        load_file = torch.load(load_file_path)
        logger.info(f"Loading model weights from {load_file_path} ({load_file['type']})")
        bb_wts = load_file['backbone']
        cl_wts = load_file['classifier'] if 'classifier' in load_file.keys() else None
        net = networks.ResNet18Sup(num_classes=12, backbone_weights=bb_wts, classifier_weights=cl_wts)
    else:
        net = networks.ResNet18Sup(num_classes=12, pretrained=exp_args['pretrained'])
    
    model = models.DualSupModel(network=net, source_loader=src_loader_train, target_loader=trgt_loader_train,
                                logger=logger, combined_batch=combined_batch,
                                scramble_target_for_classification=scramble_cl)
    
    bb_lr_wt = 0.1 if (exp_args['pretrained'] or
                       (exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']))) \
        else 1.0
    
    optimizer = torch.optim.Adam(net.backbone.parameters(), lr=bb_lr_wt * exp_args['lr'], weight_decay=exp_args['wd'])
    optimizer.add_param_group({'params': net.classifier_head.parameters(), 'lr': exp_args['lr']})

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                         total_iters=2 * steps)
    decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(num_epochs - 2) * steps)
    combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                               schedulers=[warmup_scheduler, decay_scheduler],
                                                               milestones=[2 * steps + 1])
    
    # lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer,
    #                                                  lambda x: (1. + 10 * float(x / (num_epochs * steps))) ** (-0.75))
    
    get_train_test_acc(model=model,
                       src_train_loader=src_loader_train, src_test_loader=src_loader_test,
                       trgt_train_loader=trgt_loader_train, trgt_test_loader=trgt_loader_test,
                       writer=tb_writer, step=0, logger=logger, no_save=no_save)
    model.calc_val_loss(ep=0, steps=steps, writer=tb_writer, loaders=[src_loader_test, trgt_loader_test],
                        loader_names=['tb_test', 'in12_test'], no_save=no_save)
    if not no_save:
        save_dict = {
            'type': net.__class__.__name__,
            'backbone': net.backbone.model.state_dict(),
            'classifier': net.classifier_head.state_dict(),
        }
        fname = "model_epoch_0.pt"
        torch.save(save_dict, tb_path + fname)
    
    for ep in range(1, num_epochs + 1):
        model.train(optimizer=optimizer, scheduler=combined_scheduler, steps=steps,
                    ep=ep, ep_total=num_epochs, writer=tb_writer, no_save=no_save)
        if ep % 5 == 0:
            model.calc_val_loss(ep=ep, steps=steps, writer=tb_writer, loaders=[src_loader_test, trgt_loader_test],
                                loader_names=['tb_test', 'in12_test'], no_save=no_save)
        if ep % 20 == 0 and ep != num_epochs:
            get_train_test_acc(model=model,
                               src_train_loader=src_loader_train, src_test_loader=src_loader_test,
                               trgt_train_loader=trgt_loader_train, trgt_test_loader=trgt_loader_test,
                               writer=tb_writer, step=ep * steps, logger=logger, no_save=no_save)
            
        if save_freq > 0 and ep % save_freq == 0 and not no_save:
            save_dict = {
                'type': net.__class__.__name__,
                'backbone': net.backbone.model.state_dict(),
                'classifier': net.classifier_head.state_dict(),
            }
            fname = f"model_epoch_{ep}.pt"
            torch.save(save_dict, tb_path + fname)
    
    src_tr_acc, src_te_acc, trgt_tr_acc, trgt_te_acc = get_train_test_acc(model=model,
                                                                          src_train_loader=src_loader_train,
                                                                          src_test_loader=src_loader_test,
                                                                          trgt_train_loader=trgt_loader_train,
                                                                          trgt_test_loader=trgt_loader_test,
                                                                          writer=tb_writer,
                                                                          step=num_epochs * steps, logger=logger,
                                                                          no_save=no_save)

    if not no_save:
        tb_writer.close()
        save_dict = {
            'type': net.__class__.__name__,
            'backbone': net.backbone.model.state_dict(),
            'classifier': net.classifier_head.state_dict(),
        }
        torch.save(save_dict, tb_path + "final_model.pt")

        exp_args['tb_train'] = src_tr_acc
        exp_args['tb_test'] = src_te_acc
        exp_args['in12_train'] = trgt_tr_acc
        exp_args['in12_test'] = trgt_te_acc
        exp_args['start_time'] = start_time.strftime("%b %d %Y %H:%M")
        exp_args['train_transform'] = str(src_transform_train)
        utils.save_args(path=tb_path, args=exp_args)
        logger.info("Experimental details and results saved to {}".format(tb_path))


if __name__ == "__main__":
    main()
