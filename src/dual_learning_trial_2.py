"""Module for training model to learn from noisy labels"""
import argparse
import os
import datetime

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.utils.tensorboard as tb

import datasets
import utils
import networks_noisy
import models_noisy

OUT_DIR = "../out/noisy_labels_1/"
os.makedirs(OUT_DIR, exist_ok=True)


def get_train_test_acc(model, loader_train, loader_test, logger, writer, step, noisy=False):
    """Get train and test accuracy"""
    acc_train = model.eval(loader=loader_train, noisy=noisy)
    acc_test = model.eval(loader=loader_test, noisy=noisy)
    logger.info("Noisy: {:s}  Train Acc: {:.2f}  Test Acc: {:.2f}".format(str(noisy), acc_train, acc_test))
    if not noisy:
        writer.add_scalars(main_tag="Accuracies",
                           tag_scalar_dict={
                               'tb_train': acc_train,
                               'tb_test': acc_test,
                           },
                           global_step=step)
    return acc_train, acc_test


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
    parser.add_argument("--track-gradients", default=False, action='store_true',
                        help="Use this flag to track gradients during training...")
    return vars(parser.parse_args())


class Experiment:
    """Class definition for experiment"""
    
    def __init__(self, exp_args):
        self.exp_args = exp_args
        self.start_time = datetime.datetime.now()
        self.start_time_str = self.start_time.strftime("%b_%d_%Y_%H_%M")
        self.save_dir = OUT_DIR + self.start_time_str + "/"
        self.tb_writer = tb.SummaryWriter(log_dir=self.save_dir)
        self.logger = utils.create_logger(log_file_name=self.save_dir + "log.txt", log_level_str=self.exp_args['log'])
        
        self.hypertune = not self.exp_args['final']
        
        color_jitter = transforms.ColorJitter(brightness=0.8, contrast=0.8, hue=0.2, saturation=0.8)
        self.transform_train = transforms.Compose([transforms.ToPILImage(),
                                                   transforms.RandomApply([color_jitter], p=0.8),
                                                   transforms.RandomGrayscale(p=0.2),
                                                   transforms.Resize(256),
                                                   transforms.RandomResizedCrop(size=224),
                                                   transforms.RandomHorizontalFlip(p=0.5),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)
                                                   ])
        
        self.transform_test = transforms.Compose([transforms.ToPILImage(),
                                                  transforms.Resize(224),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)
                                                  ])
        
        self.train_data = datasets.DatasetIN12(train=True, transform=self.transform_train, fraction=1.0,
                                               hypertune=self.hypertune)
        self.train_loader = torchdata.DataLoader(self.train_data, batch_size=self.exp_args['bsize'], shuffle=True,
                                                 num_workers=self.exp_args['workers'], drop_last=True)
        self.test_data = datasets.DatasetIN12(train=False, transform=self.transform_test, hypertune=self.hypertune)
        self.test_loader = torchdata.DataLoader(self.test_data, batch_size=2 * self.exp_args['bsize'], shuffle=False,
                                                num_workers=self.exp_args['workers'])
        
        if exp_args['load_path'] != "" and os.path.isdir(exp_args['load_path']):
            load_file_path = exp_args['load_path'] + "final_model.pt"
            load_file = torch.load(load_file_path)
            self.logger.info(f"Loading noisy model weights from {load_file_path} ({load_file['type']})")
            bb_wts = load_file['backbone']
            cl_wts = load_file['classifier']
            self.network_pretrained = networks_noisy.ResNet18Noisy1(num_classes=12,
                                                                    backbone_weights=bb_wts, classifier_weights=cl_wts)
        else:
            self.network_pretrained = networks_noisy.ResNet18Noisy1(num_classes=12)

        self.network_pretrained.set_eval()
        self.network_pretrained.freeze_network()

        self.network = networks_noisy.ResNet18Noisy1(num_classes=12, tb_writer=self.tb_writer, track_gradients=True)
        self.network.set_train()
        self.network.unfreeze_network()
        
        self.model = models_noisy.NoisyModel1(network_pt=self.network_pretrained, network=self.network,
                                              train_loader=self.train_loader, logger=self.logger)
        
        self.optimizer = torch.optim.Adam(self.network.backbone.parameters(), lr=self.exp_args['lr'],
                                          weight_decay=self.exp_args['wd'])
        self.optimizer.add_param_group({'params': self.network.classifier_head.parameters(), 'lr': self.exp_args['lr'],
                                        'weight_decay': self.exp_args['wd']})
        
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=self.optimizer, start_factor=0.01,
                                                                  end_factor=1.0,
                                                                  total_iters=2 * self.exp_args['iters'])
        self.decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer,
                                                                          T_max=
                                                                          (self.exp_args['epochs'] - 2) *
                                                                          self.exp_args['iters'])
        self.combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=self.optimizer,
                                                                        schedulers=[self.warmup_scheduler,
                                                                                    self.decay_scheduler],
                                                                        milestones=[2 * self.exp_args['iters'] + 1])
        
        self.exp_args['start_time'] = self.start_time_str
        self.exp_args['transform_train'] = str(self.transform_train)
        self.exp_args['transform_test'] = str(self.transform_test)
        utils.save_args(path=self.save_dir, args=self.exp_args)
        self.logger.info("Experimental details and results saved to {}".format(self.save_dir))
    
    def run_training(self):
        """Train model"""
        num_epochs = self.exp_args['epochs']
        steps = self.exp_args['iters']
        
        get_train_test_acc(model=self.model,
                           loader_train=self.train_loader, loader_test=self.test_loader,
                           logger=self.logger, writer=self.tb_writer, step=0, noisy=True)
        get_train_test_acc(model=self.model,
                           loader_train=self.train_loader, loader_test=self.test_loader,
                           logger=self.logger, writer=self.tb_writer, step=0, noisy=False)
        
        for ep in range(1, num_epochs + 1):
            self.model.train(optimizer=self.optimizer, scheduler=self.combined_scheduler, steps=steps,
                             ep=ep, ep_total=num_epochs, writer=self.tb_writer)
            if ep % 20 == 0 and ep != num_epochs:
                get_train_test_acc(model=self.model,
                                   loader_train=self.train_loader, loader_test=self.test_loader,
                                   logger=self.logger, writer=self.tb_writer, step=ep * steps)

        get_train_test_acc(model=self.model,
                           loader_train=self.train_loader, loader_test=self.test_loader,
                           logger=self.logger, writer=self.tb_writer, step=0, noisy=True)
        
        acc_train, acc_test = \
            get_train_test_acc(model=self.model,
                               loader_train=self.train_loader, loader_test=self.test_loader,
                               logger=self.logger, writer=self.tb_writer, step=num_epochs * steps)
        
        self.exp_args['tb_train'] = acc_train
        self.exp_args['tb_test'] = acc_test
        utils.save_args(path=self.save_dir, args=self.exp_args)

        save_dict = {
            'type': self.model.__class__.__name__,
            'backbone': self.network.backbone.model.state_dict(),
            'classifier': self.network.classifier_head.state_dict(),
        }
        torch.save(save_dict, self.save_dir + "final_model.pt")


def main():
    """Main method"""
    args = get_parser()
    experiment = Experiment(exp_args=args)
    experiment.run_training()


if __name__ == "__main__":
    main()
