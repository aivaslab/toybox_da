# This example requires the following dependencies to be installed:
# pip install lightly

# Note: The model and training settings do not follow the reference settings
# from the paper. The settings are chosen such that the example can easily be
# run on a small dataset with a single GPU.

import numpy as np
import time
import datetime
import os
from collections import defaultdict

import torch
import torchvision.transforms.v2 as v2
from torch import nn
import torch.utils.data as torchdata
import torch.utils.tensorboard as tb
import torch.nn.functional as func

from lightly.loss import SwaVLoss
from lightly.models.modules import SwaVProjectionHead, SwaVPrototypes
from lightly.transforms.swav_transform import SwaVTransform
from lightly.models.modules.memory_bank import MemoryBankModule

import datasets
import parsers
import utils
import linear_eval
import networks
import mmd_util


OUT_DIR = "../out/SwAV/"


def get_swav_parser():
    base_parser = parsers.get_default_parser()
    base_parser.add_argument("--tb-ssl-type", "-tbssl", choices=['self', 'transform', 'object', 'class'],
                             default='object', help="Type of SSL for toybox")
    base_parser.add_argument('--lr-sched', type=str, default='cosine', choices=['cosine', 'fixed'])
    base_parser.add_argument("--ims", "-ims", default=False, action="store_true")
    base_parser.add_argument("--sinkhorn-epsilon", "-se", default=0.05, type=float)
    base_parser.add_argument("--queue-start", "-qs", default=3, type=int)
    base_parser.add_argument("--prototype-freeze-epochs", "-pfe", default=2, type=int)
    base_parser.add_argument("--queue-len", "-qlen", default=3840, type=int)
    base_parser.add_argument("--seed", type=int, default=-1, help="Random seed")
    base_parser.add_argument("--data-seed", type=int, default=-1, help="Random seed for loading Toybox data")
    base_parser.add_argument("--model-type", choices=['swav', 'swav-pairwise'], default='swav',
                             help="Model type for training")
    base_parser.add_argument("--reproduce", default=False, action='store_true',
                             help="Use this flag for reproducibility")
    base_parser.add_argument("--mmd-alpha", "-ma", default=1.0, type=float, help="Max value of MMD alpha parameter")
    base_parser.add_argument("--mmd-alpha-start", "-mas", default=0, type=int, help="Epoch where MMD loss starts "
                                                                                    "being applied")
    base_parser.add_argument("--asymmetric-mmd", "-asym", default=False, action='store_true',
                             help="Use this flag to use the asymmetric version of MMD loss")
    return base_parser


class SwaVModel(nn.Module):
    def __init__(self, queue_start, prototype_freeze_epochs, queue_len=3840, sinkhorn_epsilon=0.05):
        super().__init__()
        self.queue_start = queue_start
        self.prototype_freeze_epochs = prototype_freeze_epochs
        self.queue_len = queue_len
        self.sinkhorn_epsilon = sinkhorn_epsilon

        self.backbone = networks.ResNet18Backbone(pretrained=False, weights=None)
        self.projection_head = SwaVProjectionHead(512, 512, 128)
        self.prototypes = SwaVPrototypes(128, 512, self.prototype_freeze_epochs)

        self.start_queue_at_epoch = self.queue_start
        self.queues = nn.ModuleList(
            [MemoryBankModule(size=(self.queue_len, 128)) for _ in range(2)]
        )
        self.criterion = SwaVLoss(sinkhorn_epsilon=sinkhorn_epsilon)
        self.swav_loss_meter = utils.AverageMeter(name="swav_loss")

    def forward(self, high_resolution, low_resolution, epoch):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]

        high_resolution_prototypes = [
            self.prototypes(x, epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def _subforward(self, images):
        features = self.backbone(images).flatten(start_dim=1)
        features = self.projection_head(features)
        features = nn.functional.normalize(features, dim=1, p=2)
        return features

    @torch.no_grad()
    def _get_queue_prototypes(self, high_resolution_features, epoch):
        if len(high_resolution_features) != len(self.queues):
            raise ValueError(
                f"The number of queues ({len(self.queues)}) should be equal to the number of high "
                f"resolution inputs ({len(high_resolution_features)}). Set `n_queues` accordingly."
            )

        # Get the queue features
        queue_features = []
        for i in range(len(self.queues)):
            _, features = self.queues[i](high_resolution_features[i], update=True)
            # Queue features are in (num_ftrs X queue_length) shape, while the high-res
            # features are in (batch_size X num_ftrs). Swap the axes for interoperability.
            features = torch.permute(features, (1, 0))
            queue_features.append(features)

        # If loss calculation with queue prototypes starts at a later epoch,
        # just queue the features and return None instead of queue prototypes.
        if self.start_queue_at_epoch > 0 and epoch < self.start_queue_at_epoch:
            return None

        # Assign prototypes
        queue_prototypes = [self.prototypes(x, epoch) for x in queue_features]
        return queue_prototypes

    def compute_loss(self, high_resolution, low_resolution, queue, **kwargs):
        loss = self.criterion(high_resolution, low_resolution, queue)
        self.swav_loss_meter.update(loss.item())
        loss_dict = {
            "loss_batch": loss.item(),
            "loss_ep":    self.swav_loss_meter.avg
        }
        loss_str = f"ssl: {self.swav_loss_meter.avg:.6f}"
        return loss, loss_dict, loss_str

    def reset_meters(self):
        self.swav_loss_meter.reset()

    def save_model(self, fpath):
        weights_dict = {
            'type': self.__class__.__name__,
            'backbone': self.backbone.model.state_dict(),
            'projection_head': self.projection_head.state_dict(),
            'prototypes': self.prototypes.state_dict(),
        }
        torch.save(weights_dict, fpath)


class SwaVModelPairwiseMMD(SwaVModel):
    def __init__(self, queue_start, prototype_freeze_epochs, queue_len=3840, sinkhorn_epsilon=0.05, asymmetric=False,
                 dist_metric='cosine'):
        super().__init__(queue_start=queue_start, prototype_freeze_epochs=prototype_freeze_epochs, queue_len=queue_len,
                         sinkhorn_epsilon=sinkhorn_epsilon)
        self.asymmetric_mmd = asymmetric
        self.dist_metric = dist_metric
        self.anchor_features = None
        self.mmd_loss_meter = utils.AverageMeter(name='mmd_loss_meter')
        self.total_loss_meter = utils.AverageMeter(name='total_loss_meter')
        self.mmd_dist_loss = mmd_util.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([mmd_util.GaussianKernel(alpha=2 ** k, track_running_stats=True) for k in range(-3, 2)],
                     ),
            linear=False,
        ).cuda()

    def forward(self, high_resolution, low_resolution, epoch):
        self.prototypes.normalize()

        high_resolution_features = [self._subforward(x) for x in high_resolution]
        low_resolution_features = [self._subforward(x) for x in low_resolution]
        self.anchor_features = high_resolution_features[0]

        high_resolution_prototypes = [
            self.prototypes(x, epoch) for x in high_resolution_features
        ]
        low_resolution_prototypes = [
            self.prototypes(x, epoch) for x in low_resolution_features
        ]
        queue_prototypes = self._get_queue_prototypes(high_resolution_features, epoch)

        return high_resolution_prototypes, low_resolution_prototypes, queue_prototypes

    def compute_mmd_loss(self):
        src_size = self.anchor_features.shape[0] // 2
        src_feats = self.anchor_features[:src_size, :]
        trgt_feats = self.anchor_features[src_size:, :]
        src_dists, trgt_dists = get_distance(src_feats, metric=self.dist_metric), \
            get_distance(trgt_feats, metric=self.dist_metric)
        src_dists, trgt_dists = torch.reshape(src_dists, (-1, 1)), torch.reshape(trgt_dists, (-1, 1))
        if self.asymmetric_mmd:
            src_dists = src_dists.clone().detach()
        mmd_loss = self.mmd_dist_loss((src_dists, ), (trgt_dists, ))
        # print(self.anchor_features.shape, src_feats.shape, trgt_feats.shape, src_dists.shape, trgt_dists.shape,
        #       mmd_loss.shape, mmd_loss)
        return mmd_loss

    def compute_loss(self, high_resolution, low_resolution, queue, **kwargs):
        ssl_loss = self.criterion(high_resolution, low_resolution, queue)
        mmd_loss = self.compute_mmd_loss()
        loss = ssl_loss + kwargs['mmd_weight'] * mmd_loss
        self.swav_loss_meter.update(ssl_loss.item())
        self.mmd_loss_meter.update(mmd_loss.item())
        self.total_loss_meter.update(loss.item())
        loss_dict = {
            "ssl_loss_batch":   ssl_loss.item(),
            "ssl_loss_ep":      self.swav_loss_meter.avg,
            "mmd_loss_batch":   mmd_loss.item(),
            "mmd_loss_ep":      self.mmd_loss_meter.avg,
            "total_loss_batch": loss.item(),
            "total_loss_ep":    self.total_loss_meter.avg,
        }

        loss_str = (f"ssl: {self.swav_loss_meter.avg:.4f}  mmd: {self.mmd_loss_meter.avg:.4f}  "
                    f"tot: {self.total_loss_meter.avg:.4f}")
        return loss, loss_dict, loss_str

    def reset_meters(self):
        self.swav_loss_meter.reset()
        self.mmd_loss_meter.reset()
        self.total_loss_meter.reset()


def set_seeds(seed, reproduce):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if reproduce:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def get_mmd_alpha(alpha_start, step, steps, ep, ep_total, max_alpha):
    if ep <= alpha_start:
        frac = 0.0
    else:
        total_steps = steps * (ep_total - alpha_start)
        curr_step = steps * (ep - 1 - alpha_start) + step
        frac = curr_step / total_steps
    return frac * max_alpha


def get_distance(feats, metric):
    if metric == "cosine":
        dist = func.cosine_similarity(feats.unsqueeze(1), feats.unsqueeze(0), dim=-1)
    elif metric == "euclidean":
        dist = torch.linalg.norm(feats.unsqueeze(1) - feats.unsqueeze(0), dim=-1, ord=2)
    else:
        dist = torch.matmul(feats, feats.transpose(0, 1))
    assert dist.shape == (feats.shape[0], feats.shape[0]), f"{dist.shape}"
    return dist


def train(args):
    epochs, steps, bsize, workers = args['epochs'], args['iters'], args['bsize'], args['workers']
    tb_ssl_type = args['tb_ssl_type']
    lr, lr_sched, wd, hypertune = args['lr'], args['lr_sched'], args['wd'], not args['final']
    no_save, save_dir, save_freq = args['no_save'], args['save_dir'], args['save_freq']
    sinkhorn_eps, queue_start, prototype_freeze_epochs = args['sinkhorn_epsilon'], args['queue_start'], \
        args['prototype_freeze_epochs']
    queue_length = args['queue_len']
    reproduce = args['reproduce']
    model_type = args['model_type']
    mmd_alpha_max, mmd_alpha_start, asymmetric_mmd = args['mmd_alpha'], args['mmd_alpha_start'], args['asymmetric_mmd']

    seed, data_seed = args['seed'], args['data_seed']
    random_rng = np.random.default_rng()
    args['seed'] = seed if seed != -1 else int(random_rng.random() * 1e8)
    args['data_seed'] = data_seed if data_seed != -1 else int(random_rng.random() * 1e8)

    set_seeds(seed=args['seed'], reproduce=reproduce)

    start_time = datetime.datetime.now()
    tb_path = OUT_DIR + "exp_" + start_time.strftime("%b_%d_%Y_%H_%M") + "/" if save_dir == "" else \
        OUT_DIR + save_dir + "/"
    assert not os.path.isdir(tb_path), f"{tb_path} already exists.."
    tb_writer = tb.SummaryWriter(log_dir=tb_path + "tensorboard/") if not no_save else None
    logger = utils.create_logger(log_level_str=args['log'], log_file_name=tb_path + "log.txt", no_save=no_save)
    if not no_save:
        logger.info(f"Experiment log and model output will be saved to {tb_path}")

    if model_type == 'swav':
        model = SwaVModel(prototype_freeze_epochs=prototype_freeze_epochs, queue_start=queue_start,
                          queue_len=queue_length, sinkhorn_epsilon=sinkhorn_eps)
    else:
        model = SwaVModelPairwiseMMD(prototype_freeze_epochs=prototype_freeze_epochs, queue_start=queue_start,
                                     queue_len=queue_length, sinkhorn_epsilon=sinkhorn_eps, asymmetric=asymmetric_mmd)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    tb_transform_train = v2.Compose([v2.ToPILImage(),
                                     SwaVTransform(crop_counts=(1, 3),
                                                   normalize={"mean": datasets.TOYBOX_MEAN,
                                                              "std": datasets.TOYBOX_STD
                                                              }
                                                   )
                                     ]
                                    )

    tb_train_dataset = datasets.ToyboxDatasetSSL(rng=np.random.default_rng(args['data_seed']),
                                                 transform=tb_transform_train,
                                                 distort=tb_ssl_type, hypertune=hypertune)

    tb_loader_train = torchdata.DataLoader(tb_train_dataset, batch_size=bsize//2, shuffle=True, num_workers=workers,
                                           pin_memory=True, drop_last=True, persistent_workers=True)

    in12_transform_train = v2.Compose([v2.ToPILImage(),
                                       SwaVTransform(crop_counts=(1, 3),
                                                     normalize={
                                                         "mean": datasets.IN12_MEAN,
                                                         "std": datasets.IN12_STD
                                                     }
                                                     )
                                       ]
                                      )

    in12_train_dataset = datasets.IN12SSLWithLabels(transform=in12_transform_train, hypertune=hypertune)

    in12_loader_train = torchdata.DataLoader(in12_train_dataset, batch_size=bsize//2, shuffle=True, num_workers=workers,
                                             pin_memory=True, drop_last=True, persistent_workers=True)

    tb_forever_loader = utils.ForeverDataLoader(tb_loader_train)
    in12_forever_loader = utils.ForeverDataLoader(in12_loader_train)

    optimizer = torch.optim.Adam(model.backbone.parameters(), lr=lr, weight_decay=wd)
    optimizer.add_param_group({'params': model.projection_head.parameters(), 'lr': lr, 'wd': wd})
    optimizer.add_param_group({'params': model.prototypes.parameters(), 'lr': lr, 'wd': wd})

    if lr_sched == "cosine":
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer=optimizer, start_factor=0.01, end_factor=1.0,
                                                             total_iters=2 * steps)
        decay_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(epochs - 2) * steps)
        combined_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer=optimizer,
                                                                   schedulers=[warmup_scheduler, decay_scheduler],
                                                                   milestones=[2 * steps + 1])
    else:
        combined_scheduler = None

    for epoch in range(1, epochs+1):
        num_batches = 0
        total_loss = 0
        start_time = time.time()
        loss_str = ""
        for it in range(steps):
            # print(it)
            tb_scalar_dicts = defaultdict(dict)
            num_batches += 1
            src_idxs, src_images, src_labels = tb_forever_loader.get_next_batch()
            trgt_idxs, trgt_images, trgt_labels = in12_forever_loader.get_next_batch()
            # print(len(images), type(images), len(images[0]), type(images[0]), type(images[0][0]), images[0][0].shape,
            #       images[0][1].shape)
            src_views = [src_images[0][0], src_images[1][0],                      # high-res images
                         src_images[0][1], src_images[0][2], src_images[0][3],    # low-res images from view 1
                         src_images[1][1], src_images[1][2], src_images[1][3]]    # low-res images from view 2
            trgt_views = [trgt_images[0][0], trgt_images[1][0],  # high-res images
                          trgt_images[0][1], trgt_images[0][2], trgt_images[0][3],  # low-res images from view 1
                          trgt_images[1][1], trgt_images[1][2], trgt_images[1][3]]  # low-res images from view 2

            src_views = [src_view.to(device) for src_view in src_views]
            trgt_views = [trgt_view.to(device) for trgt_view in trgt_views]

            src_high_resolution, src_low_resolution = src_views[:2], src_views[2:]
            trgt_high_resolution, trgt_low_resolution = trgt_views[:2], trgt_views[2:]

            # print(" ".join([str(view.shape) for view in src_high_resolution]))
            # print(" ".join([str(view.shape) for view in src_low_resolution]))
            # print(" ".join([str(view.shape) for view in trgt_high_resolution]))
            # print(" ".join([str(view.shape) for view in trgt_low_resolution]))
            high_resolution = [torch.cat([src_high_resolution[i], trgt_high_resolution[i]]) for
                               i in range(len(src_high_resolution))]
            low_resolution = [torch.cat([src_low_resolution[i], trgt_low_resolution[i]]) for
                              i in range(len(src_low_resolution))]

            if epoch == 1 and it == 0 and args['ims']:
                high_res_ims = [
                    utils.get_images(high_resolution[0], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(high_resolution[1], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True)
                ]
                concat_high_res = utils.concat_images_vertically(high_res_ims)
                concat_high_res.show()

                low_res_ims = [
                    utils.get_images(low_resolution[0], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(low_resolution[1], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(low_resolution[2], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(low_resolution[3], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(low_resolution[4], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                    utils.get_images(low_resolution[5], mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD, aug=True),
                ]
                concat_low_res = utils.concat_images_vertically(low_res_ims)
                concat_low_res.show()

            # print(" ".join([str(view.shape) for view in high_resolution]))
            # print(" ".join([str(view.shape) for view in low_resolution]))
            high_resolution, low_resolution, queue = model(
                high_resolution, low_resolution, epoch - 1
            )
            mmd_alpha = get_mmd_alpha(alpha_start=mmd_alpha_start, step=it+1, steps=steps, ep=epoch, ep_total=epochs,
                                      max_alpha=mmd_alpha_max)
            loss, losses_dict, loss_str = model.compute_loss(high_resolution, low_resolution, queue,
                                                             mmd_weight=mmd_alpha)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            if combined_scheduler is not None:
                combined_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if not no_save:
                tb_scalar_dicts["training_lr"] = {
                    "backbone": optimizer.param_groups[0]['lr'],
                    "projection": optimizer.param_groups[1]['lr'],
                    "prototypes": optimizer.param_groups[2]['lr']
                }
                tb_scalar_dicts["training_loss"] = losses_dict

                tb_writer.add_scalar("training_deets/qsize", 0 if queue is None else queue[0].size(0),
                                     global_step=(epoch - 1) * steps + num_batches)
                tb_writer.add_scalar("training_deets/proto_changing",
                                     int(model.prototypes.heads[0].weight.requires_grad),
                                     global_step=(epoch - 1) * steps + num_batches)
                tb_writer.add_scalar("training_deets/sinkhorn_epsilon", model.criterion.sinkhorn_epsilon,
                                     global_step=(epoch - 1) * steps + num_batches)
                if model_type == "swav-pairwise":
                    tb_writer.add_scalar("training_deets/div_alpha", mmd_alpha,
                                         global_step=(epoch - 1) * steps + num_batches)
            if not no_save:
                for key, val_dict in tb_scalar_dicts.items():
                    tb_writer.add_scalars(
                        main_tag=key,
                        tag_scalar_dict=val_dict,
                        global_step=(epoch - 1) * steps + num_batches
                    )
        logger_strs = [f"ep: {epoch}/{epochs}  it: {steps}/{steps}  lr: {optimizer.param_groups[0]['lr']:1.2e}",
                       loss_str, f"t: {time.time() - start_time:.0f}s"]
        if model_type == "swav-pairwise":
            logger_strs.append(f"alpha: {mmd_alpha:.3f}")
        logger.info("  ".join(logger_strs))
        model.reset_meters()

    if not no_save:
        final_model_path = f"{tb_path}/final_model.pt"
        utils.save_args(path=tb_path, args=args)
        model.save_model(fpath=final_model_path)
        logger.info(f"Experiment log and model output saved to {tb_path}")
        return tb_path
    return None


if __name__ == "__main__":
    swav_parser = get_swav_parser()
    swav_args = vars(swav_parser.parse_args())
    final_model_save_path = train(args=swav_args)
    if final_model_save_path is not None:
        linear_eval_args = {
            'epochs':   10,
            'iters':    10,
            'bsize':   128,
            'workers':   8,
            'final': False,
            'dataset': 'toybox',
            'lr': 0.001,
            'wd': 1e-6,
            'instances': -1,
            'images':  5000,
            'seed': -1,
            'log': 'info',
            'load_path': final_model_save_path,
            'model_name': 'final_model.pt',
            'pretrained': False,
            'no_save': True,
            'save_dir': ""
        }
        linear_eval.main(exp_args=linear_eval_args)
