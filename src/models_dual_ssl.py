
import time

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

import utils


class DualSSLOrientedModelV1:
    """Module implementing the SSL method for pretraining the DA model with both TB and IN-12 data"""

    def __init__(self, network, src_loader, trgt_loader, logger, no_save, decoupled, tb_alpha, in12_alpha,
                 orient_alpha, combined, num_images_in_batch):
        self.network = network
        self.src_loader = utils.ForeverDataLoader(src_loader)
        self.trgt_loader = utils.ForeverDataLoader(trgt_loader)
        self.logger = logger
        self.network.cuda()
        self.network.freeze_train()
        self.no_save = no_save
        self.decoupled = decoupled
        self.tb_alpha = tb_alpha
        self.in12_alpha = in12_alpha
        self.combined = combined
        self.orient_alpha = orient_alpha
        self.num_images_in_batch = num_images_in_batch

        num_params_trainable, num_params = self.network.count_trainable_parameters()
        self.orientation_loss = utils.OrientationLossV1(num_images_in_batch=self.num_images_in_batch)
        self.logger.info(f"{num_params_trainable} / {num_params} parameters are trainable...")

    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_loss_total = 0.0
        trgt_loss_total = 0.0
        ssl_loss_total = 0.0
        orient_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()

            src_idx, src_images, src_labels = self.src_loader.get_next_batch()
            trgt_idx, trgt_images, trgt_labels = self.trgt_loader.get_next_batch()
            assert len(src_images) == 4 and len(trgt_images) == 2

            if self.combined:
                anchors, positives = torch.cat([src_images[0], src_images[2], trgt_images[0]], dim=0), \
                    torch.cat([src_images[1], src_images[3], trgt_images[1]], dim=0)

                images = torch.cat([anchors, positives], dim=0)
                images = images.cuda()
                feats = self.network.forward(images)
                if self.decoupled:
                    ssl_loss = utils.decoupled_contrastive_loss(features=feats, temp=0.5)
                else:
                    logits, labels = utils.info_nce_loss(features=feats, temp=0.5)
                    ssl_loss = criterion(logits, labels)
                src_loss, trgt_loss = torch.tensor(0.0), torch.tensor(0.0)

                num_images_src, num_images_trgt = len(src_images[0]), len(trgt_images[0])

                src_feats_1, src_feats_2, src_feats_3, src_feats_4 = feats[:num_images_src], \
                    feats[num_images_src:2 * num_images_src], \
                    feats[2 * num_images_src + num_images_trgt:3 * num_images_src + num_images_trgt], \
                    feats[3 * num_images_src + num_images_trgt:4 * num_images_src + num_images_trgt]

                src_feats_stacked = torch.stack(
                    tensors=(src_feats_1, src_feats_2, src_feats_3, src_feats_4),
                    dim=1).view(4 * num_images_src, -1)

                trgt_feats_1, trgt_feats_2 = feats[2 * num_images_src:2 * num_images_src + num_images_trgt], \
                    feats[4 * num_images_src + num_images_trgt:4 * num_images_src + 2 * num_images_trgt]

                trgt_feats_stacked = torch.stack(
                    tensors=(trgt_feats_1, trgt_feats_2),
                    dim=1).view(2 * num_images_trgt, -1)
                orientation_loss = self.orientation_loss(src_feats=src_feats_stacked, trgt_feats=trgt_feats_stacked)
                loss = ssl_loss + self.orient_alpha * orientation_loss
                orient_loss_total += orientation_loss.item()
            else:
                src_anchors, src_positives = torch.cat([src_images[0], src_images[2]], dim=0), \
                    torch.cat([src_images[1], src_images[3]], dim=0)
                src_batch = torch.cat([src_anchors, src_positives], dim=0)
                trgt_anchors, trgt_positives = trgt_images[0], trgt_images[1]
                trgt_batch = torch.cat([trgt_anchors, trgt_positives], dim=0)

                images = torch.cat([src_batch, trgt_batch], dim=0)
                images = images.cuda()
                feats = self.network.forward(images)
                src_size = src_batch.shape[0]
                src_feats, trgt_feats = feats[:src_size], feats[src_size:]
                # print(src_feats.shape, trgt_feats.shape)
                if self.decoupled:
                    src_loss = utils.decoupled_contrastive_loss(features=src_feats, temp=0.5)
                    trgt_loss = utils.decoupled_contrastive_loss(features=trgt_feats, temp=0.5)
                else:
                    src_logits, src_labels = utils.info_nce_loss(features=src_feats, temp=0.5)
                    src_loss = criterion(src_logits, src_labels)
                    trgt_logits, trgt_labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
                    trgt_loss = criterion(trgt_logits, trgt_labels)

                num_images_src = len(src_images[0])
                src_feats_1, src_feats_2, src_feats_3, src_feats_4 = src_feats[:num_images_src], \
                    src_feats[num_images_src:2 * num_images_src], src_feats[2 * num_images_src:3 * num_images_src], \
                    src_feats[3 * num_images_src:4 * num_images_src]

                src_feats_stacked = torch.stack(
                    tensors=(src_feats_1, src_feats_2, src_feats_3, src_feats_4),
                    dim=1).view(4*num_images_src, -1)

                num_images_trgt = len(trgt_images[0])
                trgt_feats_1, trgt_feats_2 = trgt_feats[:num_images_trgt], \
                    trgt_feats[num_images_trgt:2*num_images_trgt]
                trgt_feats_stacked = torch.stack(
                    tensors=(trgt_feats_1, trgt_feats_2),
                    dim=1).view(2*num_images_trgt, -1)
                orientation_loss = self.orientation_loss(src_feats=src_feats_stacked, trgt_feats=trgt_feats_stacked)

                loss = self.tb_alpha * src_loss + self.in12_alpha * trgt_loss + self.orient_alpha * orientation_loss
                src_loss_total += src_loss.item()
                trgt_loss_total += trgt_loss.item()
                orient_loss_total += orientation_loss.item()
            loss.backward()

            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            ssl_loss_total += loss.item()
            num_batches += 1
            if 0 <= step - halfway < 1:
                if self.combined:
                    self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  "
                                     "OrL: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                                         ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         ssl_loss_total / num_batches, orient_loss_total / num_batches,
                                         time.time() - start_time)
                                     )
                else:
                    self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                                     "OrL: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                                        ep, ep_total, step, steps,
                                        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                        src_loss_total / num_batches, trgt_loss_total / num_batches,
                                        orient_loss_total / num_batches,
                                        ssl_loss_total / num_batches, time.time() - start_time)
                                     )

            if not self.no_save:
                if self.combined:
                    scalar_dict = {
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                        'orient_loss_ep': orient_loss_total / num_batches,
                        'orient_loss_batch': orientation_loss.item(),
                    }
                else:
                    scalar_dict = {
                        'src_ssl_loss_ep': src_loss_total / num_batches,
                        'src_ssl_loss_batch': src_loss.item(),
                        'trgt_ssl_loss_ep': trgt_loss_total / num_batches,
                        'trgt_ssl_loss_batch': trgt_loss.item(),
                        'orient_loss_ep': orient_loss_total / num_batches,
                        'orient_loss_batch': orientation_loss.item(),
                        'ssl_loss_ep': ssl_loss_total / num_batches,
                        'ssl_loss_batch': loss.item(),
                    }
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict=scalar_dict,
                    global_step=(ep - 1) * steps + num_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'ssl_head': optimizer.param_groups[1]['lr'],
                    },
                    global_step=(ep - 1) * steps + num_batches,
                )
        if self.combined:
            self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  "
                             "OrL: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                                ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                ssl_loss_total / num_batches, orient_loss_total / num_batches,
                                time.time() - start_time)
                             )
        else:
            self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL1: {:.3f}  SSL2: {:.3f}  "
                             "OrL: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                                ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                src_loss_total / num_batches, trgt_loss_total / num_batches,
                                orient_loss_total / num_batches,
                                ssl_loss_total / num_batches, time.time() - start_time)
                             )
