"""Module for implementing the Domain Adaptation models"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.tensorboard as tb

import tllib.alignment.jan as jan
import tllib.modules.kernels as kernels

import utils


class JANModel:
    """Module implementing the JAN architecture"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.jmmd_loss = jan.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([kernels.GaussianKernel(alpha=2**k) for k in range(-3, 2)],
                     (kernels.GaussianKernel(sigma=0.92, track_running_stats=False), )
                     ),
            linear=False,
            thetas=None,
        ).cuda()
        self.jmmd_loss.train()
        self.logger = logger
        self.combined_batch = combined_batch
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        ce_loss_total = 0.0
        jmmd_loss_total = 0.0
        comb_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = 2 / (1 + math.exp(-10 * p)) - 1
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()

            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()
            
            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                bb_feats, btl_feats, logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_bb_feats, src_btl_feats, src_logits = bb_feats[:src_size], btl_feats[:src_size], logits[:src_size]
                trgt_bb_feats, trgt_btl_feats, trgt_logits = \
                    bb_feats[src_size:], btl_feats[src_size:], logits[src_size:]
            else:
                src_bb_feats, src_btl_feats, src_logits = self.network.forward(src_images)
                trgt_bb_feats, trgt_btl_feats, trgt_logits = self.network.forward(trgt_images)
                
            src_loss = src_criterion(src_logits, src_labels)
            
            jmmd_loss = self.jmmd_loss(
                (src_bb_feats, src_btl_feats, func.softmax(src_logits, dim=1)),
                (trgt_bb_feats, trgt_btl_feats, func.softmax(trgt_logits, dim=1)))
            total_loss = src_loss + alfa * jmmd_loss
            
            total_loss.backward()
            optimizer.step()
            
            ce_loss_total += src_loss.item()
            jmmd_loss_total += jmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  JMMD: {:.4f}  "
                                 "Tot: {:.4f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    ce_loss_total / num_batches, jmmd_loss_total / num_batches,
                                    comb_loss_total / num_batches, time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
                    'jmmd_loss_ep': jmmd_loss_total / num_batches,
                    'jmmd_loss_batch': jmmd_loss.item(),
                    'comb_loss_ep': comb_loss_total / num_batches,
                    'comb_loss_batch': total_loss.item(),
                },
                global_step=total_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'bottleneck': optimizer.param_groups[1]['lr'],
                    'fc': optimizer.param_groups[2]['lr'],
                },
                global_step=total_batches,
            )
            writer.add_scalars(
                main_tag="training",
                tag_scalar_dict={
                    'p': p,
                    'lambda': alfa,
                },
                global_step=total_batches,
            )
            if scheduler is not None:
                scheduler.step()

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  JMMD: {:.4f}  "
                         "Tot: {:.4f}  T: {:.0f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[2]['lr'],
                            ce_loss_total / num_batches, jmmd_loss_total / num_batches, comb_loss_total / num_batches,
                            time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                with torch.no_grad():
                    _, _, logits = self.network.forward(images)
                loss = criterion(logits, labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
    
        writer.add_scalars(main_tag="val_loss",
                           tag_scalar_dict={
                               loader_names[0]: losses[0],
                               loader_names[1]: losses[1]
                           },
                           global_step=ep * steps)
    
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                _, _, logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class DualSupWithJANModel:
    """Module implementing the Dual Supervised Model with Jan loss"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger, no_save,
                 scramble_target_for_classification=False, scrambler_seed=None):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.jmmd_loss = jan.JointMultipleKernelMaximumMeanDiscrepancy(
            kernels=([kernels.GaussianKernel(alpha=2 ** k) for k in range(-3, 2)],
                     (kernels.GaussianKernel(sigma=0.92, track_running_stats=False),)
                     ),
            linear=False,
            thetas=None,
        ).cuda()
        self.jmmd_loss.train()
        self.logger = logger
        self.combined_batch = combined_batch
        self.no_save = no_save
        self.network.cuda()
    
        self.scramble_labels = scramble_target_for_classification
        self.scrambler_seed = scrambler_seed if scrambler_seed is not None else 42
        import scramble_labels
        self.scrambler = \
            scramble_labels.RandomToyboxScrambler(seed=self.scrambler_seed) if self.scramble_labels else None
        
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        jmmd_loss_total = 0.0
        comb_loss_total = 0.0
        ce_criterion = nn.CrossEntropyLoss()
        
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = 2 / (1 + math.exp(-10 * p)) - 1
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()
            
            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                feats, logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_feats, src_logits = feats[:src_size], logits[:src_size]
                trgt_feats, trgt_logits = feats[src_size:], logits[src_size:]
            else:
                src_feats, src_logits = self.network.forward(src_images)
                trgt_feats, trgt_logits = self.network.forward(trgt_images)
                
            if self.scrambler is not None:
                scrambled_trgt_labels = self.scrambler.scramble(trgt_labels)
            else:
                scrambled_trgt_labels = trgt_labels.clone()
            
            src_loss = ce_criterion(src_logits, src_labels)
            trgt_loss = ce_criterion(trgt_logits, scrambled_trgt_labels)
            jmmd_loss = self.jmmd_loss(
                (src_feats, func.softmax(src_logits, dim=1)), (trgt_feats, func.softmax(trgt_logits, dim=1)))
            total_loss = src_loss + trgt_loss + alfa * jmmd_loss
            
            total_loss.backward()
            optimizer.step()
            
            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            jmmd_loss_total += jmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  "
                                 "JMMD: {:.4f}  Tot: {:.4f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                    jmmd_loss_total / num_batches,
                                    comb_loss_total / num_batches, time.time() - start_time))
            
            if not self.no_save:
                writer.add_scalars(
                    main_tag="training_loss",
                    tag_scalar_dict={
                        'src_ce_loss_ep': src_ce_loss_total / num_batches,
                        'src_ce_loss_batch': src_loss.item(),
                        'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                        'trgt_ce_loss_batch': trgt_loss.item(),
                        'jmmd_loss_ep': jmmd_loss_total / num_batches,
                        'jmmd_loss_batch': jmmd_loss.item(),
                        'comb_loss_ep': comb_loss_total / num_batches,
                        'comb_loss_batch': total_loss.item(),
                    },
                    global_step=total_batches,
                )
                writer.add_scalars(
                    main_tag="training_lr",
                    tag_scalar_dict={
                        'bb': optimizer.param_groups[0]['lr'],
                        'fc': optimizer.param_groups[1]['lr'],
                    },
                    global_step=total_batches,
                )
                writer.add_scalars(
                    main_tag="training",
                    tag_scalar_dict={
                        'p': p,
                        'lambda': alfa,
                    },
                    global_step=total_batches,
                )
            if scheduler is not None:
                scheduler.step()
        
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SCE: {:.4f}  TCE: {:.4f}  JMMD: {:.4f}  "
                         "Tot: {:.4f}  T: {:.0f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                            src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                            jmmd_loss_total / num_batches, comb_loss_total / num_batches,
                            time.time() - start_time))

    def calc_val_loss(self, loaders, loader_names, ep, steps, writer: tb.SummaryWriter):
        """Calculate loss on provided dataloader"""
        self.network.set_eval()
        criterion = nn.CrossEntropyLoss()
        losses = []
        for idx, loader in enumerate(loaders):
            num_batches = 0
            total_loss = 0.0
            for _, (idxs, images, labels) in enumerate(loader):
                num_batches += 1
                images, labels = images.cuda(), labels.cuda()
                if "in12" in loader_names[idx] and self.scrambler is not None and self.scramble_labels:
                    scrambled_labels = self.scrambler.scramble(labels)
                else:
                    scrambled_labels = labels.clone()
                with torch.no_grad():
                    _, logits = self.network.forward(images)
                loss = criterion(logits, scrambled_labels)
                total_loss += loss.item()
            losses.append(total_loss / num_batches)
        self.logger.info("Validation Losses -- {:s}: {:.2f}     {:s}: {:.2f}".format(loader_names[0], losses[0],
                                                                                     loader_names[1], losses[1]))
        
        if not self.no_save:
            writer.add_scalars(main_tag="val_loss",
                               tag_scalar_dict={
                                   loader_names[0]: losses[0],
                                   loader_names[1]: losses[1]
                               },
                               global_step=ep * steps)

    def eval(self, loader, scramble=False):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            if scramble and self.scrambler is not None:
                scrambled_labels = self.scrambler.scramble(labels)
            else:
                scrambled_labels = labels.clone()
            with torch.no_grad():
                _, logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=scrambled_labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
