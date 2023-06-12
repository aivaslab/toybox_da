"""
Module with model definition for MTL tasks
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.tensorboard as tb

import utils

import tllib.alignment.jan as jan
import tllib.modules.kernels as kernels


class JANMTLModel:
    """Class definition for JAN-MTL model"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger):
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
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter, lmbda):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        ce_loss_total = 0.0
        jmmd_loss_total = 0.0
        ssl_loss_total = 0.0
        comb_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        trgt_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            alfa = 2 / (1 + math.exp(-10 * p)) - 1
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            
            trgt_idx, trgt_images = self.target_loader.get_next_batch()
            trgt_images = torch.cat(trgt_images, dim=0)
            trgt_images = trgt_images.cuda()
            
            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                feats, cl_logits, ssl_logits = self.network.forward_all(comb_images)
                src_size = src_images.shape[0]
                src_feats, src_cl_logits, src_ssl_logits = feats[:src_size], cl_logits[:src_size], ssl_logits[:src_size]
                trgt_feats, trgt_cl_logits, trgt_ssl_logits = \
                    feats[src_size:], cl_logits[src_size:], ssl_logits[src_size:]
            else:
                src_feats, src_cl_logits, src_ssl_logits = self.network.forward_all(src_images)
                trgt_feats, trgt_cl_logits, trgt_ssl_logits = self.network.forward_all(trgt_images)
            
            if ep == 1 and step == 1:
                print(src_feats.shape, trgt_feats.shape, src_cl_logits.shape, trgt_cl_logits.shape,
                      src_ssl_logits.shape, trgt_ssl_logits.shape)
            src_loss = src_criterion(src_cl_logits, src_labels)
            jmmd_loss = self.jmmd_loss(
                (src_feats, func.softmax(src_cl_logits, dim=1)), (trgt_feats, func.softmax(trgt_cl_logits, dim=1)))
            
            logits, labels = utils.info_nce_loss(features=trgt_ssl_logits, temp=0.5)
            trgt_loss = trgt_criterion(logits, labels)
            total_loss = src_loss + alfa * jmmd_loss + lmbda * trgt_loss
            
            total_loss.backward()
            optimizer.step()
            
            ce_loss_total += src_loss.item()
            ssl_loss_total += trgt_loss.item()
            jmmd_loss_total += jmmd_loss.item()
            comb_loss_total += total_loss.item()
            num_batches += 1
            total_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SLR: {:.4f}  CE: {:.4f}  "
                                 "JMMD: {:.4f}  SSL: {:.3f}  Tot: {:.4f}  T: {:.2f}s".format(
                                    ep, ep_total, step, steps,
                                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                    optimizer.param_groups[3]['lr'],
                                    ce_loss_total / num_batches, jmmd_loss_total / num_batches,
                                    ssl_loss_total / num_batches,
                                    comb_loss_total / num_batches, time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
                    'jmmd_loss_ep': jmmd_loss_total / num_batches,
                    'jmmd_loss_batch': jmmd_loss.item(),
                    'ssl_loss_ep': ssl_loss_total / num_batches,
                    'ssl_loss_batch': trgt_loss.item(),
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
                    'ssl_fc': optimizer.param_groups[3]['lr'],
                },
                global_step=total_batches,
            )
            writer.add_scalars(
                main_tag="training",
                tag_scalar_dict={
                    'p': p,
                    'lambda': alfa,
                    'lambda_ssl': lmbda
                },
                global_step=total_batches,
            )
            if scheduler is not None:
                scheduler.step()

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  SLR: {:.4f}  CE: {:.4f}  "
                         "JMMD: {:.4f}  SSL: {:.3f}  Tot: {:.4f}  T: {:.2f}s".format(
                            ep, ep_total, steps, steps,
                            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                            optimizer.param_groups[2]['lr'],
                            ce_loss_total / num_batches, jmmd_loss_total / num_batches,
                            ssl_loss_total / num_batches,
                            comb_loss_total / num_batches, time.time() - start_time))
    
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward_class(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
