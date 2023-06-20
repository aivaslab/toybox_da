"""
Module with model definition for noisy labels models
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as func
import torch.utils.tensorboard as tb

import utils


class NoisyModel1:
    """Class definition for noisy model 1"""
    
    # TODO: Add description of model
    
    def __init__(self, network_pt, network, train_loader, logger):
        self.network_pt = network_pt
        self.network = network
        self.train_loader = utils.ForeverDataLoader(train_loader)
        self.logger = logger
        self.network.cuda()
        self.network_pt.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network_pt.set_eval()
        self.network.set_train()
        num_batches = 0
        total_batches = (ep - 1) * steps
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            p = total_batches / (steps * ep_total)
            
            src_idx, src_images, src_labels = self.train_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            
            total_batches += 1
            
            with torch.no_grad():
                feats_pt, logits_pt = self.network_pt.forward(src_images)
                _, pred_pt = logits_pt.topk(1, 1, True, True)
                pred_pt = pred_pt.t()
            
            feats, logits = self.network.forward(src_images, step=total_batches)
            
            if ep == 1 and step == 1:
                print(feats.shape, logits.shape, logits_pt.shape, pred_pt[0].shape, pred_pt[0].dtype,
                      src_labels.shape, src_labels.dtype)
                
            # compute loss wrt labels from pt_model
            src_loss = src_criterion(logits, pred_pt[0])
            
            src_loss.backward()
            optimizer.step()
            
            ce_loss_total += src_loss.item()
            num_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps,
                    optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ce_loss_total / num_batches, time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
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
                },
                global_step=total_batches,
            )
            if scheduler is not None:
                scheduler.step()

        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.4f}  CLR: {:.4f}  CE: {:.4f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps,
            optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ce_loss_total / num_batches, time.time() - start_time))
    
    def eval(self, loader, noisy=False):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        self.network_pt.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                if noisy:
                    _, logits = self.network_pt.forward(images)
                else:
                    _, logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
