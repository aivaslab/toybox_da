"""Module implementing the MTL pretraining task for Toybox->IN-12"""
import time

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

import utils


class ModelLE:
    """model for linear eval of network"""
    def __init__(self, network, train_loader, test_loader, logger):
        self.network = network
        self.train_loader = utils.ForeverDataLoader(train_loader)
        self.test_loader = test_loader
        self.logger = logger
        self.network.cuda()
        
    def train(self, optimizer, scheduler, ep, ep_total, steps):
        """Train the network for 1 epoch"""
        self.network.set_linear_eval()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
    
            idx, images, labels = self.train_loader.get_next_batch()
            images, labels = images.cuda(), labels.cuda()
            logits = self.network.forward(images)
            loss = src_criterion(logits, labels)
            loss.backward()
    
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    
            ce_loss_total += loss.item()
            num_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], ce_loss_total / num_batches,
                    time.time() - start_time)
                )

        self.logger.info("Ep: {}/{}  Step: {}/{}  LR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], ce_loss_total / num_batches,
            time.time() - start_time)
        )
        
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)
        
        
class MTLModel:
    """Module implementing the MTL method for pretraining the DA model"""
    def __init__(self, network, source_loader, target_loader, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.network.cuda()
        
    def train(self, optimizer, scheduler, steps, ep, ep_total, lmbda, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        ssl_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        trgt_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps+1):
            optimizer.zero_grad()
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            src_logits = self.network.forward_class(src_images)
            src_loss = src_criterion(src_logits, src_labels)
            
            trgt_idx, trgt_images = self.target_loader.get_next_batch()
            trgt_images = torch.cat(trgt_images, dim=0)
            trgt_images = trgt_images.cuda()
            trgt_feats = self.network.forward_ssl(trgt_images)
            logits, labels = utils.info_nce_loss(features=trgt_feats, temp=0.5)
            trgt_loss = trgt_criterion(logits, labels)
            
            total_loss = src_loss + lmbda * trgt_loss
            total_loss.backward()
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            ce_loss_total += src_loss.item()
            ssl_loss_total += trgt_loss.item()
            num_batches += 1

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ssl_loss_ep': ssl_loss_total / num_batches,
                    'ssl_loss_batch': trgt_loss.item(),
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
                    'total_loss_batch': total_loss.item()
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'cl_head': optimizer.param_groups[1]['lr'],
                    'ssl_head': optimizer.param_groups[2]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{} Step:{}/{}  BLR: {:.3f}  CLR: {:.3f}  SLR: {:.3f}  CE: {:.3f}  SSL: {:.3f}"
                                 "T: {:.2f}s".
                                 format(ep, ep_total, step, steps,
                                        optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                        optimizer.param_groups[2]['lr'], ce_loss_total/num_batches,
                                        ssl_loss_total/num_batches, time.time() - start_time)
                                 )
        self.logger.info("Ep: {}/{} Step:{}/{}  BLR: {:.3f}  CLR: {:.3f}  SLR: {:.3f}  CE: {:.3f}  SSL: {:.3f}"
                         "T: {:.2f}s".
                         format(ep, ep_total, steps, steps,
                                optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                optimizer.param_groups[2]['lr'], ce_loss_total / num_batches,
                                ssl_loss_total / num_batches, time.time() - start_time)
                         )
        
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


class SSLModel:
    """Module implementing the SSL method for pretraining the DA model"""
    
    def __init__(self, network, loader, logger):
        self.network = network
        self.loader = utils.ForeverDataLoader(loader)
        self.logger = logger
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ssl_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            
            idx, images = self.loader.get_next_batch()
            images = torch.cat(images, dim=0)
            images = images.cuda()
            feats = self.network.forward(images)
            logits, labels = utils.info_nce_loss(features=feats, temp=0.5)
            loss = criterion(logits, labels)
            
            loss.backward()
            
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            ssl_loss_total += loss.item()
            num_batches += 1
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ssl_loss_total / num_batches, time.time() - start_time))

            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ssl_loss_ep': ssl_loss_total / num_batches,
                    'ssl_loss_batch': loss.item(),
                },
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
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ssl_loss_total / num_batches, time.time() - start_time))


class SupModel:
    """Module implementing the supervised pretraining on source"""
    
    def __init__(self, network, source_loader, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.logger = logger
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            src_logits = self.network.forward(src_images)
            src_loss = src_criterion(src_logits, src_labels)
            
            src_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            ce_loss_total += src_loss.item()
            num_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
                    ep, ep_total, step, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                    ce_loss_total / num_batches, time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'ce_loss_ep': ce_loss_total / num_batches,
                    'ce_loss_batch': src_loss.item(),
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'fc': optimizer.param_groups[1]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.3f}  T: {:.2f}s".format(
            ep, ep_total, steps, steps, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
            ce_loss_total / num_batches, time.time() - start_time))
        
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class DualSupModel:
    """Module implementing the combined supervised pretraining on source and target"""
    
    def __init__(self, network, source_loader, target_loader, combined_batch, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.combined_batch = combined_batch
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()
            
            if self.combined_batch:
                comb_images = torch.concat([src_images, trgt_images], dim=0)
                logits = self.network.forward(comb_images)
                src_size = src_images.shape[0]
                src_logits = logits[:src_size]
                trgt_logits = logits[src_size:]
            else:
                src_logits = self.network.forward(src_images)
                trgt_logits = self.network.forward(trgt_images)
            
            src_loss = criterion(src_logits, src_labels)
            trgt_loss = criterion(trgt_logits, trgt_labels)
            total_loss = src_loss + trgt_loss
            
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            num_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                                 "T: {:.2f}s"
                                 .format(ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                         time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'src_ce_loss_ep': src_ce_loss_total / num_batches,
                    'src_ce_loss_batch': src_loss.item(),
                    'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                    'trgt_ce_loss_batch': trgt_loss.item(),
                    'total_loss_batch': total_loss.item(),
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'fc': optimizer.param_groups[1]['lr'],
                },
                global_step=(ep - 1) * steps + num_batches,
            )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SCE: {:.3f}  TCE: {:.3f}  "
                         "T: {:.2f}s".format(ep, ep_total, steps, steps,
                                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                             src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                             time.time() - start_time))
    
    def eval(self, loader):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
        acc = n_correct / n_total
        return round(acc, 2)


class DualSupModelWithDomain:
    """Module implementing the combined supervised pretraining on source and target and domain classification"""
    
    def __init__(self, network, source_loader, target_loader, logger):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.logger = logger
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        src_ce_loss_total = 0.0
        trgt_ce_loss_total = 0.0
        dom_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        dom_criterion = nn.BCEWithLogitsLoss()
        halfway = steps / 2.0
        start_time = time.time()
        for step in range(1, steps + 1):
            optimizer.zero_grad()
            
            src_idx, src_images, src_labels = self.source_loader.get_next_batch()
            src_images, src_labels = src_images.cuda(), src_labels.cuda()
            trgt_idx, trgt_images, trgt_labels = self.target_loader.get_next_batch()
            trgt_images, trgt_labels = trgt_images.cuda(), trgt_labels.cuda()
            dom_labels = torch.cat([torch.zeros(src_images.size(0)), torch.ones(trgt_images.size(0))])
            dom_labels = dom_labels.cuda()
            
            comb_images = torch.concat([src_images, trgt_images], dim=0)
            logits, dom_logits = self.network.forward(comb_images)
            src_size = src_images.shape[0]
            src_logits = logits[:src_size]
            trgt_logits = logits[src_size:]
            
            src_loss = criterion(src_logits, src_labels)
            trgt_loss = criterion(trgt_logits, trgt_labels)
            dom_loss = dom_criterion(dom_logits, dom_labels)
            total_loss = src_loss + trgt_loss + dom_loss
            
            total_loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
            
            src_ce_loss_total += src_loss.item()
            trgt_ce_loss_total += trgt_loss.item()
            dom_loss_total += dom_loss.item()
            num_batches += 1
            
            if 0 <= step - halfway < 1:
                self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  "
                                 "SCE: {:.3f}  TCE: {:.3f}  D-BCE: {:.3f}  "
                                 "T: {:.2f}s"
                                 .format(ep, ep_total, step, steps,
                                         optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                         src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                         dom_loss_total / num_batches,
                                         time.time() - start_time))
            
            writer.add_scalars(
                main_tag="training_loss",
                tag_scalar_dict={
                    'src_ce_loss_ep': src_ce_loss_total / num_batches,
                    'src_ce_loss_batch': src_loss.item(),
                    'trgt_ce_loss_ep': trgt_ce_loss_total / num_batches,
                    'trgt_ce_loss_batch': trgt_loss.item(),
                    'dom_loss_ep': dom_loss_total / num_batches,
                    'dom_loss_batch': dom_loss.item(),
                    'total_loss_batch': total_loss.item(),
                },
                global_step=(ep - 1) * steps + num_batches,
            )
            writer.add_scalars(
                main_tag="training_lr",
                tag_scalar_dict={
                    'bb': optimizer.param_groups[0]['lr'],
                    'fc': optimizer.param_groups[1]['lr'],
                    'dom_fc': optimizer.param_groups[2]['lr']
                },
                global_step=(ep - 1) * steps + num_batches,
            )
        self.logger.info("Ep: {}/{}  Step: {}/{}  BLR: {:.3f}  CLR: {:.3f}  "
                         "SCE: {:.3f}  TCE: {:.3f}  D-BCE: {:.3f}  "
                         "T: {:.2f}s".format(ep, ep_total, steps, steps,
                                             optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                                             src_ce_loss_total / num_batches, trgt_ce_loss_total / num_batches,
                                             dom_loss_total / num_batches,
                                             time.time() - start_time))
    
    def eval(self, loader, source):
        """Evaluate the model on the provided dataloader"""
        n_total = 0
        n_correct = 0
        dom_n_correct = 0
        self.network.set_eval()
        for _, (idxs, images, labels) in enumerate(loader):
            images, labels = images.cuda(), labels.cuda()
            with torch.no_grad():
                logits, dom_logits = self.network.forward(images)
            top, pred = utils.calc_accuracy(output=logits, target=labels, topk=(1,))
            n_correct += top[0].item() * pred.shape[0]
            n_total += pred.shape[0]
            
            dom_logits_sigmoid = torch.sigmoid(dom_logits)
            dom_preds = torch.where(dom_logits_sigmoid > 0.5, 1, 0)
            n_ones = torch.sum(dom_preds)
            n_zeros = pred.shape[0] - n_ones
            if source:
                dom_n_correct += n_zeros.cpu().numpy()
            else:
                dom_n_correct += n_ones.cpu().numpy()
                
        acc = n_correct / n_total
        dom_acc = 100. * dom_n_correct / n_total
        return round(acc, 2), round(dom_acc, 2)

    