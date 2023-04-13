"""Module implementing the MTL pretraining task for Toybox->IN-12"""
import tqdm

import torch
import torch.nn as nn
import torch.utils.tensorboard as tb

import utils


class ModelFT:
    """model for finetuning network"""
    def __init__(self, network, train_loader, test_loader):
        self.network = network
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.network.cuda()
        
    def train(self, optimizer, scheduler, ep, ep_total):
        """Finetune the network for 1 epoch"""
        self.network.set_linear_eval()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        tqdm_bar = tqdm.tqdm(ncols=150, total=len(self.train_loader))
        train_loader_iter = iter(self.train_loader)
        for step in range(1, len(self.train_loader) + 1):
            optimizer.zero_grad()
    
            idx, images, labels = next(train_loader_iter)
            images, labels = images.cuda(), labels.cuda()
            logits = self.network.forward(images)
            loss = src_criterion(logits, labels)
            loss.backward()
    
            optimizer.step()
            if scheduler is not None:
                scheduler.step()
    
            ce_loss_total += loss.item()
            num_batches += 1
            tqdm_bar.update(1)
            tqdm_bar.set_description("Ep: {}/{}  LR: {:.3f}  CE: {:.3f}".format(
                ep, ep_total, optimizer.param_groups[0]['lr'], ce_loss_total / num_batches))

        tqdm_bar.close()
        
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
    def __init__(self, network, source_loader, target_loader):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.target_loader = utils.ForeverDataLoader(target_loader)
        self.network.cuda()
        
    def train(self, optimizer, scheduler, steps, ep, ep_total, lmbda):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        ssl_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        trgt_criterion = nn.CrossEntropyLoss()
        tqdm_bar = tqdm.tqdm(ncols=150, total=steps)
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
            tqdm_bar.update(1)
            tqdm_bar.set_description("Ep: {}/{}  BLR: {:.3f}  CLR: {:.3f}  SLR: {:.3f}  CE: {:.3f}  SSL: {:.3f}".format(
                ep, ep_total, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                optimizer.param_groups[2]['lr'], ce_loss_total/num_batches, ssl_loss_total/num_batches))
            
        tqdm_bar.close()
        
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
    
    def __init__(self, network, loader):
        self.network = network
        self.loader = utils.ForeverDataLoader(loader)
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ssl_loss_total = 0.0
        criterion = nn.CrossEntropyLoss()
        tqdm_bar = tqdm.tqdm(ncols=150, total=steps)
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
            tqdm_bar.update(1)
            tqdm_bar.set_description("Ep: {}/{}  BLR: {:.3f}  SLR: {:.3f}  SSL: {:.3f}".format(
                ep, ep_total, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                ssl_loss_total / num_batches))
        
        tqdm_bar.close()


class SupModel:
    """Module implementing the supervised pretraining on source"""
    
    def __init__(self, network, source_loader):
        self.network = network
        self.source_loader = utils.ForeverDataLoader(source_loader)
        self.network.cuda()
    
    def train(self, optimizer, scheduler, steps, ep, ep_total, writer: tb.SummaryWriter):
        """Train model"""
        self.network.set_train()
        num_batches = 0
        ce_loss_total = 0.0
        src_criterion = nn.CrossEntropyLoss()
        tqdm_bar = tqdm.tqdm(ncols=150, total=steps)
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
            tqdm_bar.update(1)
            tqdm_bar.set_description("Ep: {}/{}  BLR: {:.3f}  CLR: {:.3f}  CE: {:.3f}".format(
                ep, ep_total, optimizer.param_groups[0]['lr'], optimizer.param_groups[1]['lr'],
                ce_loss_total / num_batches))
            
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
        
        tqdm_bar.close()
    
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

    