"""Main training loop for ProtoPNet model"""
import os

import tqdm
import numpy as np
import datetime

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms
import torch.nn as nn

import datasets
import proto_push
import proto_network
import utils

OUT_DIR = "../temp/"


class ProtoPNetTrainer:
    """Module for implementing the ProtoPNet Trainer"""
    
    def __init__(self, args):
        self.args = args
        
        self.mean, self.std = datasets.TOYBOX_MEAN, datasets.TOYBOX_STD
        self.num_classes = 12
        
        self.network = proto_network.ResNet18Proto(num_classes=self.num_classes)
        
        self.train_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                                   transforms.Resize((224, 224)),
                                                   transforms.RandomRotation(15),
                                                   # transforms.RandomAffine(degrees=0, shear=10),
                                                   # transforms.RandomPerspective(),
                                                   transforms.RandomHorizontalFlip(),
                                                   transforms.ToTensor(),
                                                   transforms.Normalize(mean=self.mean, std=self.std)
                                                   ])
        
        self.test_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=self.mean, std=self.std)
                                                  ])
        
        self.push_transform = transforms.Compose([transforms.ToPILImage(mode='RGB'),
                                                  transforms.Resize((224, 224)),
                                                  transforms.ToTensor(),
                                                  ])
        
        self.normalize_transform = transforms.Normalize(mean=self.mean, std=self.std)
        self.network.cuda()
        
        self.train_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True,
                                                 transform=self.train_transform,
                                                 hypertune=True,
                                                 num_instances=-1,
                                                 num_images_per_class=500)
        self.train_loader = torchdata.DataLoader(self.train_data, shuffle=True, batch_size=128, num_workers=4)
        self.test_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=False,
                                                transform=self.train_transform,
                                                hypertune=True)
        self.test_loader = torchdata.DataLoader(self.test_data, shuffle=False, batch_size=128, num_workers=4)
        
        self.push_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True,
                                                transform=self.push_transform,
                                                hypertune=True,
                                                num_instances=-1,
                                                num_images_per_class=2000)
        self.push_loader = torchdata.DataLoader(self.push_data, shuffle=False, batch_size=128, num_workers=4)
        
        self.exp_time = datetime.datetime.now()
        self.out_path = OUT_DIR + self.args['dataset'].upper() + "/exp_" \
                                + self.exp_time.strftime("%b-%d-%Y-%H-%M") + "/"
        os.makedirs(self.out_path, exist_ok=False)
    
    @torch.enable_grad()
    def train(self, epoch, optimizer, total_epochs):
        """Run training depending on phase"""
        coefs = {
            'crs_ent': 1,
            'clst': 0.8,
            'sep': -0.08,
            'l1': 1e-4,
        }
        self.network.train()
        
        total_cross_entropy = 0
        total_cluster_cost = 0
        # separation cost is meaningful only for class_specific
        total_separation_cost = 0
        total_avg_separation_cost = 0
        n_examples = 0
        n_correct = 0
        n_batches = 0
        
        tqdm_bar = tqdm.tqdm(total=len(self.train_loader), ncols=150)
        last_tqdm_update = 0
        for batch_idx, (idxs, images, labels) in enumerate(self.train_loader):
            images, labels = images.cuda(), labels.cuda()
            
            output, min_distances = self.network.forward(images)
            
            # compute loss
            cross_entropy = nn.functional.cross_entropy(output, labels)
            
            max_dist = (self.network.prototype_shape[1]
                        * self.network.prototype_shape[2]
                        * self.network.prototype_shape[3])
            
            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            # calculate cluster cost
            prototypes_of_correct_class = torch.t(self.network.prototype_class_identity[:, labels]).cuda()
            inverted_distances, _ = torch.max((max_dist - min_distances) * prototypes_of_correct_class, dim=1)
            cluster_cost = torch.mean(max_dist - inverted_distances)
            
            # calculate separation cost
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class
            inverted_distances_to_nontarget_prototypes, _ = \
                torch.max((max_dist - min_distances) * prototypes_of_wrong_class, dim=1)
            separation_cost = torch.mean(max_dist - inverted_distances_to_nontarget_prototypes)
            
            # calculate avg cluster cost
            avg_separation_cost = \
                torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class,
                                                                                        dim=1)
            avg_separation_cost = torch.mean(avg_separation_cost)
            
            l1_mask = 1 - torch.t(self.network.prototype_class_identity).cuda()
            l1 = (self.network.classifier.weight * l1_mask).norm(p=1)
            
            _, predicted = torch.max(output.data, 1)
            n_examples += labels.size(0)
            n_correct += (predicted == labels).sum().item()
            
            n_batches += 1
            total_cross_entropy += cross_entropy.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            if n_batches % 20 == 0:
                tqdm_bar.set_description("Ep: {:d}/{:d}  CE: {:.3f}  Cluster: {:.3f}  Sep: {:.3f} LR: {:.6f}  "
                                         "Acc: {:.2f}"
                                         .format(epoch, total_epochs, total_cross_entropy / n_batches,
                                                 total_cluster_cost / n_batches,
                                                 total_separation_cost / n_batches, optimizer.param_groups[0]['lr'],
                                                 100. * n_correct / n_examples))
                tqdm_bar.update(n_batches - last_tqdm_update)
                last_tqdm_update = n_batches
            
            loss = coefs['crs_ent'] * cross_entropy \
                   + coefs['clst'] * cluster_cost \
                   + coefs['sep'] * separation_cost \
                   + coefs['l1'] * l1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        tqdm_bar.update(n_batches - last_tqdm_update)
        tqdm_bar.set_description("Ep: {:d}/{:d}  CE: {:.3f}  Cluster: {:.3f}  Sep: {:.3f} LR: {:.6f}  Acc: {:.2f}"
                                 .format(epoch, total_epochs, total_cross_entropy / n_batches,
                                         total_cluster_cost / n_batches,
                                         total_separation_cost / n_batches, optimizer.param_groups[0]['lr'],
                                         100. * n_correct / n_examples))
        tqdm_bar.close()
    
    def run_training(self):
        """Outer training loop"""
        warmup_epochs_num = self.args['warmup_epochs']
        total_epochs = self.args['epochs']
        last_layer_training_epochs = self.args['final_layer_epochs']
        push_steps = self.args['push_steps']
        eval_steps = self.args['eval_steps']

        joint_optimizer_lrs = {'features': 2e-4,
                               'add_on_layers': 6e-3,
                               'prototype_vectors': 6e-3}
        joint_lr_step_size = 5

        warm_optimizer_lrs = {'add_on_layers': 3e-3,
                              'prototype_vectors': 3e-3}

        last_layer_optimizer_lr = 2e-4
        
        joint_optimizer_specs = \
            [{'params': self.network.backbone.parameters(), 'lr': joint_optimizer_lrs['features'],
              'weight_decay': 1e-3},
             # bias are now also being regularized
             {'params': self.network.bottleneck.parameters(), 'lr': joint_optimizer_lrs['add_on_layers'],
              'weight_decay': 1e-3},
             {'params': self.network.prototypes, 'lr': joint_optimizer_lrs['prototype_vectors']},
             ]
        joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
        joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, step_size=joint_lr_step_size, gamma=0.1)
        
        warm_optimizer_specs = \
            [{'params': self.network.bottleneck.parameters(), 'lr': warm_optimizer_lrs['add_on_layers'],
              'weight_decay': 1e-3},
             {'params': self.network.prototypes, 'lr': warm_optimizer_lrs['prototype_vectors']},
             ]
        warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
        
        last_layer_optimizer_specs = [{'params': self.network.classifier.parameters(), 'lr': last_layer_optimizer_lr}]
        last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
        
        for epoch in range(1, total_epochs + 1):
            if epoch <= warmup_epochs_num:
                self.network.set_network_warmup()
                self.train(epoch=epoch, optimizer=warm_optimizer, total_epochs=total_epochs)
            else:
                self.network.set_network_joint()
                self.train(epoch=epoch, optimizer=joint_optimizer, total_epochs=total_epochs)
                joint_lr_scheduler.step()
            
            if epoch % push_steps == 0:
                proto_push.push_prototypes(dataloader=self.push_loader, network=self.network,
                                           transform=self.normalize_transform, out_dir=None, epochs=epoch)
                
                for ll_epoch in range(1, last_layer_training_epochs + 1):
                    self.network.set_network_last()
                    self.train(epoch=ll_epoch, optimizer=last_layer_optimizer, total_epochs=last_layer_training_epochs)
            
            if epoch % eval_steps == 0:
                train_acc, test_acc = self.eval()
                print("Train acc: {:.2f}    Test acc: {:.2f}".format(train_acc, test_acc))
        
        last_layer_optimizer.param_groups[0]['lr'] /= 2.0
        final_training_epochs = 10
        for ll_epoch in range(1, final_training_epochs + 1):
            self.network.set_network_last()
            self.train(epoch=ll_epoch, optimizer=last_layer_optimizer, total_epochs=final_training_epochs)
        
        train_acc, test_acc = self.eval()
        print("Train acc: {:.2f}    Test acc: {:.2f}".format(train_acc, test_acc))
        save_dict = {
            'network': self.network.state_dict(),
            'warmup_optimizer': warm_optimizer.state_dict(),
            'joint_optimizer': joint_optimizer.state_dict(),
            'll_optimizer': last_layer_optimizer.state_dict(),
            'joint_scheduler': joint_lr_scheduler.state_dict()
        }
        torch.save(save_dict, self.out_path + "final_training_state.pt")
    
    def eval(self):
        """Evaluate the network"""
        self.network.set_network_eval()
        self.network.eval()
        n_total_train = 0.0
        n_correct_train = 0.0
        for _, (idxs, images, labels) in enumerate(self.train_loader):
            images = images.cuda()
            with torch.no_grad():
                logits, _ = self.network.forward(images)
            _, predictions = torch.max(logits, 1)
            predictions = predictions.cpu()
            n_total_train += labels.size(0)
            n_correct_train += (predictions == labels).sum().item()
        
        n_total_test = 0.0
        n_correct_test = 0.0
        for _, (idxs, images, labels) in enumerate(self.test_loader):
            images = images.cuda()
            with torch.no_grad():
                logits, _ = self.network.forward(images)
            _, predictions = torch.max(logits, 1)
            predictions = predictions.cpu()
            n_total_test += labels.size(0)
            n_correct_test += (predictions == labels).sum().item()
        
        return 100. * (n_correct_train / n_total_train), 100. * (n_correct_test / n_total_test)


def main():
    """Main method"""
    exp_args = {
        'warmup_epochs': 3,
        'epochs': 10,
        'final_layer_epochs': 5,
        'push_steps': 3,
        'eval_steps': 10,
        'dataset': 'toybox',
    }
    trainer = ProtoPNetTrainer(args=exp_args)
    trainer.run_training()
    
    train_acc, test_acc = trainer.eval()
    print("Train acc: {:.2f}    Test acc: {:.2f}".format(train_acc, test_acc))


if __name__ == "__main__":
    main()
