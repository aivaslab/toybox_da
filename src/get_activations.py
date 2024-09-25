"""Use provided model to get the activations from provided dataset"""
import numpy as np
import os
import argparse
import gc

import torch
import torch.utils.data as torchdata
import torchvision.transforms as transforms

import datasets
import networks
import networks_da


def get_activations(net, fc_size, data, jan=False, btlnk=False, ssl=False):
    """Use the data to get activation"""
    len_train_data = len(data)
    data_loader = torchdata.DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

    activations = torch.zeros((len_train_data, fc_size), dtype=torch.float)
    indices = torch.zeros(len_train_data, dtype=torch.long)
    for (idx, act_idx), images, _ in data_loader:
        images = images.cuda()
        with torch.no_grad():
            if jan:
                if btlnk:
                    _, feats, _ = net.forward(images)
                else:
                    feats, _, _ = net.forward(images)
            elif ssl:
                feats = net.forward(images)
            else:
                feats = net.forward(images)
            feats = feats.cpu()
        indices[idx] = act_idx
        activations[idx] = feats
    del images, idx, act_idx, data_loader
    # all_activations = torch.cat([activations, mean_activations], dim=0)
    return indices.numpy(), activations.numpy()


def get_datasets():
    """Create and return the datasets as a dictionary"""
    dsets = {}

    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Resize((224, 224)),
                                           transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
                                           ])

    dsets['toybox_train'] = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True, transform=transform_toybox,
                                                   hypertune=True, num_instances=-1, num_images_per_class=1500)

    dsets['toybox_test'] = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=False, transform=transform_toybox,
                                                  hypertune=False)

    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    dsets['in12_train'] = datasets.DatasetIN12All(train=True, transform=transform_in12, hypertune=True)
    dsets['in12_test'] = datasets.DatasetIN12All(train=False, transform=transform_in12, hypertune=False)

    return dsets


def get_activations_sup(model_path, out_path, jan=False, btlnk=False):
    """Get the activations from a supervised model"""
    assert os.path.isfile(model_path)
    out_path += "backbone/activations/" if not jan or not btlnk else "bottleneck/activations/"
    os.makedirs(out_path, exist_ok=True)
    print(f"--------------------------------------------------------------------------------\nsrc: {model_path}")

    load_file = torch.load(model_path)
    if jan:
        net = networks_da.ResNet18JAN(backbone_weights=load_file['backbone'],
                                      bottleneck_weights=load_file['bottleneck'])
        fc_size = net.backbone_fc_size if not btlnk else net.bottleneck_dim
    else:
        net = networks.ResNet18Backbone(weights=load_file['backbone'], pretrained=False)
        fc_size = net.fc_size
    # net = networks.ResNet18Backbone(pretrained=True)

    net.cuda()
    net.set_eval()

    all_datasets = get_datasets()
    for dset_name, dset in all_datasets.items():
        indices, activations = get_activations(net=net, fc_size=fc_size, data=dset, jan=jan, btlnk=btlnk)
        print(f"dset: {dset}  n: {len(dset)}  indices: {indices.shape}  activations: {activations.shape}")
        np.save(file=out_path + f"{dset_name}_activations.npy", arr=activations)
        np.save(file=out_path + f"{dset_name}_indices.npy", arr=indices)

    del indices, activations, net, load_file, all_datasets
    gc.collect()
    torch.cuda.empty_cache()
    print(f"dest: {out_path}\n--------------------------------------------------------------------------------")


def get_activations_ssl(model_path, out_path):
    """Get the activations from a supervised model"""
    assert os.path.isfile(model_path)
    out_path += "ssl/activations/"
    os.makedirs(out_path, exist_ok=True)
    print(f"--------------------------------------------------------------------------------\nsrc: {model_path}")

    load_file = torch.load(model_path)
    net = networks.ResNet18SSL(backbone_weights=load_file['backbone'], ssl_weights=load_file['ssl_head'])
    fc_size = 128
    # net = networks.ResNet18Backbone(pretrained=True)

    net.cuda()
    net.set_eval()

    all_datasets = get_datasets()
    for dset_name, dset in all_datasets.items():
        indices, activations = get_activations(net=net, fc_size=fc_size, data=dset, ssl=True)
        print(f"dset: {dset}  n: {len(dset)}  indices: {indices.shape}  activations: {activations.shape}")
        np.save(file=out_path + f"{dset_name}_activations.npy", arr=activations)
        np.save(file=out_path + f"{dset_name}_indices.npy", arr=indices)

    del indices, activations, net, load_file, all_datasets
    gc.collect()
    torch.cuda.empty_cache()
    print(f"dest: {out_path}\n--------------------------------------------------------------------------------")


def get_activations_office31(out_path):
    """Get the activations from resnet-18 pretrained model for office-31 dataset"""
    out_path_office31 = out_path + "office-31/"
    os.makedirs(out_path_office31, exist_ok=True)

    net = networks.ResNet18Backbone(pretrained=True)
    fc_size = net.fc_size
    net.cuda()
    net.set_eval()

    # Get activations for Office-31 Amazon
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=datasets.OFFICE31_AMAZON_MEAN,
                                                         std=datasets.OFFICE31_AMAZON_STD)
                                    ])

    train_data = datasets.DatasetOffice31(domain='amazon', transform=transform)
    print(len(train_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=train_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path_office31 + "amazon_activations.npy", arr=activations)
    np.save(file=out_path_office31 + "amazon_indices.npy", arr=indices)

    # Get activations for Office-31 DSLR
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=datasets.OFFICE31_DSLR_MEAN,
                                                         std=datasets.OFFICE31_DSLR_STD)
                                    ])

    train_data = datasets.DatasetOffice31(domain='dslr', transform=transform)
    print(len(train_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=train_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path_office31 + "dslr_activations.npy", arr=activations)
    np.save(file=out_path_office31 + "dslr_indices.npy", arr=indices)

    # Get activations for Office-31 Webcam
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                    transforms.Resize((224, 224)),
                                    transforms.Normalize(mean=datasets.OFFICE31_WEBCAM_MEAN,
                                                         std=datasets.OFFICE31_WEBCAM_STD)
                                    ])

    train_data = datasets.DatasetOffice31(domain='webcam', transform=transform)
    print(len(train_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=train_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path_office31 + "webcam_activations.npy", arr=activations)
    np.save(file=out_path_office31 + "webcam_indices.npy", arr=indices)
    

def get_args():
    """Parser with arguments"""
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("--dir-path", "-d", type=str, required=True, help="Path where model is stored.")
    return vars(parser.parse_args())


if __name__ == "__main__":
    args = get_args()
    dir_name = args['dir_path']
    assert os.path.isdir(dir_name)
    get_activations_sup(model_path=dir_name+"final_model.pt", out_path=dir_name+"activations/")
    