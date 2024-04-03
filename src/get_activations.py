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


def get_activations(net, fc_size, data):
    """Use the data to get activation"""
    len_train_data = len(data)
    data_loader = torchdata.DataLoader(data, batch_size=512, shuffle=False, num_workers=4)

    activations = torch.zeros((len_train_data, fc_size), dtype=torch.float)
    indices = torch.zeros(len_train_data, dtype=torch.long)
    for (idx, act_idx), images, _ in data_loader:
        images = images.cuda()
        with torch.no_grad():
            feats = net.forward(images)
            feats = feats.cpu()
        indices[idx] = act_idx
        activations[idx] = feats
    del images, idx, act_idx, data_loader
    # all_activations = torch.cat([activations, mean_activations], dim=0)
    return indices.numpy(), activations.numpy()


def get_activations_sup(model_path, out_path):
    """Get the activations from a supervised model"""
    os.makedirs(out_path, exist_ok=True)

    load_file = torch.load(model_path)
    net = networks.ResNet18Backbone(weights=load_file['backbone'], pretrained=False)
    # net = networks.ResNet18Backbone(pretrained=True)
    fc_size = net.fc_size
    net.cuda()
    net.set_eval()

    transform_toybox = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(),
                                           transforms.Resize((224, 224)),
                                           transforms.Normalize(mean=datasets.TOYBOX_MEAN, std=datasets.TOYBOX_STD)
                                           ])

    toybox_train_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=True, transform=transform_toybox,
                                               hypertune=True, num_instances=-1, num_images_per_class=1500)
    print(len(toybox_train_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=toybox_train_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path + "toybox_train_activations.npy", arr=activations)
    np.save(file=out_path + "toybox_train_indices.npy", arr=indices)
    toybox_test_data = datasets.ToyboxDataset(rng=np.random.default_rng(0), train=False, transform=transform_toybox,
                                              hypertune=False)
    print(len(toybox_test_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=toybox_test_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path+"toybox_test_activations.npy", arr=activations)
    np.save(file=out_path + "toybox_test_indices.npy", arr=indices)

    transform_in12 = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Resize((224, 224)),
                                         transforms.Normalize(mean=datasets.IN12_MEAN, std=datasets.IN12_STD)])
    in12_train_data = datasets.DatasetIN12All(train=True, transform=transform_in12, hypertune=True)
    print(len(in12_train_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=in12_train_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path+"in12_train_activations.npy", arr=activations)
    np.save(file=out_path + "in12_train_indices.npy", arr=indices)

    in12_test_data = datasets.DatasetIN12All(train=False, transform=transform_in12, hypertune=False)
    print(len(in12_test_data))
    indices, activations = get_activations(net=net, fc_size=fc_size, data=in12_test_data)
    print(indices.shape, activations.shape)
    np.save(file=out_path + "in12_test_indices.npy", arr=indices)
    np.save(file=out_path+"in12_test_activations.npy", arr=activations)

    del indices, activations, net, load_file
    del toybox_train_data, toybox_test_data, in12_train_data, in12_test_data
    gc.collect()
    torch.cuda.empty_cache()
    

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
    