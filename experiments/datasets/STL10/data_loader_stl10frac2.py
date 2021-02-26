
import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from .data_loader_stl10 import DATA_DIR, CIFAR_MEAN, CIFAR_STD, MEAN, STD
from .data_loader_stl10 import Cutout


def __balanced_subdataset_idxs(train_size, validation_size, labels, reshuffle):
        num_train = len(labels)
        assert train_size + validation_size <= num_train

        classes = set(labels)
        
        labels_idxs = {c: list() for c in classes}
        ratios = {c: 0. for c in classes}
        
        for i, l in enumerate(labels):
            labels_idxs[l].append(i)
            ratios[l] += 1.
        
        train_idx = list()
        valid_idx = list()
        for c in classes:
            ratios[c] /= num_train
            
            if reshuffle:
                np.random.shuffle(labels_idxs[c])
            
            ts = int(round(train_size * ratios[c]))
            vs = int(round(validation_size * ratios[c]))

            valid_idx += labels_idxs[c][:vs]
            train_idx += labels_idxs[c][vs:vs+ts]
        
        return train_idx, valid_idx
        
        
def __build_stl10_frac_loaders(size,
                               batch_size,
                               eval_batchsize,
                               validation=True,
                               num_workers=8,
                               augment=False,
                               reshuffle=True,
                               mean=MEAN,
                               std=STD,
                               ):
    
    normalize = transforms.Normalize(
        mean=mean,
        std=std,
    )
    
    # define transforms
    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(96, padding=12),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            # Cutout(32),
            Cutout(60),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            # Cutout(24),
            Cutout(48),
            normalize,
        ])
        # train_transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     normalize,
        # ])
    
    # load the dataset
    train_dataset = datasets.STL10(
        root=DATA_DIR, split="train",
        download=True, transform=train_transform,
    )
    
    test_dataset = datasets.STL10(
        root=DATA_DIR, split="test",
        download=True, transform=valid_transform,
    )
    
    if validation:
        
        valid_dataset = datasets.STL10(
            root=DATA_DIR, split="train",
            download=True, transform=valid_transform,
        )
        
        validation_size = 1000
        train_idx, valid_idx = __balanced_subdataset_idxs(size, validation_size, train_dataset.labels, reshuffle)
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=eval_batchsize, sampler=valid_sampler,
            num_workers=num_workers, pin_memory=True,
        )
    else:
    
        train_idx, _ = __balanced_subdataset_idxs(size, 0, train_dataset.labels, reshuffle)
    
        train_sampler = SubsetRandomSampler(train_idx)
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, sampler=train_sampler,
            num_workers=num_workers, pin_memory=True,
        )
        valid_loader = None
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=eval_batchsize, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    
    n_inputs = 3
    n_classes = 10
    
    return train_loader, valid_loader, test_loader, n_inputs, n_classes


def build_stl10_frac_loaders(size,
                             batch_size,
                             eval_batchsize,
                             validation=True,
                             num_workers=8,
                             augment=False,
                             reshuffle=True,
                             ):
    return __build_stl10_frac_loaders(size, batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                      mean=MEAN, std=STD)


def build_stl10cif_frac_loaders(size,
                                batch_size,
                                eval_batchsize,
                                validation=True,
                                num_workers=8,
                                augment=False,
                                reshuffle=True,
                                ):
    return __build_stl10_frac_loaders(size, batch_size, eval_batchsize, validation, num_workers, augment, reshuffle,
                                 mean=CIFAR_MEAN, std=CIFAR_STD)

