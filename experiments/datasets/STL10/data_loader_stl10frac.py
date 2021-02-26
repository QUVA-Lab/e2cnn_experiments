import torch
import numpy as np

from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

from .data_loader_stl10 import DATA_DIR, CIFAR_MEAN, CIFAR_STD, MEAN, STD
from .data_loader_stl10 import Cutout

from torch.utils.data import Dataset, ConcatDataset


class TransformedSubsetDataset(Dataset):
    def __init__(self, dataset: Dataset, transform, indeces):
        assert max(indeces) < len(dataset)
        assert min(indeces) >= 0
        
        self.dataset = dataset
        self.transform = transform
        self.indices = list(indeces)
    
    def __getitem__(self, index):
        x, t = self.dataset[self.indices[index]]
        return self.transform(x), t

    def __len__(self):
        return len(self.indices)


def __balanced_subdataset_idxs(train_size, validation_size, labels, reshuffle):
        num_train = len(labels)
        
        test_size = num_train - train_size - validation_size
        assert test_size >= 0
        
        classes = set(labels)
        
        labels_idxs = {c: list() for c in classes}
        ratios = {c: 0. for c in classes}
        
        for i, l in enumerate(labels):
            labels_idxs[l].append(i)
            ratios[l] += 1.
        
        train_idx = list()
        valid_idx = list()
        test_idx = list()
        for c in classes:
            ratios[c] /= num_train
            
            if reshuffle:
                np.random.shuffle(labels_idxs[c])
            
            ts = int(round(train_size * ratios[c]))
            vs = int(round(validation_size * ratios[c]))

            valid_idx += labels_idxs[c][:vs]
            train_idx += labels_idxs[c][vs:vs+ts]
            test_idx += labels_idxs[c][vs+ts:]
        
        return train_idx, valid_idx, test_idx
        
        
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
    train = datasets.STL10(
        root=DATA_DIR, split="train",
        download=True, transform=None,
    )
    test = datasets.STL10(
        root=DATA_DIR, split="test",
        download=True, transform=None,
    )
    total_dataset = ConcatDataset([train, test])
    labels = np.concatenate([train.labels, test.labels])
    
    if validation:
        validation_size = 1000
    else:
        validation_size = 0
    
    train_idx, valid_idx, test_idx = __balanced_subdataset_idxs(size, validation_size, labels, reshuffle)
    
    train_dataset = TransformedSubsetDataset(total_dataset, train_transform, train_idx)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    
    if validation:
        valid_dataset = TransformedSubsetDataset(total_dataset, valid_transform, valid_idx)
        valid_loader = torch.utils.data.DataLoader(
            valid_dataset, batch_size=eval_batchsize, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    else:
        valid_loader = None

    test_dataset = TransformedSubsetDataset(total_dataset, valid_transform, test_idx)
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

