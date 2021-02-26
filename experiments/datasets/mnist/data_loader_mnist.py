import numpy as np
import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler


def build_mnist_loader(mode, batch_size, num_workers=8, augment=False, reshuffle_seed=None):
    """  """

    assert mode in ['train', 'valid', 'trainval', 'test']
    assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')
    
    rot_trans = transforms.Compose([transforms.RandomRotation(degrees=180, resample=False),
                                    transforms.ToTensor(),
                                    transforms.Normalize((0.5,), (1.0,))])

    trans = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5,), (1.0,))])
    
    if mode == "test":
        # if doesn't exist, download mnist dataset
        test_set = dset.MNIST(root='./datasets/mnist/', train=False, transform=trans, download=True)

        loader = torch.utils.data.DataLoader(
            test_set,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )
    else:
        
        # if doesn't exist, download mnist dataset
        train_set = dset.MNIST(root='./datasets/mnist/', train=True, transform=trans, download=True)

        if mode in ["valid", "train"]:
        
            valid_set = dset.MNIST(root='./datasets/mnist/', train=True, transform=trans, download=True)
            num_train = len(train_set)
            indices = list(range(num_train))
            split = int(np.floor(num_train * 5/6))

            if reshuffle_seed is not None:
                rng = np.random.RandomState(reshuffle_seed)
                rng.shuffle(indices)
        
            train_idx, valid_idx = indices[:split], indices[split:]
            train_sampler = SubsetRandomSampler(train_idx)
            valid_sampler = SubsetRandomSampler(valid_idx)
            if mode == "train":
                loader = torch.utils.data.DataLoader(
                    train_set, batch_size=batch_size, sampler=train_sampler,
                    num_workers=num_workers, pin_memory=True
                )
            else:
                loader = torch.utils.data.DataLoader(
                    valid_set, batch_size=batch_size, sampler=valid_sampler,
                    num_workers=num_workers, pin_memory=True
                )
        else:
            # mode == "trainval"
            loader = torch.utils.data.DataLoader(
                train_set, batch_size=batch_size,
                num_workers=num_workers, pin_memory=True
            )

    n_inputs = 1
    n_outputs = 10
    
    return loader, n_inputs, n_outputs

