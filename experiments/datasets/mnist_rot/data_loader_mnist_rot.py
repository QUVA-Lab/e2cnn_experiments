import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms

from . import own_transforms


class mnist_rot_dataset(data.Dataset):
    """ rotated MNIST dataset """
    
    def __init__(self, mode, transform=None, target_transform=None, reshuffle_seed=None):
        """
        :type  mode: string from ['train', 'valid', 'test']
        :param mode: determines which subset of the dataset is loaded and whether augmentation is used
        :type  transform: callable
        :param transform: transformation applied to PIL images, returning transformed version
        :type  target_transform: callable
        :param target_transform: transformation applied to labels
        :type  reshuffle_seed: int
        :param reshuffle_seed: seed to use to reshuffle train or valid sets. If None (default), they are not reshuffled
        """
        assert mode in ['train', 'valid', 'trainval', 'test']
        assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')
        
        self.mode = mode
        self.transform = transform
        self.target_transform = target_transform
        
        # load the numpy arrays
        if mode in ["train", "valid", "trainval"]:
            filename = './datasets/mnist_rot/mnist_rot_trainval.npz'
            
            data = np.load(filename)

            num_train = len(data["labels"])
            indices = np.arange(0, num_train)

            if reshuffle_seed is not None:
                rng = np.random.RandomState(reshuffle_seed)
                rng.shuffle(indices)

            split = int(np.floor(num_train * 5/6))
            
            if mode == "train":
                data = {
                    "images": data["images"][indices[:split], :],
                    "labels": data["labels"][indices[:split]]
                }
            elif mode == "valid":
                data = {
                    "images": data["images"][indices[split:], :],
                    "labels": data["labels"][indices[split:]]
                }
            
        else:
            filename = './datasets/mnist_rot/mnist_rot_test.npz'
            data = np.load(filename)

        self.images = data['images'].astype(np.float32)
        self.labels = data['labels'].astype(np.int64)
        self.num_samples = len(self.labels)
    
    def __getitem__(self, index):
        """
        :type  index: int
        :param index: index of data
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        image, label = self.images[index], self.labels[index]
        # convert to PIL Image
        image = Image.fromarray(image)
        # transform images and labels
        if self.transform is not None:
            self.transform.update_randomization()
            image = self.transform(image)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return image, label
    
    def __len__(self):
        return len(self.labels)


def build_mnist_rot_loader(mode, batch_size, num_workers=8, rot_interpol_augmentation=False, interpolation=0, reshuffle_seed=None, coords=False):
    """  """
    rng = np.random.RandomState(42)

    assert mode in ['train', 'valid', 'trainval', 'test']
    assert reshuffle_seed is None or (mode != "test" and mode != 'trainval')
    
    transform = []
    if mode in ['valid', 'test']:
        
        shuffle = False
        drop_last = False
        if rot_interpol_augmentation:
            transform = [
                own_transforms.Rotate(rng=None, interpolation=interpolation),  # only resamples image
                # own_transforms.ShiftScale(rng),
                own_transforms.GrayToTensor()
            ]
        else:
            transform = [own_transforms.GrayToTensor()]
    elif mode in ['train', 'trainval']:
        shuffle = True
        drop_last = True
        if rot_interpol_augmentation:
            transform = [
                own_transforms.Rotate(rng=rng, interpolation=interpolation),
                # own_transforms.Rotate90(rng=rng),
                # own_transforms.ShiftScale(rng),
                own_transforms.GrayToTensor()
            ]
            
        else:
            transform = [
                # own_transforms.Rotate90(rng=rng),
                own_transforms.GrayToTensor()
            ]
    else:
        raise ValueError('unknown mode for building mnist_rot loader')
    
    if coords:
        transform += [own_transforms.CoordinateField((28, 28))]
    
    transform = own_transforms.Compose(transform)
    
    dataset = mnist_rot_dataset(mode, transform=transform, reshuffle_seed=reshuffle_seed)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        # sampler=torch.utils.data.sampler.RandomSampler(dataset),
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=True
    )
    n_inputs = 1
    n_outputs = 10
    
    if coords:
        n_inputs += 2

    return loader, n_inputs, n_outputs

