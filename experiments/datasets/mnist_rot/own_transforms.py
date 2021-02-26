import numpy as np
from PIL import Image
import torch
import torch.utils.data as data
from torchvision import transforms


class Compose(transforms.Compose):
    """ Composes several transforms together.
        Adapted from torchvision.transforms to update randomization
    """
    
    def update_randomization(self):
        """ iterate over composed transforms and apply update when method update_randomization is implemented """
        for t in self.transforms:
            update_fct = getattr(t, 'update_randomization', None)
            if update_fct is not None and callable(update_fct):
                update_fct()


class Rotate(object):
    """ return image rotated by a random angle or zero degrees
        an angle of zero still gives interpolation effects, should be applied to test set when train set is rotated by random angle
    """
    
    def __init__(self, rng=None, interpolation=0):
        self.rng = rng
        assert interpolation in [0, 2, 3]  # NEAREST, BILINEAR, BICUBIC
        self.interpolation = interpolation
        self.update_randomization()
    
    def update_randomization(self):
        if self.rng:
            self.angle = self.rng.uniform(360)
        else:
            self.angle = 0
    
    def __call__(self, img):
        """
        :type  img: PIL.Image
        :param img: image to be transformed
        """
        return img.rotate(angle=self.angle, resample=self.interpolation)


class Rotate90(object):
    """ return image rotated by a random multiple of 90 degrees """
    
    def __init__(self, rng=None):
        self.rng = rng
        self.update_randomization()
    
    def update_randomization(self):
        self.multiple = self.rng.randint(4)
    
    def __call__(self, img):
        """
        :type  img: PIL.Image
        :param img: image to be rotated
        """
        img = np.rot90(img, self.multiple)
        return Image.fromarray(img)


class ShiftScale(object):
    """ return image shifted and rescaled image """
    
    def __init__(self, rng, shiftmax=1, scalemax=.025):
        self.shiftmax = shiftmax  # default shifts by 0 to 1 pixel
        self.scalemax = scalemax  # default scales between .975 to 1.025 percent
        self.rng = rng
        self.update_randomization()
    
    def update_randomization(self):
        shiftX = self.rng.uniform(self.shiftmax)
        shiftY = self.rng.uniform(self.shiftmax)
        scaleX = self.rng.uniform(1 - self.scalemax, 1 + self.scalemax)
        scaleY = self.rng.uniform(1 - self.scalemax, 1 + self.scalemax)
    
    def __call__(self, img):
        """
        :type  img: PIL.Image
        :param img: image to be transformed
        """
        return img.transform(img.size, Image.AFFINE, data=(self.scaleX, 0, self.shiftX, 0, self.scaleY, self.shiftY),
                             resample=Image.BILINEAR)


class Reflect(object):
    """ reflect image """
    
    def __init__(self, rng):
        self.rng = rng
        self.update_randomization()
    
    def update_randomization(self):
        self.flipX, self.flipY = self.rng.randint(2, size=2)  # two random numbers, each from {0,1}
    
    def __call__(self, img):
        """
        :type  img: PIL.Image
        :param img: image to be transformed
        """
        if self.flipX:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if self.flipY:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        return img


class GrayToTensor(object):
    """ converts gray image to tensor and adds channel dimension """
    
    def __call__(self, img):
        """
        :type  tensor: torch.FloatTensor
        :param tensor: image tensor to which channel is added
        """
        img = np.array(img, np.float32, copy=False)[np.newaxis, ...]  # add channel dimension
        return torch.from_numpy(img)


class CoordinateField(object):
    """ Add the x and y coordinates of each pixel as two additional scalar features """
    
    def __init__(self, shape):
        coords = [torch.arange(s) for s in shape]
        coords = torch.stack(torch.meshgrid(coords))
        coords = coords.to(dtype=torch.float)
        
        l = len(shape)
        
        assert coords.shape == (l,) + shape, coords.shape
        
        coords = coords.reshape(l, -1)

        coords -= coords.mean(dim=1, keepdim=True)
        coords /= coords.std(dim=1, keepdim=True)

        coords = coords.reshape(l, *shape)

        self.coords = coords

        self._expand_shape = tuple(-1 for _ in range(len(shape)+1))
    
    def __call__(self, img):
        """
        :type  img: torch.FloatTensor
        :param img: image tensor to which channel is added
        """
        return torch.cat([img, self.coords], dim=0)

        
        
        
    

