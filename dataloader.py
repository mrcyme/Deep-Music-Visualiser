import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import numpy as np


def image_loader(path, size=None, normalize=False):
    """convert an image to torch tensor  of desired size

    Parameters
    ----------
    path : string
        Path to image
    size : tuple
        w x h desired size
    normalize : type
        whether to normalize to imagenet
    Returns
    -------
    type
        torch tensor of dimension c x w x h

    """
    loader = transforms.ToTensor()
    if size:
        loader = transforms.Compose([loader, transforms.Resize(size, interpolation=transforms.InterpolationMode.BILINEAR)])
    if normalize:
        loader = transforms.Compose([loader, transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    image = Image.open(path)
    return loader(image)


def image_unloader(tensor, normalize=False):
    unloader = transforms.ToPILImage()
    if normalize:
        unloader = transforms.Compose([transforms.Normalize(mean=[0., 0., 0.],
                                                            std=[1/0.229, 1/0.224, 1/0.225]),
                                       transforms.Normalize(mean=[-0.485, -0.456, -0.406],
                                                            std=[1., 1., 1.]), unloader])
    return unloader(tensor)


def render_noise_image(size, normalize=False):
    img = Image.fromarray(np.uint8(np.random.uniform(150, 180, (size[0],size[1],3))))
    if normalize:
        return transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)
    return transforms.ToTensor()(img)


def plot_image_from_tensor(tensor, title=None, figsize=None, normalize=False):
    image = image_unloader(tensor, normalize=normalize)
    plt.figure(figsize=figsize)
    plt.title(title)
    plt.axis('off')
    plt.imshow(image)
    plt.show()
