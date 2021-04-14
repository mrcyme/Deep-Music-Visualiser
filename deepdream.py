import numpy as np
import torch
from utils import torch_utils
from torch.autograd import Variable
import scipy.ndimage as nd


def objective(dst, guide_features):
    if guide_features is None:
        return dst.data
    else:
        x = dst.data[0].cpu().numpy()
        y = guide_features.data[0].cpu().numpy()
        ch, w, h = x.shape
        x = x.reshape(ch, -1)
        y = y.reshape(ch, -1)
        A = x.T.dot(y)
        diff = y[:, A.argmax(1)]
        diff = torch.Tensor(np.array([diff.reshape(ch, w, h)])).cuda()
        return diff


def make_step(model, input_image, control=None, step_size=0.1, end=28, jitter=None):
    if jitter:
        ox, oy = np.random.randint(-jitter, jitter+1, 2)
        input_image = np.roll(np.roll(input_image, ox, -1), oy, -2)
    tensor = torch.Tensor(input_image).unsqueeze(0)
    image_var = Variable(tensor.to(torch_utils.get_device()), requires_grad=True)
    model.zero_grad()
    x = image_var
    for index, layer in enumerate(model.features.children()):
        x = layer(x)
        if index == end:
            break

    delta = objective(x, control)
    x.backward(delta)

    # L2 Regularization on gradients
    mean_square = torch.Tensor([torch.mean(image_var.grad.data ** 2)]).to(torch_utils.get_device())
    image_var.grad.data /= torch.sqrt(mean_square)
    image_var.data.add_(image_var.grad.data * step_size)

    result = image_var.squeeze().data.cpu().numpy()
    if jitter:
        result = np.roll(np.roll(result, -ox, -1), -oy, -2)
    return torch.Tensor(result)


def deepdream(model, input_image, n_octave=6, octave_scale=1.4,
              n_iter=10, end=28, control=None,
              step_size=1.5, jitter=None):

    return octaver_fn(
              model, input_image, n_octave=n_octave, octave_scale=octave_scale,
              n_iter=n_iter, end=end, control=control,
              step_size=step_size, jitter=jitter
           )


def octaver_fn(model, content_img, n_octave=6, octave_scale=1.4, n_iter=10, **step_args):
    octaves = [content_img.numpy()]

    for i in range(n_octave - 1):
        octaves.append(nd.zoom(octaves[-1], (1, 1.0 / octave_scale, 1.0 / octave_scale), order=1))

    detail = np.zeros_like(octaves[-1])
    for octave, octave_base in enumerate(octaves[::-1]):
        h, w = octave_base.shape[-2:]

        if octave > 0:
            h1, w1 = detail.shape[-2:]
            detail = nd.zoom(detail, (1, 1.0 * h / h1, 1.0 * w / w1), order=1)

        src = octave_base + detail

        for i in range(n_iter):
            src = make_step(model, src, **step_args)

        detail = src.numpy() - octave_base
    return src
