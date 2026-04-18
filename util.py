from torch.autograd import grad


def compute_grad(inputs, output, create_graph=True, retain_graph=True):
    """Compute gradient of the scalar output with respect to inputs.

    Args:
        inputs (torch.Tensor): torch tensor, requires_grad=True
        output (torch.Tensor): scalar output

    Returns:
        torch.Tensor: gradients with respect to each input component
    """

    assert inputs.requires_grad

    gradspred, = grad(output, inputs, grad_outputs=output.data.new(output.shape).fill_(1),
                      create_graph=create_graph, retain_graph=retain_graph)

    return gradspred


def batch_to(batch, device):
    gpu_batch = dict()
    for key, val in batch.items():
        gpu_batch[key] = val.to(device) if hasattr(val, 'to') else val
    return gpu_batch


import torch
import torch.nn as nn

def gaussian_smearing(distances, offset, widths, centered=False):

    if not centered:
        # Compute width of Gaussians (using an overlap of 1 STDDEV)
        # widths = offset[1] - offset[0]
        coeff = -0.5 / torch.pow(widths, 2)
        diff = distances - offset

    else:
        # If Gaussians are centered, use offsets to compute widths
        coeff = -0.5 / torch.pow(offset, 2)
        # If centered Gaussians are requested, don't substract anything
        diff = distances

    # Compute and return Gaussians
    gauss = torch.exp(coeff * torch.pow(diff, 2))

    return gauss


class GaussianSmearing(nn.Module):
    """
    Wrapper class of gaussian_smearing function. Places a predefined number of Gaussian functions within the
    specified limits.

    sample struct dictionary:

        struct = {'start': 0.0, 'stop':5.0, 'n_gaussians': 32, 'centered': False, 'trainable': False}

    Args:
        start (float): Center of first Gaussian.
        stop (float): Center of last Gaussian.
        n_gaussians (int): Total number of Gaussian functions.
        centered (bool):  if this flag is chosen, Gaussians are centered at the origin and the
              offsets are used to provide their widths (used e.g. for angular functions).
              Default is False.
        trainable (bool): If set to True, widths and positions of Gaussians are adjusted during training. Default
              is False.
    """

    def __init__(self, start, stop, n_gaussians, width=None, centered=False, trainable=False):
        super().__init__()
        offset = torch.linspace(start, stop, n_gaussians)
        if width is None:
            widths = torch.FloatTensor((offset[1] - offset[0]) * torch.ones_like(offset))
        else:
            widths = torch.FloatTensor(width * torch.ones_like(offset))
        if trainable:
            self.width = nn.Parameter(widths)
            self.offsets = nn.Parameter(offset)
        else:
            self.register_buffer('width', widths)
            self.register_buffer('offsets', offset)
        self.centered = centered

    def forward(self, distances):
        """
        Args:
            distances (torch.Tensor): Tensor of interatomic distances.

        Returns:
            torch.Tensor: Tensor of convolved distances.

        """
        result = gaussian_smearing(distances,
                                   self.offsets,
                                   self.width,
                                   centered=self.centered)

        return result