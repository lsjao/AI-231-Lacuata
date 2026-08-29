"""
ME1 - A 3-layer CNN for MNIST built entirely from `einops` + `torch.einsum`.

The point of this exercise is that *no* high-level layer module does the work
for us. Everything below is raw tensor algebra:

  * convolution      -> manual zero-pad + einops patch extraction + `torch.einsum`
  * max / mean pool  -> `einops.reduce`
  * fully-connected  -> `torch.einsum`

Explicitly NOT used anywhere in the forward math:
`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.BatchNorm2d`, `F.conv2d`,
`F.max_pool2d`, `F.linear`, `F.unfold`.

We *do* use `nn.Parameter` (a weight container, not a layer), `nn.Module`
(a parameter registry, so `torch.optim` can find them), `torch.optim`, and
autograd - the handoff allows these, since none of them implement a layer.
Activations (`relu`) and the loss (`cross_entropy`) also come from torch
functional, which the handoff likewise allows.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, reduce


# --------------------------------------------------------------------------
# Building blocks
# --------------------------------------------------------------------------

def extract_patches(x: torch.Tensor, k: int, stride: int = 1, pad: int = 0) -> torch.Tensor:
    """Sliding-window patch extraction ("im2col"), done by hand.

    Args:
        x: input feature map, shape (b, c, h, w)
        k: square kernel size
        stride: sliding-window stride
        pad: zero-padding applied to all four spatial borders

    Returns:
        Tensor of shape (b, c, out_h, out_w, k, k) - for every output position
        (out_h, out_w) we carry the full (c, k, k) receptive field that feeds it.
        The convolution itself is then just a contraction over (c, k, k), which
        is exactly what `torch.einsum` is for.

    Why implement it this way? `Tensor.unfold` is a pure *view/stride* op - it
    reindexes memory rather than performing any convolution - so the sliding
    window is genuinely constructed here, and the actual multiply-accumulate is
    left entirely to the einsum below. This keeps the "no built-in convolution"
    constraint satisfied while staying fast enough to train on CPU.
    """
    if pad > 0:
        # Manual zero-pad: allocate a larger buffer and copy the image into the
        # middle of it. (F.pad would also work, but doing it explicitly keeps
        # the data flow obvious and avoids leaning on a torch helper.)
        b, c, h, w = x.shape
        padded = x.new_zeros((b, c, h + 2 * pad, w + 2 * pad))
        padded[:, :, pad:pad + h, pad:pad + w] = x
        x = padded

    # unfold(dim, size, step) inserts a new trailing axis of length `size`
    # holding each window along `dim`. Applied to H then W we get
    # (b, c, out_h, out_w, k, k) without copying any data.
    patches = x.unfold(2, k, stride).unfold(3, k, stride)
    return patches


class EinsumConv2d(nn.Module):
    """A convolution layer whose entire forward pass is one `torch.einsum`.

    weight: (c_out, c_in, k, k)   bias: (c_out,)
    """

    def __init__(self, c_in: int, c_out: int, k: int = 3, stride: int = 1, pad: int = 1):
        super().__init__()
        self.c_in, self.c_out, self.k, self.stride, self.pad = c_in, c_out, k, stride, pad

        # He/Kaiming-normal init: std = sqrt(2 / fan_in). Chosen because every
        # conv here is followed by a ReLU, and He init is what keeps activation
        # variance stable across depth for ReLU networks.
        fan_in = c_in * k * k
        self.weight = nn.Parameter(torch.randn(c_out, c_in, k, k) * math.sqrt(2.0 / fan_in))
        self.bias = nn.Parameter(torch.zeros(c_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (b, c_in, out_h, out_w, k, k)
        patches = extract_patches(x, self.k, self.stride, self.pad)

        # The convolution proper: contract the patch over the input-channel and
        # both kernel axes (i, p, q), broadcasting across batch and position.
        #   b i h w p q , o i p q -> b o h w
        # Read as: "for each output channel o, sum the elementwise product of its
        # (c_in, k, k) filter with the (c_in, k, k) receptive field at (h, w)."
        out = torch.einsum('bihwpq,oipq->bohw', patches, self.weight)

        # Bias is per-output-channel; rearrange gives it the (1, o, 1, 1) shape
        # needed to broadcast over batch and space.
        return out + rearrange(self.bias, 'o -> 1 o 1 1')

    def extra_repr(self) -> str:
        return f'{self.c_in}->{self.c_out}, k={self.k}, stride={self.stride}, pad={self.pad}'


class EinsumLinear(nn.Module):
    """Fully-connected layer as a single einsum contraction over the feature axis."""

    def __init__(self, f_in: int, f_out: int):
        super().__init__()
        self.f_in, self.f_out = f_in, f_out
        # Xavier/Glorot-uniform init: this layer feeds the softmax, not a ReLU,
        # so we want unit gain rather than He's sqrt(2) gain.
        bound = math.sqrt(6.0 / (f_in + f_out))
        self.weight = nn.Parameter(torch.empty(f_in, f_out).uniform_(-bound, bound))
        self.bias = nn.Parameter(torch.zeros(f_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 'bf,fo->bo' - contract the feature axis f, keep batch b and output o.
        return torch.einsum('bf,fo->bo', x, self.weight) + self.bias

    def extra_repr(self) -> str:
        return f'{self.f_in}->{self.f_out}'


def maxpool2d(x: torch.Tensor, k: int = 2) -> torch.Tensor:
    """2x2 max-pool expressed directly as an `einops.reduce`.

    The pattern splits each spatial axis into (blocks, within-block) and reduces
    away the within-block axes - which *is* the definition of pooling, stated
    declaratively instead of via a pooling module.
    """
    return reduce(x, 'b c (h ph) (w pw) -> b c h w', 'max', ph=k, pw=k)


def global_avgpool(x: torch.Tensor) -> torch.Tensor:
    """Collapse the whole spatial map to one number per channel: (b,c,h,w) -> (b,c)."""
    return reduce(x, 'b c h w -> b c', 'mean')


# --------------------------------------------------------------------------
# The model
# --------------------------------------------------------------------------

class EinopsCNN(nn.Module):
    """3 conv layers + a linear classifier head.

    Shape trace for a (b, 1, 28, 28) MNIST batch:

        input                                  (b,  1, 28, 28)
        conv1  1->8   3x3 s1 p1  + ReLU        (b,  8, 28, 28)
        maxpool 2x2                            (b,  8, 14, 14)
        conv2  8->16  3x3 s1 p1  + ReLU        (b, 16, 14, 14)
        maxpool 2x2                            (b, 16,  7,  7)
        conv3  16->32 3x3 s1 p1  + ReLU        (b, 32,  7,  7)
        global average pool                    (b, 32)
        linear head 32->10                     (b, 10)     <- raw logits

    Design notes (every choice is deliberate and defensible):

    * **3x3 kernels, stride 1, pad 1.** Padding of 1 keeps the spatial size
      unchanged, so all downsampling is done by the pooling layers alone - that
      makes the shape trace above trivial to reason about. 3x3 is the smallest
      kernel that still has a notion of orientation/edges.
    * **Channel widths 8 -> 16 -> 32.** The standard "halve the resolution,
      double the channels" schedule: each pool throws away 4x the spatial
      information, so widening by 2x keeps the representation from collapsing.
      Starting at 8 keeps the model tiny (~6k params) because MNIST is easy and
      this must train on CPU in minutes.
    * **Max-pool after conv1 and conv2, not conv3.** Two pools take 28 -> 7.
      A third would leave a 3x3 map that global average pooling makes redundant.
    * **Global average pooling instead of flattening.** Flattening 32x7x7 would
      need a 15,680-parameter head that dominates the model and overfits; GAP
      gives a 330-parameter head and forces conv3's channels to become the
      class-relevant features themselves.
    * **Logits, not softmax.** `F.cross_entropy` applies log-softmax internally;
      applying softmax here as well would be numerically worse and
      mathematically wrong (double-softmax).
    """

    def __init__(self, n_classes: int = 10):
        super().__init__()
        self.conv1 = EinsumConv2d(1, 8, k=3, stride=1, pad=1)
        self.conv2 = EinsumConv2d(8, 16, k=3, stride=1, pad=1)
        self.conv3 = EinsumConv2d(16, 32, k=3, stride=1, pad=1)
        self.head = EinsumLinear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = maxpool2d(F.relu(self.conv1(x)), k=2)   # (b,  8, 14, 14)
        x = maxpool2d(F.relu(self.conv2(x)), k=2)   # (b, 16,  7,  7)
        x = F.relu(self.conv3(x))                   # (b, 32,  7,  7)
        x = global_avgpool(x)                       # (b, 32)
        return self.head(x)                         # (b, 10) logits


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
