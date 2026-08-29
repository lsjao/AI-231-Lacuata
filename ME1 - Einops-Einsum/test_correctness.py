"""Correctness checks for the hand-rolled einsum layers.

These use `F.conv2d` / `F.max_pool2d` as a *reference oracle only* - they are
never imported by the model. The whole point is to prove that the einsum
implementation computes exactly the same function as the built-in one, so that
"we didn't use nn.Conv2d" is a statement about the implementation, not a
disclaimer about the results.

Run with:  python test_correctness.py
"""

import torch
import torch.nn.functional as F

from einops_cnn import EinopsCNN, EinsumConv2d, count_parameters, global_avgpool, maxpool2d


def test_conv_matches_reference():
    torch.manual_seed(0)
    conv = EinsumConv2d(3, 5, k=3, stride=1, pad=1)
    x = torch.randn(2, 3, 12, 12)

    ours = conv(x)
    reference = F.conv2d(x, conv.weight, conv.bias, stride=1, padding=1)

    assert ours.shape == reference.shape, f'{ours.shape} vs {reference.shape}'
    max_err = (ours - reference).abs().max().item()
    assert max_err < 1e-5, f'conv mismatch, max abs error = {max_err}'
    print(f'  conv2d  vs F.conv2d      -> max abs err {max_err:.2e}  OK')


def test_conv_stride2_matches_reference():
    """Stride/padding bookkeeping is the easiest thing to get wrong, so check a
    non-default configuration too."""
    torch.manual_seed(1)
    conv = EinsumConv2d(2, 4, k=5, stride=2, pad=2)
    x = torch.randn(2, 2, 15, 15)

    ours = conv(x)
    reference = F.conv2d(x, conv.weight, conv.bias, stride=2, padding=2)

    assert ours.shape == reference.shape, f'{ours.shape} vs {reference.shape}'
    max_err = (ours - reference).abs().max().item()
    assert max_err < 1e-5, f'strided conv mismatch, max abs error = {max_err}'
    print(f'  conv2d  vs F.conv2d (s2) -> max abs err {max_err:.2e}  OK')


def test_maxpool_matches_reference():
    torch.manual_seed(2)
    x = torch.randn(2, 4, 8, 8)
    max_err = (maxpool2d(x, 2) - F.max_pool2d(x, 2)).abs().max().item()
    assert max_err == 0.0, f'maxpool mismatch, max abs error = {max_err}'
    print(f'  maxpool vs F.max_pool2d  -> max abs err {max_err:.2e}  OK')


def test_global_avgpool_matches_reference():
    torch.manual_seed(3)
    x = torch.randn(2, 6, 7, 7)
    max_err = (global_avgpool(x) - x.mean(dim=(2, 3))).abs().max().item()
    assert max_err < 1e-6, f'gap mismatch, max abs error = {max_err}'
    print(f'  gap     vs x.mean(2,3)   -> max abs err {max_err:.2e}  OK')


def test_model_shapes_and_gradients():
    model = EinopsCNN()
    x = torch.randn(4, 1, 28, 28)
    logits = model(x)
    assert logits.shape == (4, 10), logits.shape

    # Autograd must reach every parameter - a layer that silently detaches would
    # still "train" without ever updating, so assert gradients actually arrive.
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f'{name} received no gradient'
        assert torch.isfinite(p.grad).all(), f'{name} has non-finite gradient'
    print(f'  model   forward+backward -> logits {tuple(logits.shape)}, '
          f'all {len(list(model.parameters()))} params have finite grads  OK')
    print(f'  trainable parameters: {count_parameters(model):,}')


if __name__ == '__main__':
    print('Verifying einsum layers against torch reference implementations:')
    test_conv_matches_reference()
    test_conv_stride2_matches_reference()
    test_maxpool_matches_reference()
    test_global_avgpool_matches_reference()
    test_model_shapes_and_gradients()
    print('All checks passed.')
