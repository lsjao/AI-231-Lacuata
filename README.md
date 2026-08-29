# AI 231, Lacuata

Coursework repository for **AI 231**. Every machine exercise lives in its own subfolder and
stands on its own, carrying a self-contained notebook, a README that walks through the
reasoning, and a pinned requirements file, so any exercise can be cloned and reproduced
without touching the others.

## Contents

| Folder | Exercise | Summary |
|---|---|---|
| [`ME1 - Einops-Einsum`](./ME1%20-%20Einops-Einsum) | Machine Exercise 1 | A 3-layer CNN for MNIST in which every layer is hand-implemented with `einops` and `torch.einsum`, with no `nn.Conv2d`, `nn.Linear`, or `nn.MaxPool2d` anywhere in the forward path. Reaches **97.51%** test accuracy from 6,218 parameters. |

## ME1 at a glance

![Training and test learning curves for the einops/einsum CNN](./ME1%20-%20Einops-Einsum/figures/learning_curves.png)

Five epochs on CPU take the model from 82% to 97.51% on the held-out test set, with the
train and test curves staying close together throughout. The
[full writeup](./ME1%20-%20Einops-Einsum/README.md) derives each layer as an index
contraction, documents every hyperparameter choice, and discusses openly where the
experiment could have been tighter.
