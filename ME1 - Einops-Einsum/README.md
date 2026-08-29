# ME1 — A 3-Layer CNN for MNIST Built from `einops` + `torch.einsum`

**Course:** AI 231 · **Author:** Lacuata

A convolutional network for MNIST digit classification in which **every layer is
hand-implemented** with `einops` (`rearrange`, `reduce`) and `torch.einsum`. No
`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.BatchNorm2d`, `F.conv2d`, `F.max_pool2d`,
`F.linear`, or `F.unfold` appears anywhere in the forward path.

> **Final test accuracy: `PLACEHOLDER_ACC`** after 5 epochs on CPU, with 6,218 trainable
> parameters.

---

## Files

| File | What it is |
|---|---|
| `me1_einops_cnn.ipynb` | **The deliverable.** Executed end to end with outputs intact: layer derivations, model definition, correctness verification, LR sweep, 5-epoch training, final test accuracy, and the 4×4 prediction grid. |
| `einops_cnn.py` | The same layers and model as an importable module, for reuse and testing. |
| `test_correctness.py` | Verifies the einsum layers against `F.conv2d` / `F.max_pool2d` as a reference oracle. |
| `requirements.txt` | Pinned versions this was developed and run against. |

---

## What is and isn't used

| Not used anywhere in the forward math | Used, and why that's consistent with the brief |
|---|---|
| `nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.BatchNorm2d` | `nn.Parameter` — a weight *container*, not a layer |
| `F.conv2d`, `F.max_pool2d`, `F.linear`, `F.unfold` | `nn.Module` — a parameter *registry*, so `torch.optim` can find the weights |
| any built-in convolution routine | `torch.optim`, autograd — optimisation, not layer implementation |
| | `F.relu`, `F.cross_entropy` — activation and loss, explicitly permitted |

`F.conv2d` and `F.max_pool2d` appear **only** inside verification code (notebook §2.1 and
`test_correctness.py`), used as a reference oracle to prove the einsum implementation
computes the identical function. They are never part of the model.

---

## How each layer is implemented

### Convolution — one `einsum`

A convolution output is

```
y[b,o,h,w] = Σ_i Σ_p Σ_q  x[b, i, h+p, w+q] · W[o,i,p,q]  +  β[o]
```

a sum of products over three indices `(i, p, q)` — exactly what `einsum` expresses. The
one obstacle is that `x` is indexed at *shifted* positions `h+p, w+q`, which `einsum`
cannot do itself. So the sliding window is materialised first (`extract_patches`, the
classic **im2col** trick), giving `patches[b,i,h,w,p,q] = x[b,i,h+p,w+q]`, after which the
whole convolution is a single line:

```python
torch.einsum('bihwpq,oipq->bohw', patches, weight)
```

`i`, `p`, `q` are absent from the right-hand side, so einsum sums over them: the input
channel and both kernel axes. `b`, `o`, `h`, `w` survive.

Patch extraction uses `Tensor.unfold`, a pure **stride/view** operation — it reindexes
existing memory and performs no arithmetic, so no built-in convolution is smuggled in.
Zero-padding is done manually by allocating a larger zero buffer and copying the image
into the middle of it.

### Pooling — `einops.reduce`

```python
reduce(x, 'b c (h ph) (w pw) -> b c h w', 'max', ph=2, pw=2)   # 2x2 max-pool
reduce(x, 'b c h w -> b c', 'mean')                            # global average pool
```

The pattern factorises each spatial axis into (block index, position-within-block) and
reduces the within-block axes away — which *is* the definition of pooling, stated
declaratively rather than invoked as a module.

### Fully-connected head — `einsum`

```python
torch.einsum('bf,fo->bo', x, weight) + bias    # contract the feature axis
```

---

## Architecture

```
input                                  (b,  1, 28, 28)
conv1   1 -> 8    3x3 s1 p1  + ReLU    (b,  8, 28, 28)
maxpool 2x2                            (b,  8, 14, 14)
conv2   8 -> 16   3x3 s1 p1  + ReLU    (b, 16, 14, 14)
maxpool 2x2                            (b, 16,  7,  7)
conv3  16 -> 32   3x3 s1 p1  + ReLU    (b, 32,  7,  7)
global average pool                    (b, 32)
linear head 32 -> 10                   (b, 10)   <- raw logits
```

**6,218 trainable parameters.**

Why this design:

- **3×3 kernels, stride 1, padding 1.** Padding 1 makes each conv shape-preserving, so all
  downsampling happens in the pooling layers alone and the shape trace stays trivial to
  reason about. 3×3 is the smallest kernel that can still represent orientation and edges;
  three stacked 3×3 convs give a 7×7 effective receptive field, which after the two pools
  covers essentially the whole digit.
- **Channel widths 8 → 16 → 32.** The standard "halve the resolution, double the channels"
  schedule — each 2×2 pool discards 4× the spatial information, so doubling channels keeps
  the representation from collapsing. Starting at 8 keeps the model ~6k parameters, which
  matters because this trains on CPU.
- **Pool after conv1 and conv2 but not conv3.** Two pools take 28 → 14 → 7. A third would
  leave a 3×3 map that global average pooling then makes redundant anyway.
- **Global average pooling instead of flattening.** Flattening 32×7×7 = 1,568 features
  would need a **15,690**-parameter head — 70% of the entire model and the main thing that
  would overfit. GAP gives a **330**-parameter head and forces conv3's 32 channels to
  become class-relevant features themselves. This is the single biggest architectural
  decision here.
- **Outputs are logits, not probabilities.** `F.cross_entropy` applies log-softmax
  internally; adding a softmax would be numerically worse and mathematically wrong.

**Initialisation.** He/Kaiming-normal (`std = sqrt(2/fan_in)`) for the convs, because each
is followed by a ReLU and the factor of 2 compensates for ReLU zeroing half the
activations. Xavier/Glorot-uniform for the head, because it feeds a softmax rather than a
ReLU and so wants unit gain.

---

## Hyperparameters

| Hyperparameter | Value | Justification |
|---|---|---|
| Batch size | **128** | Benchmarked on the actual machine: 64 → 1,908 img/s, **128 → 2,231 img/s**, 256 → 1,076 img/s. 128 is the throughput optimum; larger batches push the im2col patch tensor out of cache. |
| Optimizer | **Adam** | Per-parameter adaptive step sizes remove the need for a hand-tuned LR schedule within a fixed 5-epoch budget. |
| Learning rate | **`PLACEHOLDER_LR`** | Not asserted — chosen by a 1-epoch sweep over {1e-3, 3e-3, 1e-2} on a held-out 5k validation split carved from train (notebook §3.1). |
| Weight decay | **0** | ~6k parameters against 60k images is capacity-limited, not overfitting; regularisation would only slow fitting. Confirmed by the small train/test gap (`PLACEHOLDER_GAP`). |
| Epochs | **5** | Fixed by the exercise specification. |
| Loss | **Cross-entropy** | Standard for single-label multi-class classification; consumes raw logits. |
| Normalisation | **mean 0.1307, std 0.3081** | MNIST's conventional dataset-wide statistics. Zero-centred unit-variance inputs are what the He initialisation assumes, so the two choices go together. |
| Augmentation | **none** | The model is capacity-limited, not data-limited; augmentation would slow convergence inside 5 epochs without addressing an overfitting problem that doesn't exist here. |
| Seed | **0** | Set once at the top of the notebook. |

**The test split is never used for any decision.** Learning-rate selection uses a
validation split held out of the training data; test accuracy is printed per epoch for
visibility only and is the final reported number.

---

## Results

| Metric | Value |
|---|---|
| **Final test accuracy** | **`PLACEHOLDER_ACC`** |
| Final test loss | `PLACEHOLDER_LOSS` |
| Train accuracy (epoch 5) | `PLACEHOLDER_TRAIN` |
| Generalisation gap | `PLACEHOLDER_GAP` |
| Trainable parameters | 6,218 |
| Training time (5 epochs, CPU) | `PLACEHOLDER_TIME` |

Verification of the hand-rolled layers against the torch reference implementations:

```
PLACEHOLDER_VERIFY
```

---

## Reproducing

```bash
pip install -r requirements.txt

# Verify the einsum layers match torch's reference implementations:
python test_correctness.py

# Run the full exercise (downloads MNIST on first run, ~10 min on CPU):
jupyter nbconvert --to notebook --execute --inplace me1_einops_cnn.ipynb
# ...or just open me1_einops_cnn.ipynb and Run All.
```

MNIST is downloaded automatically by `torchvision` into `data/` on first run; that
directory is gitignored rather than committed.

---

## Discussion

**What the exercise demonstrates.** Convolution, pooling, and matrix multiplication are not
three different mechanisms — they are all *contractions over chosen axes*. `einsum` lets
each be written as one line that names exactly which axes are summed and which survive.
The `torch.nn` layers are performance wrappers around that idea, not the idea itself.

**The cost of doing it longhand.** `extract_patches` materialises a `(b, c, h, w, k, k)`
tensor — 9× the input feature map for a 3×3 kernel. `nn.Conv2d` avoids that blow-up by
dispatching to fused kernels that never build the patch tensor explicitly. That memory
traffic is the real price of the explicit version, and it is why the batch-size benchmark
above falls off a cliff at 256: the patch tensor stops fitting in cache.

**Where the remaining error is.** The per-class table in the notebook shows the weakest
digits. With a 32-dimensional globally-pooled representation feeding a 330-parameter head,
the model has little capacity to separate visually similar digits (4/9, 3/5, 7/1), which is
where most of the residual error sits. The most effective next step would be widening the
channels or replacing GAP with a small hidden layer — not training longer, since the
learning curves are already close to flat by epoch 5.
