# ME1 — A 3-Layer CNN for MNIST Built from `einops` + `torch.einsum`

**Course:** AI 231 · **Author:** Lacuata

A convolutional network for MNIST digit classification in which **every layer is
hand-implemented** with `einops` (`rearrange`, `reduce`) and `torch.einsum`. No
`nn.Conv2d`, `nn.Linear`, `nn.MaxPool2d`, `nn.BatchNorm2d`, `F.conv2d`, `F.max_pool2d`,
`F.linear`, or `F.unfold` appears anywhere in the forward path.

> ### Final test accuracy: **97.51%** (9,751 / 10,000)
> 5 epochs on CPU · 6,218 trainable parameters · 432 s training time

---

## Files

| File | What it is |
|---|---|
| `me1_einops_cnn.ipynb` | **The deliverable.** Executed end to end with outputs intact: layer derivations, model definition, correctness verification, learning-rate sweep, 5-epoch training, final test accuracy, per-class breakdown, and the 4×4 prediction grid. |
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

a sum of products over three indices `(i, p, q)` — exactly what `einsum` expresses. The one
obstacle is that `x` is indexed at *shifted* positions `h+p, w+q`, which `einsum` cannot do
itself. So the sliding window is materialised first (`extract_patches`, the classic
**im2col** trick), giving `patches[b,i,h,w,p,q] = x[b,i,h+p,w+q]`, after which the whole
convolution is a single line:

```python
torch.einsum('bihwpq,oipq->bohw', patches, weight)
```

`i`, `p`, `q` are absent from the right-hand side, so einsum sums over them: the input
channel and both kernel axes. `b`, `o`, `h`, `w` survive.

Patch extraction uses `Tensor.unfold`, a pure **stride/view** operation — it reindexes
existing memory and performs no arithmetic, so no built-in convolution is smuggled in.
Zero-padding is done manually by allocating a larger zero buffer and copying the image into
the middle of it.

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

**6,218 trainable parameters** (conv1 80, conv2 1,168, conv3 4,640, head 330).

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
| Learning rate | **1e-2** | Selected by a sweep over {1e-3, 3e-3, 1e-2} on a held-out 5k validation split (88.84% / 93.14% / **95.58%** after 1 epoch). **See the caveat below — this is the least rigorous choice in the project.** |
| Weight decay | **0** | ~6k parameters against 60k images is capacity-limited, not overfitting; regularisation would only slow fitting. Confirmed by the +0.41% train/test gap. |
| Epochs | **5** | Fixed by the exercise specification. |
| Loss | **Cross-entropy** | Standard for single-label multi-class classification; consumes raw logits. |
| Normalisation | **mean 0.1307, std 0.3081** | MNIST's conventional dataset-wide statistics. Zero-centred unit-variance inputs are what the He initialisation assumes, so the two choices go together. |
| Augmentation | **none** | The model is capacity-limited, not data-limited; augmentation would slow convergence inside 5 epochs without addressing an overfitting problem that doesn't exist here. |
| Seed | **0** | Set once at the top of the notebook. |

**The test split is never used for any decision.** Learning-rate selection uses a
validation split held out of the training data; test accuracy is printed per epoch for
visibility only, and is the final reported number.

### ⚠️ Caveat: the learning-rate sweep is the weakest part of this project

Stated openly rather than papered over, because it is the first thing a careful reader
should question:

1. **The winner sits on the edge of the grid.** The sweep selected `1e-2`, the largest value
   tried. When the best value is an endpoint, the honest reading is that the grid was too
   narrow — the true optimum may lie outside it. A sound sweep is widened until the winner
   is *interior* to the range.
2. **A 1-epoch proxy structurally favours large learning rates.** A large step size makes
   rapid early progress, which is exactly what a 1-epoch score rewards, but it is also more
   likely to become unstable later. Selecting under a budget different from the one actually
   deployed means selecting for the wrong thing.

**The effect is visible in the results, not hypothetical.** Test accuracy peaks at **97.63%
in epoch 2** and then oscillates down to **97.51%** by epoch 5 instead of converging
cleanly — the signature of a step size slightly too large for the later part of training.

**The correct fix**, not run here purely on compute budget (~30 min vs ~1 min): widen the
grid (e.g. add `3e-2`) and give each candidate the full 5-epoch budget, selecting on
final-epoch validation accuracy and confirming the winner is not on a boundary.

**Why it still ships.** The exercise's objective is the einsum/einops implementation, and
97.51% is a genuine held-out number from a fully specified, seeded, reproducible procedure.
The learning rate is simply not tuned as tightly as it could be — likely costing a few
tenths of a percent. It should be described as *reasonable and evidence-informed*, not as
*tuned*.

---

## Results

| Metric | Value |
|---|---|
| **Final test accuracy** | **97.51%** (9,751 / 10,000) |
| Final test loss | 0.0772 |
| Train accuracy (epoch 5) | 97.92% |
| Generalisation gap | +0.41% |
| Trainable parameters | 6,218 |
| Training time (5 epochs, 6 CPU threads) | 432.3 s |

Per epoch:

| Epoch | Train loss | Train acc | Test loss | Test acc | Time |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.5306 | 82.17% | 0.1630 | 94.83% | 65.0 s |
| 2 | 0.1227 | 96.28% | 0.0752 | **97.63%** | 81.3 s |
| 3 | 0.0904 | 97.23% | 0.0769 | 97.58% | 91.6 s |
| 4 | 0.0794 | 97.53% | 0.0986 | 97.04% | 93.9 s |
| 5 | 0.0678 | 97.92% | 0.0772 | 97.51% | 100.6 s |

Per-class test accuracy — weakest digit is **3 (93.17%)**, strongest is **4 (99.59%)**:

| digit | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 |
|---|---|---|---|---|---|---|---|---|---|---|
| acc | 96.43% | 99.03% | 99.03% | **93.17%** | 99.59% | 98.99% | 97.91% | 96.50% | 97.23% | 97.22% |

Verification of the hand-rolled layers against the torch reference implementations:

```
einsum conv (k=3,s=1,p=1) vs F.conv2d      max abs err = 7.15e-07
einsum conv (k=5,s=2,p=2) vs F.conv2d      max abs err = 1.19e-06
einops maxpool            vs F.max_pool2d  max abs err = 0.00e+00
einops global avgpool     vs x.mean(2,3)   max abs err = 0.00e+00

gradients reached all 8 parameter tensors, all finite
```

Agreement at float32 precision means the einsum implementation is *correct*, not merely
plausible — "we didn't use `nn.Conv2d`" is a statement about the implementation, not a
disclaimer about the results.

---

## Reproducing

```bash
pip install -r requirements.txt

# Verify the einsum layers match torch's reference implementations:
python test_correctness.py

# Run the full exercise (downloads MNIST on first run, ~9 min on CPU):
jupyter nbconvert --to notebook --execute --inplace me1_einops_cnn.ipynb
# ...or just open me1_einops_cnn.ipynb and Run All.
```

Everything is seeded (`SEED = 0`), so a re-run reproduces the numbers above. MNIST is
downloaded automatically by `torchvision` into `data/` on first run; that directory is
gitignored rather than committed.

---

## Discussion

**What the exercise demonstrates.** Convolution, pooling, and matrix multiplication are not
three different mechanisms — they are all *contractions over chosen axes*. `einsum` lets
each be written as one line that names exactly which axes are summed and which survive. The
`torch.nn` layers are performance wrappers around that idea, not the idea itself.

**The cost of doing it longhand.** `extract_patches` materialises a `(b, c, h, w, k, k)`
tensor — 9× the input feature map for a 3×3 kernel. `nn.Conv2d` avoids that blow-up by
dispatching to fused kernels that never build the patch tensor explicitly. That memory
traffic is the real price of the explicit version, and it is why the batch-size benchmark
falls off a cliff at 256: the patch tensor stops fitting in cache.

**Where the remaining error is.** With a 32-dimensional globally-pooled representation
feeding a 330-parameter head, the model has little capacity to separate visually similar
digits. Digit 3 (93.17%) is the clear outlier — every other class is above 96% — though
identifying *which* digits it is being confused with would need a confusion matrix, which
this notebook does not compute. The most effective next step would be widening the channels
or replacing GAP with a small hidden layer, **not** training longer: the learning curves are
already flat by epoch 5, and the model is capacity-limited rather than underfit.
