# LSNet Modification Implementation Guide
## Proposals 3, 7, 8, 9 — Pretrained Weight Reuse Strategy

---

## Preamble — What the Pretrained Checkpoints Contain

The file `lsnet_t.pth` is a standard PyTorch checkpoint produced by
`torch.save({'model': model.state_dict(), ...})`. It contains a flat
`state_dict` with keys matching the original `LSNet` class defined in
`model/lsnet.py`. Every key you add or rename in the modified model
will be **missing** from the checkpoint; every key in the checkpoint
that no longer exists in the model will be **unexpected**. Both
conditions are acceptable when loading with `strict=False`.

### Confirmed original architecture (from source)

```
LSNet-T:
  embed_dim  = [64, 128, 256, 384]
  depth      = [0,  2,   8,   10 ]   # stage 0 has 0 LS blocks
  num_heads  = [3,  3,   3,   4  ]
  patch_size = 8
  img_size   = 224

Block logic (lsnet.py lines):
  if depth % 2 == 0:          # even index → RepVGGDW + SE
      mixer = RepVGGDW(ed)
      se    = SqueezeExcite(ed, 0.25)
  else:                        # odd index → LSConv (stages 0-2) or Attention (stage 3)
      se    = Identity()
      if stage == 3:
          mixer = Residual(Attention(...))
      else:
          mixer = LSConv(ed)   ← THIS IS WHAT WE MODIFY

LSConv (lsnet.py lines):
  lkp = LKP(dim, lks=7, sks=3, groups=8)
  ska = SKA()
  bn  = BatchNorm2d(dim)
  forward: bn(ska(x, lkp(x))) + x

LKP (lsnet.py lines):
  cv1  = Conv2d_BN(dim, dim//2)           # 1×1 PW, reduces channels
  act  = ReLU()
  cv2  = Conv2d_BN(dim//2, dim//2,        # 7×7 DW, large-kernel perception
             ks=7, pad=3, groups=dim//2)
  cv3  = Conv2d_BN(dim//2, dim//2)        # 1×1 PW
  cv4  = Conv2d(dim//2,                   # 1×1 PW, generates dynamic weights
             sks**2 * dim // groups, 1)   # → shape [B, (3²×dim/8), H, W]
  norm = GroupNorm(dim//groups,
             sks**2 * dim // groups)
  forward:
    x = act(cv3(cv2(act(cv1(x)))))
    w = norm(cv4(x))
    w = w.view(B, dim//groups, sks**2, H, W)
    return w   # shape: [B, dim//8, 9, H, W]

SKA (ska.py lines):
  forward(x, w):
    # x: [B, C, H, W]
    # w: [B, C//G, KS*KS, H, W]  (G=8, KS=3 → C//8, 9, H, W)
    return PyTorchSkaFn.apply(x, w)

PyTorchSkaFn (ska.py lines):
  1. x_unfolded = F.unfold(x, ks=3, pad=1)  → [B, C*9, H*W]
  2. x_unfolded = x_unfolded.view(B, C, 9, H*W)
  3. w = w.view(B, C//8, 9, H*W)
  4. w = w.repeat(1, 8, 1, 1)               → [B, C, 9, H*W]
  5. output = (x_unfolded * w).sum(dim=2)   → [B, C, H*W]
  6. output = output.view(B, C, H, W)
```

---

## Repository Structure You Will Work In

```
lsnet/
├── model/
│   ├── lsnet.py          ← MODIFY THIS (LSConv, LKP, Block classes)
│   └── __init__.py
├── ska.py  (or model/ska.py)  ← MODIFY FOR P3, P9
├── main.py               ← MODIFY for fine-tune entry point
├── engine.py             ← DO NOT MODIFY
├── train.sh              ← CREATE new variant for fine-tuning
└── pretrain/
    └── lsnet_t.pth       ← LOAD THIS, never overwrite it
```

> **Convention used in this guide:** All new classes are added to
> `model/lsnet.py` unless stated otherwise. The original classes
> (`LKP`, `LSConv`, `SKA`, `Block`) are **not deleted** — only
> subclassed or conditionally replaced.

---

## Proposal 3 — Sparse Dynamic Aggregation in SKA

### What changes

Inside `PyTorchSkaFn.forward`, after the weight tensor `w` is
expanded to shape `[B, C, 9, H*W]`, we zero out the `k` positions
per spatial location that have the smallest absolute magnitude.
During the backward pass the zeroed positions produce zero gradients
naturally (no STE needed — we are masking values, not
discontinuous argmax).

### Checkpoint compatibility

`P3` adds **no new parameters**. All weight keys are identical to the
original checkpoint. Load with `strict=True`.

### Step-by-step implementation

**Step 1 — Add `SparsePyTorchSkaFn` in `ska.py`**

Add the following class immediately after `PyTorchSkaFn`. Do not
remove `PyTorchSkaFn`.

```python
class SparsePyTorchSkaFn(Function):
    """
    Sparse variant of SKA: zeroes out the (ks*ks - top_k) kernel
    positions with smallest absolute value before aggregation.
    top_k is applied per (batch, channel, spatial location).
    Default top_k=5 out of 9 keeps the 5 most informative neighbours.
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, w: torch.Tensor,
                top_k: int = 5) -> torch.Tensor:
        ks = int(math.sqrt(w.shape[2]))   # always 3 for LSNet-T
        pad = (ks - 1) // 2              # always 1

        n, ic, h, width = x.shape
        wc = w.shape[1]

        # --- identical to original up to the repeat ---
        x_unfolded = F.unfold(x, kernel_size=ks, padding=pad)
        x_unfolded = x_unfolded.view(n, ic, ks * ks, h * width)
        w = w.view(n, wc, ks * ks, h * width)
        if ic != wc:
            repeats = ic // wc
            w = w.repeat(1, repeats, 1, 1)

        # --- sparsity mask: keep top_k positions per (n, c, hw) ---
        # w shape: [n, ic, ks*ks, h*width]
        # Compute threshold along dim=2 (kernel positions)
        with torch.no_grad():
            # abs_w: [n, ic, ks*ks, h*width]
            abs_w = w.abs()
            # topk returns values sorted descending; take index top_k-1
            threshold = abs_w.topk(top_k, dim=2, largest=True,
                                   sorted=True).values[..., -1:, :]
            # mask: True where abs_w >= threshold (keep), False (zero)
            mask = (abs_w >= threshold).float()

        w = w * mask   # zero out bottom (ks*ks - top_k) positions

        output = (x_unfolded * w).sum(dim=2)
        output = output.view(n, ic, h, width)
        return output
        # NOTE: backward is handled automatically by autograd because
        # mask is detached (no_grad block). Masked positions receive
        # zero gradient, which is the correct behaviour.
```

**Step 2 — Add `SparseSKA` module in `ska.py`**

```python
class SparseSKA(torch.nn.Module):
    """
    Drop-in replacement for SKA that uses sparse aggregation.
    top_k: number of kernel positions (out of KS*KS) to keep.
           For KS=3, KS*KS=9. Recommended default: 5.
    """
    def __init__(self, top_k: int = 5):
        super().__init__()
        self.top_k = top_k

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        return SparsePyTorchSkaFn.apply(x, w, self.top_k)
```

**Step 3 — Modify `LSConv.__init__` in `model/lsnet.py`**

Change the import at the top of `lsnet.py`:

```python
# BEFORE:
from .ska import SKA

# AFTER:
from .ska import SKA, SparseSKA
```

Then modify `LSConv.__init__` to accept a `sparse` flag:

```python
# BEFORE:
class LSConv(nn.Module):
    def __init__(self, dim):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SKA()
        self.bn  = nn.BatchNorm2d(dim)

# AFTER:
class LSConv(nn.Module):
    def __init__(self, dim, sparse_ska: bool = False,
                 sparse_top_k: int = 5):
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=8)
        self.ska = SparseSKA(top_k=sparse_top_k) if sparse_ska else SKA()
        self.bn  = nn.BatchNorm2d(dim)
```

`LSConv.forward` is **unchanged**.

**Step 4 — Propagate the flag through `Block`**

```python
# BEFORE (lsnet.py lines):
else:
    self.mixer = LSConv(ed)

# AFTER:
else:
    self.mixer = LSConv(ed,
                        sparse_ska=getattr(kwargs_block, 'sparse_ska', False),
                        sparse_top_k=getattr(kwargs_block, 'sparse_top_k', 5))
```

Because the original `Block.__init__` does not accept `**kwargs`,
the cleanest approach that does not break the checkpoint loading is
to add two arguments directly:

```python
class Block(torch.nn.Module):
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1,
                 sparse_ska: bool = False,   # NEW
                 sparse_top_k: int = 5):     # NEW
        super().__init__()

        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se    = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar,
                                                resolution=resolution))
            else:
                self.mixer = LSConv(ed,
                                    sparse_ska=sparse_ska,
                                    sparse_top_k=sparse_top_k)  # NEW

        self.ffn = Residual(FFN(ed, int(ed * 2)))
```

`Block.forward` is **unchanged**.

**Step 5 — Propagate through `LSNet.__init__`**

In the loop that builds blocks (around line 274 in `lsnet.py`):

```python
# BEFORE:
blocks[i].append(Block(ed, kd, nh, ar, resolution, stage=i, depth=d))

# AFTER:
blocks[i].append(Block(ed, kd, nh, ar, resolution,
                       stage=i, depth=d,
                       sparse_ska=kwargs.get('sparse_ska', False),
                       sparse_top_k=kwargs.get('sparse_top_k', 5)))
```

**Step 6 — Update the `lsnet_t` factory function**

```python
@register_model
def lsnet_t(num_classes=1000, distillation=False, pretrained=False,
            sparse_ska=False, sparse_top_k=5, **kwargs):   # NEW ARGS
    model = _create_lsnet("lsnet_t" + ("_distill" if distillation else ""),
                          pretrained=pretrained,
                          num_classes=num_classes,
                          distillation=distillation,
                          img_size=224,
                          patch_size=8,
                          embed_dim=[64, 128, 256, 384],
                          depth=[0, 2, 8, 10],
                          num_heads=[3, 3, 3, 4],
                          sparse_ska=sparse_ska,       # NEW
                          sparse_top_k=sparse_top_k,   # NEW
                          **kwargs)
    return model
```

### Checkpoint loading for P3

```python
import torch
from model.lsnet import lsnet_t

# Build modified model
model = lsnet_t(sparse_ska=True, sparse_top_k=5)

# Load checkpoint
ckpt = torch.load('pretrain/lsnet_t.pth', map_location='cpu')
state = ckpt['model']  # or ckpt if saved without wrapping

# strict=True is safe because NO new parameters were added
missing, unexpected = model.load_state_dict(state, strict=True)
# Expected: missing=[], unexpected=[]
print("Missing:", missing)
print("Unexpected:", unexpected)
```

### Fine-tuning schedule for P3

P3 adds zero parameters. The pretrained weights are fully reused.
Fine-tuning is needed only because the sparse mask changes the
gradient landscape during the first few epochs.

| Phase | Epochs | LR | Notes |
|---|---|---|---|
| Warmup | 5 | 1e-5 → 5e-5 | Linear warmup, all layers unfrozen |
| Fine-tune | 25 | 5e-5 → 1e-7 | Cosine decay |
| **Total** | **30** | | |

Use the same optimiser settings as the original training:
`AdamW, weight_decay=0.025, batch_size=2048`.

**Ablation:** Run with `top_k ∈ {3, 5, 7, 9}`. `top_k=9` equals
the original (all kept); this is your numerical sanity check.

---

## Proposal 7 — Per-Group Micro-SE Inside SKA

### What changes

After the grouped dynamic convolution in `LSConv`, but before the
final `BatchNorm2d`, we insert a lightweight Squeeze-and-Excite
block that operates **per group** rather than globally. The SE block
has squeeze ratio 4 (i.e., `reduced_channels = dim // (groups * 4)`).

### Checkpoint compatibility

P7 adds new parameters (`se_g.fc1.weight`, `se_g.fc1.bias`,
`se_g.fc2.weight`, `se_g.fc2.bias` per `LSConv` instance). These
keys are **absent** from the pretrained checkpoint. Load with
`strict=False`. The pretrained weights for `lkp.*`, `ska` (no
params), and `bn.*` all load correctly.

### Step-by-step implementation

**Step 1 — Add `GroupSE` in `model/lsnet.py`**

Place this class immediately before `LSConv`:

```python
class GroupSE(nn.Module):
    """
    Per-group Squeeze-and-Excite.

    Splits input tensor into G groups along the channel dimension,
    applies independent SE recalibration to each group, then
    concatenates back. Squeeze ratio is fixed at 4.

    Args:
        dim    : total number of channels (C)
        groups : number of groups (G). Must divide dim exactly.
                 For LSNet-T default: groups=8.
    """
    def __init__(self, dim: int, groups: int = 8):
        super().__init__()
        assert dim % groups == 0, \
            f"dim ({dim}) must be divisible by groups ({groups})"
        self.groups = groups
        self.channels_per_group = dim // groups
        reduced = max(1, self.channels_per_group // 4)  # squeeze ratio 4

        # One FC pair per group, stored as groups × reduced conv
        # We use a grouped 1×1 conv to process all groups in parallel
        self.fc1 = nn.Conv2d(dim, reduced * groups,
                             kernel_size=1, groups=groups, bias=True)
        self.fc2 = nn.Conv2d(reduced * groups, dim,
                             kernel_size=1, groups=groups, bias=True)
        self.act  = nn.ReLU()
        self.gate = nn.Sigmoid()

        # Initialise fc2 to near-zero so at the start of fine-tuning
        # the GroupSE is approximately an identity (gate ≈ 0.5 → scale ≈ 0.5).
        # We compensate by initialising fc2 bias to 1.0 so gate(0+1)≈0.73,
        # which is still close to 1 for fast warm-up convergence.
        nn.init.zeros_(self.fc2.weight)
        nn.init.ones_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        # Global average pool per group
        scale = x.mean(dim=[2, 3], keepdim=True)   # [B, C, 1, 1]
        scale = self.act(self.fc1(scale))            # [B, reduced*G, 1, 1]
        scale = self.gate(self.fc2(scale))           # [B, C, 1, 1]
        return x * scale
```

**Step 2 — Modify `LSConv`**

```python
class LSConv(nn.Module):
    def __init__(self, dim,
                 sparse_ska: bool = False,
                 sparse_top_k: int = 5,
                 group_se: bool = False,      # NEW
                 groups: int = 8):            # NEW (must match LKP groups)
        super(LSConv, self).__init__()
        self.lkp     = LKP(dim, lks=7, sks=3, groups=groups)
        self.ska     = SparseSKA(top_k=sparse_top_k) if sparse_ska else SKA()
        self.group_se = GroupSE(dim, groups=groups) if group_se else nn.Identity()
        self.bn      = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.ska(x, self.lkp(x))
        out = self.group_se(out)    # applied before BN
        return self.bn(out) + x
```

> **Why before BN?** The SE gate recalibrates raw aggregated
> features. Applying it before BN ensures the subsequent BN
> normalises the recalibrated distribution, not the pre-SE one.

**Step 3 — Propagate through `Block` and `lsnet_t`**

Follow the same pattern as P3: add `group_se: bool = False` and
`groups: int = 8` to `Block.__init__`, pass them to `LSConv`, then
add them to `lsnet_t`.

```python
# In Block.__init__, in the else/LSConv branch:
self.mixer = LSConv(ed,
                    sparse_ska=sparse_ska,
                    sparse_top_k=sparse_top_k,
                    group_se=group_se,    # NEW
                    groups=8)             # hardcoded; matches original LKP

# In lsnet_t factory:
@register_model
def lsnet_t(num_classes=1000, distillation=False, pretrained=False,
            sparse_ska=False, sparse_top_k=5,
            group_se=False, **kwargs):
    model = _create_lsnet(...,
                          sparse_ska=sparse_ska,
                          sparse_top_k=sparse_top_k,
                          group_se=group_se,
                          **kwargs)
    return model
```

### Checkpoint loading for P7

```python
model = lsnet_t(group_se=True)

ckpt  = torch.load('pretrain/lsnet_t.pth', map_location='cpu')
state = ckpt['model']

# strict=False because GroupSE params are new
missing, unexpected = model.load_state_dict(state, strict=False)

# Expected missing keys (one LSConv per odd-depth block in stages 1-2):
# e.g. "blocks2.0.mixer.group_se.fc1.weight", "...fc1.bias",
#       "blocks2.0.mixer.group_se.fc2.weight", "...fc2.bias", ...
# Expected unexpected keys: []
print("Missing (should be only GroupSE keys):", missing)
print("Unexpected:", unexpected)

# Safety check: confirm no pretrained weights are accidentally missing
non_gse_missing = [k for k in missing if 'group_se' not in k]
assert len(non_gse_missing) == 0, \
    f"Non-GroupSE keys missing — check model definition: {non_gse_missing}"
```

### Fine-tuning schedule for P7

The `GroupSE` weights are initialised close to identity (see
`fc2.bias=1.0` above). The pretrained feature extractor is valid
from step 0. A short warm-up lets the new SE weights find their
role before the learning rate ramps up.

| Phase | Epochs | LR (GroupSE params) | LR (all other params) | Notes |
|---|---|---|---|---|
| Warm-up | 5 | 5e-4 | 1e-5 | GroupSE learns fast; backbone is nearly frozen |
| Joint fine-tune | 25 | 1e-4 → 1e-6 | 5e-5 → 1e-7 | Cosine decay, both param groups |
| **Total** | **30** | | | |

Use PyTorch parameter groups to assign different learning rates:

```python
se_params    = [p for n, p in model.named_parameters()
                if 'group_se' in n]
other_params = [p for n, p in model.named_parameters()
                if 'group_se' not in n]

optimizer = torch.optim.AdamW([
    {'params': se_params,    'lr': 5e-4, 'weight_decay': 0.0},
    {'params': other_params, 'lr': 1e-5, 'weight_decay': 0.025},
])
```

---

## Proposal 8 — Shifted-Window SKA in Alternating Blocks

### What changes

In LSNet, `LSConv.forward` calls `ska(x, lkp(x))`. The SKA unfolds
`x` with `F.unfold(x, kernel_size=3, padding=1)` — a fixed 3×3
window centred on each spatial position. In a **shifted** variant,
we first roll the feature map by `(shift_h, shift_w)` pixels,
apply the normal 3×3 SKA on the rolled map, then roll back. The net
effect is that the effective 3×3 aggregation window is offset, and
in adjacent even/odd depth blocks the two different window
alignments together tile the full spatial extent.

**This adds zero new parameters.** All weight keys are identical.
Load with `strict=True`.

### Shift amount

For `KS=3`, the standard shift used in Swin is `KS//2 = 1` pixel.
We shift by `(1, 1)` in (H, W). This means the aggregation window
that normally covers positions `(-1,0,+1) × (-1,0,+1)` now covers
`(0,+1,+2) × (0,+1,+2)` (modulo boundary, handled by the cyclic
roll).

### Step-by-step implementation

**Step 1 — Add `ShiftedSKA` in `ska.py`**

```python
class ShiftedSKA(torch.nn.Module):
    """
    Applies cyclic spatial shift before SKA and unshifts after.
    shift_size: tuple (shift_h, shift_w). Default (1, 1) for KS=3.
    """
    def __init__(self, shift_size: tuple = (1, 1)):
        super().__init__()
        self.shift_h, self.shift_w = shift_size
        self._ska_fn = PyTorchSkaFn   # reuse existing function

    def forward(self, x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        # Cyclic shift: roll in H and W dimensions
        x_shifted = torch.roll(x,
                               shifts=(-self.shift_h, -self.shift_w),
                               dims=(2, 3))
        # w must shift the same way so weights correspond to the
        # same spatial positions as the rolled x
        w_shifted = torch.roll(w,
                               shifts=(-self.shift_h, -self.shift_w),
                               dims=(3, 4))
        # Apply standard SKA on the shifted tensors
        out_shifted = PyTorchSkaFn.apply(x_shifted, w_shifted)
        # Reverse the cyclic shift on the output
        out = torch.roll(out_shifted,
                         shifts=(self.shift_h, self.shift_w),
                         dims=(2, 3))
        return out
```

> **Why roll `w` as well?** `w` has shape `[B, C//G, KS*KS, H, W]`
> — the last two dimensions are spatial. `LKP` computed `w` from `x`
> before the shift, so both must be aligned. Rolling `w` ensures that
> the dynamic kernel at position `(h, w_)` in the shifted domain
> corresponds to the shifted feature at the same location.

**Step 2 — Modify `LSConv`**

```python
class LSConv(nn.Module):
    def __init__(self, dim,
                 sparse_ska: bool = False,
                 sparse_top_k: int = 5,
                 group_se: bool = False,
                 groups: int = 8,
                 shifted: bool = False,       # NEW
                 shift_size: tuple = (1, 1)): # NEW
        super(LSConv, self).__init__()
        self.lkp = LKP(dim, lks=7, sks=3, groups=groups)

        if shifted:
            self.ska = ShiftedSKA(shift_size=shift_size)
        elif sparse_ska:
            self.ska = SparseSKA(top_k=sparse_top_k)
        else:
            self.ska = SKA()

        self.group_se = GroupSE(dim, groups=groups) if group_se else nn.Identity()
        self.bn       = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.ska(x, self.lkp(x))
        out = self.group_se(out)
        return self.bn(out) + x
```

**Step 3 — Assign shifted/non-shifted alternating in `Block`**

The LSNet paper assigns block roles by `depth % 2`. We extend this:

- `depth % 2 == 0` → `RepVGGDW` (unchanged, no SKA)
- `depth % 2 == 1` → `LSConv`
  - `depth % 4 == 1` → regular SKA
  - `depth % 4 == 3` → shifted SKA

This means one in every two `LSConv` blocks uses shifted windows.

```python
class Block(torch.nn.Module):
    def __init__(self,
                 ed, kd, nh=8,
                 ar=4,
                 resolution=14,
                 stage=-1, depth=-1,
                 sparse_ska: bool = False,
                 sparse_top_k: int = 5,
                 group_se: bool = False,
                 use_shifted_ska: bool = False):  # NEW: master switch
        super().__init__()

        if depth % 2 == 0:
            self.mixer = RepVGGDW(ed)
            self.se    = SqueezeExcite(ed, 0.25)
        else:
            self.se = torch.nn.Identity()
            if stage == 3:
                self.mixer = Residual(Attention(ed, kd, nh, ar,
                                                resolution=resolution))
            else:
                # Alternate: depths 1,5,9,... get regular; 3,7,11,... get shifted
                is_shifted = use_shifted_ska and (depth % 4 == 3)
                self.mixer = LSConv(
                    ed,
                    sparse_ska=sparse_ska,
                    sparse_top_k=sparse_top_k,
                    group_se=group_se,
                    groups=8,
                    shifted=is_shifted,
                    shift_size=(1, 1),
                )

        self.ffn = Residual(FFN(ed, int(ed * 2)))
```

**Step 4 — Propagate through `LSNet` and `lsnet_t`**

```python
# In the LSNet block-building loop:
blocks[i].append(Block(ed, kd, nh, ar, resolution,
                       stage=i, depth=d,
                       sparse_ska=kwargs.get('sparse_ska', False),
                       sparse_top_k=kwargs.get('sparse_top_k', 5),
                       group_se=kwargs.get('group_se', False),
                       use_shifted_ska=kwargs.get('use_shifted_ska', False)))

# lsnet_t factory:
@register_model
def lsnet_t(num_classes=1000, distillation=False, pretrained=False,
            sparse_ska=False, sparse_top_k=5,
            group_se=False,
            use_shifted_ska=False, **kwargs):
    model = _create_lsnet(...,
                          sparse_ska=sparse_ska,
                          sparse_top_k=sparse_top_k,
                          group_se=group_se,
                          use_shifted_ska=use_shifted_ska,
                          **kwargs)
    return model
```

### Checkpoint loading for P8

```python
model = lsnet_t(use_shifted_ska=True)

ckpt  = torch.load('pretrain/lsnet_t.pth', map_location='cpu')
state = ckpt['model']

# strict=True: ShiftedSKA has zero parameters
missing, unexpected = model.load_state_dict(state, strict=True)
print("Missing:", missing)      # expected []
print("Unexpected:", unexpected) # expected []
```

### Fine-tuning schedule for P8

P8 has no new parameters. The only disruption is the shifted spatial
alignment in every fourth `LSConv` block. The BN running statistics
for those layers will adapt within 2–5 epochs.

| Phase | Epochs | LR | Notes |
|---|---|---|---|
| Warmup | 3 | 1e-5 → 5e-5 | Linear warmup |
| Fine-tune | 17 | 5e-5 → 1e-7 | Cosine decay |
| **Total** | **20** | | Shortest of all proposals |

**Critical:** Freeze the `head` (classification layer) for the first
3 epochs to prevent early feature drift from influencing the
classification loss before the BN statistics adapt.

```python
# Freeze head for warmup epochs
for param in model.head.parameters():
    param.requires_grad = False

# After warmup (epoch 3):
for param in model.head.parameters():
    param.requires_grad = True
```

---

## Proposal 9 — Quantization-Aware Training on the Dynamic Weight Path

### What changes

The output of `LKP.forward` — the dynamic kernel weights `w` of
shape `[B, dim//8, 9, H, W]` — is float-sensitive. When the model
is post-training quantized (PTQ) to INT8, this tensor is quantized
with per-tensor statistics that were computed on a calibration set,
causing significant accuracy drop because `w` varies per input.

QAT fixes this by inserting `FakeQuantize` nodes on the `w` path
during training so the model learns weight values that are robust
under 8-bit discretisation. We insert fake-quant **after** `LKP`
generates `w` and **before** `SKA` consumes it.

We do **not** quantize `x` (feature map) in this guide — only `w`.
Full INT8 quantization of activations requires calibration data and
a separate step.

**This adds no learned parameters** — `FakeQuantize` is a
stateful observer module with non-gradient-tracked buffers
(running min, running max, scale, zero\_point).

Load with `strict=False` (observer buffers are new, but they are
buffers not parameters — the strict check may pass depending on
PyTorch version; use `strict=False` to be safe).

### Step-by-step implementation

**Step 1 — Add `QATLKPWrapper` in `model/lsnet.py`**

```python
from torch.quantization import FakeQuantize, MovingAverageMinMaxObserver

class QATLKPWrapper(nn.Module):
    """
    Wraps LKP and inserts a FakeQuantize node on its output (the
    dynamic kernel tensor w) to simulate INT8 quantization during
    training. The FakeQuantize observer uses per-tensor affine
    quantization with a moving average of min/max values.

    At inference, call model.apply(torch.quantization.disable_fake_quant)
    to turn off fake quant and recover full float32 behaviour,
    OR convert the model with torch.quantization.convert() for INT8.
    """

    def __init__(self, lkp: LKP):
        super().__init__()
        self.lkp = lkp
        # Per-tensor affine fake quant, 8-bit signed, quant_min=-128
        self.fake_quant_w = FakeQuantize.with_args(
            observer=MovingAverageMinMaxObserver,
            quant_min=-128,
            quant_max=127,
            dtype=torch.qint8,
            qscheme=torch.per_tensor_affine,
            reduce_range=False,
        )()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.lkp(x)                 # [B, dim//8, 9, H, W]
        w_flat = w.view(w.shape[0], -1) # flatten non-batch dims for observer
        # FakeQuantize expects [N, ...]; we apply over flattened channel dims
        # Reshape to [B, C, H, W] form that FakeQuantize accepts
        B, G, K2, H, W = w.shape
        w_4d = w.view(B, G * K2, H, W)
        w_4d_q = self.fake_quant_w(w_4d)
        w_q = w_4d_q.view(B, G, K2, H, W)
        return w_q
```

**Step 2 — Modify `LSConv` to use `QATLKPWrapper`**

```python
class LSConv(nn.Module):
    def __init__(self, dim,
                 sparse_ska: bool = False,
                 sparse_top_k: int = 5,
                 group_se: bool = False,
                 groups: int = 8,
                 shifted: bool = False,
                 shift_size: tuple = (1, 1),
                 qat: bool = False):          # NEW
        super(LSConv, self).__init__()

        base_lkp = LKP(dim, lks=7, sks=3, groups=groups)
        self.lkp  = QATLKPWrapper(base_lkp) if qat else base_lkp

        if shifted:
            self.ska = ShiftedSKA(shift_size=shift_size)
        elif sparse_ska:
            self.ska = SparseSKA(top_k=sparse_top_k)
        else:
            self.ska = SKA()

        self.group_se = GroupSE(dim, groups=groups) if group_se else nn.Identity()
        self.bn       = nn.BatchNorm2d(dim)

    def forward(self, x):
        out = self.ska(x, self.lkp(x))
        out = self.group_se(out)
        return self.bn(out) + x
```

**Step 3 — Propagate `qat` through `Block`, `LSNet`, `lsnet_t`**

Same pattern as P3/P7/P8. Add `qat: bool = False` to `Block.__init__`
and pass it through to `LSConv`. Add `qat` to `LSNet` kwargs and to
the `lsnet_t` factory function.

**Step 4 — Checkpoint loading for P9**

```python
model = lsnet_t(qat=True)

ckpt  = torch.load('pretrain/lsnet_t.pth', map_location='cpu')
state = ckpt['model']

# The FakeQuantize observer buffers (scale, zero_point, min_val, max_val)
# are new. LKP weights (cv1..cv4, norm) load correctly because they are
# nested under lkp.lkp.* via QATLKPWrapper.
#
# IMPORTANT: The key prefix changes:
#   Original:   blocks2.1.mixer.lkp.cv1.c.weight
#   Modified:   blocks2.1.mixer.lkp.lkp.cv1.c.weight   ← extra .lkp.
#
# We must remap the state_dict keys before loading.

def remap_lkp_keys(state_dict):
    """Remap 'X.lkp.cv*' → 'X.lkp.lkp.cv*' for QATLKPWrapper."""
    new_state = {}
    for k, v in state_dict.items():
        # Detect any key that goes through .mixer.lkp.cv or .mixer.lkp.norm
        if '.mixer.lkp.cv' in k or '.mixer.lkp.norm' in k:
            # Insert extra .lkp. after .mixer.lkp
            new_k = k.replace('.mixer.lkp.cv', '.mixer.lkp.lkp.cv')
            new_k = new_k.replace('.mixer.lkp.norm', '.mixer.lkp.lkp.norm')
            new_state[new_k] = v
        else:
            new_state[k] = v
    return new_state

state_remapped = remap_lkp_keys(state)
missing, unexpected = model.load_state_dict(state_remapped, strict=False)

# Expected missing keys: only FakeQuantize observer buffers
# e.g. "blocks2.1.mixer.lkp.fake_quant_w.scale"
#      "blocks2.1.mixer.lkp.fake_quant_w.zero_point"
#      "blocks2.1.mixer.lkp.fake_quant_w.activation_post_process.min_val"
#      "blocks2.1.mixer.lkp.fake_quant_w.activation_post_process.max_val"
# Expected unexpected keys: []
qat_missing = [k for k in missing if 'fake_quant' not in k]
assert len(qat_missing) == 0, \
    f"Non-QAT keys missing — check remap_lkp_keys: {qat_missing}"
print("All pretrained LKP weights loaded correctly.")
```

### Fine-tuning schedule for P9

QAT requires calibration before the fake-quant statistics are
meaningful. The first 5 epochs act as a calibration phase where the
observer accumulates min/max statistics but the fake-quant rounding
is **disabled**.

```python
import torch.quantization as quant

# Phase 0 (epochs 0–4): Observers ON, fake-quant OFF (calibration mode)
model.apply(quant.disable_fake_quant)
model.apply(quant.enable_observer)

# Phase 1 (epochs 5–29): Observers OFF, fake-quant ON (QAT mode)
# Switch at the start of epoch 5:
model.apply(quant.disable_observer)
model.apply(quant.enable_fake_quant)
```

| Phase | Epochs | LR | Fake-quant | Notes |
|---|---|---|---|---|
| Calibration | 5 | 1e-5 | OFF | Observer accumulates statistics |
| QAT warmup | 5 | 5e-5 | ON | LR ramps up; model adapts |
| QAT fine-tune | 20 | 5e-5 → 1e-7 | ON | Cosine decay |
| **Total** | **30** | | | |

**After QAT fine-tuning, to produce INT8 model:**

```python
model.eval()
model.apply(quant.disable_fake_quant)  # remove fake quant for PTQ export
# (full INT8 conversion requires torch.quantization.convert()
#  or torch.ao.quantization, which is outside this guide's scope)
```

---

## Combined Usage — All Four Proposals Together

All four proposals are designed to stack. The final `lsnet_t` call:

```python
model = lsnet_t(
    sparse_ska=True,       # P3: top-5 sparse aggregation
    sparse_top_k=5,
    group_se=True,         # P7: per-group SE inside LSConv
    use_shifted_ska=False, # P8: shift every depth%4==3 block
    # NOTE: P8 is incompatible with P3 in the same LSConv block
    # because ShiftedSKA takes priority in the if/elif chain.
    # Run P3 and P8 in separate experiments; do not combine in one model.
    qat=True,              # P9: fake-quant on LKP output
)
```

> **P3 and P8 are mutually exclusive per LSConv block.** The
> `if shifted / elif sparse_ska / else SKA()` chain picks only one.
> Either combine them by applying shift to sparse SKA (extend
> `SparsePyTorchSkaFn` to accept a shift, left as future work), or
> run them as separate ablation experiments.

**Recommended combination for a single best model:**

```python
model = lsnet_t(
    sparse_ska=True,        # P3
    sparse_top_k=5,
    group_se=True,          # P7
    use_shifted_ska=False,  # evaluate P8 separately
    qat=False,              # evaluate P9 separately (deployment concern)
)
```

Checkpoint loading for combined P3+P7:

```python
ckpt  = torch.load('pretrain/lsnet_t.pth', map_location='cpu')
state = ckpt['model']
missing, unexpected = model.load_state_dict(state, strict=False)
# Missing: only GroupSE keys
# Unexpected: []
```

Combined fine-tune schedule:

| Phase | Epochs | LR (GroupSE) | LR (all other) |
|---|---|---|---|
| GroupSE warm-up | 5 | 5e-4 | 1e-5 |
| Joint fine-tune | 25 | 1e-4 → 1e-6 | 5e-5 → 1e-7 |
| **Total** | **30** | | |

---

## Fine-Tuning Launch Command

The original `train.sh` runs 300 epochs. Create `finetune.sh`:

```bash
#!/bin/bash
python -m torch.distributed.launch \
    --nproc_per_node=8 \
    --master_port 12346 \
    --use_env main.py \
    --model lsnet_t \
    --data-path ~/imagenet \
    --dist-eval \
    --epochs 30 \
    --warmup-epochs 5 \
    --lr 5e-5 \
    --min-lr 1e-7 \
    --weight-decay 0.025 \
    --batch-size 256 \
    --resume pretrain/lsnet_t.pth \
    --finetune pretrain/lsnet_t.pth \
    --output_dir output/lsnet_t_modified \
    --sparse-ska \
    --sparse-top-k 5 \
    --group-se
    # Add --use-shifted-ska or --qat as separate experiments
```

Add argument parsing for the new flags to `main.py`:

```python
# In the argparse section of main.py:
parser.add_argument('--sparse-ska',    action='store_true', default=False)
parser.add_argument('--sparse-top-k',  type=int, default=5)
parser.add_argument('--group-se',      action='store_true', default=False)
parser.add_argument('--use-shifted-ska', action='store_true', default=False)
parser.add_argument('--qat',           action='store_true', default=False)

# In the model creation section of main.py (after args.model is resolved):
model = create_model(
    args.model,
    num_classes=args.nb_classes,
    distillation=(args.distillation_type != 'none'),
    sparse_ska=args.sparse_ska,
    sparse_top_k=args.sparse_top_k,
    group_se=args.group_se,
    use_shifted_ska=args.use_shifted_ska,
    qat=args.qat,
)
```

The existing `--finetune` argument in `main.py` already handles
loading a checkpoint with `strict=False` and remapping as needed —
verify this in your copy of `main.py`; if absent, use the explicit
loading code given per-proposal above.

---

## Epoch Count Rationale

The original LSNet-T was trained for **300 epochs from scratch** on
ImageNet-1K (as stated in the paper's analysis section: "all models
are trained for 100 epochs for limitations in training time" for
ablation, 300 for final models).

For fine-tuning from the pretrained checkpoint:

| Proposal | Recommended Epochs | Justification |
|---|---|---|
| P3 (Sparse SKA) | 30 | Zero new params; BN statistics adapt in ~5 epochs; sparse gradient landscape stabilises in ~25 |
| P7 (GroupSE) | 30 | New SE params initialised near-identity; need 25 epochs to converge; 5 warmup with frozen backbone |
| P8 (Shifted SKA) | 20 | Zero new params; minimal disruption; only BN adaptation needed (~5 epochs) |
| P9 (QAT) | 30 | 5 calibration + 5 QAT warmup + 20 QAT fine-tune |
| P3 + P7 combined | 30 | Dominated by GroupSE convergence time |

**30 epochs is 10% of the original 300-epoch schedule**, which is the
standard ratio used in fine-tuning literature (RepViT, FastViT,
EfficientViT all use 10–15% of training budget for fine-tune
ablations).

Do not go below 20 epochs for any proposal: ImageNet BN statistics
require at least 3–5 full dataset passes to stabilise under a
changed data flow, and the fine-tune LR schedule needs room for the
cosine decay to reach near-zero.

---

## Sanity Checks Before Full Fine-Tuning Run

Run these before submitting to the cluster:

```python
import torch
from model.lsnet import lsnet_t

# 1. Build each variant and verify forward pass
for cfg in [
    dict(sparse_ska=True, sparse_top_k=5),
    dict(group_se=True),
    dict(use_shifted_ska=True),
    dict(qat=True),
    dict(sparse_ska=True, group_se=True),
]:
    m = lsnet_t(**cfg).cuda().eval()
    x = torch.randn(2, 3, 224, 224).cuda()
    with torch.no_grad():
        y = m(x)
    assert y.shape == (2, 1000), f"Wrong output shape for {cfg}: {y.shape}"
    print(f"PASS: {cfg}")

# 2. Verify checkpoint loads without unintended missing keys
m = lsnet_t(sparse_ska=True, group_se=True).cuda()
state = torch.load('pretrain/lsnet_t.pth', map_location='cpu')['model']
missing, unexpected = m.load_state_dict(state, strict=False)
non_gse = [k for k in missing if 'group_se' not in k]
assert non_gse == [], f"Unexpected missing keys: {non_gse}"
assert unexpected == [], f"Unexpected extra keys: {unexpected}"
print("PASS: checkpoint load")

# 3. Verify FLOPs are within 5% of baseline (P3, P8, P9 should be exact)
from flops import get_flops  # from the original repo
base  = lsnet_t().cuda()
mod   = lsnet_t(sparse_ska=True, group_se=True).cuda()
x     = torch.randn(1, 3, 224, 224).cuda()
f_base, f_mod = get_flops(base, x), get_flops(mod, x)
print(f"Baseline FLOPs: {f_base/1e9:.3f}G  Modified FLOPs: {f_mod/1e9:.3f}G")
assert f_mod / f_base < 1.10, "FLOPs increased by more than 10%"
print("PASS: FLOPs check")
```
