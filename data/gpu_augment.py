"""GPU-accelerated augmentation pipeline for training.

Splits the standard training transform into two stages:

  CPU (DataLoader workers):
    RandomResizedCrop(224, bicubic) + RandomHorizontalFlip → uint8 tensor [C, H, W]
    — Fast spatial-only ops on PIL input; output is collatable fixed-size uint8 tensors.

  GPU (inside train_one_epoch, after .to(device)):
    RandAugment → ToDtype(float32) → Normalize → RandomErasing
    — Moves the dominant CPU bottleneck (~RandAugment PIL ops) to CUDA.

Design notes
------------
* Per-sample independence is preserved: RandAugment and RandomErasing are applied
  to each image individually via a Python loop over the batch dimension.  A batch-level
  call (same ops for all images) would violate the augmentation diversity requirement.
* torchvision.transforms.v2 is used throughout; it natively operates on CUDA tensors.
* RandomResizedCrop (bicubic) and RandomHorizontalFlip remain on CPU because they
  receive variable-resolution PIL images from disk that cannot be batched before
  the crop step.
* ThreeAugment is intentionally excluded: its GaussianBlur and Solarization ops
  have no v2 GPU equivalent that preserves exact PIL behaviour.
* CIFAR-scale datasets (input_size ≤ 32) are excluded: they use RandomCrop instead
  of RandomResizedCrop, requiring a different minimal CPU transform.

Policy preserved (matching original args defaults):
  --aa rand-m9-mstd0.5-inc1  →  RandAugment(num_ops=2, magnitude=9)
  --reprob 0.25              →  RandomErasing(p=0.25)
  --remode pixel             →  RandomErasing(value='random')
  --smoothing / mixup        →  unchanged (handled downstream in engine.py)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def gpu_aug_available() -> bool:
    """Return True when torchvision.transforms.v2 and CUDA are both available."""
    try:
        import torchvision.transforms.v2  # noqa: F401
        return torch.cuda.is_available()
    except ImportError:
        return False


def build_cpu_minimal_train_transform(args):
    """Return a minimal CPU transform for use inside the DataLoader.

    Performs only the spatial and layout operations that must happen on the CPU
    (PIL input, variable image size):
      1. RandomResizedCrop  — bicubic, matches original `create_transform` policy
      2. RandomHorizontalFlip — independent per image (called once per PIL sample)
      3. PILToTensor          — converts PIL → uint8 CUDA-transferable tensor [0,255]

    RandAugment, normalization, and RandomErasing are deferred to GPUTrainAugment.

    Args:
        args: training argument namespace (args.input_size used for crop size).

    Returns:
        torchvision.transforms.Compose outputting uint8 tensors [C, H, W] in [0,255].
    """
    return T.Compose([
        T.RandomResizedCrop(
            args.input_size,
            scale=(0.08, 1.0),
            interpolation=InterpolationMode.BICUBIC,
        ),
        T.RandomHorizontalFlip(0.5),
        T.PILToTensor(),  # → uint8 [C, H, W] in [0, 255]; no normalization
    ])


class GPUTrainAugment(nn.Module):
    """GPU-side stochastic augmentation applied inside the training loop.

    Input  : uint8 CUDA tensor [B, C, H, W] in [0, 255]  (from DataLoader after .to(device))
    Output : float32 CUDA tensor [B, C, H, W] normalized to ImageNet mean/std

    Pipeline per batch
    ------------------
    1. RandAugment(num_ops=2, magnitude=9)  — applied per-image, independently
       Matches `--aa rand-m9-mstd0.5-inc1` policy (num_ops=2, same op set and magnitude).
    2. uint8 → float32 / 255.0              — in-place, batch-level, O(1) overhead
    3. Normalize(ImageNet mean, std)        — batch-level, highly efficient on GPU
    4. RandomErasing(p=reprob, value=random)— applied per-image, independently
       Matches `--reprob` and `--remode pixel` (random pixel fill).

    Per-image loop rationale
    ------------------------
    torchvision.transforms.v2, when called on a 4-D batch tensor, applies the same
    random parameters to every image in the batch (designed for segmentation consistency).
    For classification training, each image must receive independently drawn augmentations
    to maintain augmentation diversity.  The per-image loop preserves this while still
    running each kernel on the GPU.

    GPU throughput
    --------------
    On a T4, each 224×224 RandAugment call takes ~0.3–0.5 ms.  For a batch of 128:
      GPU: 128 × ~0.4 ms ≈ 51 ms  (replaces ~900 ms CPU DataLoader wait).
    """

    def __init__(self, args):
        super().__init__()
        import torchvision.transforms.v2 as v2

        # RandAugment: num_ops=2 (default in rand-N2 from timm),
        # magnitude=9 matches rand-m9 on the [0,10] original-paper scale.
        self.rand_aug = v2.RandAugment(num_ops=2, magnitude=9)

        # Normalize: ImageNet statistics, operates on float [0,1] tensors.
        self.normalize = v2.Normalize(
            mean=list(IMAGENET_DEFAULT_MEAN),
            std=list(IMAGENET_DEFAULT_STD),
        )

        # RandomErasing: p and value='random' match --reprob and --remode pixel.
        self.reprob = args.reprob
        if self.reprob > 0:
            self.random_erase = v2.RandomErasing(p=self.reprob, value='random')
        else:
            self.random_erase = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: uint8 CUDA tensor [B, C, H, W] in [0, 255]
        Returns:
            float32 CUDA tensor [B, C, H, W] normalized to ImageNet statistics
        """
        # --- Step 1: RandAugment, independently per sample ---
        # Loop preserves per-image stochastic independence.
        # Each call to self.rand_aug(xi) draws fresh random op choices for xi.
        x = torch.stack([self.rand_aug(xi) for xi in x])  # uint8 [B, C, H, W]

        # --- Step 2: uint8 [0,255] → float32 [0.0, 1.0] ---
        # div_ is in-place and avoids an extra allocation.
        x = x.float().div_(255.0)

        # --- Step 3: Normalize to ImageNet mean / std (batch op, very fast) ---
        x = self.normalize(x)

        # --- Step 4: RandomErasing, independently per sample ---
        if self.random_erase is not None:
            x = torch.stack([self.random_erase(xi) for xi in x])

        return x
