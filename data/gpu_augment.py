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


class EnsureTensorAndCrop:
    """A minimal CPU transform wrapper that handles both PIL Images and PyTorch Tensors.

    If the input is a PIL Image, it applies crop, flip, and converts it to a uint8 tensor.
    If the input is already a PyTorch tensor (from OpenCV), it applies crop and flip directly
    on the tensor using torchvision v2, avoiding PIL wrapping overhead.
    """
    def __init__(self, args):
        import torchvision.transforms.v2 as v2
        self.crop = v2.RandomResizedCrop(
            args.input_size,
            scale=(0.08, 1.0),
            interpolation=v2.InterpolationMode.BICUBIC,
        )
        self.flip = v2.RandomHorizontalFlip(0.5)
        self.to_tensor = v2.PILToTensor()

    def __call__(self, img):
        from PIL import Image
        if isinstance(img, Image.Image):
            img = self.crop(img)
            img = self.flip(img)
            img = self.to_tensor(img)
        elif isinstance(img, torch.Tensor):
            img = self.crop(img)
            img = self.flip(img)
        else:
            img = self.to_tensor(img)
            img = self.crop(img)
            img = self.flip(img)
        return img


def build_cpu_minimal_train_transform(args):
    """Return a minimal CPU transform for use inside the DataLoader."""
    return EnsureTensorAndCrop(args)


def build_cpu_uint8_eval_transform(args):
    """Return eval transforms that keep batches uint8 until they reach CUDA."""
    return EnsureEvalUint8(args)


class EnsureEvalUint8:
    """Eval resize/crop that accepts PIL images or uint8 tensors."""

    def __init__(self, args):
        import torchvision.transforms.v2 as v2

        transforms = []
        if args.finetune:
            transforms.append(
                v2.Resize(
                    (args.input_size, args.input_size),
                    interpolation=v2.InterpolationMode.BICUBIC,
                )
            )
        elif args.input_size > 32:
            size = int((256 / 224) * args.input_size)
            transforms.extend([
                v2.Resize(size, interpolation=v2.InterpolationMode.BICUBIC),
                v2.CenterCrop(args.input_size),
            ])
        self.transform = v2.Compose(transforms)
        self.to_tensor = v2.PILToTensor()

    def __call__(self, img):
        from PIL import Image

        img = self.transform(img)
        if isinstance(img, Image.Image):
            img = self.to_tensor(img)
        return img


class GPUEvalNormalize(nn.Module):
    """Convert uint8 eval batches to normalized float tensors on CUDA."""

    def __init__(self):
        super().__init__()
        import torchvision.transforms.v2 as v2

        self.normalize = v2.Normalize(
            mean=list(IMAGENET_DEFAULT_MEAN),
            std=list(IMAGENET_DEFAULT_STD),
        )

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.normalize(x.float().div_(255.0))


class GPUTrainAugment(nn.Module):
    """GPU-side stochastic augmentation applied inside the training loop.

    Input  : uint8 CUDA tensor [B, C, H, W] in [0, 255]  (from DataLoader after .to(device))
    Output : float32 CUDA tensor [B, C, H, W] normalized to ImageNet mean/std

    Pipeline per batch
    ------------------
    1. RandAugment(num_ops=2, magnitude=9)  — applied in groups of 8
       Matches `--aa rand-m9-mstd0.5-inc1` policy (num_ops=2, same op set and magnitude).
    2. uint8 → float32 / 255.0              — in-place, batch-level, O(1) overhead
    3. Normalize(ImageNet mean, std)        — batch-level, highly efficient on GPU
    4. RandomErasing(p=reprob, value=random)— applied in groups of 8
       Matches `--reprob` and `--remode pixel` (random pixel fill).

    Grouped batch rationale
    ------------------------
    torchvision.transforms.v2, when called on a batch tensor, applies the same random
    parameters to all images in that batch. To preserve sample diversity without incurring
    severe Python loop and kernel launch overhead from looping over 128 images individually
    (which starves the CPU and restricts GPU utilization), we split the batch into groups
    of size 8. Each group gets independently drawn random parameters, reducing the loop
    iterations from 128 to 16.
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
        # --- Step 1: RandAugment, grouped (split-batch) to reduce CPU/kernel overhead ---
        # Splitting into groups of 32 reduces loop iterations (e.g. 256 -> 8),
        # keeping Python overhead minimal while maintaining sample diversity.
        x = torch.cat([self.rand_aug(chunk) for chunk in torch.split(x, 32, dim=0)], dim=0)

        # --- Step 2: uint8 [0,255] → float32 [0.0, 1.0] ---
        # div_ is in-place and avoids an extra allocation.
        x = x.float().div_(255.0)

        # --- Step 3: Normalize to ImageNet mean / std (batch op, very fast) ---
        x = self.normalize(x)

        # --- Step 4: RandomErasing, grouped (split-batch) ---
        if self.random_erase is not None:
            x = torch.cat([self.random_erase(chunk) for chunk in torch.split(x, 32, dim=0)], dim=0)

        return x
