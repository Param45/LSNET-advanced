import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

from losses import DistillationLoss
import utils

class CudaPrefetcher:
    """Move the next batch to CUDA while the current batch is training."""

    def __init__(self, loader: Iterable, device: torch.device, gpu_transform=None):
        self.loader = loader
        self.device = device
        self.gpu_transform = gpu_transform

    def __len__(self):
        return len(self.loader)

    def __iter__(self):
        stream = torch.cuda.Stream(device=self.device)
        iterator = iter(self.loader)
        next_samples = None
        next_targets = None

        def preload():
            try:
                samples, targets = next(iterator)
            except StopIteration:
                return None, None

            with torch.cuda.stream(stream):
                samples = samples.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                if self.gpu_transform is not None:
                    samples = self.gpu_transform(samples)
            return samples, targets

        next_samples, next_targets = preload()
        while next_samples is not None:
            torch.cuda.current_stream(self.device).wait_stream(stream)
            samples, targets = next_samples, next_targets
            next_samples, next_targets = preload()
            yield samples, targets


def set_bn_state(model):
    for m in model.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()

def train_one_epoch(model: torch.nn.Module, criterion: DistillationLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler,
                    clip_grad: float = 0,
                    clip_mode: str = 'norm',
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True,
                    set_bn_eval=False,
                    gpu_transform=None):
    model.train(set_training_mode)
    if set_bn_eval:
        set_bn_state(model)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(
        window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 100
    iterable = data_loader
    if device.type == "cuda":
        iterable = CudaPrefetcher(data_loader, device, gpu_transform)

    num_steps = len(iterable)
    for step, (samples, targets) in enumerate(metric_logger.log_every(
            iterable, print_freq, header)):
        if device.type != "cuda":
            samples = samples.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # GPU augmentation is only available on CUDA; keep this fallback for
            # non-CUDA smoke tests.
            if gpu_transform is not None:
                samples = gpu_transform(samples)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type="cuda"):
            outputs = model(samples)
            loss = criterion(samples, outputs, targets)

        optimizer.zero_grad(set_to_none=True)

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(
            optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=clip_grad, clip_mode=clip_mode,
                    parameters=model.parameters(), create_graph=is_second_order)

        if model_ema is not None:
            model_ema.update(model)

        if step % print_freq == 0 or step == num_steps - 1:
            loss_value = loss.item()
            if not math.isfinite(loss_value):
                print("Loss is {}, stopping training".format(loss_value))
                sys.exit(1)
            metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.amp.autocast(enabled=True, dtype=torch.float16, device_type="cuda"):
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
