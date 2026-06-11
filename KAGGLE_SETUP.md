# Kaggle Setup and Execution Guide

This document describes how to set up, attach datasets, and run training, evaluation, and inference inside a Kaggle notebook environment.

---

## 1. Required Pip Installs

To run all modules (Classification, Detection, Segmentation) successfully, install the following libraries at the start of your notebook:

```bash
!pip install timm wandb fvcore
```

For Object Detection (MMDetection) and Semantic Segmentation (MMSegmentation), install MMCV:

```bash
!pip install -U openmim
!mim install mmcv-full=="1.7.2"
!pip install mmdet=="2.28.2"
!pip install mmseg=="0.30.0"
```

---

## 2. Dataset Attachment Instructions

Under the **"Data"** panel in the Kaggle Notebook sidebar:
1. Click **"Add Data"**
2. Search for the relevant datasets:
   * For classification: **ImageNet** or **CIFAR-100**
   * For detection: **COCO 2017**
   * For segmentation: **ADE20K**
3. Select and add them. The datasets will be mounted under `/kaggle/input/<dataset-name>`.

Our dynamic configuration `kaggle_config.py` automatically detects variations in attachment names (e.g. `coco-2017-dataset` instead of `coco`).

---

## 3. Expected Kaggle Folder Structure

```text
/kaggle
├── input/
│   ├── imagenet/           # ImageNet training/validation tar files or directories
│   ├── coco-2017-dataset/  # COCO images and annotations
│   └── ade20k-dataset/     # ADEChallengeData2016 directory
└── working/                # Writable workspace
    └── lsnet-master/       # Repository root containing this code
        ├── kaggle_run.py   # Main runner script
        ├── main.py         # Classification entry point
        ├── detection/      # MMDetection models & configs
        └── segmentation/   # MMSegmentation tools & configs
```

---

## 4. Execution Commands via `kaggle_run.py`

You can run all operations from a notebook cell using `kaggle_run.py`. This script automatically manages search paths (`PYTHONPATH`), loads mappings from `kaggle_config.py`, and streams logs in real-time.

### A. Image Classification (timm / main.py)

#### Training (ImageNet)
```bash
!python kaggle_run.py --action train_clf --model lsnet_t --extra-args "--data-set IMNET"
```

#### Evaluation (ImageNet)
```bash
!python kaggle_run.py --action eval_clf --model lsnet_t --checkpoint /kaggle/input/lsnet-checkpoints/lsnet_t.pth
```

#### Robustness Evaluation (ImageNet-A, ImageNet-R, ImageNet-Sketch, ImageNet-C)
```bash
!python kaggle_run.py --action robust_clf --model lsnet_t --checkpoint /kaggle/input/lsnet-checkpoints/lsnet_t.pth
```

---

### B. Object Detection (MMDetection)

#### Training
```bash
!python kaggle_run.py --action train_det --config detection/configs/retinanet_lsnet_t_fpn_1x_coco.py
```

#### Evaluation
```bash
!python kaggle_run.py --action test_det --config detection/configs/retinanet_lsnet_t_fpn_1x_coco.py --checkpoint /kaggle/working/work_dirs/retinanet_lsnet_t_fpn_1x_coco/epoch_12.pth --extra-args "--eval bbox"
```

---

### C. Semantic Segmentation (MMSegmentation)

#### Training
```bash
!python kaggle_run.py --action train_seg --config segmentation/configs/sem_fpn/fpn_lsnet_t_ade20k_40k.py
```

#### Evaluation
```bash
!python kaggle_run.py --action test_seg --config segmentation/configs/sem_fpn/fpn_lsnet_t_ade20k_40k.py --checkpoint /kaggle/working/work_dirs/fpn_lsnet_t_ade20k_40k/iter_40000.pth --extra-args "--eval mIoU"
```

---

## Notes on Outputs
All logs, checkpoints, and visualization outputs are automatically directed under `/kaggle/working` (e.g. `/kaggle/working/checkpoints` or `/kaggle/working/work_dirs`) so that they are saved and can be committed or downloaded.
