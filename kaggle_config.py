import os

# Centralized Kaggle Configuration
KAGGLE_DATASET_ROOT = os.environ.get(
    "KAGGLE_DATASET_ROOT",
    "/kaggle/input"
)

WORK_DIR = "/kaggle/working"
CHECKPOINT_DIR = "/kaggle/working/checkpoints"
LOG_DIR = "/kaggle/working/logs"

def get_dataset_path(default_name, search_keywords=None):
    if not os.path.exists(KAGGLE_DATASET_ROOT):
        return os.path.join(KAGGLE_DATASET_ROOT, default_name)

    exact_path = os.path.join(KAGGLE_DATASET_ROOT, default_name)

    if os.path.exists(exact_path):
        return exact_path

    if search_keywords:
        try:
            for item in os.listdir(KAGGLE_DATASET_ROOT):
                item_path = os.path.join(KAGGLE_DATASET_ROOT, item)

                if os.path.isdir(item_path):
                    for keyword in search_keywords:
                        if keyword.lower() in item.lower():
                            return item_path
        except Exception:
            pass

    return exact_path


IMAGENET_PATH = "/kaggle/input/datasets/sautkin"

COCO_PATH = get_dataset_path("coco", ["coco"])
ADE20K_PATH = get_dataset_path("ade20k", ["ade20k", "ade"])
CIFAR_PATH = get_dataset_path("cifar100", ["cifar", "cifar100"])

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)