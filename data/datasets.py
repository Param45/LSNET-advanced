'''
Build trainining/testing datasets
'''
import os
import json

from torchvision import datasets, transforms
from torchvision.datasets.folder import ImageFolder, default_loader
import torch

from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform

try:
    from timm.data import TimmDatasetTar
except ImportError:
    # for higher version of timm
    from timm.data import ImageDataset as TimmDatasetTar

class INatDataset(ImageFolder):
    def __init__(self, root, train=True, year=2018, transform=None, target_transform=None,
                 category='name', loader=default_loader):
        self.transform = transform
        self.loader = loader
        self.target_transform = target_transform
        self.year = year
        # assert category in ['kingdom','phylum','class','order','supercategory','family','genus','name']
        path_json = os.path.join(
            root, f'{"train" if train else "val"}{year}.json')
        with open(path_json) as json_file:
            data = json.load(json_file)

        with open(os.path.join(root, 'categories.json')) as json_file:
            data_catg = json.load(json_file)

        path_json_for_targeter = os.path.join(root, f"train{year}.json")

        with open(path_json_for_targeter) as json_file:
            data_for_targeter = json.load(json_file)

        targeter = {}
        indexer = 0
        for elem in data_for_targeter['annotations']:
            king = []
            king.append(data_catg[int(elem['category_id'])][category])
            if king[0] not in targeter.keys():
                targeter[king[0]] = indexer
                indexer += 1
        self.nb_classes = len(targeter)

        self.samples = []
        for elem in data['images']:
            cut = elem['file_name'].split('/')
            target_current = int(cut[2])
            path_current = os.path.join(root, cut[0], cut[2], cut[3])

            categors = data_catg[target_current]
            target_current_true = targeter[categors[category]]
            self.samples.append((path_current, target_current_true))

    # __getitem__ and __len__ inherited from ImageFolder

from torch.utils.data import Dataset
from PIL import Image


class KaggleImageNetDataset(Dataset):
    def __init__(self, roots, transform=None):
        self.transform = transform
        self.samples = []

        all_classes = set()

        for root in roots:
            all_classes.update(os.listdir(root))

        classes = sorted(list(all_classes))

        self.class_to_idx = {
            cls_name: idx
            for idx, cls_name in enumerate(classes)
        }

        for root in roots:

            for cls_name in os.listdir(root):

                cls_dir = os.path.join(root, cls_name)

                label = self.class_to_idx[cls_name]

                for fname in os.listdir(cls_dir):

                    self.samples.append(
                        (
                            os.path.join(cls_dir, fname),
                            label
                        )
                    )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, target = self.samples[idx]

        img = None
        try:
            import cv2
            img_cv = cv2.imread(path, cv2.IMREAD_COLOR)
            if img_cv is not None:
                img_cv = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
                img = torch.from_numpy(img_cv).permute(2, 0, 1)  # uint8 tensor [C, H, W]
        except Exception:
            pass

        if img is None:
            img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, target


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == 'CIFAR':
        dataset = datasets.CIFAR100(
            args.data_path, train=is_train, transform=transform)
        nb_classes = 100
    # elif args.data_set == 'IMNET':
    #     prefix = 'train' if is_train else 'val'
    #     data_dir = os.path.join(args.data_path, f'{prefix}.tar')
    #     if os.path.exists(data_dir):
    #         dataset = TimmDatasetTar(data_dir, transform=transform)
    #     else:
    #         root = os.path.join(args.data_path, 'train' if is_train else 'val')
    #         dataset = datasets.ImageFolder(root, transform=transform)
    #     nb_classes = 1000
    elif args.data_set == 'IMNET':

        if "/kaggle/input/datasets/sautkin" in args.data_path:

            if is_train:

                dataset = KaggleImageNetDataset(
                    roots=[
                        "/kaggle/input/datasets/sautkin/imagenet1k0",
                        "/kaggle/input/datasets/sautkin/imagenet1k1",
                        "/kaggle/input/datasets/sautkin/imagenet1k2",
                        "/kaggle/input/datasets/sautkin/imagenet1k3",
                    ],
                    transform=transform
                )

            else:

                dataset = KaggleImageNetDataset(
                    roots=[
                        "/kaggle/input/datasets/sautkin/imagenet1kvalid"
                    ],
                    transform=transform
                )

        else:

            prefix = 'train' if is_train else 'val'
            data_dir = os.path.join(args.data_path, f'{prefix}.tar')

            if os.path.exists(data_dir):
                dataset = TimmDatasetTar(data_dir, transform=transform)
            else:
                root = os.path.join(
                    args.data_path,
                    'train' if is_train else 'val'
                )
                dataset = datasets.ImageFolder(
                    root,
                    transform=transform
                )

        nb_classes = 1000

    elif args.data_set == 'IMNETEE':
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 10
    elif args.data_set == 'FLOWERS':
        root = os.path.join(args.data_path, 'train' if is_train else 'test')
        dataset = datasets.ImageFolder(root, transform=transform)
        if is_train:
            dataset = torch.utils.data.ConcatDataset(
                [dataset for _ in range(100)])
        nb_classes = 102
    elif args.data_set == 'INAT':
        dataset = INatDataset(args.data_path, train=is_train, year=2018,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    elif args.data_set == 'INAT19':
        dataset = INatDataset(args.data_path, train=is_train, year=2019,
                              category=args.inat_category, transform=transform)
        nb_classes = dataset.nb_classes
    # Apply dataset fraction (training split only)
    fraction = getattr(args, 'dataset_fraction', 1.0)
    if is_train and fraction < 1.0:
        n_total = len(dataset)
        n_keep = max(1, int(n_total * fraction))
        # Fixed seed so the subset is reproducible across ranks
        generator = torch.Generator()
        generator.manual_seed(42)
        indices = torch.randperm(n_total, generator=generator)[:n_keep].tolist()
        dataset = torch.utils.data.Subset(dataset, indices)

    return dataset, nb_classes


class ToTensorIfNeeded:
    def __call__(self, pic):
        if isinstance(pic, torch.Tensor):
            if pic.dtype == torch.uint8:
                return pic.to(torch.float32).div(255.0)
            return pic
        import torchvision.transforms.functional as F
        return F.to_tensor(pic)


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if args.finetune:
        t.append(
            transforms.Resize((args.input_size, args.input_size),
                                interpolation=3)
        )
    else:
        if resize_im:
            size = int((256 / 224) * args.input_size)
            t.append(
                # to maintain same ratio w.r.t. 224 images
                transforms.Resize(size, interpolation=3),
            )
            t.append(transforms.CenterCrop(args.input_size))
    
    t.append(ToTensorIfNeeded())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
