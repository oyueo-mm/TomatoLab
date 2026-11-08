"""Dataset loading and identical preprocessing conventions."""
from pathlib import Path

from torchvision import datasets, transforms

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
IMAGE_SIZE = 224


def image_transform(training: bool = False):
    common = [transforms.Resize((256, 256))]
    if training:
        common.extend([
            transforms.RandomResizedCrop(IMAGE_SIZE, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
    else:
        common.append(transforms.CenterCrop(IMAGE_SIZE))
    common.extend([transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    return transforms.Compose(common)


def load_split(data_dir: str, split: str, training: bool = False):
    path = Path(data_dir) / split
    if not path.is_dir():
        raise FileNotFoundError(f"Missing dataset split: {path}. Run scripts/prepare_data.py first.")
    return datasets.ImageFolder(str(path), transform=image_transform(training=training))


def check_same_classes(*datasets_to_compare):
    reference = datasets_to_compare[0].classes
    for dataset in datasets_to_compare[1:]:
        if dataset.classes != reference:
            raise ValueError("Dataset splits have different class names or class order.")
    return reference
