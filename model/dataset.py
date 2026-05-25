import medmnist
from medmnist import INFO
from torchvision import transforms
from torch.utils.data import DataLoader, random_split


def get_transforms(image_size: int) -> dict:
    """Returns train and val transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    return {"train": train_transforms, "val": val_transforms}


def get_dataloaders(config: dict) -> tuple[DataLoader, DataLoader, int]:
    """
    Downloads PathMNIST and returns train/val dataloaders.
    Returns: (train_loader, val_loader, num_classes)
    """
    dataset_name = config["dataset"]["name"]
    image_size = config["dataset"]["image_size"]
    batch_size = config["dataset"]["batch_size"]
    num_workers = config["dataset"]["num_workers"]

    info = INFO[dataset_name]
    num_classes = len(info["label"])
    DataClass = getattr(medmnist, info["python_class"])

    tf = get_transforms(image_size)

    train_dataset = DataClass(split="train", transform=tf["train"], download=True)
    val_dataset = DataClass(split="val", transform=tf["val"], download=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return train_loader, val_loader, num_classes