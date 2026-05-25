from PIL import Image
from torchvision import transforms
import io


def get_transforms(image_size: int = 64):
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def preprocess_image(image_bytes: bytes, image_size: int = 64):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    transform = get_transforms(image_size)
    tensor = transform(image)
    return tensor.unsqueeze(0)  # add batch dimension → (1, C, H, W)