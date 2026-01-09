import torch
import torchvision.models as models
from torchvision import transforms
import numpy as np
from PIL import Image

# Устройство
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Глобальный трансформ для всех изображений
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def load_model(model_type="resnet50", num_classes=11):
    """
    Загружает модель DeepLabV3 с заданной архитектурой и количеством классов.
    """
    if model_type == "resnet50":
        model = models.segmentation.deeplabv3_resnet50(
            weights=None, pretrained_backbone=False, aux_loss=False
        )
        weights = "models/DeepLabV3_ResNet50_best.pth"
    else:
        model = models.segmentation.deeplabv3_mobilenet_v3_large(
            weights=None, pretrained_backbone=False, aux_loss=False
        )
        weights = "models/DeepLabV3_MobileNetV3_best.pth"

    # Меняем классификатор под нужное количество классов
    model.classifier[4] = torch.nn.Conv2d(256, num_classes, 1)

    # Загружаем веса, пропуская несоответствующие ключи
    state_dict = torch.load(weights, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)

    model.to(DEVICE)
    model.eval()
    return model

def predict(image: Image.Image, model):
    """
    Делает предсказание маски для одного изображения PIL.Image.
    Возвращает numpy-массив с индексами классов [H, W].
    """
    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        out = model(x)["out"]
        mask = out.argmax(1).squeeze().cpu().numpy()

    return mask