from Models.models_1 import SimpleCNN, IMG_SIZE as IMG_SIZE_1
from Models.models_2 import ModifiedCNN, IMG_SIZE as IMG_SIZE_2
from Models.models_3 import ConvNeXtTiny, IMG_SIZE as IMG_SIZE_3, LR as LR_3

# daftar model yang bisa dipilih lewat argumen --model
MODELS = {
    1: {"name": "model_1", "class": SimpleCNN, "img_size": IMG_SIZE_1, "lr": 1e-3},
    2: {"name": "model_2", "class": ModifiedCNN, "img_size": IMG_SIZE_2, "lr": 1e-3},
    3: {"name": "model_3", "class": ConvNeXtTiny, "img_size": IMG_SIZE_3, "lr": LR_3},
}


def build_model(model_id, num_classes, img_size, for_training=True):
    cfg = MODELS[model_id]
    if model_id == 3:
        # saat test, bobot pretrained tidak perlu diunduh (diambil dari .pth)
        return cfg["class"](input_dim=img_size, output=num_classes, pretrained=for_training)
    return cfg["class"](input_dim=img_size, output=num_classes)