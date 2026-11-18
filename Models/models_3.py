import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights

IMG_SIZE = 224   # ukuran resize gambar untuk model 3
LR = 1e-4        # learning rate fine-tuning (lebih kecil dari model 1 & 2)


class ConvNeXtTiny(nn.Module):
    def __init__(self, input_dim=224, input_c=3, output=6, dropout=0.3, pretrained=True, device='cpu'):
        super(ConvNeXtTiny, self).__init__()

        if pretrained:
            try:
                self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            except Exception as e:
                raise RuntimeError(
                    "Gagal mengunduh bobot pretrained ConvNeXt-Tiny. "
                    "Pastikan laptop terhubung internet, lalu jalankan ulang.") from e
        else:
            # dipakai test.py: arsitektur saja, bobot diambil dari model_3_best.pth
            self.backbone = convnext_tiny(weights=None)

        # ganti layer klasifikasi terakhir: 1000 kelas ImageNet -> jumlah kelas MSLD
        in_features = self.backbone.classifier[2].in_features   # 768
        self.backbone.classifier[2] = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, output),
        )
        self.logSoftmax = nn.LogSoftmax(dim=1)

        # normalisasi ImageNet
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = x.flip(1)                        # BGR (cv2) -> RGB
        x = (x - self.mean) / self.std       # normalisasi ImageNet
        x = self.backbone(x)
        x = self.logSoftmax(x)
        return x


if __name__ == "__main__":
    model = ConvNeXtTiny(output=6, pretrained=False)
    out = model(torch.rand(2, 3, IMG_SIZE, IMG_SIZE))
    print("Output shape :", out.shape)
    print("Total params :", sum(p.numel() for p in model.parameters()))