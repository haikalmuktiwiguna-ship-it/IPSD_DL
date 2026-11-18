"""
Models/models_1.py
------------------
Model 1: SimpleCNN — arsitektur SAMA PERSIS dengan materi (Models/CNN.py).

Satu-satunya yang berbeda adalah nilai saat dipanggil:
    output = 6  (MSLD v2.0 punya 6 kelas, materi 5 kelas dino)

Ukuran input harus 100 x 100 (sama seperti getData.py materi), karena
layer fc ditulis tetap 50 * 9 * 9:
    100 -> Conv5x5 -> 96 -> MaxPool3 -> 32 -> Conv5x5 -> 28 -> MaxPool3 -> 9
"""

import torch
import torch.nn as nn

IMG_SIZE = 100   # ukuran resize gambar untuk model 1 (mengikuti materi)


class SimpleCNN(nn.Module):
    def __init__(self, input_dim=200, input_c=3, output=5, hidden_dim=128, dropout=0.5, device='cpu'):
        super(SimpleCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=20, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(3, 3)),
        )
        self.fc = nn.Linear(in_features=50 * 9 * 9, out_features=output)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)

        x = self.fc(torch.flatten(x, 1))
        x = self.logSoftmax(x)

        return x


if __name__ == "__main__":
    model = SimpleCNN(output=6)
    out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    print(model)
    print("Output shape :", out.shape)
    print("Total params :", sum(p.numel() for p in model.parameters()))
