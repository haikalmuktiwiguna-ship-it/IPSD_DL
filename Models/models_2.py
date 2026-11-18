"""
Models/models_2.py
------------------
Model 2: ModifiedCNN — hasil MODIFIKASI dari SimpleCNN materi (Model 1).

Perbandingan dengan Model 1:
| Bagian            | Model 1 (materi)         | Model 2 (modifikasi)                     |
|-------------------|--------------------------|------------------------------------------|
| Ukuran input      | 100 x 100                | 128 x 128                                |
| Aktivasi          | ReLU                     | LeakyReLU(0.1)                           |
| Kernel conv       | 5x5, tanpa padding       | 3x3, padding 1                           |
| Jumlah blok conv  | 2 blok (20, 50 filter)   | 3 blok (32, 64, 128 filter)              |
| Normalisasi       | -                        | BatchNorm2d di tiap blok                 |
| Pooling           | MaxPool 3x3              | MaxPool 2x2                              |
| Classifier        | Linear langsung ke kelas | Linear(hidden_dim) -> LeakyReLU          |
|                   |                          | -> Dropout(0.5) -> Linear ke kelas       |
| Output            | LogSoftmax               | LogSoftmax (sama)                        |

Catatan: di materi, parameter hidden_dim dan dropout sudah ada di __init__
tapi tidak dipakai. Di Model 2 keduanya benar-benar dipakai.

Alur ukuran: 128 -> pool -> 64 -> pool -> 32 -> pool -> 16  => 128 * 16 * 16
"""

import torch
import torch.nn as nn

IMG_SIZE = 128   # ukuran resize gambar untuk model 2


class ModifiedCNN(nn.Module):
    def __init__(self, input_dim=128, input_c=3, output=6, hidden_dim=128, dropout=0.5, device='cpu'):
        super(ModifiedCNN, self).__init__()

        self.convBlock1 = nn.Sequential(
            nn.Conv2d(in_channels=input_c, out_channels=32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        self.convBlock2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        # blok tambahan (tidak ada di materi)
        self.convBlock3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=(2, 2)),
        )

        flat = 128 * (input_dim // 8) * (input_dim // 8)    # 128 * 16 * 16
        self.fc1 = nn.Linear(in_features=flat, out_features=hidden_dim)
        self.act = nn.LeakyReLU(0.1)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output)
        self.logSoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):
        x = self.convBlock1(x)
        x = self.convBlock2(x)
        x = self.convBlock3(x)

        x = torch.flatten(x, 1)
        x = self.dropout(self.act(self.fc1(x)))
        x = self.fc2(x)
        x = self.logSoftmax(x)

        return x


if __name__ == "__main__":
    model = ModifiedCNN(output=6)
    out = model(torch.randn(2, 3, IMG_SIZE, IMG_SIZE))
    print(model)
    print("Output shape :", out.shape)
    print("Total params :", sum(p.numel() for p in model.parameters()))
