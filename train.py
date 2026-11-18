import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from Models import MODELS, build_model
from Utils.Getdata import Getdata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "archive")


class Trainer:
    def __init__(self, model, train_loader, valid_loader, device,
                 lr=1e-3, weight_decay=1e-4, class_weights=None):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.device = device
        if class_weights is not None:
            class_weights = class_weights.to(device)
        # model mengeluarkan LogSoftmax (sesuai materi) -> pasangannya NLLLoss
        self.criterion = nn.NLLLoss(weight=class_weights)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                          weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="min", factor=0.5, patience=3)
        self.history = []

    def train_one_epoch(self):
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in self.train_loader:
            images = images.to(self.device)
            labels = labels.argmax(1).to(self.device)   # one-hot -> indeks kelas

            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    @torch.no_grad()
    def evaluate(self, loader):
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.argmax(1).to(self.device)   # one-hot -> indeks kelas
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
        return total_loss / total, correct / total

    def fit(self, epochs, save_path, patience=7, extra_info=None):
        best_acc, wait = 0.0, 0
        for epoch in range(1, epochs + 1):
            start = time.time()
            train_loss, train_acc = self.train_one_epoch()
            val_loss, val_acc = self.evaluate(self.valid_loader)
            self.scheduler.step(val_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            self.history.append({"epoch": epoch, "train_loss": train_loss,
                                 "train_acc": train_acc, "val_loss": val_loss,
                                 "val_acc": val_acc, "lr": lr})
            print(f"Epoch {epoch:>3}/{epochs} | "
                  f"train loss {train_loss:.4f} acc {train_acc:.4f} | "
                  f"val loss {val_loss:.4f} acc {val_acc:.4f} | "
                  f"lr {lr:.1e} | {time.time() - start:.1f}s")

            if val_acc > best_acc:
                best_acc, wait = val_acc, 0
                torch.save({"model_state": self.model.state_dict(),
                            "epoch": epoch, "val_acc": val_acc,
                            **(extra_info or {})}, save_path)
                print(f"   -> model terbaik disimpan ({save_path})")
            else:
                wait += 1
                if wait >= patience:
                    print(f"Early stopping: val acc tidak naik {patience} epoch.")
                    break
        print(f"Selesai. Val acc terbaik = {best_acc:.4f}")
        return self.history

    def save_history(self, out_dir):
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "history.csv"), "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.history[0].keys())
            writer.writeheader()
            writer.writerows(self.history)

        epochs = [h["epoch"] for h in self.history]
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        ax[0].plot(epochs, [h["train_loss"] for h in self.history], label="train")
        ax[0].plot(epochs, [h["val_loss"] for h in self.history], label="valid")
        ax[0].set_title("Loss"); ax[0].set_xlabel("Epoch"); ax[0].legend()
        ax[1].plot(epochs, [h["train_acc"] for h in self.history], label="train")
        ax[1].plot(epochs, [h["val_acc"] for h in self.history], label="valid")
        ax[1].set_title("Accuracy"); ax[1].set_xlabel("Epoch"); ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "training_curve.png"), dpi=150)
        plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description="Training CNN MSLD v2.0")
    p.add_argument("--model", type=int, choices=[1, 2, 3], default=1)
    p.add_argument("--data", default=DEFAULT_DATA, help="folder archive dataset")
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all-folds", action="store_true",
                   help="latih di fold 1 sampai 5 berurutan (5-fold cross validation)")
    p.add_argument("--original-only", action="store_true",
                   help="training hanya dengan Train original (default: original + augmented)")
    p.add_argument("--no-clahe", action="store_true",
                   help="matikan pre-processing CLAHE (default: CLAHE aktif)")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=None,
                   help="default: 1e-3 untuk model 1 & 2, 1e-4 untuk model 3")
    p.add_argument("--patience", type=int, default=7)
    p.add_argument("--weighted-loss", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def train_one_fold(args, fold, device):
    torch.manual_seed(args.seed)
    cfg = MODELS[args.model]
    print("\n" + "#" * 60)
    print(f"# {cfg['name']} | FOLD {fold}")
    print("#" * 60)

    data = Getdata(root=args.data, img_size=cfg["img_size"],
                   batch_size=args.batch_size, fold=fold,
                   use_augmented=not args.original_only,
                   clahe=not args.no_clahe, seed=args.seed)
    train_loader, valid_loader, _ = data.get_loaders()
    data.summary()

    model = build_model(args.model, len(data.classes), cfg["img_size"])
    print("Total parameter:", f"{sum(p.numel() for p in model.parameters()):,}")

    lr = args.lr if args.lr is not None else cfg["lr"]
    print("Learning rate:", lr)
    trainer = Trainer(model, train_loader, valid_loader, device, lr=lr,
                      class_weights=data.class_weights() if args.weighted_loss else None)

    # file per fold: Models/model_1_fold1_best.pth, Results/model_1/fold1/
    save_path = os.path.join(BASE_DIR, "Models", f"{cfg['name']}_fold{fold}_best.pth")
    trainer.fit(args.epochs, save_path, patience=args.patience,
                extra_info={"classes": data.classes, "img_size": cfg["img_size"],
                            "model_id": args.model, "fold": fold,
                            "clahe": not args.no_clahe})
    out_dir = os.path.join(BASE_DIR, "Results", cfg["name"], f"fold{fold}")
    trainer.save_history(out_dir)
    print("Riwayat training disimpan di", out_dir)


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg = MODELS[args.model]
    print(f"Device: {device} | Model: {cfg['name']} | input {cfg['img_size']}x{cfg['img_size']}")

    folds = [1, 2, 3, 4, 5] if args.all_folds else [args.fold]
    for fold in folds:
        train_one_fold(args, fold, device)


if __name__ == "__main__":
    main()
