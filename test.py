import argparse
import csv
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, ConfusionMatrixDisplay,
                             precision_recall_fscore_support, roc_auc_score, roc_curve)
from sklearn.preprocessing import label_binarize

from Models import MODELS, build_model
from Utils.Getdata import Getdata

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(BASE_DIR, "archive")
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_macro"]


def checkpoint_path(model_name, fold):
    """Cari file bobot untuk fold tertentu."""
    path = os.path.join(BASE_DIR, "Models", f"{model_name}_fold{fold}_best.pth")
    old = os.path.join(BASE_DIR, "Models", f"{model_name}_best.pth")   # format lama (fold 1)
    if os.path.exists(path):
        return path
    if fold == 1 and os.path.exists(old):
        return old
    return path


def plot_roc(y_true, probs, classes, title, path):
    """Kurva ROC one-vs-rest untuk tiap kelas."""
    y_bin = label_binarize(y_true, classes=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(7, 6))
    for i, cls in enumerate(classes):
        if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        auc = roc_auc_score(y_bin[:, i], probs[:, i])
        ax.plot(fpr, tpr, label=f"{cls} (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title(title); ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def macro_roc_auc(y_true, probs, n_classes):
    """ROC-AUC macro one-vs-rest (kelas yang tidak ada di test dilewati)."""
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    aucs = [roc_auc_score(y_bin[:, i], probs[:, i])
            for i in range(n_classes) if 0 < y_bin[:, i].sum() < len(y_true)]
    return float(np.mean(aucs)) if aucs else float("nan")


def plot_cm(cm, classes, title, path):
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(cm, display_labels=classes).plot(
        ax=ax, cmap="Blues", xticks_rotation=45, colorbar=False)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


class Tester:
    def __init__(self, model_id, data_root, fold=1, batch_size=32, device=None, assume_clahe=True):
        self.cfg = MODELS[model_id]
        self.fold = fold
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.ckpt_path = checkpoint_path(self.cfg["name"], fold)
        if not os.path.exists(self.ckpt_path):
            raise FileNotFoundError(f"Belum ada model terlatih: {self.ckpt_path}. "
                                    f"Jalankan dulu: python train.py --model {model_id} --fold {fold}")

        ckpt = torch.load(self.ckpt_path, map_location=self.device)
        self.classes = ckpt["classes"]
        img_size = ckpt.get("img_size", self.cfg["img_size"])

        self.model = build_model(model_id, len(self.classes), img_size, for_training=False)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(self.device).eval()

        # test hanya butuh Test original -> tidak perlu memuat data augmented
        # pre-processing test harus sama dengan saat training (CLAHE on/off)
        # jika file .pth tidak menyimpan info CLAHE, pakai --assume-clahe
        self.clahe = ckpt.get("clahe", assume_clahe)
        data = Getdata(root=data_root, img_size=img_size, batch_size=batch_size,
                       fold=fold, use_augmented=False, clahe=self.clahe)
        _, _, self.test_loader = data.get_loaders()
        if data.classes != self.classes:
            raise RuntimeError(f"Kelas dataset {data.classes} beda dengan kelas model {self.classes}")
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def predict(self):
        y_true, y_pred, probs = [], [], []
        start = time.time()
        for images, labels in self.test_loader:
            outputs = self.model(images.to(self.device))
            probs.append(outputs.exp().cpu())        # LogSoftmax -> probabilitas
            y_pred += outputs.argmax(1).cpu().tolist()
            y_true += labels.argmax(1).tolist()   # one-hot -> indeks kelas
        self.infer_ms = (time.time() - start) / len(y_true) * 1000
        return y_true, y_pred, torch.cat(probs).numpy()

    def run(self):
        y_true, y_pred, probs = self.predict()
        labels = list(range(len(self.classes)))
        auc = macro_roc_auc(y_true, probs, len(self.classes))
        acc = accuracy_score(y_true, y_pred)
        p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels,
                                                      average="macro", zero_division=0)
        report = classification_report(y_true, y_pred, labels=labels,
                                       target_names=self.classes, digits=4,
                                       zero_division=0)

        out_dir = os.path.join(BASE_DIR, "Results", self.cfg["name"], f"fold{self.fold}")
        os.makedirs(out_dir, exist_ok=True)
        text = (f"Model          : {self.cfg['name']} | fold {self.fold} | "
                f"CLAHE {'ON' if self.clahe else 'OFF'}\n"
                f"Jumlah test    : {len(y_true)} gambar\n"
                f"Parameter      : {self.n_params:,}\n"
                f"Accuracy       : {acc:.4f}\n"
                f"Precision (macro): {p:.4f}\n"
                f"Recall (macro) : {r:.4f}\n"
                f"F1 (macro)     : {f1:.4f}\n"
                f"ROC-AUC (macro, OvR): {auc:.4f}\n"
                f"Waktu inferensi: {self.infer_ms:.2f} ms/gambar\n\n{report}")
        print(text)
        with open(os.path.join(out_dir, "test_metrics.txt"), "w") as f:
            f.write(text)

        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plot_cm(cm, self.classes, f"Confusion Matrix - {self.cfg['name']} (fold {self.fold})",
                os.path.join(out_dir, "confusion_matrix.png"))
        plot_roc(y_true, probs, self.classes,
                 f"ROC Curve - {self.cfg['name']} (fold {self.fold})",
                 os.path.join(out_dir, "roc_curve.png"))
        print("Hasil test disimpan di", out_dir)

        return {"model": self.cfg["name"], "fold": self.fold,
                "accuracy": round(acc, 4), "precision_macro": round(p, 4),
                "recall_macro": round(r, 4), "f1_macro": round(f1, 4),
                "roc_auc_macro": round(auc, 4),
                "params": self.n_params,
                "inference_ms_per_img": round(self.infer_ms, 2)}, cm


def summarize_folds(model_id, rows, cms, classes):
    """Simpan metrik tiap fold + mean/std, dan confusion matrix gabungan."""
    name = MODELS[model_id]["name"]
    out_dir = os.path.join(BASE_DIR, "Results", name)
    os.makedirs(out_dir, exist_ok=True)

    summary = {"model": name, "n_folds": len(rows), "params": rows[0]["params"]}
    for m in METRICS + ["inference_ms_per_img"]:
        vals = np.array([r[m] for r in rows])
        summary[f"{m}_mean"] = round(float(vals.mean()), 4)
        summary[f"{m}_std"] = round(float(vals.std()), 4)

    with open(os.path.join(out_dir, "summary_folds.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
        f.write("\n")
        f.write("mean," + ",".join(str(summary.get(f"{k}_mean", "")) for k in list(rows[0].keys())[1:]) + "\n")
        f.write("std," + ",".join(str(summary.get(f"{k}_std", "")) for k in list(rows[0].keys())[1:]) + "\n")

    total_cm = np.sum(cms, axis=0)
    plot_cm(total_cm, classes, f"Confusion Matrix - {name} (gabungan {len(rows)} fold)",
            os.path.join(out_dir, "confusion_matrix_all_folds.png"))

    print(f"\n=== {name}: rata-rata {len(rows)} fold ===")
    for m in METRICS:
        print(f"{m:<16}: {summary[m + '_mean']:.4f} ± {summary[m + '_std']:.4f}")
    print("Ringkasan disimpan di", out_dir)
    return summary


def parse_args():
    p = argparse.ArgumentParser(description="Testing CNN MSLD v2.0")
    p.add_argument("--model", type=int, choices=[1, 2, 3])
    p.add_argument("--compare", action="store_true", help="uji & bandingkan semua model")
    p.add_argument("--fold", type=int, default=1, choices=[1, 2, 3, 4, 5])
    p.add_argument("--all-folds", action="store_true", help="uji fold 1-5 lalu hitung rata-rata")
    p.add_argument("--data", default=DEFAULT_DATA)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--assume-clahe", choices=["on", "off"], default="on",
                   help="dipakai hanya jika file .pth tidak menyimpan info CLAHE")
    args = p.parse_args()
    if not args.model and not args.compare:
        args.compare = True   # tombol Run -> otomatis bandingkan semua model
    return args


def main():
    args = parse_args()
    model_ids = list(MODELS) if args.compare else [args.model]
    folds = [1, 2, 3, 4, 5] if args.all_folds else [args.fold]

    comparison = []
    for model_id in model_ids:
        rows, cms, classes = [], [], None
        for fold in folds:
            try:
                tester = Tester(model_id, args.data, fold, args.batch_size,
                                assume_clahe=(args.assume_clahe == "on"))
                row, cm = tester.run()
                rows.append(row); cms.append(cm); classes = tester.classes
            except FileNotFoundError as e:
                print("Lewati:", e)
        if not rows:
            continue
        if args.all_folds:
            comparison.append(summarize_folds(model_id, rows, cms, classes))
        else:
            comparison.extend(rows)

    if len(comparison) > 0 and (args.compare or args.all_folds):
        path = os.path.join(BASE_DIR, "Results", "comparison.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=comparison[0].keys())
            writer.writeheader()
            writer.writerows(comparison)
        print("\n=== PERBANDINGAN MODEL ===")
        for r in comparison:
            print(r)
        print("Tabel perbandingan disimpan di", path)


if __name__ == "__main__":
    main()
