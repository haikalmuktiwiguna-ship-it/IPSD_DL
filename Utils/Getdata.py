import os
import re
from collections import Counter
from os import listdir

import cv2 as cv
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_NAMES = {"train": "train", "valid": "valid", "val": "valid",
               "validation": "valid", "test": "test"}


def apply_clahe(image, clip_limit=2.0, tile_grid=(8, 8)):
    """
    CLAHE (Contrast Limited Adaptive Histogram Equalization).
    Kontras diperjelas per area kecil (tile) hanya pada kanal L (kecerahan)
    di ruang warna LAB, sehingga warna asli kulit/lesi tidak berubah.
    """
    lab = cv.cvtColor(image, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l = clahe.apply(l)
    return cv.cvtColor(cv.merge((l, a, b)), cv.COLOR_LAB2BGR)


class getImageLabel(Dataset):
    """
    Versi materi: folder berisi sub-folder per kelas.
    `folder` boleh satu path atau list path (mis. Train original + Train augmented).
    """

    def __init__(self, folder, classes, img_size=100, clahe=False):
        self.dataset = []
        self.labels = []
        folders = folder if isinstance(folder, (list, tuple)) else [folder]

        # initiate the one hot encoding for class
        to_one_hot = np.eye(len(classes))     # number of class (MSLD v2.0 = 6)
        for folder in folders:
          for i, cls in enumerate(classes):
            cls_dir = os.path.join(folder, cls)
            if not os.path.isdir(cls_dir):
                continue
            for image_name in sorted(listdir(cls_dir)):
                if not image_name.lower().endswith(IMG_EXT):
                    continue
                # read image & resize (normalisasi 0-1 dilakukan di __getitem__)
                image = cv.imread(os.path.join(cls_dir, image_name))
                if image is None:            # lewati file yang rusak
                    continue
                if clahe:                    # pre-processing: perjelas kontras lesi
                    image = apply_clahe(image)
                image = cv.resize(image, dsize=(img_size, img_size))
                # append feature image and one hot encoded label
                self.dataset.append([image, to_one_hot[i]])
                self.labels.append(i)

    def __getitem__(self, item):
        feature, label = self.dataset[item]
        feature = feature.astype(np.float32) / 255          # normalize 0-1
        feature = np.transpose(feature, (2, 0, 1))          # (H,W,C) -> (C,H,W)
        return (torch.tensor(feature, dtype=torch.float32),
                torch.tensor(label, dtype=torch.float32))

    def __len__(self):
        return len(self.dataset)


class _SubsetFromSamples(getImageLabel):
    """Dipakai saat dataset tidak punya folder fold (split otomatis 70:20:10)."""

    def __init__(self, items, n_classes):
        to_one_hot = np.eye(n_classes)
        self.dataset = [[img, to_one_hot[lbl]] for img, lbl in items]
        self.labels = [lbl for _, lbl in items]


class Getdata:
    """
    Contoh:
        data = Getdata(root="D:/Tes python/.venv/archive", img_size=100)
        train_loader, valid_loader, test_loader = data.get_loaders()
        data.summary()
    """

    def __init__(self, root, img_size=100, batch_size=32, fold=1,
                 use_augmented=True, clahe=True, seed=42):
        if not os.path.isdir(root):
            raise FileNotFoundError(f"Folder dataset tidak ditemukan: {root}")
        self.root = root
        self.img_size = img_size
        self.batch_size = batch_size
        self.fold = fold
        self.use_augmented = use_augmented
        self.clahe = clahe
        self.seed = seed
        self.classes = []

    # ------------------------- pencarian folder ------------------------- #
    def _find_dir(self, keyword):
        for dirpath, dirnames, _ in os.walk(self.root):
            for d in sorted(dirnames):
                if keyword.lower() in d.lower():
                    return os.path.join(dirpath, d)
        return None

    def _find_fold_splits(self, base):
        if base is None:
            return {}
        pattern = re.compile(rf"fold[_\s-]*{self.fold}(?!\d)", re.IGNORECASE)
        for dirpath, dirnames, _ in os.walk(base):
            for d in sorted(dirnames):
                if pattern.search(d):
                    fold_dir = os.path.join(dirpath, d)
                    splits = {}
                    for sub in listdir(fold_dir):
                        key = SPLIT_NAMES.get(sub.lower())
                        if key and os.path.isdir(os.path.join(fold_dir, sub)):
                            splits[key] = os.path.join(fold_dir, sub)
                    if splits:
                        return splits
        return {}

    def _find_class_root(self, base):
        for dirpath, dirnames, _ in os.walk(base):
            if any("fold" in d.lower() for d in dirnames):
                continue
            subdirs = [os.path.join(dirpath, d) for d in dirnames]
            if len(subdirs) >= 2 and all(
                    any(f.lower().endswith(IMG_EXT) for f in listdir(s)) for s in subdirs):
                return dirpath
        return None

    @staticmethod
    def _class_dirs(folder):
        return sorted(d for d in listdir(folder) if os.path.isdir(os.path.join(folder, d)))

    # ----------------------------- loading ------------------------------ #
    def load_datasets(self):
        original_base = self._find_dir("original")
        augmented_base = self._find_dir("augmented")
        orig = self._find_fold_splits(original_base)
        aug = self._find_fold_splits(augmented_base) if self.use_augmented else {}

        if orig:
            # Ketentuan tugas:
            #   training   = Train original (+ Train augmented)
            #   validasi   = Valid original
            #   testing    = Test original
            orig_train, valid_dir, test_dir = orig.get("train"), orig.get("valid"), orig.get("test")
            if not (orig_train and valid_dir and test_dir):
                raise RuntimeError(f"Folder Original fold{self.fold} tidak lengkap: "
                                   f"train={orig_train}, valid={valid_dir}, test={test_dir}")
            train_dirs = [orig_train]
            if self.use_augmented:
                aug_train = aug.get("train")
                if not aug_train:
                    raise RuntimeError(f"Folder Train di Augmented Images fold{self.fold} tidak ditemukan.")
                train_dirs.append(aug_train)

            self.classes = sorted(set(self._class_dirs(orig_train)) |
                                  set(self._class_dirs(valid_dir)) |
                                  set(self._class_dirs(test_dir)))
            for d in train_dirs[1:]:
                beda = set(self._class_dirs(d)) - set(self.classes)
                if beda:
                    raise RuntimeError(f"Nama kelas di {d} tidak cocok dengan data original: {sorted(beda)}")

            self.source_info = {"train": " + ".join(train_dirs), "valid": valid_dir, "test": test_dir}
            print("Membaca gambar... (bisa agak lama untuk data augmented)")
            self.train_set = getImageLabel(train_dirs, self.classes, self.img_size, self.clahe)
            self.valid_set = getImageLabel(valid_dir, self.classes, self.img_size, self.clahe)
            self.test_set = getImageLabel(test_dir, self.classes, self.img_size, self.clahe)
        else:
            class_root = self._find_class_root(original_base or self.root)
            if class_root is None:
                raise RuntimeError("Struktur dataset tidak dikenali. Cek isi 'Original Images'.")
            self.classes = self._class_dirs(class_root)
            self.source_info = {"all (split 70:20:10)": class_root}
            full = getImageLabel(class_root, self.classes, self.img_size, self.clahe)
            self.train_set, self.valid_set, self.test_set = self._split(full)
        return self.train_set, self.valid_set, self.test_set

    def _split(self, full, ratios=(0.7, 0.2, 0.1)):
        rng = np.random.default_rng(self.seed)
        parts = ([], [], [])
        for c in range(len(self.classes)):
            idx = [i for i, l in enumerate(full.labels) if l == c]
            rng.shuffle(idx)
            n_tr, n_va = int(len(idx) * ratios[0]), int(len(idx) * ratios[1])
            for part, sel in zip(parts, (idx[:n_tr], idx[n_tr:n_tr + n_va], idx[n_tr + n_va:])):
                part += [(full.dataset[i][0], c) for i in sel]
        return tuple(_SubsetFromSamples(p, len(self.classes)) for p in parts)

    def get_loaders(self):
        self.load_datasets()
        train_loader = DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True)
        valid_loader = DataLoader(self.valid_set, batch_size=self.batch_size, shuffle=False)
        test_loader = DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False)
        return train_loader, valid_loader, test_loader

    def class_weights(self):
        """Bobot kelas untuk data tidak seimbang (dari data train)."""
        counts = Counter(self.train_set.labels)
        total, n = sum(counts.values()), len(self.classes)
        return torch.tensor([total / (n * counts.get(i, 1)) for i in range(n)],
                            dtype=torch.float32)

    def summary(self):
        print("=" * 60)
        print("Dataset :", self.root)
        print("Mode    :", "Train original + augmented" if self.use_augmented else "Train original saja",
              f"| fold {self.fold} | resize {self.img_size}x{self.img_size}"
              f" | CLAHE {'ON' if self.clahe else 'OFF'}")
        for k, v in getattr(self, "source_info", {}).items():
            print(f"  {k:<6}: {v}")
        print("Kelas   :", self.classes)
        for name, ds in [("Train", self.train_set), ("Valid", self.valid_set),
                         ("Test", self.test_set)]:
            c = Counter(ds.labels)
            detail = ", ".join(f"{self.classes[i]}={c.get(i, 0)}" for i in range(len(self.classes)))
            print(f"{name:<6}: {len(ds):>5} gambar  ({detail})")
        print("=" * 60)


def save_clahe_example(root, out_path, n=6):
    """Simpan gambar perbandingan sebelum & sesudah CLAHE (1 contoh per kelas)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = Getdata(root=root, use_augmented=False)
    base = data._find_dir("original")
    splits = data._find_fold_splits(base)
    train_dir = splits.get("train")
    if train_dir is None:
        print("Folder Train fold1 tidak ditemukan, contoh CLAHE dilewati.")
        return
    classes = data._class_dirs(train_dir)[:n]
    fig, axes = plt.subplots(2, len(classes), figsize=(3 * len(classes), 6))
    for j, cls in enumerate(classes):
        cls_dir = os.path.join(train_dir, cls)
        name = sorted(f for f in listdir(cls_dir) if f.lower().endswith(IMG_EXT))[0]
        img = cv.imread(os.path.join(cls_dir, name))
        for i, (title, im) in enumerate([("Asli", img), ("CLAHE", apply_clahe(img))]):
            axes[i, j].imshow(cv.cvtColor(im, cv.COLOR_BGR2RGB))
            axes[i, j].set_title(f"{cls}\n{title}" if i == 0 else title, fontsize=9)
            axes[i, j].axis("off")
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Contoh CLAHE disimpan di", out_path)


if __name__ == "__main__":
    # Cek cepat: python Utils/Getdata.py
    program_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root = os.path.join(program_dir, "archive")
    data = Getdata(root=root)
    data.get_loaders()
    data.summary()
    save_clahe_example(root, os.path.join(program_dir, "Results", "clahe_example.png"))
