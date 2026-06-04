"""Video-clip dataset for STQD-Det.

Wraps rfdetr_video.sequence_dataset to yield T-frame ImageNet-normalised
tensors + per-frame YOLO targets. Augmentations match the RF-DETR-video baseline.
"""

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from rfdetr_video.sequence_dataset import (
    build_sequence_index,
    build_train_augmentation,
    build_windows,
    load_yolo_labels,
    pascal_to_cxcywh_norm,
    yolo_to_pascal,
)
from albumentations import ReplayCompose

from .config import Config


class VideoSequenceDataset(Dataset):
    """T-frame windows + per-frame targets.

    Returns frames (T,3,H,W) and a list of T target dicts with normalised
    cxcywh boxes and all-zero int64 labels.
    """

    def __init__(self, split: str, cfg: Config):
        self.cfg = cfg
        self.split = split
        self.img_dir: Path = cfg.data_root / split / "images"
        self.lbl_dir: Path = cfg.data_root / split / "labels"
        if not self.img_dir.is_dir():
            raise FileNotFoundError(f"Image dir not found: {self.img_dir}")

        sequences = build_sequence_index(self.img_dir)
        self.windows: List[List[Path]] = build_windows(sequences, cfg.T)
        if not self.windows:
            raise RuntimeError(
                f"No T={cfg.T} windows found under {self.img_dir}"
            )

        self.augment = (
            build_train_augmentation(cfg.img_size) if split == "train" else None
        )
        self.centre = cfg.T // 2
        self.mean = torch.tensor(cfg.pixel_mean, dtype=torch.float32).view(3, 1, 1)
        self.std = torch.tensor(cfg.pixel_std, dtype=torch.float32).view(3, 1, 1)

        print(
            f"[stqd_det/{split}] {len(sequences)} sequences, "
            f"{sum(len(s[2]) for s in sequences)} frames, "
            f"{len(self.windows)} windows of T={cfg.T}"
        )

    def __len__(self) -> int:
        return len(self.windows)

    def _to_imagenet_tensor(self, img: np.ndarray) -> torch.Tensor:
        img = img.astype(np.float32) / 255.0
        img_3ch = np.stack([img, img, img], axis=0)
        return (torch.from_numpy(img_3ch) - self.mean) / self.std

    def __getitem__(self, idx: int) -> dict:
        paths = self.windows[idx]
        images: List[torch.Tensor] = []
        targets: List[dict] = []
        replay = None

        for img_path in paths:
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise FileNotFoundError(f"Cannot read {img_path}")
            if img.shape != (self.cfg.img_size, self.cfg.img_size):
                img = cv2.resize(
                    img,
                    (self.cfg.img_size, self.cfg.img_size),
                    interpolation=cv2.INTER_AREA,
                )

            lbl_path = self.lbl_dir / (img_path.stem + ".txt")
            yolo = load_yolo_labels(lbl_path)
            bboxes, class_ids = yolo_to_pascal(
                yolo, self.cfg.img_size, self.cfg.img_size
            )
            bboxes, class_ids = bboxes.tolist(), class_ids.tolist()

            if self.augment is not None:
                if replay is None:
                    res = self.augment(image=img, bboxes=bboxes, class_ids=class_ids)
                    replay = res["replay"]
                else:
                    res = ReplayCompose.replay(
                        replay, image=img, bboxes=bboxes, class_ids=class_ids
                    )
                img = res["image"]
                bboxes = res["bboxes"]
                class_ids = res["class_ids"]

            images.append(self._to_imagenet_tensor(img))

            if len(bboxes) > 0:
                boxes_cxcywh = pascal_to_cxcywh_norm(
                    np.array(bboxes, dtype=np.float32),
                    self.cfg.img_size,
                    self.cfg.img_size,
                )
                boxes = torch.from_numpy(boxes_cxcywh)
            else:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(len(boxes), dtype=torch.int64)
            targets.append({"boxes": boxes, "labels": labels})

        frames = torch.stack(images, dim=0)
        return {"frames": frames, "targets": targets, "paths": [str(p) for p in paths]}


def collate_video(batch: List[dict]) -> dict:
    """Stack frames along a batch axis; keep per-sample target lists nested."""
    frames = torch.stack([b["frames"] for b in batch], dim=0)  # (B, T, 3, H, W)
    targets = [b["targets"] for b in batch]                    # list len B, each list len T
    paths = [b["paths"] for b in batch]
    return {"frames": frames, "targets": targets, "paths": paths}


def get_dataloader(
    split: str,
    cfg: Config,
    shuffle: bool = None,
    drop_last: bool = None,
) -> DataLoader:
    if shuffle is None:
        shuffle = split == "train"
    if drop_last is None:
        drop_last = split == "train"
    ds = VideoSequenceDataset(split, cfg)
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=cfg.num_workers,
        drop_last=drop_last,
        collate_fn=collate_video,
        pin_memory=True,
    )
