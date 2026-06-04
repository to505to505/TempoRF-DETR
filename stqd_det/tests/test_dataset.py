"""Smoke tests for stqd_det.dataset.

Runs against the real ``data/dataset2_split`` tree because it must be
present for any training/eval anyway. Skips if the data dir is absent.
"""

from pathlib import Path

import pytest
import torch

from stqd_det.config import Config
from stqd_det.dataset import VideoSequenceDataset, collate_video, get_dataloader


DATA_ROOT = Path("data/dataset2_split")
pytestmark = pytest.mark.skipif(
    not (DATA_ROOT / "test" / "images").is_dir(),
    reason="data/dataset2_split not symlinked into worktree",
)


def _cfg(split_T: int = 5, img_size: int = 256) -> Config:
    # Small img_size keeps the smoke test fast.
    return Config(T=split_T, img_size=img_size, batch_size=2, num_workers=0)


def test_dataset_loads_test_split_first_sample():
    cfg = _cfg()
    ds = VideoSequenceDataset("test", cfg)
    assert len(ds) > 0

    sample = ds[0]
    assert set(sample.keys()) == {"frames", "targets", "paths"}

    frames = sample["frames"]
    assert isinstance(frames, torch.Tensor)
    assert frames.shape == (cfg.T, 3, cfg.img_size, cfg.img_size)
    assert frames.dtype == torch.float32

    targets = sample["targets"]
    assert isinstance(targets, list)
    assert len(targets) == cfg.T
    for t in targets:
        assert set(t.keys()) == {"boxes", "labels"}
        assert t["boxes"].dtype == torch.float32
        assert t["boxes"].ndim == 2 and t["boxes"].shape[1] == 4
        assert t["labels"].dtype == torch.int64
        assert t["labels"].shape[0] == t["boxes"].shape[0]
        # boxes are normalised cxcywh so values must lie in [0, 1]
        if t["boxes"].numel() > 0:
            assert float(t["boxes"].min()) >= 0.0
            assert float(t["boxes"].max()) <= 1.0

    assert len(sample["paths"]) == cfg.T


def test_collate_video_batches_frames_and_keeps_target_lists():
    cfg = _cfg()
    ds = VideoSequenceDataset("test", cfg)
    batch = collate_video([ds[0], ds[1]])
    assert batch["frames"].shape == (2, cfg.T, 3, cfg.img_size, cfg.img_size)
    assert len(batch["targets"]) == 2
    assert all(len(per_sample) == cfg.T for per_sample in batch["targets"])


def test_get_dataloader_one_iter():
    cfg = _cfg()
    loader = get_dataloader("test", cfg, shuffle=False, drop_last=False)
    batch = next(iter(loader))
    assert batch["frames"].shape[0] == cfg.batch_size
    assert batch["frames"].shape[1] == cfg.T


def test_train_split_runs_with_augmentation():
    cfg = _cfg()
    ds = VideoSequenceDataset("train", cfg)
    assert len(ds) > 0
    sample = ds[0]
    assert sample["frames"].shape == (cfg.T, 3, cfg.img_size, cfg.img_size)
    # All T frames must share the SAME augmentation replay -> identical resolution.
    # (We only assert shapes here; geometric replay correctness is tested via
    # rfdetr_video's own tests.)


def test_window_centre_index_matches_config():
    cfg = _cfg()
    ds = VideoSequenceDataset("test", cfg)
    assert ds.centre == cfg.T // 2
