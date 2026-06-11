"""Scene-only LIBERO pair dataset for Cosmos SCVC training.

This dataset consumes the Phase-2 pair future-frame manifest produced by
``render_libero_pair_future_frames.py`` and returns the standard Cosmos LIBERO
sample dictionary plus:

* ``video_pair``: the paired view's 7-frame scene-only video tensor
* ``pair_valid``: 1 for matched demo-pair samples, 0 for pass-through rollout samples
* ``pair_id`` and pair camera metadata for audits

Rollout mixture (Plan A, researcher-ratified 2026-06-11): when ``rollout_data_dir``
is provided, indices [0, num_pairs) yield demo-pair samples and indices
[num_pairs, 2*num_pairs) yield rollout samples drawn from an embedded
rollout-only ``LIBERODataset`` (``demonstration_sampling_prob=0.0``).  With a
shuffling sampler this preserves Cosmos's original 0.5:0.5 demo:rollout
mixture.  Rollout samples pass through single-view semantics: ``video_pair``
is a copy of ``video`` and ``pair_valid=0``, so the SCVC trainer's FM term is
exactly the single-branch loss (identical branches under shared noise) and the
CV term is masked out.

Augmentation: pair branches use photometric-only augmentation with independent
draws per branch (B6b locked semantics: spatial aug off for pair branches,
photometric independent across views).  Rollout pass-through samples keep the
original LIBERODataset augmentation unchanged.
"""

from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from cosmos_policy.datasets.dataset_common import compute_monte_carlo_returns, get_action_chunk_with_padding
from cosmos_policy.datasets.dataset_utils import preprocess_image, rescale_episode_data
from cosmos_policy.datasets.libero_dataset import LIBERODataset
from cosmos_policy.utils.utils import duplicate_array


class LIBEROPairDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        pair_manifest_path: str,
        repo_root: str = ".",
        chunk_size: int = 16,
        final_image_size: int = 224,
        t5_text_embeddings_path: str = "",
        normalize_images: bool = False,
        normalize_actions: bool = True,
        normalize_proprio: bool = True,
        use_image_aug: bool = True,
        use_stronger_image_aug: bool = True,
        use_proprio: bool = True,
        num_duplicates_per_image: int = 4,
        return_value_function_returns: bool = True,
        gamma: float = 0.99,
        rollout_data_dir: str = "",
        success_rollout_sampling_prob: float = 0.5,
    ) -> None:
        self.data_dir = data_dir
        self.pair_manifest_path = pair_manifest_path
        self.repo_root = Path(repo_root).resolve()
        self.chunk_size = int(chunk_size)
        self.final_image_size = int(final_image_size)
        self.t5_text_embeddings_path = t5_text_embeddings_path
        self.normalize_images = normalize_images
        self.normalize_actions = normalize_actions
        self.normalize_proprio = normalize_proprio
        self.use_image_aug = use_image_aug
        self.use_stronger_image_aug = use_stronger_image_aug
        self.use_proprio = use_proprio
        self.num_duplicates_per_image = int(num_duplicates_per_image)
        self.return_value_function_returns = return_value_function_returns
        self.gamma = float(gamma)

        self.rows = self._read_manifest(Path(pair_manifest_path))
        if not self.rows:
            raise ValueError(f"No rows found in pair manifest: {pair_manifest_path}")

        stats_path = Path(data_dir) / "dataset_statistics.json"
        if not stats_path.exists():
            raise FileNotFoundError(f"Expected dataset statistics at {stats_path}")
        self.dataset_stats = self._load_stats(stats_path)

        if t5_text_embeddings_path:
            with open(t5_text_embeddings_path, "rb") as f:
                self.t5_text_embeddings = pickle.load(f)
        else:
            self.t5_text_embeddings = {}

        # Plan A rollout mixture: embedded rollout-only LIBERODataset (demo loading skipped because
        # demonstration_sampling_prob=0.0). Rollout samples pass through with original augmentation.
        self.rollout_dataset: LIBERODataset | None = None
        if rollout_data_dir:
            self.rollout_dataset = LIBERODataset(
                data_dir=data_dir,
                chunk_size=chunk_size,
                final_image_size=final_image_size,
                t5_text_embeddings_path=t5_text_embeddings_path,
                normalize_images=normalize_images,
                normalize_actions=normalize_actions,
                normalize_proprio=normalize_proprio,
                use_image_aug=use_image_aug,
                use_stronger_image_aug=use_stronger_image_aug,
                use_wrist_images=False,
                use_third_person_images=True,
                use_proprio=use_proprio,
                num_duplicates_per_image=num_duplicates_per_image,
                rollout_data_dir=rollout_data_dir,
                demonstration_sampling_prob=0.0,
                success_rollout_sampling_prob=success_rollout_sampling_prob,
                treat_success_rollouts_as_demos=False,
                return_value_function_returns=return_value_function_returns,
                gamma=gamma,
            )

    @staticmethod
    def _read_manifest(path: Path) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows

    @staticmethod
    def _load_stats(path: Path) -> dict[str, np.ndarray]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return {key: np.asarray(value, dtype=np.float32) for key, value in raw.items()}

    def __len__(self) -> int:
        # With rollout mixture: [0, num_pairs) = pair samples, [num_pairs, 2*num_pairs) = rollout
        # samples; a shuffling sampler then yields the 0.5:0.5 demo:rollout mixture in expectation.
        if self.rollout_dataset is not None:
            return 2 * len(self.rows)
        return len(self.rows)

    def _resolve_path(self, raw_path: str) -> Path:
        path = Path(raw_path)
        if path.is_absolute():
            return path
        return self.repo_root / path

    @staticmethod
    def _load_rgb(path: Path) -> np.ndarray:
        return np.asarray(Image.open(path).convert("RGB"), dtype=np.uint8)

    def _load_episode_labels(self, row: dict[str, Any]) -> dict[str, Any]:
        hdf5_path = Path(row["hdf5_path"])
        demo_key = str(row["demo_key"])
        with h5py.File(hdf5_path, "r") as f:
            demo = f[f"data/{demo_key}"]
            actions = demo["actions"][:].astype(np.float32)
            proprio = demo["robot_states"][:].astype(np.float32)
            num_steps = int(len(actions))

        if self.normalize_actions:
            actions = rescale_episode_data({"actions": actions}, self.dataset_stats, "actions")
        if self.normalize_proprio:
            proprio = rescale_episode_data({"proprio": proprio}, self.dataset_stats, "proprio")

        returns = None
        if self.return_value_function_returns:
            returns = compute_monte_carlo_returns(num_steps, terminal_reward=1.0, gamma=self.gamma)
        return {
            "actions": actions,
            "proprio": proprio,
            "num_steps": num_steps,
            "returns": returns,
            "command": row.get("language", ""),
        }

    def _build_video(self, current_image: np.ndarray, future_image: np.ndarray) -> torch.Tensor:
        image_list: list[np.ndarray] = []
        first_input_image = np.expand_dims(np.zeros_like(current_image), axis=0)
        image_list.append(first_input_image)

        if self.use_proprio:
            image_list.append(duplicate_array(np.zeros_like(current_image), total_num_copies=self.num_duplicates_per_image))

        image_list.append(duplicate_array(current_image, total_num_copies=self.num_duplicates_per_image))
        image_list.append(duplicate_array(np.zeros_like(current_image), total_num_copies=self.num_duplicates_per_image))

        if self.use_proprio:
            image_list.append(duplicate_array(np.zeros_like(current_image), total_num_copies=self.num_duplicates_per_image))

        image_list.append(duplicate_array(future_image, total_num_copies=self.num_duplicates_per_image))

        if self.return_value_function_returns:
            image_list.append(duplicate_array(np.zeros_like(current_image), total_num_copies=self.num_duplicates_per_image))

        images = np.concatenate(image_list, axis=0)
        # B6b locked semantics for pair branches: spatial aug OFF, photometric aug with independent
        # draws per branch (this method is called once per branch, so draws are independent).
        return preprocess_image(
            images,
            final_image_size=self.final_image_size,
            normalize_images=self.normalize_images,
            use_image_aug=self.use_image_aug,
            stronger_image_aug=self.use_stronger_image_aug,
            photometric_only_aug=True,
        )

    def _get_rollout_item(self, idx: int) -> dict[str, Any]:
        assert self.rollout_dataset is not None
        rollout_idx = (idx - len(self.rows)) % len(self.rollout_dataset)
        sample = self.rollout_dataset[rollout_idx]
        # Single-view pass-through: identical branches => FM degenerates to the standard single-branch
        # loss under shared noise; CV is masked out via pair_valid=0.
        sample["video_pair"] = sample["video"].clone()
        sample["pair_valid"] = torch.tensor(0, dtype=torch.int64)
        sample["pair_id"] = f"rollout_{rollout_idx}"
        sample["pair_type"] = "rollout_passthrough"
        sample["camera_category"] = ""
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.rollout_dataset is not None and idx >= len(self.rows):
            return self._get_rollout_item(idx)
        row = self.rows[idx]
        labels = self._load_episode_labels(row)
        t = int(row["timestep"])
        future_t = int(row["future_timestep"])
        num_steps = int(labels["num_steps"])

        current_a = self._load_rgb(self._resolve_path(row["current_img_a_path"]))
        current_b = self._load_rgb(self._resolve_path(row["current_img_b_path"]))
        future_a = self._load_rgb(self._resolve_path(row["future_img_a_path"]))
        future_b = self._load_rgb(self._resolve_path(row["future_img_b_path"]))

        action_chunk = get_action_chunk_with_padding(labels["actions"], t, self.chunk_size, num_steps)
        next_t = min(t + self.chunk_size, num_steps - 1)
        next_action_chunk = get_action_chunk_with_padding(labels["actions"], next_t, self.chunk_size, num_steps)
        next_future_t = min(next_t + self.chunk_size, num_steps - 1)

        value_function_return = float("-100")
        next_value_function_return = float("-100")
        if self.return_value_function_returns and labels["returns"] is not None:
            value_function_return = float(labels["returns"][future_t])
            next_value_function_return = float(labels["returns"][next_future_t])

        command = str(labels["command"])
        if command not in self.t5_text_embeddings:
            raise KeyError(f"Command {command!r} not found in T5 embeddings: {self.t5_text_embeddings_path}")

        # Scene-only P2 latent layout:
        # 0 blank | 1 proprio | 2 current scene | 3 action | 4 future proprio | 5 future scene | 6 value
        sample = {
            "video": self._build_video(current_a, future_a),
            "video_pair": self._build_video(current_b, future_b),
            "pair_valid": torch.tensor(1, dtype=torch.int64),
            "pair_id": str(row["pair_id"]),
            "pair_type": str(row.get("pair_type", "matched")),
            "camera_category": str(row.get("camera_category", "")),
            "actions": action_chunk,
            "t5_text_embeddings": torch.squeeze(self.t5_text_embeddings[command]),
            "t5_text_mask": torch.ones(512, dtype=torch.int64),
            "fps": 16,
            "padding_mask": torch.zeros(1, self.final_image_size, self.final_image_size),
            "image_size": self.final_image_size * torch.ones(4),
            "proprio": labels["proprio"][t] if self.use_proprio else np.zeros_like(labels["proprio"][t]),
            "future_proprio": labels["proprio"][future_t] if self.use_proprio else np.zeros_like(labels["proprio"][future_t]),
            "__key__": idx,
            "rollout_data_mask": 0,
            "rollout_data_success_mask": 0,
            "world_model_sample_mask": 0,
            "value_function_sample_mask": 0,
            "global_rollout_idx": -1,
            "action_latent_idx": 3,
            "value_latent_idx": 6 if self.return_value_function_returns else -1,
            "current_proprio_latent_idx": 1 if self.use_proprio else -1,
            "current_wrist_image_latent_idx": -1,
            "current_image_latent_idx": 2,
            "future_proprio_latent_idx": 4 if self.use_proprio else -1,
            "future_wrist_image_latent_idx": -1,
            "future_image_latent_idx": 5,
            "value_function_return": value_function_return,
            "next_action_chunk": next_action_chunk,
            "next_value_function_return": next_value_function_return,
        }
        return sample


if __name__ == "__main__":
    data_root = os.environ.get("BASE_DATASETS_DIR", ".")
    manifest = os.environ.get("PAIR_MANIFEST", "")
    if not manifest:
        raise SystemExit("Set PAIR_MANIFEST to inspect a LIBEROPairDataset sample.")
    dataset = LIBEROPairDataset(
        data_dir=os.path.join(data_root, "LIBERO-Cosmos-Policy", "success_only"),
        pair_manifest_path=manifest,
        repo_root=".",
        t5_text_embeddings_path=os.path.join(data_root, "LIBERO-Cosmos-Policy", "success_only", "t5_embeddings.pkl"),
        use_image_aug=False,
    )
    item = dataset[0]
    print({key: (tuple(value.shape) if hasattr(value, "shape") else type(value).__name__) for key, value in item.items()})
