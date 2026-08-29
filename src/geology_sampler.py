from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
from torch.utils.data import Sampler


@dataclass(frozen=True)
class StrataDescription:
    labels: np.ndarray
    background_label: int
    label_to_name: dict[int, str]


def build_multilabel_strata(
    metadata_matrix: np.ndarray,
    metadata_keys: Sequence[str],
    threshold: float,
    active_key_indices: Sequence[int],
    max_active_keys_per_stratum: int = 2,
) -> StrataDescription:
    if metadata_matrix.ndim != 2:
        raise ValueError("metadata_matrix must be 2D [num_examples, num_features].")
    if metadata_matrix.shape[0] <= 0:
        raise ValueError("metadata_matrix must contain at least one sample.")
    if metadata_matrix.shape[1] != len(metadata_keys):
        raise ValueError("metadata_matrix feature count must match metadata_keys length.")

    if max_active_keys_per_stratum < 1:
        raise ValueError("max_active_keys_per_stratum must be >= 1.")

    active_key_indices = tuple(int(v) for v in active_key_indices)
    if not active_key_indices:
        active_key_indices = tuple(range(metadata_matrix.shape[1]))

    key_names = [str(k) for k in metadata_keys]
    sample_count = int(metadata_matrix.shape[0])

    labels = np.zeros((sample_count,), dtype=np.int64)
    signature_to_label: dict[tuple[str, ...], int] = {}
    label_to_name: dict[int, str] = {}

    background_signature = ("background",)
    signature_to_label[background_signature] = 0
    label_to_name[0] = "background"
    next_label = 1

    for idx in range(sample_count):
        selected_values = metadata_matrix[idx, list(active_key_indices)]
        selected_active = selected_values > float(threshold)
        if not np.any(selected_active):
            labels[idx] = 0
            continue

        selected_key_ids = np.asarray(active_key_indices, dtype=np.int64)[selected_active]
        if selected_key_ids.size > max_active_keys_per_stratum:
            all_values = metadata_matrix[idx, selected_key_ids]
            keep_order = np.argsort(-all_values)[:max_active_keys_per_stratum]
            selected_key_ids = selected_key_ids[keep_order]

        selected_key_ids = np.sort(selected_key_ids)
        signature = tuple(key_names[int(key_idx)] for key_idx in selected_key_ids)
        if signature not in signature_to_label:
            signature_to_label[signature] = next_label
            label_to_name[next_label] = "+".join(signature)
            next_label += 1
        labels[idx] = signature_to_label[signature]

    return StrataDescription(labels=labels, background_label=0, label_to_name=label_to_name)


def _normalize_weights(weights: np.ndarray) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.clip(arr, a_min=0.0, a_max=None)
    weight_sum = float(arr.sum())
    if weight_sum <= 0.0:
        arr = np.ones_like(arr, dtype=np.float64)
        weight_sum = float(arr.sum())
    return arr / weight_sum


class GeologyAwareBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        strata_labels: np.ndarray,
        batch_size: int,
        num_batches: int,
        seed: int,
        sample_weights: Optional[np.ndarray] = None,
        background_fraction: float = 0.20,
        hard_fraction: float = 0.20,
        hard_top_quantile: float = 0.20,
        min_negative_strata: int = 2,
        require_positive_pair: bool = True,
        allow_duplicates: bool = False,
    ):
        self.strata_labels = np.asarray(strata_labels, dtype=np.int64).reshape(-1)
        if self.strata_labels.size <= 0:
            raise ValueError("strata_labels must not be empty.")
        if batch_size <= 1:
            raise ValueError("batch_size must be >= 2.")
        if num_batches <= 0:
            raise ValueError("num_batches must be positive.")

        self.batch_size = int(batch_size)
        self.num_batches = int(num_batches)
        self.seed = int(seed)
        self.epoch = 0
        self.background_fraction = float(np.clip(background_fraction, 0.0, 1.0))
        self.hard_fraction = float(np.clip(hard_fraction, 0.0, 1.0))
        self.hard_top_quantile = float(np.clip(hard_top_quantile, 0.01, 0.99))
        self.min_negative_strata = max(1, int(min_negative_strata))
        self.require_positive_pair = bool(require_positive_pair)
        self.allow_duplicates = bool(allow_duplicates)

        if sample_weights is None:
            self.sample_weights = np.ones((self.strata_labels.shape[0],), dtype=np.float64)
        else:
            sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
            if sample_weights.shape[0] != self.strata_labels.shape[0]:
                raise ValueError("sample_weights must match strata_labels length.")
            self.sample_weights = sample_weights

        self._sample_prob = _normalize_weights(self.sample_weights)
        self._indices = np.arange(self.strata_labels.shape[0], dtype=np.int64)
        self._label_ids = np.unique(self.strata_labels)
        self._label_to_indices = {
            int(label): self._indices[self.strata_labels == label]
            for label in self._label_ids
        }

        self._last_epoch_stats: dict[str, float] = {
            "positive_pair_batch_rate": 0.0,
            "negative_strata_batch_rate": 0.0,
            "avg_unique_strata_per_batch": 0.0,
            "fallback_positive_pair_count": 0.0,
            "fallback_duplicate_fill_count": 0.0,
            "background_fraction_achieved": 0.0,
            "hard_fraction_achieved": 0.0,
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def update_sample_weights(self, sample_weights: np.ndarray) -> None:
        sample_weights = np.asarray(sample_weights, dtype=np.float64).reshape(-1)
        if sample_weights.shape[0] != self.strata_labels.shape[0]:
            raise ValueError("sample_weights must match strata_labels length.")
        self.sample_weights = sample_weights
        self._sample_prob = _normalize_weights(self.sample_weights)

    def get_last_epoch_stats(self) -> dict[str, float]:
        return dict(self._last_epoch_stats)

    def __len__(self) -> int:
        return self.num_batches

    def _draw_from_pool(
        self,
        rng: np.random.Generator,
        pool: np.ndarray,
        used: set[int],
    ) -> Optional[int]:
        if pool.size <= 0:
            return None

        if not self.allow_duplicates:
            pool = np.asarray([idx for idx in pool.tolist() if int(idx) not in used], dtype=np.int64)
            if pool.size <= 0:
                return None

        weights = self._sample_prob[pool]
        weights = _normalize_weights(weights)
        chosen = int(rng.choice(pool, p=weights))
        return chosen

    def __iter__(self) -> Iterable[list[int]]:
        rng = np.random.default_rng(self.seed + self.epoch)
        hard_threshold = float(np.quantile(self.sample_weights, 1.0 - self.hard_top_quantile))
        hard_indices = self._indices[self.sample_weights >= hard_threshold]

        background_label = 0
        non_background_labels = [int(v) for v in self._label_ids.tolist() if int(v) != background_label]
        positive_candidate_labels = [
            label for label in non_background_labels
            if int(self._label_to_indices[label].shape[0]) >= 2
        ]

        batches_with_positive_pair = 0
        batches_with_negative_strata = 0
        fallback_positive_pair_count = 0
        fallback_duplicate_fill_count = 0
        sum_unique_strata = 0.0
        sum_background_fraction = 0.0
        sum_hard_fraction = 0.0

        for _ in range(self.num_batches):
            batch: list[int] = []
            used: set[int] = set()
            strata_in_batch: set[int] = set()

            target_background = int(round(self.batch_size * self.background_fraction))
            target_hard = int(round(self.batch_size * self.hard_fraction))
            target_background = min(target_background, self.batch_size)
            target_hard = min(target_hard, self.batch_size)

            positive_label = None
            if self.require_positive_pair and positive_candidate_labels:
                positive_label = int(rng.choice(np.asarray(positive_candidate_labels, dtype=np.int64)))
                for _pair_pick in range(2):
                    chosen = self._draw_from_pool(rng, self._label_to_indices[positive_label], used)
                    if chosen is None:
                        break
                    batch.append(chosen)
                    used.add(chosen)
                    strata_in_batch.add(int(self.strata_labels[chosen]))
                if len(batch) >= 2:
                    batches_with_positive_pair += 1
                else:
                    fallback_positive_pair_count += 1
            elif self.require_positive_pair:
                fallback_positive_pair_count += 1

            background_pool = self._label_to_indices.get(background_label, np.empty((0,), dtype=np.int64))
            for _ in range(target_background):
                if len(batch) >= self.batch_size:
                    break
                chosen = self._draw_from_pool(rng, background_pool, used)
                if chosen is None:
                    break
                batch.append(chosen)
                used.add(chosen)
                strata_in_batch.add(int(self.strata_labels[chosen]))

            available_negative_labels = [
                label for label in non_background_labels
                if label != positive_label and self._label_to_indices[label].size > 0
            ]
            rng.shuffle(available_negative_labels)
            for label in available_negative_labels[: self.min_negative_strata]:
                if len(batch) >= self.batch_size:
                    break
                chosen = self._draw_from_pool(rng, self._label_to_indices[label], used)
                if chosen is None:
                    continue
                batch.append(chosen)
                used.add(chosen)
                strata_in_batch.add(int(self.strata_labels[chosen]))

            if len([s for s in strata_in_batch if s != background_label]) >= self.min_negative_strata:
                batches_with_negative_strata += 1

            while len(batch) < self.batch_size and hard_indices.size > 0 and target_hard > 0:
                chosen = self._draw_from_pool(rng, hard_indices, used)
                if chosen is None:
                    break
                batch.append(chosen)
                used.add(chosen)
                strata_in_batch.add(int(self.strata_labels[chosen]))
                target_hard -= 1

            while len(batch) < self.batch_size:
                chosen = self._draw_from_pool(rng, self._indices, used)
                if chosen is None:
                    if not self.allow_duplicates:
                        fallback_duplicate_fill_count += 1
                        chosen = int(rng.choice(self._indices, p=self._sample_prob))
                    else:
                        break
                batch.append(chosen)
                used.add(chosen)
                strata_in_batch.add(int(self.strata_labels[chosen]))

            batch_arr = np.asarray(batch, dtype=np.int64)
            batch_labels = self.strata_labels[batch_arr]
            sum_unique_strata += float(np.unique(batch_labels).shape[0])
            sum_background_fraction += float(np.mean(batch_labels == background_label))
            if hard_indices.size > 0:
                hard_mask = np.isin(batch_arr, hard_indices)
                sum_hard_fraction += float(np.mean(hard_mask))

            yield batch

        batch_count = max(1, self.num_batches)
        self._last_epoch_stats = {
            "positive_pair_batch_rate": float(batches_with_positive_pair) / float(batch_count),
            "negative_strata_batch_rate": float(batches_with_negative_strata) / float(batch_count),
            "avg_unique_strata_per_batch": float(sum_unique_strata) / float(batch_count),
            "fallback_positive_pair_count": float(fallback_positive_pair_count),
            "fallback_duplicate_fill_count": float(fallback_duplicate_fill_count),
            "background_fraction_achieved": float(sum_background_fraction) / float(batch_count),
            "hard_fraction_achieved": float(sum_hard_fraction) / float(batch_count),
        }
