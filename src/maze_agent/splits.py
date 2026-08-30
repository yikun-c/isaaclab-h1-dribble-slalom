"""Seed manifests that prevent accidental final-test leakage."""

from __future__ import annotations

from dataclasses import dataclass
from random import Random


@dataclass(frozen=True)
class SplitManifest:
    generator_version: str
    master_seed: int
    train_seeds: tuple[int, ...]
    development_seeds: tuple[int, ...]
    iid_final_seeds: tuple[int, ...]
    ood_final_seeds: tuple[int, ...]

    def __post_init__(self) -> None:
        groups = (self.train_seeds, self.development_seeds, self.iid_final_seeds, self.ood_final_seeds)
        flattened = [seed for group in groups for seed in group]
        if len(flattened) != len(set(flattened)):
            raise ValueError("maze split seeds must be disjoint")
        if not all(group for group in groups):
            raise ValueError("every split must contain at least one seed")

    @property
    def sealed_final_seeds(self) -> frozenset[int]:
        return frozenset(self.iid_final_seeds + self.ood_final_seeds)

    def assert_trainable(self, seed: int) -> None:
        if seed in self.sealed_final_seeds:
            raise PermissionError(f"seed {seed} is sealed for final evaluation")


def build_split_manifest(
    master_seed: int,
    train_count: int = 2000,
    development_count: int = 200,
    iid_final_count: int = 500,
    ood_final_count: int = 500,
) -> SplitManifest:
    counts = (train_count, development_count, iid_final_count, ood_final_count)
    if any(count <= 0 for count in counts):
        raise ValueError("all split sizes must be positive")
    total = sum(counts)
    rng = Random(master_seed)
    population = list(range(1, total * 20 + 1))
    rng.shuffle(population)
    selected = population[:total]
    cursor = 0
    groups: list[tuple[int, ...]] = []
    for count in counts:
        groups.append(tuple(sorted(selected[cursor : cursor + count])))
        cursor += count
    return SplitManifest("maze-v1", master_seed, *groups)
