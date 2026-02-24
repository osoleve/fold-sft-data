"""Shared train/eval split helpers for SFT dataset generators."""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Dict, List, Set, Tuple


def spread_indices(n: int, k: int) -> Set[int]:
    if k <= 0:
        return set()
    if k >= n:
        return set(range(n))
    if k == 1:
        return {n // 2}
    idxs = {round(i * (n - 1) / (k - 1)) for i in range(k)}
    cursor = 0
    while len(idxs) < k:
        if cursor not in idxs:
            idxs.add(cursor)
        cursor += 1
    return idxs


def _initial_eval_ids(
    samples: List[Dict[str, object]],
    eval_ratio: float,
    eval_min_by_family: Dict[str, int],
) -> Set[str]:
    by_family: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for sample in samples:
        by_family[str(sample["family"])].append(sample)

    def eval_quota_for_family(family: str, family_size: int) -> int:
        if family_size <= 0:
            return 0
        target = round(family_size * eval_ratio)
        floor = eval_min_by_family.get(family, 1)
        return min(family_size, max(floor, target))

    eval_ids: Set[str] = set()
    for family, family_samples in by_family.items():
        picked = spread_indices(
            len(family_samples),
            eval_quota_for_family(family, len(family_samples)),
        )
        for idx, sample in enumerate(family_samples):
            if idx in picked:
                eval_ids.add(str(sample["id"]))
    return eval_ids


def _find(parent: List[int], i: int) -> int:
    while parent[i] != i:
        parent[i] = parent[parent[i]]
        i = parent[i]
    return i


def _union(parent: List[int], a: int, b: int) -> None:
    ra = _find(parent, a)
    rb = _find(parent, b)
    if ra != rb:
        parent[rb] = ra


def _leakage_components(
    samples: List[Dict[str, object]]
) -> Tuple[List[Dict[str, object]], Dict[str, int]]:
    n = len(samples)
    parent = list(range(n))

    by_key: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        by_key[("ground_truth", str(sample["ground_truth"]).strip())].append(idx)
        by_key[("verify_expr", str(sample["verify_expr"]).strip())].append(idx)

    for indices in by_key.values():
        if len(indices) < 2:
            continue
        head = indices[0]
        for idx in indices[1:]:
            _union(parent, head, idx)

    groups: Dict[int, List[int]] = defaultdict(list)
    for idx in range(n):
        groups[_find(parent, idx)].append(idx)

    ordered_groups = sorted(
        groups.values(),
        key=lambda ids: (min(str(samples[i]["id"]) for i in ids), len(ids)),
    )

    components: List[Dict[str, object]] = []
    id_to_component: Dict[str, int] = {}
    for comp_idx, indices in enumerate(ordered_groups):
        ids = {str(samples[i]["id"]) for i in indices}
        families = Counter(str(samples[i]["family"]) for i in indices)
        sources = Counter(str(samples[i]["source_function"]) for i in indices)
        anchor = min(ids)
        components.append(
            {
                "ids": ids,
                "families": families,
                "sources": sources,
                "size": len(ids),
                "anchor": anchor,
            }
        )
        for sid in ids:
            id_to_component[sid] = comp_idx

    return components, id_to_component


def _split_counts(
    components: List[Dict[str, object]],
    selected: Set[int],
) -> Tuple[Counter, Counter, int]:
    family_counts: Counter = Counter()
    source_counts: Counter = Counter()
    total = 0
    for idx in selected:
        component = components[idx]
        family_counts.update(component["families"])  # type: ignore[arg-type]
        source_counts.update(component["sources"])  # type: ignore[arg-type]
        total += int(component["size"])
    return family_counts, source_counts, total


def compute_leakage_aware_eval_ids(
    samples: List[Dict[str, object]],
    eval_ratio: float,
    eval_min_by_family: Dict[str, int],
    enforce_source_function_coverage: bool = True,
) -> Set[str]:
    if not samples:
        return set()

    seed_eval_ids = _initial_eval_ids(samples, eval_ratio, eval_min_by_family)
    components, _ = _leakage_components(samples)

    selected_components: Set[int] = set()
    for idx, component in enumerate(components):
        ids = component["ids"]  # type: ignore[assignment]
        in_eval = sum(1 for sid in ids if sid in seed_eval_ids)
        if in_eval == 0:
            continue
        if in_eval == len(ids):
            selected_components.add(idx)
            continue

        in_train = len(ids) - in_eval
        if in_eval > in_train:
            selected_components.add(idx)
        elif in_eval == in_train:
            anchor = str(component["anchor"])
            if sum(ord(ch) for ch in anchor) % 2 == 0:
                selected_components.add(idx)

    all_source_functions = sorted({str(sample["source_function"]) for sample in samples})
    target_eval_size = max(1, round(len(samples) * eval_ratio))

    def unmet_constraints(
        selected: Set[int],
    ) -> Tuple[Dict[str, int], Set[str], int]:
        family_counts, source_counts, total = _split_counts(components, selected)
        family_deficits = {
            family: max(0, minimum - family_counts[family])
            for family, minimum in eval_min_by_family.items()
        }
        missing_sources = set()
        if enforce_source_function_coverage:
            missing_sources = {fn for fn in all_source_functions if source_counts[fn] == 0}
        size_deficit = max(0, target_eval_size - total)
        return family_deficits, missing_sources, size_deficit

    max_steps = len(components) * 4
    for _ in range(max_steps):
        family_deficits, missing_sources, size_deficit = unmet_constraints(selected_components)
        if not any(family_deficits.values()) and not missing_sources and size_deficit == 0:
            break

        best_idx = -1
        best_score: Tuple[int, int, int, int, str] | None = None
        for idx, component in enumerate(components):
            if idx in selected_components:
                continue

            component_families = component["families"]  # type: ignore[assignment]
            component_sources = component["sources"]  # type: ignore[assignment]
            component_size = int(component["size"])

            family_gain = sum(
                min(deficit, component_families.get(family, 0))
                for family, deficit in family_deficits.items()
                if deficit > 0
            )
            source_gain = len(missing_sources & set(component_sources.keys()))
            size_gain = min(size_deficit, component_size) if size_deficit > 0 else 0

            if family_gain == 0 and source_gain == 0 and size_gain == 0:
                continue

            overshoot = (
                max(0, component_size - size_deficit)
                if size_deficit > 0
                else component_size
            )
            score = (
                source_gain,
                family_gain,
                -overshoot,
                size_gain,
                str(component["anchor"]),
            )
            if best_score is None or score > best_score:
                best_idx = idx
                best_score = score

        if best_idx < 0:
            remaining = [idx for idx in range(len(components)) if idx not in selected_components]
            if not remaining:
                break
            remaining.sort(
                key=lambda idx: (
                    int(components[idx]["size"]),
                    str(components[idx]["anchor"]),
                )
            )
            best_idx = remaining[0]

        selected_components.add(best_idx)

    def constraints_satisfied(selected: Set[int]) -> bool:
        family_counts, source_counts, total = _split_counts(components, selected)
        if total < target_eval_size:
            return False
        for family, minimum in eval_min_by_family.items():
            if family_counts[family] < minimum:
                return False
        if enforce_source_function_coverage:
            for fn in all_source_functions:
                if source_counts[fn] == 0:
                    return False
        return True

    if not constraints_satisfied(selected_components):
        raise ValueError("unable to satisfy leakage-aware eval split constraints")

    changed = True
    while changed:
        changed = False
        ordered = sorted(
            selected_components,
            key=lambda idx: (
                int(components[idx]["size"]),
                str(components[idx]["anchor"]),
            ),
            reverse=True,
        )
        for idx in ordered:
            trial = set(selected_components)
            trial.remove(idx)
            if constraints_satisfied(trial):
                selected_components = trial
                changed = True
                break

    eval_ids: Set[str] = set()
    for idx in selected_components:
        eval_ids.update(components[idx]["ids"])  # type: ignore[arg-type]
    return eval_ids
