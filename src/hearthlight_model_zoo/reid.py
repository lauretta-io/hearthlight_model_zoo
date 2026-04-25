from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


def _cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


def _centroid_distance(box_a, box_b) -> float:
    ax = (float(box_a[0]) + float(box_a[2])) / 2.0
    ay = (float(box_a[1]) + float(box_a[3])) / 2.0
    bx = (float(box_b[0]) + float(box_b[2])) / 2.0
    by = (float(box_b[1]) + float(box_b[3])) / 2.0
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


@dataclass
class _Entity:
    entity_id: int
    feature: np.ndarray | None
    feature_count: int = 0
    bbox: np.ndarray | None = None


class _VectorIndex:
    def __init__(self):
        self._vectors: dict[int, np.ndarray] = {}

    def update(self, entity_id: int, feature: np.ndarray | None) -> None:
        if feature is None:
            return
        self._vectors[entity_id] = feature.astype(np.float32)

    def get_nearest_ids(self, feature: np.ndarray, max_matches: int, match_threshold: float) -> list[int]:
        scored = []
        for entity_id, stored in self._vectors.items():
            score = _cosine_similarity(feature, stored)
            if score >= match_threshold:
                scored.append((score, entity_id))
        scored.sort(reverse=True)
        return [entity_id for _, entity_id in scored[:max_matches]]


class _Strategy:
    def __init__(self):
        self.vectors = _VectorIndex()


class _EntityStore:
    def __init__(self):
        self._rows: dict[int, _Entity] = {}
        self._track_assignments: dict[int, int] = {}

    def set_assignment(self, track_id: int, entity_id: int) -> None:
        self._track_assignments[track_id] = entity_id

    def get_ids(self, tracks) -> list[int]:
        return [self._track_assignments.get(track.track_id, -1) for track in tracks]

    def get(self, entity_id: int):
        return self._rows.get(entity_id)

    def put(self, entity: _Entity) -> None:
        self._rows[entity.entity_id] = entity

    def pop(self, entity_id: int):
        return self._rows.pop(entity_id, None)

    def items(self):
        return self._rows.items()


class _BaseReIDModule:
    def __init__(self):
        self.entities = _EntityStore()
        self.strategy = _Strategy()
        self._last_promotions: dict[int, int] = {}

    def get_temp_to_real_update(self) -> dict[int, int]:
        updates = dict(self._last_promotions)
        self._last_promotions.clear()
        return updates


class _TransReIDPersonModule(_BaseReIDModule):
    def __init__(self, cfg):
        super().__init__()
        self.match_threshold = float(cfg.get("high_threshold", cfg.get("threshold", 0.76)))
        self.predict_threshold = float(cfg.get("low_threshold", 0.71))
        self.min_features = int(cfg.get("min_features", 10))
        self._next_temp_id = -1
        self._next_real_id = 1

    def _match_existing(self, feature: np.ndarray | None) -> int | None:
        if feature is None:
            return None
        best_id = None
        best_score = 0.0
        for entity_id, entity in self.entities.items():
            if entity.feature is None:
                continue
            score = _cosine_similarity(feature, entity.feature)
            if score > best_score:
                best_score = score
                best_id = entity_id
        if best_id is not None and best_score >= self.match_threshold:
            return best_id
        return None

    def _promote_if_ready(self, entity: _Entity) -> int:
        if entity.entity_id > 0 or entity.feature_count < self.min_features:
            return entity.entity_id
        previous_id = entity.entity_id
        new_id = self._next_real_id
        self._next_real_id += 1
        self.entities.pop(previous_id)
        entity.entity_id = new_id
        self.entities.put(entity)
        self.strategy.vectors.update(new_id, entity.feature)
        self._last_promotions[previous_id] = new_id
        return new_id

    def reid(self, tracks) -> None:
        for track in tracks:
            feature = getattr(track, "feature", None)
            feature = None if feature is None else np.asarray(feature, dtype=np.float32)
            entity_id = self._match_existing(feature)
            if entity_id is None:
                entity_id = self._next_temp_id
                self._next_temp_id -= 1
                entity = _Entity(entity_id=entity_id, feature=feature, feature_count=0)
                self.entities.put(entity)
            entity = self.entities.get(entity_id)
            assert entity is not None
            if feature is not None:
                entity.feature = feature if entity.feature is None else ((entity.feature * entity.feature_count) + feature) / (entity.feature_count + 1)
                self.strategy.vectors.update(entity.entity_id, entity.feature)
            entity.feature_count += 1
            assigned_id = self._promote_if_ready(entity)
            self.entities.set_assignment(track.track_id, assigned_id)

    def predict(self, entity_id: int):
        entity = self.entities.get(entity_id)
        if entity is None or entity.feature is None:
            return None
        candidates = self.strategy.vectors.get_nearest_ids(entity.feature, 1, self.predict_threshold)
        return candidates[0] if candidates else None


class _HybridBagModule(_BaseReIDModule):
    def __init__(self, cfg):
        super().__init__()
        self.distance_threshold = float(cfg.get("distance_threshold", 80.0))
        self.min_features = int(cfg.get("min_features", 3))
        self._next_temp_id = -1
        self._next_real_id = 1

    def _match_existing(self, bbox: np.ndarray) -> int | None:
        best_id = None
        best_distance = float("inf")
        for entity_id, entity in self.entities.items():
            if entity.bbox is None:
                continue
            distance = _centroid_distance(entity.bbox, bbox)
            if distance < best_distance:
                best_distance = distance
                best_id = entity_id
        if best_id is not None and best_distance <= self.distance_threshold:
            return best_id
        return None

    def reid(self, tracks) -> None:
        for track in tracks:
            bbox = np.asarray(track.bbox, dtype=np.float32)
            entity_id = self._match_existing(bbox)
            if entity_id is None:
                entity_id = self._next_temp_id
                self._next_temp_id -= 1
                self.entities.put(_Entity(entity_id=entity_id, feature=None, feature_count=0, bbox=bbox))
            entity = self.entities.get(entity_id)
            assert entity is not None
            entity.bbox = bbox
            entity.feature_count += 1
            if entity.entity_id < 0 and entity.feature_count >= self.min_features:
                previous_id = entity.entity_id
                new_id = self._next_real_id
                self._next_real_id += 1
                self.entities.pop(previous_id)
                entity.entity_id = new_id
                self.entities.put(entity)
                self._last_promotions[previous_id] = new_id
            self.entities.set_assignment(track.track_id, entity.entity_id)

    def predict(self, entity_id: int):
        return None


class PersonReIDBundle:
    def __init__(self, cfg, registration: dict, namespace: int = 0):
        self.namespace = namespace
        self.registration = registration
        self.person_reid = _TransReIDPersonModule(cfg.reid["person"])
        self.bag_reid = _HybridBagModule(cfg.reid["bag"])

    def search(self, feature, max_matches: int, match_threshold: float) -> list[int]:
        if feature is None:
            return []
        return self.person_reid.strategy.vectors.get_nearest_ids(
            np.asarray(feature, dtype=np.float32),
            max_matches,
            match_threshold,
        )
