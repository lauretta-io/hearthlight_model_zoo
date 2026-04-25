from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    left = max(float(box_a[0]), float(box_b[0]))
    top = max(float(box_a[1]), float(box_b[1]))
    right = min(float(box_a[2]), float(box_b[2]))
    bottom = min(float(box_a[3]), float(box_b[3]))
    if right <= left or bottom <= top:
        return 0.0
    intersection = (right - left) * (bottom - top)
    area_a = max(0.0, float(box_a[2] - box_a[0])) * max(0.0, float(box_a[3] - box_a[1]))
    area_b = max(0.0, float(box_b[2] - box_b[0])) * max(0.0, float(box_b[3] - box_b[1]))
    union = area_a + area_b - intersection
    if union <= 0:
        return 0.0
    return intersection / union


@dataclass
class _TrackState:
    track_id: int
    bbox: np.ndarray
    feature: np.ndarray | None = None
    missed_frames: int = 0


def _cosine_similarity(left: np.ndarray | None, right: np.ndarray | None) -> float:
    if left is None or right is None:
        return 0.0
    left_norm = np.linalg.norm(left)
    right_norm = np.linalg.norm(right)
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return float(np.dot(left, right) / (left_norm * right_norm))


class CommodityTracker:
    def __init__(
        self,
        *,
        match_thresh: float = 0.6,
        track_buffer: int = 30,
        feature_weight: float = 0.0,
    ):
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.feature_weight = feature_weight
        self._next_track_id = 1
        self._tracks: list[_TrackState] = []

    def _score_track(self, track: _TrackState, bbox: np.ndarray, feature: np.ndarray | None) -> float:
        iou_score = _iou(track.bbox, bbox)
        if feature is None or track.feature is None or self.feature_weight <= 0:
            return iou_score
        feature_score = max(0.0, _cosine_similarity(feature, track.feature))
        return ((1.0 - self.feature_weight) * iou_score) + (self.feature_weight * feature_score)

    def _assign_track(self, bbox: np.ndarray, feature: np.ndarray | None) -> int:
        best_track = None
        best_score = 0.0
        for track in self._tracks:
            score = self._score_track(track, bbox, feature)
            if score > best_score:
                best_score = score
                best_track = track
        if best_track is not None and best_score >= self.match_thresh:
            best_track.bbox = bbox
            best_track.feature = feature
            best_track.missed_frames = 0
            return best_track.track_id

        track = _TrackState(track_id=self._next_track_id, bbox=bbox, feature=feature)
        self._next_track_id += 1
        self._tracks.append(track)
        return track.track_id

    def update(self, detections: np.ndarray, features: np.ndarray | None = None) -> np.ndarray:
        for track in self._tracks:
            track.missed_frames += 1

        if detections.size == 0:
            self._tracks = [
                track for track in self._tracks if track.missed_frames <= self.track_buffer
            ]
            return np.empty((0, 5), dtype=np.float32)

        outputs = []
        for index, det in enumerate(detections):
            bbox = np.array(det[:4], dtype=np.float32)
            feature = None
            if features is not None and index < len(features):
                feature = np.asarray(features[index], dtype=np.float32)
            track_id = self._assign_track(bbox, feature)
            outputs.append(np.concatenate([bbox, np.array([track_id], dtype=np.float32)]))

        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self.track_buffer
        ]
        return np.vstack(outputs).astype(np.float32)


class ByteTrackTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.6, track_buffer=30, feature_weight=0.0)


class FastByteTrackTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.5, track_buffer=18, feature_weight=0.0)


class BalancedByteTrackTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.55, track_buffer=24, feature_weight=0.0)


class OCSortTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.45, track_buffer=20, feature_weight=0.0)


class BoTSORTTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.52, track_buffer=30, feature_weight=0.2)


class StrongSORTTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.48, track_buffer=36, feature_weight=0.45)


class CMTrackTracker(CommodityTracker):
    def __init__(self):
        super().__init__(match_thresh=0.65, track_buffer=45, feature_weight=0.15)


def get_tracker(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"bytetrack", "builtin_bytetrack"}:
        return ByteTrackTracker()
    if normalized in {"bytetrack-s", "bytetrack-fast"}:
        return FastByteTrackTracker()
    if normalized in {"bytetrack-balanced", "hearthlight-balanced"}:
        return BalancedByteTrackTracker()
    if normalized in {"ocsort", "oc-sort", "ocsort-tuned"}:
        return OCSortTracker()
    if normalized in {"botsort", "bot-sort"}:
        return BoTSORTTracker()
    if normalized in {"strongsort", "strong-sort"}:
        return StrongSORTTracker()
    if normalized in {"cmtrack", "builtin_cmtrack"}:
        return CMTrackTracker()
    raise ValueError(f"unsupported tracker {name}")
