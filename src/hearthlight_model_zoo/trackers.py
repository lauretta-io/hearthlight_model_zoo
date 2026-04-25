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
    missed_frames: int = 0


class ByteTrackTracker:
    def __init__(self, *, match_thresh: float = 0.6, track_buffer: int = 30):
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self._next_track_id = 1
        self._tracks: list[_TrackState] = []

    def _assign_track(self, bbox: np.ndarray) -> int:
        best_track = None
        best_iou = 0.0
        for track in self._tracks:
            score = _iou(track.bbox, bbox)
            if score > best_iou:
                best_iou = score
                best_track = track
        if best_track is not None and best_iou >= self.match_thresh:
            best_track.bbox = bbox
            best_track.missed_frames = 0
            return best_track.track_id

        track = _TrackState(track_id=self._next_track_id, bbox=bbox)
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
        for det in detections:
            bbox = np.array(det[:4], dtype=np.float32)
            track_id = self._assign_track(bbox)
            outputs.append(np.concatenate([bbox, np.array([track_id], dtype=np.float32)]))

        self._tracks = [
            track for track in self._tracks if track.missed_frames <= self.track_buffer
        ]
        return np.vstack(outputs).astype(np.float32)


def get_tracker(name: str):
    normalized = str(name).strip().lower()
    if normalized in {"bytetrack", "bytetrack-s", "builtin_bytetrack", "cmtrack"}:
        return ByteTrackTracker()
    raise ValueError(f"unsupported tracker {name}")
