from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import logging

import numpy as np

from .artifacts import ensure_artifact

logger = logging.getLogger(__name__)

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import onnxruntime as ort  # type: ignore
except Exception:  # pragma: no cover
    ort = None


def _bbox_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
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


class PoseDetector:
    def __init__(self, cfg: Any):
        logger.debug("Initializing", extra={"task": self.__class__.__name__})
        pose_cfg = getattr(cfg, "pose", cfg)
        device = str(pose_cfg.get("device", "cpu"))
        providers = ["CUDAExecutionProvider"] if "cuda" in device else []
        providers.append("CPUExecutionProvider")
        self.available = ort is not None and cv2 is not None
        self.shape_attrs = {}
        self.session = None

        if not self.available:
            logger.warning("Pose runtime dependencies are unavailable; pose detector will be pass-through")
            return

        model_path = ensure_artifact(str(pose_cfg.get("model_name", "rtmo-s")))
        if not model_path.exists():
            logger.warning("Pose artifact %s is unavailable; pose detector will be pass-through", model_path)
            return
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        logger.debug("Initialized", extra={"task": self.__class__.__name__})

    def get_shape_attrs(self, image):
        if image.shape not in self.shape_attrs:
            self.shape_attrs[image.shape] = self.create_shape_attrs(image)
        return self.shape_attrs[image.shape]

    def create_shape_attrs(self, image, model_shape=(640, 640)):
        original_shape = (image.shape[1], image.shape[0])
        ratio = float(max(model_shape)) / max(original_shape)
        new_shape = tuple(int(x * ratio) for x in original_shape)
        delta_w = model_shape[0] - new_shape[0]
        delta_h = model_shape[1] - new_shape[1]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        return SimpleNamespace(
            ratio=ratio,
            new_shape=new_shape,
            top=top,
            bottom=bottom,
            left=left,
            right=right,
        )

    def __call__(self, frames, tracks):
        if self.session is None or cv2 is None:
            return tracks
        frame_arrays = [frame.array for frame in frames.frames]
        bboxes, keypoints = self.get_keypoints(frame_arrays)
        return self.link_keypoints(keypoints, bboxes, tracks)

    def get_keypoints(self, images):
        input_array = self.preprocess(images)
        raw_bboxes, raw_keypoints = self.session.run(None, {"input": input_array})
        return self.postprocess(raw_bboxes, raw_keypoints, images)

    def preprocess(self, images):
        padded_arrays = [self.resize_with_pad(image) for image in images]
        rgb_arrays = [cv2.cvtColor(array, cv2.COLOR_BGR2RGB) for array in padded_arrays]
        inputs = [array.astype(np.float32).transpose(2, 0, 1) for array in rgb_arrays]
        return np.array(inputs)

    def resize_with_pad(self, image, pad_color=(114, 114, 114)):
        shape_attrs = self.get_shape_attrs(image)
        image = cv2.resize(image, shape_attrs.new_shape)
        return cv2.copyMakeBorder(
            image,
            shape_attrs.top,
            shape_attrs.bottom,
            shape_attrs.left,
            shape_attrs.right,
            cv2.BORDER_CONSTANT,
            value=pad_color,
        )

    def postprocess(self, bboxes, keypoints, images):
        bbox_list = [np.unique(cam_bboxes, axis=0) for cam_bboxes in bboxes]
        keypoints_list = [np.unique(cam_keypoints, axis=0) for cam_keypoints in keypoints]
        processed_bboxes, processed_keypoints = [], []
        for cam_bboxes, cam_keypoints, image in zip(bbox_list, keypoints_list, images):
            shape_attrs = self.get_shape_attrs(image)
            cam_bboxes[..., 0] -= shape_attrs.left
            cam_bboxes[..., 2] -= shape_attrs.left
            cam_bboxes[..., 1] -= shape_attrs.top
            cam_bboxes[..., 3] -= shape_attrs.top
            cam_bboxes[..., 0:4] /= shape_attrs.ratio
            cam_keypoints[..., 0] -= shape_attrs.left
            cam_keypoints[..., 1] -= shape_attrs.top
            cam_keypoints[..., 0:2] /= shape_attrs.ratio
            processed_bboxes.append(cam_bboxes)
            processed_keypoints.append(cam_keypoints)
        return processed_bboxes, processed_keypoints

    def link_keypoints(self, keypoints, bboxes, tracks):
        for cam_keys, cam_bboxes, cam_tracks in zip(keypoints, bboxes, tracks):
            if len(cam_bboxes) == 0 or not cam_tracks:
                continue
            person_tracks = [
                track
                for track in cam_tracks
                if str(getattr(track, "clss", "")).upper() == "PERSON"
            ]
            if not person_tracks:
                continue
            unmatched_tracks = list(range(len(person_tracks)))
            for key_index, key_bbox in enumerate(cam_bboxes[:, :4]):
                best_track_index = None
                best_score = 0.0
                for track_index in unmatched_tracks:
                    score = _bbox_iou(key_bbox, np.asarray(person_tracks[track_index].bbox))
                    if score > best_score:
                        best_score = score
                        best_track_index = track_index
                if best_track_index is None or best_score < 0.1:
                    continue
                unmatched_tracks.remove(best_track_index)
                track = person_tracks[best_track_index]
                keys = cam_keys[key_index]
                track.keypoints = keys
                track.body_visible = self.check_body(keys)
                track.face_visible = self.check_face(keys)
        return tracks

    def check_body(self, keypoints):
        if keypoints is None:
            return False
        return bool(np.mean(keypoints[:, 2]) >= 0.7)

    def check_face(self, keypoints):
        if keypoints is None or keypoints.shape[0] < 5:
            return False
        return bool(keypoints[4, 0] <= keypoints[3, 0])
