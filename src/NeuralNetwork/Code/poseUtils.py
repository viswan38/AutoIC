"""
Shared pose extraction and feature engineering for the AutoIC CoreSet (SARD) pipeline.

Used by processSARD.py, testSARDPose.py, EvaluateCoreDataset.py and trainSequenceModel.py
so every script samples frames, picks the worker, and builds features the same way.

Main design decisions (and why):
- Frames are read sequentially at a fixed target rate (default 15 fps) instead of seeking
  to 30 evenly spaced frames. Hammering and sawing repeat 2-4 times per second; sampling
  every ~0.3 s aliases that motion, and seeking is unreliable for .webm files.
- The same worker is followed across frames (highest overlap with the previous box)
  instead of re-picking the largest person in every frame.
- Keypoints are expressed relative to the hip midpoint and scaled by the worker's median
  box height, so features describe body motion rather than camera distance or resolution.
- Detection counts and missing-value flags are kept as quality-control columns only. In the
  old dataset they predicted the action better than the motion features did, which means
  the model was learning camera framing rather than worker movement.
"""

import math
import os
import re
import warnings

import cv2
import numpy as np

# COCO keypoint order used by YOLO pose models
KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]
NUM_KEYPOINTS = 17
L_SHOULDER, R_SHOULDER = 5, 6
L_ELBOW, R_ELBOW = 7, 8
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12
L_KNEE, R_KNEE = 13, 14
L_ANKLE, R_ANKLE = 15, 16

# Left/right pairs, used to mirror a skeleton for data augmentation
FLIP_PAIRS = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10), (11, 12), (13, 14), (15, 16)]

VIDEO_EXTENSIONS = (".mp4", ".webm", ".avi", ".mov", ".mkv")

DEFAULT_TARGET_FPS = 15.0
DEFAULT_MAX_SECONDS = 20.0
DEFAULT_KEYPOINT_CONF = 0.3

# Columns that describe the recording or detection quality, not worker motion.
# These are never used as model inputs.
META_COLUMNS = [
    "video", "action", "source_video", "sequence_file",
    "sampled_frames", "detected_frames", "detection_rate",
    "effective_fps", "duration_s", "scale_px",
    # columns from the older CSV formats
    "total_frames", "valid_wrist_measurements", "valid_elbow_measurements",
    "valid_hip_measurements", "valid_body_measurements", "fold",
]


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------

def source_video_id(filename):
    """
    Return the original YouTube video a clip was cut from.

    Clips such as 'MNry7eqwc_Q_0_20_0_31.mp4' and 'MNry7eqwc_Q_0_32_0_50.mp4' come from the
    same recording, so they must stay on the same side of a train/test split.
    """
    name = re.sub(r"\.(mp4|webm|avi|mov|mkv)$", "", str(filename), flags=re.IGNORECASE)
    return re.sub(r"(_\d+){4}$", "", name)


def list_videos(dataset_path):
    """Yield (action, filename, full_path) for every video in dataset_path/<action>/."""
    for action in sorted(os.listdir(dataset_path)):
        action_path = os.path.join(dataset_path, action)
        if not os.path.isdir(action_path):
            continue
        for filename in sorted(os.listdir(action_path)):
            if filename.lower().endswith(VIDEO_EXTENSIONS) and not filename.startswith("."):
                yield action, filename, os.path.join(action_path, filename)


def assign_folds(df, n_splits=5, seed=42):
    """
    Assign each recording to a cross-validation fold, keeping all clips from the same
    source video in the same fold and balancing classes across folds.

    Every training script calls this, so all models are compared on identical folds.
    """
    from sklearn.model_selection import StratifiedGroupKFold

    groups = df["video"].map(source_video_id).values
    folds = np.full(len(df), -1, dtype=int)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (_, test_index) in enumerate(splitter.split(df, df["action"], groups)):
        folds[test_index] = fold
    return folds, groups


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def load_pose_model(model_path="yolo11n-pose.pt"):
    from ultralytics import YOLO
    return YOLO(model_path)


def _box_iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def select_person(boxes, previous_box=None, min_iou=0.3):
    """
    Pick which detected person to follow.

    If we were already following someone, keep the box that overlaps them most.
    Otherwise (first frame, or the worker was lost) take the largest person.
    """
    if len(boxes) == 0:
        return None
    if previous_box is not None:
        ious = [_box_iou(box, previous_box) for box in boxes]
        best = int(np.argmax(ious))
        if ious[best] >= min_iou:
            return best
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return int(np.argmax(areas))


def extract_sequence(model, video_path, target_fps=DEFAULT_TARGET_FPS,
                     max_seconds=DEFAULT_MAX_SECONDS, detection_conf=0.25,
                     device=None, image_size=640):
    """
    Run the pose model on a video at a fixed sampling rate and follow one worker.

    Frames are decoded sequentially (no seeking). Videos longer than max_seconds are
    trimmed to their middle portion.

    Returns a dict of numpy arrays, with NaN wherever the worker was not detected:
        keypoints  (T, 17, 2)  pixel coordinates
        confidence (T, 17)     keypoint confidence
        boxes      (T, 4)      person box x1, y1, x2, y2
        times      (T,)        seconds from the start of the video
        num_people (T,)        people detected in each frame
    plus the scalars fps, effective_fps, frame_width, frame_height, total_frames.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError("Could not open video: " + str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Some .webm files report 0 or absurd fps; fall back to 30
    if not fps or fps <= 1 or fps > 240 or math.isnan(fps):
        fps = 30.0

    step = max(1, int(round(fps / target_fps)))
    effective_fps = fps / step
    max_samples = max(1, int(round(max_seconds * effective_fps)))

    # For long videos, start so the analysed window sits in the middle
    window_frames = max_samples * step
    start_frame = 0
    if total_frames > window_frames:
        start_frame = (total_frames - window_frames) // 2

    keypoints, confidence, boxes, times, num_people = [], [], [], [], []
    previous_box = None
    frame_index = -1

    predict_kwargs = {"verbose": False, "conf": detection_conf, "imgsz": image_size}
    if device is not None:
        predict_kwargs["device"] = device

    while len(times) < max_samples:
        if not cap.grab():
            break
        frame_index += 1
        if frame_index < start_frame or (frame_index - start_frame) % step != 0:
            continue

        ok, frame = cap.retrieve()
        if not ok:
            break

        result = model(frame, **predict_kwargs)[0]

        frame_keypoints = np.full((NUM_KEYPOINTS, 2), np.nan, dtype=np.float32)
        frame_confidence = np.zeros(NUM_KEYPOINTS, dtype=np.float32)
        frame_box = np.full(4, np.nan, dtype=np.float32)
        people = 0

        if result.boxes is not None and len(result.boxes) > 0 and result.keypoints is not None:
            all_boxes = result.boxes.xyxy.cpu().numpy()
            all_keypoints = result.keypoints.xy.cpu().numpy()
            all_confidence = (
                result.keypoints.conf.cpu().numpy()
                if result.keypoints.conf is not None
                else np.ones(all_keypoints.shape[:2], dtype=np.float32)
            )
            people = len(all_boxes)
            chosen = select_person(all_boxes, previous_box)

            if chosen is not None and chosen < len(all_keypoints):
                frame_keypoints = all_keypoints[chosen].astype(np.float32)
                frame_confidence = all_confidence[chosen].astype(np.float32)
                frame_box = all_boxes[chosen].astype(np.float32)
                # YOLO reports undetected keypoints as (0, 0)
                frame_keypoints[frame_confidence <= 0] = np.nan
                previous_box = frame_box
        else:
            # Worker lost; next detection may be someone new, so pick the largest again
            previous_box = None

        keypoints.append(frame_keypoints)
        confidence.append(frame_confidence)
        boxes.append(frame_box)
        times.append(frame_index / fps)
        num_people.append(people)

    cap.release()

    return {
        "keypoints": np.array(keypoints, dtype=np.float32).reshape(-1, NUM_KEYPOINTS, 2),
        "confidence": np.array(confidence, dtype=np.float32).reshape(-1, NUM_KEYPOINTS),
        "boxes": np.array(boxes, dtype=np.float32).reshape(-1, 4),
        "times": np.array(times, dtype=np.float64),
        "num_people": np.array(num_people, dtype=np.int32),
        "fps": float(fps),
        "effective_fps": float(effective_fps),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "total_frames": total_frames,
    }


def save_sequence(path, sequence):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez_compressed(path, **sequence)


def load_sequence(path):
    with np.load(path) as data:
        sequence = {key: data[key] for key in data.files}
    for key in ("fps", "effective_fps"):
        sequence[key] = float(sequence[key])
    for key in ("frame_width", "frame_height", "total_frames"):
        sequence[key] = int(sequence[key])
    return sequence


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _nanmean_rows(values, axis):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(values, axis=axis)


def normalize_sequence(sequence, keypoint_conf=DEFAULT_KEYPOINT_CONF):
    """
    Convert pixel keypoints into body-relative coordinates.

    local[t, j] = (keypoint - body centre) / worker height, so 0.1 means
    10% of the worker's height. The body centre is the hip midpoint, falling back to
    the box centre when the hips are not visible. Image y points down.
    """
    keypoints = sequence["keypoints"].astype(np.float64).copy()
    confidence = np.nan_to_num(sequence["confidence"].astype(np.float64), nan=0.0)
    keypoints[confidence < keypoint_conf] = np.nan
    boxes = sequence["boxes"].astype(np.float64)

    heights = boxes[:, 3] - boxes[:, 1]
    widths = boxes[:, 2] - boxes[:, 0]
    finite_heights = heights[np.isfinite(heights) & (heights > 1)]
    scale = float(np.median(finite_heights)) if len(finite_heights) else np.nan

    hip_mid = _nanmean_rows(keypoints[:, [L_HIP, R_HIP]], axis=1)
    box_center = np.stack([(boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2], axis=1)
    from_hip = np.isfinite(hip_mid).all(axis=1)
    center = np.where(from_hip[:, None], hip_mid, box_center)

    local = (keypoints - center[:, None, :]) / scale
    return {
        "local": local,
        "center": center / scale,
        "center_from_hip": from_hip,
        "confidence": confidence,
        "aspect": widths / heights,
        "times": sequence["times"].astype(np.float64),
        "effective_fps": sequence["effective_fps"],
        "scale": scale,
        "detected": np.isfinite(boxes).all(axis=1),
    }


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _stats(values, prefix):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {prefix + s: np.nan for s in ("_mean", "_std", "_p10", "_p50", "_p90", "_max")}
    return {
        prefix + "_mean": values.mean(),
        prefix + "_std": values.std(),
        prefix + "_p10": np.percentile(values, 10),
        prefix + "_p50": np.percentile(values, 50),
        prefix + "_p90": np.percentile(values, 90),
        prefix + "_max": values.max(),
    }


def _velocity(positions, times, same_reference):
    """Per-step velocity (T-1, 2) in body-heights per second; NaN across gaps."""
    dt = np.diff(times)
    with np.errstate(invalid="ignore", divide="ignore"):
        velocity = np.diff(positions, axis=0) / dt[:, None]
    ok = np.isfinite(velocity).all(axis=1) & (dt > 0) & same_reference
    velocity[~ok] = np.nan
    return velocity


def _speed(velocity):
    return np.linalg.norm(velocity, axis=1)


def _angle(a, b, c):
    """Joint angle at b in degrees for (T, 2) arrays; 180 = fully straight."""
    v1, v2 = a - b, c - b
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = (v1 * v2).sum(axis=1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))
    return np.degrees(np.arccos(np.clip(cosine, -1, 1)))


def _rhythm(signal, sample_rate, min_hz=0.4, max_hz=6.0):
    """
    Dominant frequency (Hz) of a 1-D motion signal and how periodic it is
    (fraction of in-band spectral power near the peak; ~1 = perfectly rhythmic).
    """
    signal = np.asarray(signal, dtype=np.float64)
    valid = np.isfinite(signal)
    if len(signal) < 16 or valid.mean() < 0.5:
        return np.nan, np.nan
    index = np.arange(len(signal))
    filled = np.interp(index, index[valid], signal[valid])
    filled = filled - np.polyval(np.polyfit(index, filled, 1), index)
    spectrum = np.abs(np.fft.rfft(filled * np.hanning(len(filled)))) ** 2
    freqs = np.fft.rfftfreq(len(filled), d=1.0 / sample_rate)
    band = (freqs >= min_hz) & (freqs <= min(max_hz, sample_rate / 2))
    if band.sum() < 2 or spectrum[band].sum() <= 0:
        return np.nan, np.nan
    band_freqs, band_power = freqs[band], spectrum[band]
    peak = int(np.argmax(band_power))
    near_peak = band_power[max(0, peak - 1): peak + 2].sum()
    return float(band_freqs[peak]), float(near_peak / band_power.sum())


def compute_features(sequence, keypoint_conf=DEFAULT_KEYPOINT_CONF):
    """
    Summarise one recording as a fixed-length feature vector.

    Feature groups:
      speed     how fast hands, elbows, knees, feet and head move relative to the body
      direction whether hand motion is mostly vertical (hammering) or horizontal (sawing)
      rhythm    dominant frequency and periodicity of the most active hand
      posture   elbow/knee angles, trunk lean, hand height, hand spacing, body shape
      travel    how the whole body moves through the frame (carrying, climbing)
    """
    norm = normalize_sequence(sequence, keypoint_conf)
    local, times = norm["local"], norm["times"]
    fps = norm["effective_fps"]
    n = len(times)
    detected = int(norm["detected"].sum())

    features = {
        "sampled_frames": n,
        "detected_frames": detected,
        "detection_rate": detected / n if n else 0.0,
        "effective_fps": fps,
        "duration_s": float(times[-1] - times[0]) if n > 1 else 0.0,
        "scale_px": norm["scale"],
    }

    if n < 2:
        same_reference = np.zeros(0, dtype=bool)
    else:
        same_reference = norm["center_from_hip"][1:] == norm["center_from_hip"][:-1]

    def joint_velocity(j):
        if n < 2:
            return np.full((0, 2), np.nan)
        return _velocity(local[:, j], times, same_reference)

    # --- speed -----------------------------------------------------------
    v_lw, v_rw = joint_velocity(L_WRIST), joint_velocity(R_WRIST)
    s_lw, s_rw = _speed(v_lw), _speed(v_rw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        active_hand = np.fmax(s_lw, s_rw)    # faster hand in each step
        support_hand = np.fmin(s_lw, s_rw)   # slower hand in each step
        elbow = np.nanmean(np.stack([_speed(joint_velocity(L_ELBOW)), _speed(joint_velocity(R_ELBOW))]), axis=0)
        knee = np.nanmean(np.stack([_speed(joint_velocity(L_KNEE)), _speed(joint_velocity(R_KNEE))]), axis=0)
        ankle = np.nanmean(np.stack([_speed(joint_velocity(L_ANKLE)), _speed(joint_velocity(R_ANKLE))]), axis=0)
        head = _speed(joint_velocity(0))

    features.update(_stats(active_hand, "active_hand_speed"))
    features.update(_stats(support_hand, "support_hand_speed"))
    features.update(_stats(elbow, "elbow_speed"))
    features.update(_stats(knee, "knee_speed"))
    features.update(_stats(ankle, "ankle_speed"))
    features.update(_stats(head, "head_speed"))
    with np.errstate(invalid="ignore", divide="ignore"):
        features["hand_symmetry"] = np.nanmean(support_hand) / np.nanmean(active_hand) \
            if np.isfinite(active_hand).any() else np.nan

    # --- direction -------------------------------------------------------
    hand_v = np.concatenate([v_lw, v_rw]) if n >= 2 else np.full((0, 2), np.nan)
    hand_v = hand_v[np.isfinite(hand_v).all(axis=1)]
    if len(hand_v):
        abs_x, abs_y = np.abs(hand_v[:, 0]).sum(), np.abs(hand_v[:, 1]).sum()
        features["hand_vertical_fraction"] = abs_y / (abs_x + abs_y) if abs_x + abs_y > 0 else np.nan
    else:
        features["hand_vertical_fraction"] = np.nan

    # --- rhythm (most active hand with enough data) -------------------------
    best_hand, best_score = None, -1.0
    for j, s in ((L_WRIST, s_lw), (R_WRIST, s_rw)):
        coverage = np.isfinite(local[:, j, 1]).mean() if n else 0
        score = np.nanmean(s) if np.isfinite(s).any() else -1
        if coverage >= 0.5 and score > best_score:
            best_hand, best_score = j, score
    if best_hand is not None:
        freq_y, period_y = _rhythm(local[:, best_hand, 1], fps)
        freq_x, period_x = _rhythm(local[:, best_hand, 0], fps)
    else:
        freq_y = period_y = freq_x = period_x = np.nan
    features["hand_freq_vertical_hz"] = freq_y
    features["hand_periodicity_vertical"] = period_y
    features["hand_freq_horizontal_hz"] = freq_x
    features["hand_periodicity_horizontal"] = period_x

    # --- posture ---------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        elbow_angles = np.stack([
            _angle(local[:, L_SHOULDER], local[:, L_ELBOW], local[:, L_WRIST]),
            _angle(local[:, R_SHOULDER], local[:, R_ELBOW], local[:, R_WRIST]),
        ])
        knee_angles = np.stack([
            _angle(local[:, L_HIP], local[:, L_KNEE], local[:, L_ANKLE]),
            _angle(local[:, R_HIP], local[:, R_KNEE], local[:, R_ANKLE]),
        ])
        shoulder_mid = np.nanmean(local[:, [L_SHOULDER, R_SHOULDER]], axis=1)
        hip_mid = np.nanmean(local[:, [L_HIP, R_HIP]], axis=1)
        trunk = shoulder_mid - hip_mid
        trunk_lean = np.degrees(np.arctan2(np.abs(trunk[:, 0]), -trunk[:, 1]))  # 0 = upright

        wrists_y = local[:, [L_WRIST, R_WRIST], 1]
        hand_above_shoulder = shoulder_mid[:, 1:2] - wrists_y      # positive = above shoulders
        hand_above_hip = hip_mid[:, 1:2] - wrists_y
        highest_hand = np.nanmax(hand_above_shoulder, axis=1)
        hand_gap = np.linalg.norm(local[:, L_WRIST] - local[:, R_WRIST], axis=1)

        features.update(_stats(np.nanmean(elbow_angles, axis=0), "elbow_angle"))
        features.update(_stats(np.nanmean(knee_angles, axis=0), "knee_angle"))
        features.update(_stats(trunk_lean, "trunk_lean"))
        features.update(_stats(highest_hand, "hand_above_shoulder"))
        features.update(_stats(np.nanmean(hand_above_hip, axis=1), "hand_above_hip"))
        features.update(_stats(hand_gap, "hand_gap"))
        features.update(_stats(norm["aspect"], "body_aspect"))

        # range of motion of each hand around the body
        hand_range = [np.nanstd(local[:, j], axis=0) for j in (L_WRIST, R_WRIST)]
        hand_range = np.array(hand_range)
        features["hand_range_x"] = float(np.nanmax(hand_range[:, 0])) if np.isfinite(hand_range[:, 0]).any() else np.nan
        features["hand_range_y"] = float(np.nanmax(hand_range[:, 1])) if np.isfinite(hand_range[:, 1]).any() else np.nan

        # how much the overall body configuration changes
        pose_variation = np.nanstd(local, axis=0)  # (17, 2)
        features["pose_variation"] = float(np.nanmean(pose_variation)) if np.isfinite(pose_variation).any() else np.nan

    # --- travel through the frame ------------------------------------------
    if n >= 2:
        center_velocity = _velocity(norm["center"], times, same_reference)
        features.update(_stats(_speed(center_velocity), "body_travel_speed"))
        cv = center_velocity[np.isfinite(center_velocity).all(axis=1)]
        if len(cv):
            ax, ay = np.abs(cv[:, 0]).sum(), np.abs(cv[:, 1]).sum()
            features["body_travel_vertical_fraction"] = ay / (ax + ay) if ax + ay > 0 else np.nan
            features["body_net_rise_per_s"] = float(-np.nanmean(cv[:, 1]))   # positive = moving up
        else:
            features["body_travel_vertical_fraction"] = np.nan
            features["body_net_rise_per_s"] = np.nan
    else:
        features.update(_stats([], "body_travel_speed"))
        features["body_travel_vertical_fraction"] = np.nan
        features["body_net_rise_per_s"] = np.nan

    return features


def model_feature_columns(df):
    """Numeric columns that describe worker motion (everything except META_COLUMNS)."""
    return [
        c for c in df.columns
        if c not in META_COLUMNS and np.issubdtype(df[c].dtype, np.number)
    ]