import argparse
import os

import cv2
import numpy as np
import pandas as pd

import poseUtils
import trainSequenceModel as tsm

DEFAULT_DATASET = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/CoreSet"
DEFAULT_DATA_DIR = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/src/NeuralNetwork/Data"

BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12),
         (11, 13), (13, 15), (12, 14), (14, 16), (0, 5), (0, 6)]
SNAPSHOT_SECONDS = [0.5, 1.0, 2.0, 3.0]

# BGR colours
GREEN = (90, 200, 60)
ORANGE = (40, 150, 245)
WHITE = (245, 245, 245)
GREY = (150, 150, 150)
DARK = (32, 32, 32)
PANEL = (24, 24, 24)
CYAN = (230, 200, 60)
MAGENTA = (200, 80, 220)

VIDEO_HEIGHT = 540
PANEL_WIDTH = 500
CHART_HEIGHT = 230


# ---------------------------------------------------------------------------
# Choosing the video and the model
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="folder with sard_pose_features.csv, sequences/ and fold_models/")
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="folder with the original videos")
    parser.add_argument("--video", default=None, help="video file name (as in the features CSV)")
    parser.add_argument("--action", default="Saw", help="used when --video is not given")
    parser.add_argument("--pick", choices=["typical", "best", "worst", "random"], default="typical",
                        help="typical = correctly recognised with median confidence (default)")
    parser.add_argument("--output", default=None, help="default: <data-dir>/demo/<video>_demo.mp4")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def choose_video(df, args):
    if args.video:
        match = df[(df["video"] == args.video) | (df["video"] == os.path.basename(args.video))]
        if match.empty:
            raise SystemExit(f"{args.video} is not in the features CSV")
        return match.iloc[0], "chosen with --video"

    rows = df[df["action"] == args.action]
    if rows.empty:
        raise SystemExit(f"No videos for action '{args.action}'. Options: {sorted(df['action'].unique())}")

    oof_path = os.path.join(args.data_dir, "oof_sequence_probs.csv")
    if not os.path.exists(oof_path) or args.pick == "random":
        return rows.sample(1, random_state=args.seed).iloc[0], "random clip"

    oof = pd.read_csv(oof_path)
    prob_columns = [c for c in oof.columns if c.startswith("prob_")]
    oof["true_prob"] = oof.apply(lambda r: r.get("prob_" + r["action"], np.nan), axis=1)
    oof["predicted"] = oof[prob_columns].idxmax(axis=1).str[len("prob_"):]
    candidates = rows.merge(oof[["video", "true_prob", "predicted"]], on="video")
    if "detection_rate" in candidates:
        well_detected = candidates[candidates["detection_rate"] >= 0.7]
        if len(well_detected):
            candidates = well_detected
    if candidates.empty:
        return rows.sample(1, random_state=args.seed).iloc[0], "random clip"

    correct = candidates[candidates["predicted"] == candidates["action"]]
    if args.pick == "best":
        return candidates.sort_values("true_prob").iloc[-1], "highest-confidence clip (cherry-picked)"
    if args.pick == "worst":
        return candidates.sort_values("true_prob").iloc[0], "lowest-confidence clip"
    pool = correct if len(correct) else candidates
    pool = pool.sort_values("true_prob").reset_index(drop=True)
    note = "typical correctly recognised clip" if len(correct) else "typical clip (none recognised correctly)"
    return pool.iloc[len(pool) // 2], note


def load_fold_network(models_dir, sequence_file, device):
    import torch

    if not os.path.isdir(models_dir):
        raise SystemExit(f"{models_dir} not found. Run trainSequenceModel.py first; it saves one model per fold.")
    for name in sorted(os.listdir(models_dir)):
        if not name.endswith(".pt"):
            continue
        checkpoint = torch.load(os.path.join(models_dir, name), map_location="cpu", weights_only=False)
        if sequence_file in checkpoint["test_sequence_files"]:
            network = tsm.build_network(len(checkpoint["classes"]), width=checkpoint["width"])
            network.load_state_dict(checkpoint["state_dict"])
            return network.to(device).eval(), checkpoint
    raise SystemExit(f"No fold model held out {sequence_file}. Re-run trainSequenceModel.py on this CSV.")


# ---------------------------------------------------------------------------
# Predictions over time
# ---------------------------------------------------------------------------

def predictions_over_time(network, checkpoint, skeleton, raw_times, device):
    """Class probabilities after each sampled frame, using only frames seen so far."""
    classes = checkpoint["classes"]
    fps = checkpoint["fps"]
    probabilities = []
    for t in raw_times:
        seen = (t - raw_times[0]) + 1.0 / fps
        sample = tsm.truncate(skeleton, seen, fps)
        probabilities.append(tsm.predict(network, [sample], checkpoint["window"], device, len(classes))[0])
    return np.array(probabilities)


def locked_in_time(probabilities, seen_seconds, true_index):
    """Earliest time after which the prediction stays correct until the end, or None."""
    correct = probabilities.argmax(1) == true_index
    if not correct[-1]:
        return None
    wrong = np.flatnonzero(~correct)
    first = 0 if len(wrong) == 0 else wrong[-1] + 1
    return float(seen_seconds[first])


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def text(img, s, org, scale=0.55, color=WHITE, thickness=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_skeleton(frame, keypoints, confidence, box, trail, scale, min_conf):
    kp = keypoints * scale
    visible = np.isfinite(kp).all(axis=1) & (confidence >= min_conf)
    if np.isfinite(box).all():
        x1, y1, x2, y2 = (box * scale).astype(int)
        cv2.rectangle(frame, (x1, y1), (x2, y2), GREEN, 2)
    for a, b in BONES:
        if visible[a] and visible[b]:
            cv2.line(frame, tuple(kp[a].astype(int)), tuple(kp[b].astype(int)), WHITE, 2, cv2.LINE_AA)
    for j in range(len(kp)):
        if visible[j]:
            color = CYAN if j == poseUtils.L_WRIST else MAGENTA if j == poseUtils.R_WRIST else GREEN
            cv2.circle(frame, tuple(kp[j].astype(int)), 5 if j in (9, 10) else 3, color, -1, cv2.LINE_AA)
    for joint, color in ((poseUtils.L_WRIST, CYAN), (poseUtils.R_WRIST, MAGENTA)):
        points = [p * scale for p in trail[joint] if p is not None]
        for i in range(1, len(points)):
            cv2.line(frame, tuple(points[i - 1].astype(int)), tuple(points[i].astype(int)),
                     color, max(1, int(3 * i / len(points))), cv2.LINE_AA)


def draw_panel(panel, probs, classes, true_index, seen, locked, video_name, note):
    panel[:] = PANEL
    text(panel, "Live action recognition", (20, 34), 0.75, WHITE, 2)
    text(panel, "skeleton TCN, using only frames seen so far", (20, 58), 0.48, GREY)

    predicted = int(probs.argmax())
    correct = predicted == true_index
    text(panel, f"Seen: {seen:4.1f} s", (20, 98), 0.65)
    text(panel, "Prediction:", (20, 132), 0.6, GREY)
    text(panel, f"{classes[predicted]}  {probs[predicted] * 100:.0f}%", (140, 132), 0.65,
         GREEN if correct else ORANGE, 2)
    text(panel, "True action:", (20, 162), 0.6, GREY)
    text(panel, classes[true_index], (140, 162), 0.65, WHITE, 2)
    if locked is not None and seen >= locked - 1e-9:
        text(panel, f"Correct from {locked:.1f} s onward", (20, 192), 0.55, GREEN)

    # top-5 probability bars
    order = np.argsort(probs)[::-1][:5]
    if true_index not in order:
        order = np.append(order[:4], true_index)
    top, bar_x, bar_w = 225, 190, PANEL_WIDTH - 190 - 70
    for row, k in enumerate(order):
        y = top + row * 38
        name = classes[k] if len(classes[k]) <= 19 else classes[k][:18] + "."
        text(panel, name, (20, y + 18), 0.5, WHITE if k == true_index else GREY)
        cv2.rectangle(panel, (bar_x, y + 4), (bar_x + bar_w, y + 24), DARK, -1)
        fill = int(bar_w * probs[k])
        color = GREEN if k == true_index else (ORANGE if k == predicted else GREY)
        cv2.rectangle(panel, (bar_x, y + 4), (bar_x + fill, y + 24), color, -1)
        text(panel, f"{probs[k] * 100:3.0f}%", (bar_x + bar_w + 8, y + 20), 0.5)

    text(panel, "green = true action, orange = wrong top guess", (20, top + 5 * 38 + 22), 0.45, GREY)
    text(panel, "Held-out video (not used to train this model)", (20, VIDEO_HEIGHT - 44), 0.45, GREY)
    text(panel, (video_name if len(video_name) <= 48 else video_name[:45] + "...") + f"  [{note}]",
         (20, VIDEO_HEIGHT - 20), 0.4, GREY)


def draw_chart(chart, probabilities, seen_seconds, upto, classes, true_index):
    chart[:] = PANEL
    h, w = chart.shape[:2]
    left, right, top, bottom = 60, w - 20, 30, h - 35
    duration = max(seen_seconds[-1], 1e-6)
    to_x = lambda s: int(left + (right - left) * s / duration)
    to_y = lambda p: int(bottom - (bottom - top) * p)

    text(chart, "Probability of each action over time", (left, 20), 0.5, WHITE)
    for p in (0.0, 0.5, 1.0):
        cv2.line(chart, (left, to_y(p)), (right, to_y(p)), DARK, 1)
        text(chart, f"{p:.1f}", (18, to_y(p) + 5), 0.4, GREY)
    chance = 1.0 / len(classes)
    cv2.line(chart, (left, to_y(chance)), (right, to_y(chance)), (70, 70, 70), 1, cv2.LINE_AA)
    text(chart, "chance", (right - 55, to_y(chance) - 5), 0.38, GREY)
    for s in SNAPSHOT_SECONDS:
        if s <= duration:
            cv2.line(chart, (to_x(s), top), (to_x(s), bottom), (55, 55, 55), 1)
            text(chart, f"{s:g}s", (to_x(s) - 10, h - 14), 0.4, GREY)
    text(chart, f"{duration:.1f}s", (right - 30, h - 14), 0.4, GREY)

    xs = [to_x(s) for s in seen_seconds[: upto + 1]]
    for k in range(len(classes)):
        if k == true_index:
            continue
        ys = [to_y(p) for p in probabilities[: upto + 1, k]]
        for i in range(1, len(xs)):
            cv2.line(chart, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), (95, 95, 95), 1, cv2.LINE_AA)
    ys = [to_y(p) for p in probabilities[: upto + 1, true_index]]
    for i in range(1, len(xs)):
        cv2.line(chart, (xs[i - 1], ys[i - 1]), (xs[i], ys[i]), GREEN, 3, cv2.LINE_AA)
    cv2.line(chart, (xs[-1], top), (xs[-1], bottom), WHITE, 1)
    text(chart, classes[true_index] + " (true)", (left + 8, top + 18), 0.45, GREEN)
    text(chart, "other actions", (left + 8, top + 36), 0.45, GREY)


def open_writer(path, fps, size):
    for codec in ("avc1", "mp4v"):     # avc1 (H.264) plays everywhere; mp4v is the fallback
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*codec), fps, size)
        if writer.isOpened():
            return writer, codec
    raise SystemExit("Could not open a video writer for " + path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    features_path = os.path.join(args.data_dir, "sard_pose_features.csv")
    df = pd.read_csv(features_path)
    row, note = choose_video(df, args)
    action, video_name, sequence_file = row["action"], row["video"], row["sequence_file"]
    print(f"Video: {action}/{video_name}  ({note})")

    device = tsm.pick_device(args.device)
    network, checkpoint = load_fold_network(os.path.join(args.data_dir, "fold_models"), sequence_file, device)
    classes = checkpoint["classes"]
    true_index = classes.index(action)
    print(f"Using fold {checkpoint['fold']} model (this video was in its test set) on {device}")

    npz_path = os.path.join(args.data_dir, "sequences", sequence_file)
    raw = poseUtils.load_sequence(npz_path)
    skeleton = tsm.load_skeleton(npz_path, keypoint_conf=checkpoint["keypoint_conf"], fps=checkpoint["fps"])
    times = raw["times"]
    if len(times) == 0:
        raise SystemExit("This video has no sampled frames.")
    seen_seconds = times - times[0]

    print(f"Predicting after each of {len(times)} frames...")
    probabilities = predictions_over_time(network, checkpoint, skeleton, times, device)
    locked = locked_in_time(probabilities, seen_seconds, true_index)

    # frames of the original video at the sampled times (blank background if the video is missing)
    video_path = os.path.join(args.dataset, action, video_name)
    cap = cv2.VideoCapture(video_path) if os.path.exists(video_path) else None
    if cap is None:
        print("Original video not found at", video_path, "- drawing the skeleton on a blank background")
    frame_w, frame_h = raw["frame_width"] or 640, raw["frame_height"] or 480
    scale = VIDEO_HEIGHT / frame_h
    video_w = int(round(frame_w * scale / 2) * 2)
    video_w = min(video_w, 960)
    scale = min(scale, video_w / frame_w)

    wanted = np.round(times * raw["fps"]).astype(int)
    out_dir = os.path.join(args.data_dir, "demo")
    os.makedirs(out_dir, exist_ok=True)
    stem = os.path.splitext(video_name)[0][:60]
    output = args.output or os.path.join(out_dir, f"{stem}_demo.mp4")
    size = (video_w + PANEL_WIDTH, VIDEO_HEIGHT + CHART_HEIGHT)
    writer, codec = open_writer(output, raw["effective_fps"], size)

    trail = {poseUtils.L_WRIST: [], poseUtils.R_WRIST: []}
    snapshots, snapshot_targets = {}, [s for s in SNAPSHOT_SECONDS if s <= seen_seconds[-1]]
    frame_index, frame = -1, None
    for i, target in enumerate(wanted):
        if cap is not None:
            while frame_index < target:
                ok, next_frame = cap.read()
                if not ok:
                    break
                frame_index += 1
                frame = next_frame
        background = np.full((VIDEO_HEIGHT, video_w, 3), 15, np.uint8)
        if frame is not None:
            resized = cv2.resize(frame, (int(frame_w * scale), int(frame_h * scale)))
            background[: resized.shape[0], : resized.shape[1]] = resized[:VIDEO_HEIGHT, :video_w]

        conf = np.nan_to_num(raw["confidence"][i])
        for joint in trail:
            point = raw["keypoints"][i, joint]
            trail[joint].append(point if np.isfinite(point).all() and conf[joint] >= checkpoint["keypoint_conf"] else None)
            trail[joint] = trail[joint][-15:]
        draw_skeleton(background, raw["keypoints"][i], conf, raw["boxes"][i], trail, scale, checkpoint["keypoint_conf"])

        canvas = np.zeros((size[1], size[0], 3), np.uint8)
        canvas[:VIDEO_HEIGHT, :video_w] = background
        draw_panel(canvas[:VIDEO_HEIGHT, video_w:], probabilities[i], classes, true_index,
                   seen_seconds[i], locked, video_name, note)
        draw_chart(canvas[VIDEO_HEIGHT:], probabilities, seen_seconds, i, classes, true_index)
        writer.write(canvas)

        for s in snapshot_targets:
            if s not in snapshots and seen_seconds[i] >= s - 1e-9:
                snapshots[s] = canvas[:VIDEO_HEIGHT].copy()
    writer.release()
    if cap is not None:
        cap.release()

    # storyboard: snapshots at 0.5 / 1 / 2 / 3 s and the final frame, two per row
    snapshots["end"] = canvas[:VIDEO_HEIGHT].copy()
    tiles = []
    for key, image in snapshots.items():
        tile = image.copy()
        label = "end of clip" if key == "end" else f"after {key:g} s"
        cv2.rectangle(tile, (0, 0), (190, 36), (0, 0, 0), -1)
        text(tile, label, (10, 25), 0.7, WHITE, 2)
        tiles.append(cv2.resize(tile, (tile.shape[1] // 2, tile.shape[0] // 2)))
    if len(tiles) % 2:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[k: k + 2]) for k in range(0, len(tiles), 2)]
    storyboard = np.vstack(rows + [cv2.resize(canvas[VIDEO_HEIGHT:], (rows[0].shape[1], CHART_HEIGHT))])
    storyboard_path = os.path.splitext(output)[0] + "_storyboard.png"
    cv2.imwrite(storyboard_path, storyboard)

    final = probabilities[-1].argmax()
    print(f"\nFinal prediction: {classes[final]} ({probabilities[-1][final] * 100:.0f}%) | true: {action}")
    print("Correct from", f"{locked:.1f} s onward" if locked is not None else "- never stays correct")
    print(f"Saved video ({codec}): {output}")
    print(f"Saved storyboard: {storyboard_path}")


if __name__ == "__main__":
    main()