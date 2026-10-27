
import argparse
import os
import time

import numpy as np
import pandas as pd

import poseUtils

DEFAULT_DATASET = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/CoreSet"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--video", default=None, help="test a single video instead of the dataset")
    parser.add_argument("--per-action", type=int, default=3)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--target-fps", type=float, default=poseUtils.DEFAULT_TARGET_FPS)
    parser.add_argument("--max-seconds", type=float, default=poseUtils.DEFAULT_MAX_SECONDS)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def check_video(model, action, video_path, args):
    started = time.time()
    sequence = poseUtils.extract_sequence(
        model, video_path, target_fps=args.target_fps,
        max_seconds=args.max_seconds, device=args.device,
    )
    features = poseUtils.compute_features(sequence)
    wrist_conf = sequence["confidence"][:, [poseUtils.L_WRIST, poseUtils.R_WRIST]]
    detected = np.isfinite(sequence["boxes"]).all(axis=1)

    warnings = []
    if features["detection_rate"] < 0.5:
        warnings.append("worker found in <50% of frames")
    if features["effective_fps"] < 8:
        warnings.append("sampling <8 fps, fast strokes may alias")
    if detected.any() and np.nanmean(wrist_conf[detected]) < 0.5:
        warnings.append("wrists poorly visible")
    if (sequence["num_people"] > 1).mean() > 0.5:
        warnings.append("several people in frame")
    if features["sampled_frames"] < 16:
        warnings.append("very short clip, rhythm not measurable")

    return {
        "action": action,
        "video": os.path.basename(video_path)[:40],
        "frames": features["sampled_frames"],
        "fps": round(features["effective_fps"], 1),
        "detected_%": round(features["detection_rate"] * 100, 1),
        "wrist_conf": round(float(np.nanmean(wrist_conf[detected])), 2) if detected.any() else 0.0,
        "people": round(float(sequence["num_people"].mean()), 1),
        "hand_hz": round(features["hand_freq_vertical_hz"], 2)
        if np.isfinite(features["hand_freq_vertical_hz"]) else None,
        "seconds": round(time.time() - started, 1),
        "warnings": "; ".join(warnings),
    }


def main():
    args = parse_args()
    model = poseUtils.load_pose_model(args.model)

    if args.video:
        targets = [("(single video)", args.video)]
    else:
        per_action = {}
        for action, _, path in poseUtils.list_videos(args.dataset):
            per_action.setdefault(action, [])
            if len(per_action[action]) < args.per_action:
                per_action[action].append(path)
        targets = [(a, p) for a, paths in per_action.items() for p in paths]

    rows = []
    for action, path in targets:
        try:
            rows.append(check_video(model, action, path, args))
        except Exception as error:
            rows.append({"action": action, "video": os.path.basename(path)[:40], "warnings": "FAILED: " + str(error)})
        print(rows[-1])

    report = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print("\n" + report.to_string(index=False))

    if "detected_%" in report:
        print("\nAverage detection rate:", round(report["detected_%"].mean(), 1), "%")
        print("Videos with warnings:", int((report["warnings"] != "").sum()), "of", len(report))
        if "seconds" in report and not args.video:
            total_videos = sum(1 for _ in poseUtils.list_videos(args.dataset))
            estimate = report["seconds"].mean() * total_videos / 60
            print(f"Estimated full extraction time for {total_videos} videos: ~{estimate:.0f} min")


if __name__ == "__main__":
    main()