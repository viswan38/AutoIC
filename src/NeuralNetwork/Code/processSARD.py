

import argparse
import os
import time

import pandas as pd

import poseUtils

DEFAULT_DATASET = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/CoreSet"
DEFAULT_OUTPUT_DIR = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/src/NeuralNetwork/Data"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default=DEFAULT_DATASET, help="folder with one subfolder per action")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model", default="yolo11n-pose.pt")
    parser.add_argument("--target-fps", type=float, default=poseUtils.DEFAULT_TARGET_FPS,
                        help="sampling rate; keep >= 10 so 2-4 Hz tool strokes are not aliased")
    parser.add_argument("--max-seconds", type=float, default=poseUtils.DEFAULT_MAX_SECONDS,
                        help="longer videos are trimmed to their middle portion")
    parser.add_argument("--keypoint-conf", type=float, default=poseUtils.DEFAULT_KEYPOINT_CONF)
    parser.add_argument("--device", default=None,
                        help="e.g. cpu, cuda:0. Ultralytics advises against mps for pose models.")
    parser.add_argument("--overwrite", action="store_true", help="re-run YOLO even if a .npz exists")
    return parser.parse_args()


def main():
    args = parse_args()
    sequence_dir = os.path.join(args.output_dir, "sequences")
    features_path = os.path.join(args.output_dir, "sard_pose_features.csv")
    os.makedirs(sequence_dir, exist_ok=True)

    videos = list(poseUtils.list_videos(args.dataset))
    if not videos:
        raise SystemExit("No videos found in " + args.dataset)
    print("Found", len(videos), "videos in", len({a for a, _, _ in videos}), "action folders")

    model = None
    rows, failures = [], []
    start = time.time()
    extracted = 0

    for number, (action, filename, video_path) in enumerate(videos, start=1):
        stem = os.path.splitext(filename)[0]
        relative_npz = os.path.join(action, stem + ".npz")
        npz_path = os.path.join(sequence_dir, relative_npz)

        try:
            if os.path.exists(npz_path) and not args.overwrite:
                sequence = poseUtils.load_sequence(npz_path)
                status = "cached"
            else:
                if model is None:
                    model = poseUtils.load_pose_model(args.model)
                sequence = poseUtils.extract_sequence(
                    model, video_path,
                    target_fps=args.target_fps,
                    max_seconds=args.max_seconds,
                    device=args.device,
                )
                poseUtils.save_sequence(npz_path, sequence)
                extracted += 1
                status = "extracted"
        except Exception as error:  # keep going; one bad file should not stop a 2-hour run
            failures.append((action, filename, str(error)))
            print(f"[{number}/{len(videos)}] FAILED {action}/{filename}: {error}")
            continue

        features = poseUtils.compute_features(sequence, keypoint_conf=args.keypoint_conf)
        row = {
            "video": filename,
            "action": action,
            "source_video": poseUtils.source_video_id(filename),
            "sequence_file": relative_npz,
        }
        row.update(features)
        rows.append(row)

        elapsed = time.time() - start
        remaining = ""
        if extracted:
            per_video = elapsed / extracted
            remaining = f" | ~{per_video * (len(videos) - number) / 60:.0f} min left"
        print(
            f"[{number}/{len(videos)}] {status:9s} {action}/{filename} | "
            f"{features['sampled_frames']} frames @ {features['effective_fps']:.1f} fps | "
            f"detected {features['detection_rate'] * 100:.0f}%{remaining}"
        )

        # write progress regularly so an interrupted run still leaves a usable CSV
        if number % 25 == 0:
            pd.DataFrame(rows).to_csv(features_path, index=False)

    data = pd.DataFrame(rows)
    data.to_csv(features_path, index=False)

    print("\nSaved", len(data), "rows and", len(poseUtils.model_feature_columns(data)),
          "model features to", features_path)
    print("Sequences saved under", sequence_dir)
    print("Average detection rate:", round(data["detection_rate"].mean(), 3))
    print("Videos with no detections:", int((data["detected_frames"] == 0).sum()))
    if failures:
        print("\nVideos that could not be processed:")
        for action, filename, error in failures:
            print(" ", action, "/", filename, "-", error)


if __name__ == "__main__":
    main()