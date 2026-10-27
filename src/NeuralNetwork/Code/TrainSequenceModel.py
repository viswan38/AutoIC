import argparse
import os
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

import poseUtils

DEFAULT_FEATURES = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/src/NeuralNetwork/Data/sard_pose_features.csv"

MODEL_FPS = 15.0
NUM_CHANNELS = 17 * 2 + 17 + 17 * 2 + 2   # positions, visibility, velocities, body velocity


# ---------------------------------------------------------------------------
# Data preparation (numpy only)
# ---------------------------------------------------------------------------

def load_skeleton(path, keypoint_conf=poseUtils.DEFAULT_KEYPOINT_CONF, fps=MODEL_FPS):
    """
    Load a saved sequence and return body-relative arrays on a fixed 15 fps time grid:
        local   (T, 17, 2) joint positions, NaN where not visible
        center  (T, 2)     body centre position (in body heights)
        same    (T,)       True where the body centre came from the hips (for velocity)
    """
    norm = poseUtils.normalize_sequence(poseUtils.load_sequence(path), keypoint_conf)
    times = norm["times"]
    if len(times) == 0:
        return {"local": np.full((1, 17, 2), np.nan), "center": np.full((1, 2), np.nan),
                "same": np.zeros(1, dtype=bool)}

    grid = np.arange(times[0], times[-1] + 1e-6, 1.0 / fps)
    nearest = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    previous = np.clip(nearest - 1, 0, len(times) - 1)
    use_previous = np.abs(times[previous] - grid) < np.abs(times[nearest] - grid)
    index = np.where(use_previous, previous, nearest)

    return {
        "local": norm["local"][index],
        "center": norm["center"][index],
        "same": norm["center_from_hip"][index],
    }


def augment(sample, rng):
    """Random mirror, rotation, scale, speed change, joint dropout and jitter."""
    local = sample["local"].copy()
    center = sample["center"].copy()
    same = sample["same"].copy()

    if rng.random() < 0.5:                               # mirror left/right
        local[..., 0] *= -1
        center[:, 0] *= -1
        for a, b in poseUtils.FLIP_PAIRS:
            local[:, [a, b]] = local[:, [b, a]]

    angle = np.radians(rng.uniform(-10, 10))
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    zoom = rng.uniform(0.9, 1.1)
    local = local @ rotation.T * zoom
    center = center @ rotation.T * zoom

    speed = rng.uniform(0.85, 1.15)                      # slightly faster/slower worker
    index = np.clip(np.round(np.arange(0, len(local), speed)).astype(int), 0, len(local) - 1)
    local, center, same = local[index], center[index], same[index]

    drop = rng.random(local.shape[:2]) < 0.05            # occasionally lose a joint
    local[drop] = np.nan
    local = local + rng.normal(0, 0.005, local.shape)
    return {"local": local, "center": center, "same": same}


def mirror(sample):
    local = sample["local"].copy()
    center = sample["center"].copy()
    local[..., 0] *= -1
    center[:, 0] *= -1
    for a, b in poseUtils.FLIP_PAIRS:
        local[:, [a, b]] = local[:, [b, a]]
    return {"local": local, "center": center, "same": sample["same"]}


def to_channels(sample, fps=MODEL_FPS):
    """Build the (channels, T) input array and a (T,) mask of frames with a visible worker."""
    local, center, same = sample["local"], sample["center"], sample["same"]
    T = len(local)
    visible = np.isfinite(local).all(axis=2)                       # (T, 17)

    velocity = np.zeros_like(local)
    body_velocity = np.zeros_like(center)
    if T > 1:
        step = (local[1:] - local[:-1]) * fps
        step[~(same[1:] == same[:-1])] = np.nan
        velocity[1:] = step
        body_step = (center[1:] - center[:-1]) * fps
        body_step[~(same[1:] == same[:-1])] = np.nan
        body_velocity[1:] = body_step

    channels = np.concatenate([
        np.nan_to_num(local).reshape(T, -1),
        visible.astype(np.float64),
        np.nan_to_num(velocity).reshape(T, -1) / 5.0,          # ~unit scale
        np.nan_to_num(body_velocity) / 5.0,
    ], axis=1)
    channels = np.clip(channels, -5, 5).astype(np.float32).T  # (C, T)
    frame_mask = visible.any(axis=1)
    return channels, frame_mask


def crop(channels, mask, window, start):
    """Fixed-length window; shorter sequences are zero-padded and masked."""
    C, T = channels.shape
    out = np.zeros((C, window), dtype=np.float32)
    out_mask = np.zeros(window, dtype=bool)
    end = min(T, start + window)
    out[:, : end - start] = channels[:, start:end]
    out_mask[: end - start] = mask[start:end]
    return out, out_mask


def random_window(sample, window, rng):
    channels, mask = to_channels(augment(sample, rng))
    T = channels.shape[1]
    visible = np.flatnonzero(mask)
    if T <= window:
        start = 0
    elif len(visible):
        # centre the window on a random visible frame so crops are rarely empty
        start = int(np.clip(rng.choice(visible) - window // 2, 0, T - window))
    else:
        start = int(rng.integers(0, T - window + 1))
    return crop(channels, mask, window, start)


def test_windows(sample, window):
    """Overlapping windows over the whole sequence, plus the mirrored sequence."""
    windows = []
    for version in (sample, mirror(sample)):
        channels, mask = to_channels(version)
        T = channels.shape[1]
        starts = [0] if T <= window else list(range(0, T - window + 1, window // 2))
        if T > window and starts[-1] != T - window:
            starts.append(T - window)
        windows += [crop(channels, mask, window, s) for s in starts]
    return windows


# ---------------------------------------------------------------------------
# Model and training (PyTorch)
# ---------------------------------------------------------------------------

def build_network(num_classes, width=64, dropout=0.2):
    import torch
    from torch import nn

    class ResidualBlock(nn.Module):
        def __init__(self, c_in, c_out, dilation):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv1d(c_in, c_out, 5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(c_out), nn.GELU(), nn.Dropout(dropout),
                nn.Conv1d(c_out, c_out, 5, padding=2 * dilation, dilation=dilation),
                nn.BatchNorm1d(c_out),
            )
            self.skip = nn.Identity() if c_in == c_out else nn.Conv1d(c_in, c_out, 1)
            self.act = nn.GELU()

        def forward(self, x):
            return self.act(self.net(x) + self.skip(x))

    class SkeletonTCN(nn.Module):
        def __init__(self):
            super().__init__()
            self.stem = nn.Sequential(nn.Conv1d(NUM_CHANNELS, width, 1), nn.BatchNorm1d(width), nn.GELU())
            self.blocks = nn.Sequential(
                ResidualBlock(width, width, 1),
                ResidualBlock(width, width * 2, 2),
                ResidualBlock(width * 2, width * 2, 4),
                ResidualBlock(width * 2, width * 2, 8),
            )
            self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(width * 4, num_classes))

        def forward(self, x, mask):
            h = self.blocks(self.stem(x))                      # (B, C, T)
            m = mask.unsqueeze(1).float()                      # (B, 1, T)
            count = m.sum(dim=2)
            has_frames = (count > 0).float()
            mean = (h * m).sum(dim=2) / count.clamp(min=1)
            maxed = h.masked_fill(m == 0, -1e4).max(dim=2).values * has_frames
            return self.head(torch.cat([mean, maxed], dim=1))

    return SkeletonTCN()


def pick_device(requested):
    import torch
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def train_network(samples, labels, num_classes, args, device, seed):
    import torch
    from torch import nn

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    network = build_network(num_classes, width=args.width).to(device)

    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    class_weights = torch.tensor(counts.sum() / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)
    loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device), label_smoothing=0.1)

    steps_per_epoch = int(np.ceil(len(samples) * args.crops_per_video / args.batch_size))
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.epochs * steps_per_epoch, pct_start=0.15)

    order = np.repeat(np.arange(len(samples)), args.crops_per_video)
    for epoch in range(args.epochs):
        network.train()
        rng.shuffle(order)
        total_loss = 0.0
        for step in range(steps_per_epoch):
            batch = order[step * args.batch_size: (step + 1) * args.batch_size]
            if len(batch) < 2:      # BatchNorm needs at least 2 examples
                continue
            windows = [random_window(samples[i], args.window, rng) for i in batch]
            x = torch.from_numpy(np.stack([w[0] for w in windows])).to(device)
            mask = torch.from_numpy(np.stack([w[1] for w in windows])).to(device)
            y = torch.from_numpy(labels[batch]).long().to(device)

            optimizer.zero_grad()
            loss = loss_fn(network(x, mask), y)
            loss.backward()
            nn.utils.clip_grad_norm_(network.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += loss.item()
        if args.verbose and (epoch + 1) % 10 == 0:
            print(f"    epoch {epoch + 1}/{args.epochs}  loss {total_loss / steps_per_epoch:.3f}")
    return network


def predict(network, samples, window, device, num_classes):
    """Average class probabilities over all test windows of each video."""
    import torch

    network.eval()
    probabilities = np.zeros((len(samples), num_classes))
    with torch.no_grad():
        for i, sample in enumerate(samples):
            windows = test_windows(sample, window)
            x = torch.from_numpy(np.stack([w[0] for w in windows])).to(device)
            mask_np = np.stack([w[1] for w in windows])
            mask = torch.from_numpy(mask_np).to(device)
            probs = torch.softmax(network(x, mask), dim=1).cpu().numpy()
            # windows where the worker is visible longer count more
            weight = mask_np.sum(axis=1).astype(np.float64)
            weight = weight if weight.sum() > 0 else np.ones(len(windows))
            probabilities[i] = (probs * weight[:, None]).sum(axis=0) / weight.sum()
    return probabilities


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=DEFAULT_FEATURES, help="CSV written by processSARD.py")
    parser.add_argument("--sequences", default=None, help="default: <features folder>/sequences")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--crops-per-video", type=int, default=4, help="random windows per video per epoch")
    parser.add_argument("--window", type=int, default=48, help="frames per window (48 = 3.2 s at 15 fps)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--device", default=None, help="cpu, cuda or mps (default: best available)")
    parser.add_argument("--fuse", default=None,
                        help="out-of-fold probabilities from EvaluateCoreDataset.py "
                             "(default: <features folder>/oof_feature_probs.csv if it exists)")
    parser.add_argument("--save-model", action="store_true",
                        help="after cross-validation, train on all videos and save the network")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main():
    import torch

    args = parse_args()
    data_dir = os.path.dirname(os.path.abspath(args.features))
    sequence_dir = args.sequences or os.path.join(data_dir, "sequences")

    df = pd.read_csv(args.features)
    if "sequence_file" not in df:
        raise SystemExit("This CSV has no 'sequence_file' column. Run the new processSARD.py first.")
    df = df.dropna(subset=["action"]).reset_index(drop=True)

    exists = df["sequence_file"].map(lambda p: os.path.exists(os.path.join(sequence_dir, p)))
    if (~exists).any():
        print(f"Warning: {int((~exists).sum())} sequence files are missing and will be skipped")
    # folds are assigned on the full CSV so they match EvaluateCoreDataset.py exactly
    folds_all, _ = poseUtils.assign_folds(df, n_splits=args.folds, seed=args.seed)
    df["fold"] = folds_all
    df = df[exists].reset_index(drop=True)

    classes = sorted(df["action"].unique())
    labels = df["action"].map({c: i for i, c in enumerate(classes)}).values
    folds = df["fold"].values
    device = pick_device(args.device)

    print(f"Loading {len(df)} skeleton sequences...")
    samples = [load_skeleton(os.path.join(sequence_dir, p)) for p in df["sequence_file"]]
    lengths = np.array([len(s["local"]) for s in samples])
    print(f"Frames per video at {MODEL_FPS:.0f} fps: median {np.median(lengths):.0f}, "
          f"min {lengths.min()}, max {lengths.max()}  |  window {args.window} frames")
    print(f"Training on {device} | {len(classes)} classes | {args.folds} grouped folds\n")

    probabilities = np.zeros((len(df), len(classes)))
    for fold in sorted(np.unique(folds)):
        started = time.time()
        train, test = np.flatnonzero(folds != fold), np.flatnonzero(folds == fold)
        network = train_network([samples[i] for i in train], labels[train], len(classes),
                                args, device, seed=args.seed + fold)
        probabilities[test] = predict(network, [samples[i] for i in test], args.window, device, len(classes))
        fold_acc = accuracy_score(labels[test], probabilities[test].argmax(1))
        print(f"Fold {fold + 1}/{args.folds}: accuracy {fold_acc:.3f} "
              f"({len(test)} videos, {time.time() - started:.0f} s)")

    predictions = probabilities.argmax(1)
    print("\n" + "=" * 70)
    print("SKELETON TCN (out-of-fold, grouped by source video)")
    print("=" * 70)
    print(f"Accuracy {accuracy_score(labels, predictions):.3f} | macro-F1 {f1_score(labels, predictions, average='macro'):.3f}\n")
    print(classification_report(labels, predictions, labels=range(len(classes)),
                                target_names=classes, zero_division=0, digits=3))
    matrix = pd.DataFrame(confusion_matrix(labels, predictions, labels=range(len(classes))),
                          index=classes, columns=[c[:7] for c in classes])
    pd.set_option("display.width", 200)
    print("Confusion matrix (rows = true action, columns = predicted):")
    print(matrix.to_string())

    output = os.path.join(data_dir, "oof_sequence_probs.csv")
    saved = pd.DataFrame(probabilities, columns=["prob_" + c for c in classes])
    saved.insert(0, "fold", folds)
    saved.insert(0, "action", df["action"])
    saved.insert(0, "video", df["video"])
    saved.to_csv(output, index=False)
    print("\nSaved out-of-fold probabilities to", output)

    # -------------------------------------------------- late fusion
    fuse_path = args.fuse or os.path.join(data_dir, "oof_feature_probs.csv")
    if os.path.exists(fuse_path):
        feature_probs = pd.read_csv(fuse_path)
        merged = saved.merge(feature_probs, on=["video", "action"], suffixes=("_seq", "_feat"))
        same_folds = (merged["fold_seq"] == merged["fold_feat"]).all()
        if len(merged) and same_folds and all("prob_" + c in feature_probs for c in classes):
            seq = merged[[f"prob_{c}_seq" for c in classes]].values
            feat = merged[[f"prob_{c}_feat" for c in classes]].values
            truth = merged["action"].map({c: i for i, c in enumerate(classes)}).values
            print("\n" + "=" * 70)
            print(f"LATE FUSION with {os.path.basename(fuse_path)} ({len(merged)} videos, equal weights)")
            print("=" * 70)
            for name, probs in [("Feature model", feat), ("Skeleton TCN", seq), ("Fused", (seq + feat) / 2)]:
                p = probs.argmax(1)
                print(f"{name:14s} accuracy {accuracy_score(truth, p):.3f} | macro-F1 {f1_score(truth, p, average='macro'):.3f}")
        else:
            print("\nSkipping fusion: the feature probabilities do not match these videos/folds. "
                  "Re-run EvaluateCoreDataset.py on the same features CSV.")
    else:
        print("\nRun EvaluateCoreDataset.py on the same CSV first to also get a fused result.")

    # -------------------------------------------------- final model
    if args.save_model:
        print("\nTraining final network on all videos...")
        network = train_network(samples, labels, len(classes), args, device, seed=args.seed)
        path = os.path.join(data_dir, "skeleton_tcn.pt")
        torch.save({
            "state_dict": network.state_dict(),
            "classes": classes,
            "window": args.window,
            "width": args.width,
            "fps": MODEL_FPS,
            "keypoint_conf": poseUtils.DEFAULT_KEYPOINT_CONF,
        }, path)
        print("Saved", path)


if __name__ == "__main__":
    main()