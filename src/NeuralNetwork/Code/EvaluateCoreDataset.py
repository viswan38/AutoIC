import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.svm import SVC

import poseUtils

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

DEFAULT_FEATURES = "/Users/srivatsaviswanadhula/Fall_2026_Semester/AutoIC/src/NeuralNetwork/Data/sard_pose_features.csv"

# Quality-control columns used only for the shortcut check
QUALITY_COLUMNS = [
    "detection_rate", "detected_frames", "sampled_frames",
    "valid_wrist_measurements", "valid_elbow_measurements",
    "valid_hip_measurements", "valid_body_measurements",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--features", default=DEFAULT_FEATURES)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default=None,
                        help="where to save out-of-fold probabilities (default: next to the features CSV)")
    return parser.parse_args()


def scaled(estimator):
    """Impute, reduce skew (speeds are long-tailed), standardise, then fit."""
    return make_pipeline(
        SimpleImputer(strategy="median"),
        PowerTransformer(method="yeo-johnson", standardize=True),
        estimator,
    )


def tree(estimator):
    return make_pipeline(SimpleImputer(strategy="median"), estimator)


def build_models(seed):
    return {
        "Logistic regression": scaled(LogisticRegression(C=0.5, max_iter=5000, class_weight="balanced")),
        "SVM (RBF)": scaled(CalibratedClassifierCV(
            SVC(C=3.0, gamma="scale", class_weight="balanced"), method="sigmoid", cv=3, ensemble=False)),
        "MLP neural network": scaled(MLPClassifier(
            hidden_layer_sizes=(128, 64), alpha=1e-2, max_iter=2000,
            early_stopping=True, validation_fraction=0.15, random_state=seed)),
        "Random forest": tree(RandomForestClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed)),
        "Extra trees": tree(ExtraTreesClassifier(
            n_estimators=500, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed)),
        "Gradient boosting": HistGradientBoostingClassifier(
            max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0,
            class_weight="balanced", random_state=seed),
    }


def cross_validate(model, X, y, folds, classes):
    """Out-of-fold class probabilities and per-fold accuracy for one model."""
    probabilities = np.zeros((len(y), len(classes)))
    fold_accuracy = []
    for fold in sorted(np.unique(folds)):
        train, test = folds != fold, folds == fold
        fitted = clone(model).fit(X[train], y[train])
        fold_probs = np.zeros((test.sum(), len(classes)))
        fold_probs[:, [classes.index(c) for c in fitted.classes_]] = fitted.predict_proba(X[test])
        probabilities[test] = fold_probs
        fold_accuracy.append(accuracy_score(y[test], np.array(classes)[fold_probs.argmax(1)]))
    return probabilities, np.array(fold_accuracy)


def summarise(name, y, probabilities, fold_accuracy, classes):
    predictions = np.array(classes)[probabilities.argmax(1)]
    return {
        "model": name,
        "accuracy": accuracy_score(y, predictions),
        "accuracy_fold_std": fold_accuracy.std() if fold_accuracy is not None else np.nan,
        "macro_f1": f1_score(y, predictions, average="macro"),
    }


def main():
    args = parse_args()
    df = pd.read_csv(args.features)
    df = df.dropna(subset=["action"]).reset_index(drop=True)
    features = poseUtils.model_feature_columns(df)
    classes = sorted(df["action"].unique())
    y = df["action"].values

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    # ------------------------------------------------------------------
    print("=" * 70)
    print("DATASET")
    print("=" * 70)
    print("File:", args.features)
    print("Recordings:", len(df), "| Action classes:", len(classes),
          "| Source videos:", df["video"].map(poseUtils.source_video_id).nunique())
    print("Model features:", len(features))
    print("\nRecordings per action:")
    print(df["action"].value_counts().to_string())

    if "detection_rate" in df:
        print("\nDETECTION QUALITY")
        print("Average detection rate:", round(df["detection_rate"].mean(), 3))
        print("Median detection rate:", round(df["detection_rate"].median(), 3))
        print("Recordings with no detections:", int((df["detected_frames"] == 0).sum()))
        print("Recordings detected in <50% of frames:", int((df["detection_rate"] < 0.5).sum()))
        print("\nDetection rate by action:")
        print(df.groupby("action")["detection_rate"].mean().round(2).sort_values().to_string())

    missing = df[features].isna().mean().sort_values(ascending=False)
    print("\nFeatures with the most missing values (fraction of recordings):")
    print(missing.head(8).round(3).to_string())

    folds, _ = poseUtils.assign_folds(df, n_splits=args.folds, seed=args.seed)
    chance = df["action"].value_counts().max() / len(df)
    print(f"\nMajority-class baseline accuracy: {chance:.3f}  |  uniform-guess macro-F1 ~ {1 / len(classes):.3f}")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("SHORTCUT CHECK: can detection quality alone predict the action?")
    print("=" * 70)
    quality = [c for c in QUALITY_COLUMNS if c in df]
    shortcut_score = None
    if quality:
        quality_model = scaled(LogisticRegression(max_iter=5000, class_weight="balanced"))
        probs, accs = cross_validate(quality_model, df[quality].values, y, folds, classes)
        shortcut_score = summarise("quality only", y, probs, accs, classes)
        print(f"Detection-quality columns only ({', '.join(quality)}):")
        print(f"  accuracy {shortcut_score['accuracy']:.3f} | macro-F1 {shortcut_score['macro_f1']:.3f}")
        print("These columns are NOT used by the models below. If this score is close to the motion")
        print("models' score, the classes differ more in camera framing than in worker movement.")

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"MODEL COMPARISON ({args.folds}-fold cross-validation grouped by source video)")
    print("=" * 70)
    X = df[features].values.astype(np.float64)
    results, all_probabilities = [], {}
    for name, model in build_models(args.seed).items():
        probs, accs = cross_validate(model, X, y, folds, classes)
        all_probabilities[name] = probs
        results.append(summarise(name, y, probs, accs, classes))
        print(f"{name:22s} accuracy {results[-1]['accuracy']:.3f} (±{accs.std():.3f} across folds) "
              f"| macro-F1 {results[-1]['macro_f1']:.3f}")

    # Soft-voting ensemble of the three best individual models
    top3 = [r["model"] for r in sorted(results, key=lambda r: r["macro_f1"], reverse=True)[:3]]
    ensemble = np.mean([all_probabilities[m] for m in top3], axis=0)
    ensemble_name = "Ensemble (top 3)"
    all_probabilities[ensemble_name] = ensemble
    results.append(summarise(ensemble_name, y, ensemble, None, classes))
    print(f"{ensemble_name:22s} accuracy {results[-1]['accuracy']:.3f} | macro-F1 {results[-1]['macro_f1']:.3f}"
          f"  ({', '.join(top3)})")

    table = pd.DataFrame(results).sort_values("macro_f1", ascending=False)
    best = table.iloc[0]["model"]
    best_probs = all_probabilities[best]
    predictions = np.array(classes)[best_probs.argmax(1)]

    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("BEST MODEL:", best)
    print("=" * 70)
    print(classification_report(y, predictions, labels=classes, zero_division=0, digits=3))

    print("Confusion matrix (rows = true action, columns = predicted):")
    matrix = pd.DataFrame(confusion_matrix(y, predictions, labels=classes),
                          index=classes, columns=[c[:7] for c in classes])
    print(matrix.to_string())

    pairs = []
    for i, a in enumerate(classes):
        for j, b in enumerate(classes):
            if i < j:
                pairs.append((matrix.iloc[i, j] + matrix.iloc[j, i], a, b))
    print("\nMost confused pairs:")
    for count, a, b in sorted(pairs, reverse=True)[:5]:
        print(f"  {a} <-> {b}: {count}")

    # Which features the forest relies on (impurity importance; indicative only)
    forest = tree(RandomForestClassifier(n_estimators=500, min_samples_leaf=2, n_jobs=-1, random_state=args.seed))
    forest.fit(X, y)
    importance = pd.Series(forest[-1].feature_importances_, index=features).sort_values(ascending=False)
    print("\nMost informative features (random forest importance):")
    print(importance.head(15).round(4).to_string())

    if shortcut_score is not None:
        gap = table.iloc[0]["macro_f1"] - shortcut_score["macro_f1"]
        print(f"\nMotion features beat the detection-quality shortcut by {gap:+.3f} macro-F1.")

    # ------------------------------------------------------------------
    output = args.output or os.path.join(os.path.dirname(os.path.abspath(args.features)), "oof_feature_probs.csv")
    saved = pd.DataFrame(best_probs, columns=["prob_" + c for c in classes])
    saved.insert(0, "fold", folds)
    saved.insert(0, "action", df["action"])
    saved.insert(0, "video", df["video"])
    saved.attrs["model"] = best
    saved.to_csv(output, index=False)
    print("\nSaved out-of-fold probabilities of", best, "to", output)
    print("(trainSequenceModel.py --fuse uses this file to combine both models)")


if __name__ == "__main__":
    main()