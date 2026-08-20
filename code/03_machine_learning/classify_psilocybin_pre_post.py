#!/usr/bin/env python3
"""Psilocybin pre-post XGBoost classification over 1,000 random paired hold-outs."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


N_SUBJECTS = 22
N_SHUFFLES = 1_000
N_PERMUTATIONS = 10_000
PERMUTATION_BATCH_SIZE = 256
SEED = 12_345

NUM_FEATURES = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, 500,
    750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]

FILE_PATTERNS = {
    "signal": "model_hopf_FC_{subject:03d}.mat",
    "vortex": "model_vortex_partition_social_{subject:03d}.mat",
}


def parse_arguments():
    parser = ArgumentParser(
        description="Paired pre–post XGBoost classification with fold-wise PPERM."
    )
    parser.add_argument("--space", required=True, choices=("signal", "vortex"))
    parser.add_argument("--before-dir", required=True, type=Path)
    parser.add_argument("--after-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def extract_edges(mat_path):
    if not mat_path.is_file():
        raise FileNotFoundError(f"Missing connectivity file: {mat_path}")

    data = sio.loadmat(mat_path)
    if "Ceff_task1" not in data:
        raise KeyError(f"'Ceff_task1' was not found in {mat_path}")

    matrix = np.asarray(data["Ceff_task1"], dtype=np.float64)
    if matrix.shape != (100, 100):
        raise ValueError(
            f"Expected a 100 x 100 connectivity matrix in {mat_path}, "
            f"received {matrix.shape}."
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"Non-finite connectivity values found in {mat_path}")

    return matrix[np.tril_indices(100, k=-1)]


def load_group(directory, space):
    pattern = FILE_PATTERNS[space]
    subjects = [
        extract_edges(directory / pattern.format(subject=subject))
        for subject in range(1, N_SUBJECTS + 1)
    ]
    return np.stack(subjects)


def signed_rank_statistics(differences):
    ranks = np.zeros_like(differences, dtype=np.float64)

    for feature in range(differences.shape[1]):
        nonzero = differences[:, feature] != 0
        if np.any(nonzero):
            ranks[nonzero, feature] = rankdata(
                np.abs(differences[nonzero, feature]),
                method="average",
            )

    observed = np.abs(np.sum(np.sign(differences) * ranks, axis=0))
    return ranks, observed


def wilcoxon_signed_rank_pvalues(X_before, X_after, rng):
    differences = X_after - X_before
    ranks, observed = signed_rank_statistics(differences)
    exceedances = np.zeros(differences.shape[1], dtype=np.int64)

    for start in range(0, N_PERMUTATIONS, PERMUTATION_BATCH_SIZE):
        batch_size = min(
            PERMUTATION_BATCH_SIZE,
            N_PERMUTATIONS - start,
        )
        signs = rng.choice(
            np.array([-1.0, 1.0]),
            size=(batch_size, differences.shape[0]),
        )
        permuted = np.abs(signs @ ranks)
        exceedances += np.sum(permuted >= observed[None, :], axis=0)

    p_values = (exceedances + 1.0) / (N_PERMUTATIONS + 1.0)
    return p_values, observed


def rank_features(X_before, X_after, rng):
    p_values, statistics = wilcoxon_signed_rank_pvalues(
        X_before,
        X_after,
        rng,
    )
    ranking = np.lexsort((-statistics, p_values))
    return ranking.astype(np.int32), p_values


def build_classifier(seed):
    return XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
    )


def main():
    args = parse_arguments()
    X_before = load_group(args.before_dir, args.space)
    X_after = load_group(args.after_dir, args.space)

    if X_before.shape != X_after.shape:
        raise ValueError(
            f"Pre- and post-treatment arrays differ in shape: "
            f"{X_before.shape} versus {X_after.shape}."
        )

    n_features_total = X_before.shape[1]
    if max(NUM_FEATURES) > n_features_total:
        raise ValueError(
            f"Top-{max(NUM_FEATURES)} was requested, but only "
            f"{n_features_total} features are available."
        )

    n_subsets = len(NUM_FEATURES)
    accuracies = np.empty((n_subsets, N_SHUFFLES), dtype=np.float64)
    confusions = np.empty((n_subsets, N_SHUFFLES, 2, 2), dtype=np.int32)
    rankings = np.empty((N_SHUFFLES, n_features_total), dtype=np.int32)
    feature_pvalues = np.empty(
        (N_SHUFFLES, n_features_total),
        dtype=np.float64,
    )
    subjects_tested = np.empty(N_SHUFFLES, dtype=np.int32)

    all_subjects = np.arange(N_SUBJECTS)
    split_rng = np.random.default_rng(SEED)

    for iteration in range(N_SHUFFLES):
        held_out = int(split_rng.integers(0, N_SUBJECTS))
        train_subjects = all_subjects[all_subjects != held_out]

        X_before_train = X_before[train_subjects]
        X_after_train = X_after[train_subjects]
        X_train = np.vstack((X_before_train, X_after_train))
        y_train = np.concatenate((
            np.zeros(train_subjects.size, dtype=np.int32),
            np.ones(train_subjects.size, dtype=np.int32),
        ))

        X_test = np.vstack((
            X_before[held_out],
            X_after[held_out],
        ))
        y_test = np.array([0, 1], dtype=np.int32)

        fold_seed = SEED + iteration + 1
        ranking, p_values = rank_features(
            X_before_train,
            X_after_train,
            np.random.default_rng(fold_seed),
        )
        rankings[iteration] = ranking
        feature_pvalues[iteration] = p_values
        subjects_tested[iteration] = held_out + 1

        for subset_index, top_n in enumerate(NUM_FEATURES):
            selected = ranking[:top_n]
            classifier = build_classifier(fold_seed)
            classifier.fit(X_train[:, selected], y_train)
            predictions = classifier.predict(X_test[:, selected])

            accuracies[subset_index, iteration] = np.mean(
                predictions == y_test
            )
            confusions[subset_index, iteration] = confusion_matrix(
                y_test,
                predictions,
                labels=[0, 1],
            )

        completed = iteration + 1
        if completed == 1 or completed % 10 == 0 or completed == N_SHUFFLES:
            print(
                f"Completed shuffle {completed}/{N_SHUFFLES}",
                flush=True,
            )

    accuracy_mean = accuracies.mean(axis=1)
    accuracy_std = accuracies.std(axis=1)
    confusion_mean = confusions.mean(axis=1)
    best_index = int(np.argmax(accuracy_mean))

    feature_counts = np.vstack([
        np.bincount(
            rankings[:, :top_n].ravel(),
            minlength=n_features_total,
        )
        for top_n in NUM_FEATURES
    ])

    output_path = args.out or Path(
        f"results_xgb_pperm_psilocybin_pre_post_"
        f"{args.space}.mat"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(
        output_path,
        {
            "task": "psilocybin pre versus post",
            "space": args.space,
            "feature_selection": "pperm",
            "n_feature_permutations": N_PERMUTATIONS,
            "n_shuffles": N_SHUFFLES,
            "num_features": np.asarray(NUM_FEATURES, dtype=np.int32),
            "accuracy_mean": accuracy_mean,
            "accuracy_std": accuracy_std,
            "conf_matrix_mean": confusion_mean,
            "accuracies_all": accuracies,
            "confusions_all": confusions,
            "feature_rankings": rankings,
            "feature_pvalues": feature_pvalues,
            "feature_counts": feature_counts,
            "subjects_tested": subjects_tested,
            "best_num_features": NUM_FEATURES[best_index],
            "best_accuracy_mean": accuracy_mean[best_index],
            "random_seed": SEED,
        },
        do_compression=True,
    )

    print(
        f"Best subset: Top-{NUM_FEATURES[best_index]} | "
        f"accuracy={accuracy_mean[best_index]:.4f} ± "
        f"{accuracy_std[best_index]:.4f}",
        flush=True,
    )
    print(f"Saved results to {output_path}", flush=True)


if __name__ == "__main__":
    main()
