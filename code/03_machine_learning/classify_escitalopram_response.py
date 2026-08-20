#!/usr/bin/env python3
"""XGBoost classification of escitalopram responders versus non-responders using PPERM feature selection."""

from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.stats import rankdata
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier


N_SHUFFLES = 1_000
N_PERMUTATIONS = 10_000
PERMUTATION_BATCH_SIZE = 256
SEED = 12_345
N_RESPONDERS = 8
N_NON_RESPONDERS = 12

NUM_FEATURES = [1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 400, 500,
    750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 3000]

FILE_PATTERNS = {
    "signal": "model_hopf_FC_*.mat",
    "vortex": "model_vortex_partition_social_*.mat",
}


def parse_arguments():
    parser = ArgumentParser(
        description="Independent-group XGBoost classification with fold-wise PPERM."
    )
    parser.add_argument("--space", required=True, choices=("signal", "vortex"))
    parser.add_argument("--responders-dir", required=True, type=Path)
    parser.add_argument("--non-responders-dir", required=True, type=Path)
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


def load_group(directory, space, expected_size):
    if not directory.is_dir():
        raise NotADirectoryError(f"Group directory not found: {directory}")

    files = sorted(directory.glob(FILE_PATTERNS[space]))
    if len(files) != expected_size:
        raise ValueError(
            f"Expected {expected_size} connectivity files in {directory}, "
            f"found {len(files)}."
        )

    matrices = [extract_edges(path) for path in files]
    return np.stack(matrices)


def rank_sum_statistics(X_group_1, X_group_2):
    pooled = np.vstack((X_group_1, X_group_2))
    ranks = rankdata(pooled, axis=0, method="average")
    expected = X_group_1.shape[0] * (pooled.shape[0] + 1.0) / 2.0
    observed = np.abs(np.sum(ranks[:X_group_1.shape[0]], axis=0) - expected)
    return ranks, observed, expected


def wilcoxon_rank_sum_pvalues(X_group_1, X_group_2, rng):
    ranks, observed, expected = rank_sum_statistics(
        X_group_1,
        X_group_2,
    )
    n_group_1 = X_group_1.shape[0]
    n_total = ranks.shape[0]
    exceedances = np.zeros(ranks.shape[1], dtype=np.int64)

    for start in range(0, N_PERMUTATIONS, PERMUTATION_BATCH_SIZE):
        batch_size = min(
            PERMUTATION_BATCH_SIZE,
            N_PERMUTATIONS - start,
        )
        priorities = rng.random((batch_size, n_total))
        selected = np.argpartition(
            priorities,
            n_group_1 - 1,
            axis=1,
        )[:, :n_group_1]

        selectors = np.zeros((batch_size, n_total), dtype=np.float64)
        selectors[
            np.arange(batch_size)[:, None],
            selected,
        ] = 1.0

        permuted = np.abs(selectors @ ranks - expected)
        exceedances += np.sum(permuted >= observed[None, :], axis=0)

    p_values = (exceedances + 1.0) / (N_PERMUTATIONS + 1.0)
    return p_values, observed


def rank_features(X_group_1, X_group_2, rng):
    p_values, statistics = wilcoxon_rank_sum_pvalues(
        X_group_1,
        X_group_2,
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
    X_group_1 = load_group(
        args.responders_dir,
        args.space,
        N_RESPONDERS,
    )
    X_group_2 = load_group(
        args.non_responders_dir,
        args.space,
        N_NON_RESPONDERS,
    )

    if X_group_1.shape[1] != X_group_2.shape[1]:
        raise ValueError(
            f"Groups differ in feature count: {X_group_1.shape[1]} versus "
            f"{X_group_2.shape[1]}."
        )

    n_features_total = X_group_1.shape[1]
    if max(NUM_FEATURES) > n_features_total:
        raise ValueError(
            f"Top-{max(NUM_FEATURES)} was requested, but only "
            f"{n_features_total} features are available."
        )

    n_group_1 = X_group_1.shape[0]
    n_group_2 = X_group_2.shape[0]
    balanced_size = min(n_group_1, n_group_2)
    n_subsets = len(NUM_FEATURES)

    accuracies = np.empty((n_subsets, N_SHUFFLES), dtype=np.float64)
    confusions = np.empty((n_subsets, N_SHUFFLES, 2, 2), dtype=np.int32)
    rankings = np.empty((N_SHUFFLES, n_features_total), dtype=np.int32)
    feature_pvalues = np.empty(
        (N_SHUFFLES, n_features_total),
        dtype=np.float64,
    )
    split_rng = np.random.default_rng(SEED)

    for iteration in range(N_SHUFFLES):
        order_1 = split_rng.permutation(n_group_1)
        order_2 = split_rng.permutation(n_group_2)

        train_1 = order_1[:balanced_size - 1]
        train_2 = order_2[:balanced_size - 1]
        test_1 = order_1[balanced_size - 1]
        test_2 = order_2[balanced_size - 1]

        X_train_1 = X_group_1[train_1]
        X_train_2 = X_group_2[train_2]
        X_train = np.vstack((X_train_1, X_train_2))
        y_train = np.concatenate((
            np.zeros(balanced_size - 1, dtype=np.int32),
            np.ones(balanced_size - 1, dtype=np.int32),
        ))

        X_test = np.vstack((
            X_group_1[test_1],
            X_group_2[test_2],
        ))
        y_test = np.array([0, 1], dtype=np.int32)

        fold_seed = SEED + iteration + 1
        ranking, p_values = rank_features(
            X_train_1,
            X_train_2,
            np.random.default_rng(fold_seed),
        )
        rankings[iteration] = ranking
        feature_pvalues[iteration] = p_values

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

        if (iteration + 1) % 10 == 0 or iteration == 0:
            print(
                f"Completed shuffle {iteration + 1}/{N_SHUFFLES}",
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
        f"results_xgb_pperm_escitalopram_response_"
        f"{args.space}.mat"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sio.savemat(
        output_path,
        {
            "task": "escitalopram responders versus non-responders",
            "space": args.space,
            "class_names": np.asarray(
                ["responders", "non-responders"],
                dtype=object,
            ),
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
