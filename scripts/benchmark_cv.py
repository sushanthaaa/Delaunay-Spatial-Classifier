#!/usr/bin/env python3
"""
Multi-Seed Cross-Validation Benchmark for Delaunay Triangulation Classifier

Regenerates datasets with multiple random seeds and runs the full classifier
comparison on each, then aggregates results as mean +/- std across seeds.
This is the statistically-sound alternative to single-seed k-fold CV: it
accounts for variance in BOTH the data generation process and the train/test
split, rather than only the fold partitioning.

Compared classifiers:
  - KNN (k=5)                — sklearn KNeighborsClassifier
  - SVM (RBF)                — sklearn SVC, default (C=1, gamma='scale')
  - Decision Tree            — sklearn DecisionTreeClassifier, adaptive depth
  - Random Forest            — sklearn RandomForestClassifier (100 trees)
  - Delaunay (Ours)          — C++ binary via subprocess

Reports (per dataset):
  - Mean +/- std accuracy across seeds
  - Mean +/- std inference time (microseconds per query)
  - Per-class precision/recall/F1 mean +/- std across seeds
  - Aggregated confusion matrix (concatenated predictions across all seeds)
  - 95% confidence intervals
  - Paired t-test and Wilcoxon signed-rank significance tests
  - Bonferroni-corrected p-values

Usage:
  python scripts/benchmark_cv.py                                     # All datasets, all seeds
  python scripts/benchmark_cv.py --datasets moons,spiral,sfcrime     # Specific datasets
  python scripts/benchmark_cv.py --seeds 42,123,456                  # Custom seeds
  python scripts/benchmark_cv.py --datasets wine --seeds 42          # Single run (debug)

Outputs:
  results/cv_summary.csv                          — Mean +/- std accuracy for all algorithms
  results/significance_tests.csv                  — Statistical significance test results
  results/per_class_metrics_{dataset}.csv         — Per-class P/R/F1 for each classifier
  results/confusion_matrix_{dataset}_{alg}.csv    — Aggregated confusion matrices

"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import time
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

# =============================================================================
# Configuration constants
# =============================================================================

# the paper for reproducibility experiments.
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]

# to include 'sfcrime' (the new real-world spatial dataset added in #26).
ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake', 'sfcrime',
    'wine', 'cancer', 'bloodmnist'
]


# =============================================================================
# Data loading / dataset regeneration
# =============================================================================

def load_data(path):
    """Load CSV data with x, y, label columns (no header)."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values


def regenerate_datasets_for_seed(seed, requested_datasets, project_root):
    """Regenerate datasets with the given seed into a fresh temp directory.

    each other, and so we don't destroy the user's committed seed=42 data
    in data/train/ and data/test/.

    The data/cached/ directory from the project root is copied into the
    temp dir first, so generate_datasets.py finds the cached earthquake,
    sfcrime, and bloodmnist artifacts and doesn't re-download them. This
    makes the multi-seed run fully offline-safe as long as the user has
    already populated data/cached/ at least once.

    Returns the path to the temp directory, which has the same structure
    as project_root/data/ (i.e. data/train/, data/test/, data/cached/).
    The caller is responsible for cleaning it up after use.
    """
    tmp_root = tempfile.mkdtemp(prefix=f"benchcv_seed_{seed}_")

    # Copy the cache directory so real-world datasets (earthquake, sfcrime,
    # bloodmnist) can load from cache instead of hitting the network.
    src_cache = os.path.join(project_root, "data", "cached")
    dst_cache = os.path.join(tmp_root, "data", "cached")
    if os.path.isdir(src_cache):
        os.makedirs(os.path.dirname(dst_cache), exist_ok=True)
        shutil.copytree(src_cache, dst_cache)
    else:
        print(f"  Warning: {src_cache} does not exist. "
              f"Real-world datasets will be re-downloaded from their APIs.")

    # Build the --type argument for generate_datasets.py. If the user
    # requested a subset via --datasets, we only regenerate that subset
    # to save time.
    if set(requested_datasets) == set(ALL_DATASETS):
        type_arg = "all"
    else:
        type_arg = ",".join(requested_datasets)

    generate_script = os.path.join(
        project_root, "scripts", "generate_datasets.py")

    cmd = [
        sys.executable,
        generate_script,
        "--seed", str(seed),
        "--out_dir", tmp_root,
        "--type", type_arg,
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        # Surface the generator's error output so the user can diagnose
        # failures without having to run the generator manually.
        print(f"  ERROR: generate_datasets.py failed for seed={seed}")
        print(f"  stdout: {proc.stdout[-500:]}")
        print(f"  stderr: {proc.stderr[-500:]}")
        shutil.rmtree(tmp_root, ignore_errors=True)
        raise RuntimeError(
            f"Dataset regeneration failed for seed={seed}")

    return tmp_root


# =============================================================================
# C++ Delaunay subprocess wrapper
# =============================================================================

def run_cpp_delaunay(train_X, train_y, test_X, cpp_exe, results_dir):
    """Run C++ Delaunay classifier via subprocess and return predictions + timing.

    Writes train and test data to temp CSVs, invokes the C++ binary in
    'static' mode, and parses the output CSV for predictions. The C++ binary
    prints an "Avg Time Per Point" line which we scrape for timing.
    """
    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(train_X)):
            f.write(f"{train_X[i, 0]},{train_X[i, 1]},{int(train_y[i])}\n")

    with tempfile.NamedTemporaryFile(
            mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(test_X)):
            f.write(f"{test_X[i, 0]},{test_X[i, 1]}\n")

    pred_path = os.path.join(results_dir, "predictions.csv")
    if os.path.exists(pred_path):
        os.remove(pred_path)

    cmd = [cpp_exe, "static", train_path, test_path, results_dir]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    avg_time_us = 0.0
    for line in proc.stdout.split('\n'):
        if "Avg Time Per Point" in line:
            try:
                avg_time_us = float(line.split()[-2])
            except (ValueError, IndexError):
                pass

    predictions = None
    if os.path.exists(pred_path):
        predictions = pd.read_csv(pred_path, header=None).values.ravel()

    # Clean up temp files even on failure.
    try:
        os.remove(train_path)
    except OSError:
        pass
    try:
        os.remove(test_path)
    except OSError:
        pass

    return predictions, avg_time_us


# =============================================================================
# Per-seed measurement
# =============================================================================

#
# KNN grid: same k values as C++ FlannKNN::fit() CV tuning.
KNN_K_GRID = [1, 3, 5, 7, 9, 11, 15]

# SVM C grid: same values as C++ SVMClassifier::fit() grid search.
SVM_C_GRID = [0.1, 1.0, 10.0, 100.0]

# SVM gamma multiplier grid: these multiply sklearn's 'scale' heuristic
# (= 1 / (n_features * variance)) to produce the actual gamma values.
# Matches the C++ benchmark's gamma_mult grid.
SVM_GAMMA_MULT_GRID = [0.1, 1.0, 10.0, 100.0]


def _compute_scale_gamma(X_train):
    """Compute sklearn's 'scale' gamma heuristic from training data.

    For 2D data this equals 1 / (2 * X.var()) where X.var() is the variance
    of the flattened feature matrix. This matches both:
      - sklearn's SVC(gamma='scale') default (which uses X.var())
      - benchmark.cpp's reference gamma formula
    """
    variance = float(X_train.var())
    if variance < 1e-12:
        return 1.0
    n_features = X_train.shape[1]
    return 1.0 / (n_features * variance)


def build_classifiers(X_train, seed):
    """Construct the classifier dict with CV-tuned hyperparameters.

    of benchmark.cpp. This closes the methodological gap between the
    Python and C++ benchmark scripts so they can be compared directly in
    the paper's results table.

    - KNN: 5-fold CV over k in KNN_K_GRID (7 values).
    - SVM: 5-fold CV over C x gamma_mult grid (4 x 4 = 16 combinations),
            where gamma = gamma_mult * scale_gamma and scale_gamma is
            computed from the training data variance (same formula as
            sklearn's 'scale' default and benchmark.cpp's reference).
    - Decision Tree: sklearn's default exhaustive split selection already
            matches benchmark.cpp's DecisionTreeCpp, so no CV is applied.
    - Random Forest: 100 trees matches benchmark.cpp's RandomForestCpp,
            and bagging provides natural hyperparameter robustness without
            explicit CV.

    Runtime note: adding SVM grid search increases total benchmark runtime
    from ~60 min to ~2-3 hours for the full 12-dataset x 5-seed sweep,
    dominated by BloodMNIST (5 min SVM CV per seed at ~12K training points).
    """
    scale_gamma = _compute_scale_gamma(X_train)
    svm_gamma_values = [m * scale_gamma for m in SVM_GAMMA_MULT_GRID]

    # Adaptive Decision Tree depth matches the C++ benchmark heuristic:
    # roughly 2 * log2(n), clamped to [5, 20].
    n_total = len(X_train)
    adaptive_depth = min(20, max(5, int(2.0 * np.log2(max(2, n_total)))))

    classifiers = {
        # KNN with CV-tuned k. n_jobs=-1 parallelizes the CV folds.
        "KNN (CV-tuned k)": GridSearchCV(
            estimator=KNeighborsClassifier(),
            param_grid={'n_neighbors': KNN_K_GRID},
            cv=5,
            n_jobs=-1,
            refit=True,
        ),
        # SVM with CV-tuned C and gamma. The grid matches benchmark.cpp.
        "SVM (RBF, CV-tuned)": GridSearchCV(
            estimator=SVC(kernel='rbf', random_state=seed),
            param_grid={
                'C': SVM_C_GRID,
                'gamma': svm_gamma_values,
            },
            cv=5,
            n_jobs=-1,
            refit=True,
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=adaptive_depth, random_state=seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1),
    }

    return classifiers


def measure_classifiers_on_seed(dataset_name, seed_root, cpp_exe, results_dir,
                                seed):
    """Train and evaluate all classifiers on one seed's version of a dataset.

    Returns a dict mapping algorithm name to:
      {
        'accuracy': float,
        'time_us': float,
        'y_true': np.array of true labels (for aggregated confusion matrix),
        'y_pred': np.array of predicted labels,
      }

    This is called once per seed. The caller (run_cv_benchmark) accumulates
    these across seeds and computes mean/std at the end.
    """
    train_csv = f"{seed_root}/data/train/{dataset_name}_train.csv"
    test_csv = f"{seed_root}/data/test/{dataset_name}_test_y.csv"

    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)

    if X_train is None or X_test is None:
        return None

    classifiers = build_classifiers(X_train, seed)

    seed_results = {}

    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)

        start = time.perf_counter()
        y_pred = clf.predict(X_test)
        end = time.perf_counter()

        acc = accuracy_score(y_test, y_pred)
        avg_time = ((end - start) / len(X_test)) * 1_000_000

        seed_results[name] = {
            'accuracy': acc,
            'time_us': avg_time,
            'y_true': y_test.copy(),
            'y_pred': y_pred.copy(),
        }

    # C++ Delaunay
    if os.path.exists(cpp_exe):
        preds, cpp_time_us = run_cpp_delaunay(
            X_train, y_train, X_test, cpp_exe, results_dir)
        if preds is not None and len(preds) == len(y_test):
            acc = accuracy_score(y_test, preds)
            seed_results["Delaunay (Ours)"] = {
                'accuracy': acc,
                'time_us': cpp_time_us,
                'y_true': y_test.copy(),
                'y_pred': preds.copy(),
            }
        else:
            print(f"    Warning: Delaunay returned invalid predictions "
                  f"for seed={seed}")
    else:
        print(f"    Warning: C++ executable not found at {cpp_exe}")

    return seed_results


# =============================================================================
# Multi-seed aggregation
# =============================================================================

def run_cv_benchmark(dataset_name, seed_roots, project_root, seeds):
    """Run the multi-seed benchmark for a single dataset.

    (passed in as seed_roots, a dict mapping seed -> temp_root). We
    measure each seed once and accumulate results.

    Returns a dict mapping algorithm name to:
      {
        'accuracy': list of floats (one per seed),
        'time_us': list of floats (one per seed),
        'y_true_concat': concatenated true labels across all seeds,
        'y_pred_concat': concatenated predictions across all seeds,
        'per_seed_prf': list of (precision, recall, f1, support) tuples
                         from sklearn, one per seed,
      }

    The y_*_concat fields are used for the aggregated confusion matrix
    . The per_seed_prf list is used to compute mean +/- std
    per-class precision/recall/F1.
    """
    cpp_exe = f"{project_root}/build/main"
    results_dir = f"{project_root}/results"

    print(f"\n{'='*72}")
    print(f"Dataset: {dataset_name.upper()}")
    print(f"{'='*72}")

    # Accumulator for all classifier results across seeds.
    # Structure: {alg_name: {'accuracy': [...], 'time_us': [...],
    #                        'y_true_concat': [...], 'y_pred_concat': [...],
    #                        'per_seed_prf': [...]}}
    accumulator = {}

    for seed in seeds:
        seed_root = seed_roots[seed]
        print(f"  Seed {seed}...", end=" ", flush=True)

        seed_results = measure_classifiers_on_seed(
            dataset_name, seed_root, cpp_exe, results_dir, seed)

        if seed_results is None:
            print("SKIPPED (data not found)")
            continue

        for alg_name, result in seed_results.items():
            if alg_name not in accumulator:
                accumulator[alg_name] = {
                    'accuracy': [],
                    'time_us': [],
                    'y_true_concat': [],
                    'y_pred_concat': [],
                    'per_seed_prf': [],
                }

            accumulator[alg_name]['accuracy'].append(result['accuracy'])
            accumulator[alg_name]['time_us'].append(result['time_us'])
            accumulator[alg_name]['y_true_concat'].append(result['y_true'])
            accumulator[alg_name]['y_pred_concat'].append(result['y_pred'])

            # Use labels=sorted(unique) to ensure consistent class ordering
            # even when some folds/seeds might not see all classes.
            all_labels = np.unique(
                np.concatenate([result['y_true'], result['y_pred']]))
            prf = precision_recall_fscore_support(
                result['y_true'], result['y_pred'],
                labels=all_labels, average=None, zero_division=0)
            accumulator[alg_name]['per_seed_prf'].append((prf, all_labels))

        print("Done")

    # Finalize: concatenate y arrays into single vectors for confusion matrix.
    for alg_name in accumulator:
        acc_data = accumulator[alg_name]
        if acc_data['y_true_concat']:
            acc_data['y_true_concat'] = np.concatenate(
                acc_data['y_true_concat'])
            acc_data['y_pred_concat'] = np.concatenate(
                acc_data['y_pred_concat'])

    return accumulator


# =============================================================================
# Statistics
# =============================================================================

def compute_statistics(accumulator):
    """Compute mean, std, and 95% CI for accuracy and time across seeds."""
    stats_results = {}
    for name, data in accumulator.items():
        accs = np.array(data["accuracy"])
        times = np.array(data["time_us"])

        if len(accs) == 0:
            continue

        stats_results[name] = {
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs, ddof=1) if len(accs) > 1 else 0.0,
            "ci_acc": (1.96 * np.std(accs, ddof=1) / np.sqrt(len(accs))
                       if len(accs) > 1 else 0.0),
            "mean_time": np.mean(times),
            "std_time": np.std(times, ddof=1) if len(times) > 1 else 0.0,
            "raw_acc": accs,
            "raw_time": times,
            "y_true_concat": data["y_true_concat"],
            "y_pred_concat": data["y_pred_concat"],
            "per_seed_prf": data["per_seed_prf"],
        }

    return stats_results


def compute_per_class_metrics(stats_results):
    """Aggregate per-class precision/recall/F1 as mean +/- std across seeds.

    columns: algorithm, class, precision_mean, precision_std, recall_mean,
    recall_std, f1_mean, f1_std, support_mean.
    """
    records = []
    for alg_name, data in stats_results.items():
        per_seed_prf = data.get("per_seed_prf", [])
        if not per_seed_prf:
            continue

        # Find the union of all labels seen across all seeds. Then stack
        # per-seed metrics into arrays indexed by this label set, filling
        # missing classes with NaN so they don't corrupt the mean.
        all_labels = set()
        for prf, labels in per_seed_prf:
            all_labels.update(labels.tolist())
        all_labels = sorted(all_labels)

        n_seeds = len(per_seed_prf)
        n_classes = len(all_labels)
        prec_mat = np.full((n_seeds, n_classes), np.nan)
        rec_mat = np.full((n_seeds, n_classes), np.nan)
        f1_mat = np.full((n_seeds, n_classes), np.nan)
        sup_mat = np.full((n_seeds, n_classes), np.nan)

        for seed_idx, (prf, labels) in enumerate(per_seed_prf):
            prec, rec, f1, sup = prf
            for class_idx, label in enumerate(labels):
                col = all_labels.index(label)
                prec_mat[seed_idx, col] = prec[class_idx]
                rec_mat[seed_idx, col] = rec[class_idx]
                f1_mat[seed_idx, col] = f1[class_idx]
                sup_mat[seed_idx, col] = sup[class_idx]

        for col, label in enumerate(all_labels):
            records.append({
                "algorithm": alg_name,
                "class": int(label),
                "precision_mean": float(np.nanmean(prec_mat[:, col])),
                "precision_std": float(np.nanstd(prec_mat[:, col], ddof=1))
                                  if n_seeds > 1 else 0.0,
                "recall_mean": float(np.nanmean(rec_mat[:, col])),
                "recall_std": float(np.nanstd(rec_mat[:, col], ddof=1))
                               if n_seeds > 1 else 0.0,
                "f1_mean": float(np.nanmean(f1_mat[:, col])),
                "f1_std": float(np.nanstd(f1_mat[:, col], ddof=1))
                           if n_seeds > 1 else 0.0,
                "support_mean": float(np.nanmean(sup_mat[:, col])),
            })

    return records


def run_significance_tests(stats_results, n_datasets,
                           baseline="Delaunay (Ours)"):
    """Paired t-test and Wilcoxon signed-rank test vs the baseline algorithm.

    Operates on the per-seed accuracy vectors (not per-fold as in the old
    version). With 5 seeds, this gives 5 paired observations per baseline
    comparison, which is the statistical minimum for a Wilcoxon test.

    Bonferroni correction is applied across (n_comparisons * n_datasets)
    to control family-wise error rate.
    """
    sig_tests = []

    if baseline not in stats_results:
        return sig_tests

    our_accs = stats_results[baseline]["raw_acc"]
    n_comparisons = (len(stats_results) - 1) * n_datasets

    for name, data in stats_results.items():
        if name == baseline:
            continue

        other_accs = data["raw_acc"]

        if len(our_accs) != len(other_accs) or len(our_accs) < 2:
            continue

        # Paired t-test
        t_stat, t_pval = stats.ttest_rel(our_accs, other_accs)

        # Wilcoxon signed-rank test (requires at least one non-zero diff)
        try:
            diff = our_accs - other_accs
            if np.all(diff == 0):
                w_stat, w_pval = 0.0, 1.0
            else:
                w_stat, w_pval = stats.wilcoxon(our_accs, other_accs)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        bonferroni_threshold = 0.05 / max(1, n_comparisons)
        significant = (not np.isnan(t_pval)) and (t_pval < bonferroni_threshold)

        sig_tests.append({
            "baseline": name,
            "t_stat": float(t_stat) if not np.isnan(t_stat) else np.nan,
            "t_pval": float(t_pval) if not np.isnan(t_pval) else np.nan,
            "w_stat": float(w_stat) if not np.isnan(w_stat) else np.nan,
            "w_pval": float(w_pval) if not np.isnan(w_pval) else np.nan,
            "bonferroni_threshold": float(bonferroni_threshold),
            "significant": bool(significant),
        })

    return sig_tests


# =============================================================================
# Reporting
# =============================================================================

def print_results(dataset_name, stats_results, sig_tests, seeds):
    """Print formatted results table for a single dataset."""
    print(f"\n{'='*95}")
    print(f"MULTI-SEED BENCHMARK RESULTS: {dataset_name.upper()} "
          f"(seeds={seeds})")
    print(f"{'-'*95}")
    print(f"{'Algorithm':<25} | {'Accuracy (mean +/- std)':<25} | "
          f"{'Inference Time (us)':<25}")
    print(f"{'-'*95}")

    for name, data in stats_results.items():
        acc_str = (f"{data['mean_acc']*100:.2f} +/- "
                   f"{data['std_acc']*100:.2f}%")
        time_str = f"{data['mean_time']:.4f} +/- {data['std_time']:.4f}"
        print(f"{name:<25} | {acc_str:<25} | {time_str:<25}")

    print(f"{'='*95}")

    if sig_tests:
        print(f"\nSTATISTICAL SIGNIFICANCE (vs Delaunay)")
        print(f"{'-'*85}")
        print(f"{'Baseline':<25} | {'t-test p':<15} | {'Wilcoxon p':<15} | "
              f"{'Bonf. thr.':<12} | {'Sig.'}")
        print(f"{'-'*85}")
        for test in sig_tests:
            sig_str = "YES" if test['significant'] else "no"
            t_p = test['t_pval']
            w_p = test['w_pval']
            t_str = f"{t_p:.6f}" if not np.isnan(t_p) else "nan"
            w_str = f"{w_p:.6f}" if not np.isnan(w_p) else "nan"
            print(f"{test['baseline']:<25} | {t_str:<15} | {w_str:<15} | "
                  f"{test['bonferroni_threshold']:<12.6f} | {sig_str}")
        print(f"{'-'*85}")


def save_per_class_metrics(dataset_name, per_class_records, results_dir):
    """Write per_class_metrics_{dataset}.csv ."""
    if not per_class_records:
        return
    df = pd.DataFrame(per_class_records)
    out_path = f"{results_dir}/per_class_metrics_{dataset_name}.csv"
    df.to_csv(out_path, index=False)
    print(f"  Per-class metrics saved to: {out_path}")


def save_confusion_matrices(dataset_name, stats_results, results_dir):
    """Write confusion_matrix_{dataset}_{alg}.csv for each algorithm .

    Uses concatenated predictions across all seeds, which is the statistically
    correct aggregation (element-wise averaging of normalized matrices would
    double-count off-diagonal entries).
    """
    for alg_name, data in stats_results.items():
        y_true = data.get("y_true_concat")
        y_pred = data.get("y_pred_concat")
        if (y_true is None or y_pred is None
                or len(y_true) == 0 or len(y_pred) == 0):
            continue

        # Determine the set of labels present in either vector.
        labels = sorted(np.unique(np.concatenate([y_true, y_pred])).tolist())
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        df = pd.DataFrame(
            cm,
            index=[f"true_{int(l)}" for l in labels],
            columns=[f"pred_{int(l)}" for l in labels],
        )
        # Sanitize algorithm name for filename (remove spaces and parens).
        safe_alg = (alg_name.replace(" ", "_").replace("(", "")
                    .replace(")", "").replace("/", "_"))
        out_path = (f"{results_dir}/confusion_matrix_{dataset_name}"
                    f"_{safe_alg}.csv")
        df.to_csv(out_path)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-seed cross-validation benchmark for "
                    "Delaunay classifier")
    parser.add_argument(
        '--datasets', default='all',
        help='Comma-separated dataset names or "all" (default: all)')
    parser.add_argument(
        '--seeds', default=None,
        help=f'Comma-separated seeds (default: '
             f'{",".join(str(s) for s in DEFAULT_SEEDS)})')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.datasets == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]
        for name in datasets:
            if name not in ALL_DATASETS:
                print(f"Error: Unknown dataset '{name}'. "
                      f"Available: {', '.join(ALL_DATASETS)}")
                sys.exit(1)

    if args.seeds is None:
        seeds = DEFAULT_SEEDS
    else:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
        except ValueError:
            print("Error: --seeds must be a comma-separated list of integers")
            sys.exit(1)

    print("=" * 72)
    print("MULTI-SEED CROSS-VALIDATION BENCHMARK")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Seeds:    {seeds}")
    print("=" * 72)

    # -------------------------------------------------------------------------
    # Phase 1: Regenerate all datasets for all seeds into temp directories.
    # -------------------------------------------------------------------------
    #
    # We do this up-front (rather than lazily per-dataset) so that a partial
    # failure in generate_datasets.py fails fast before any benchmark work
    # begins. The temp directories are remembered in seed_roots and cleaned
    # up at the end of the run.
    seed_roots = {}
    try:
        print("\n[Phase 1/2] Regenerating datasets for all seeds...")
        for seed in seeds:
            print(f"  seed={seed}...", end=" ", flush=True)
            tmp = regenerate_datasets_for_seed(seed, datasets, project_root)
            seed_roots[seed] = tmp
            print(f"-> {tmp}")

        # ---------------------------------------------------------------------
        # Phase 2: Run the classifier suite on each dataset, aggregating
        # across seeds.
        # ---------------------------------------------------------------------
        print("\n[Phase 2/2] Running benchmark...")

        all_results = []
        all_sig_tests = []
        all_per_class = []

        for dataset in datasets:
            accumulator = run_cv_benchmark(
                dataset, seed_roots, project_root, seeds)
            if not accumulator:
                continue

            stats_results = compute_statistics(accumulator)
            sig_tests = run_significance_tests(stats_results, len(datasets))
            per_class_records = compute_per_class_metrics(stats_results)

            print_results(dataset, stats_results, sig_tests, seeds)

            # Save per-dataset artifacts immediately so a crash in a later
            # dataset doesn't destroy earlier results.
            results_dir = f"{project_root}/results"
            os.makedirs(results_dir, exist_ok=True)
            save_per_class_metrics(dataset, per_class_records, results_dir)
            save_confusion_matrices(dataset, stats_results, results_dir)

            for name, data in stats_results.items():
                all_results.append({
                    "dataset": dataset,
                    "algorithm": name,
                    "mean_acc": data["mean_acc"],
                    "std_acc": data["std_acc"],
                    "ci_acc": data["ci_acc"],
                    "mean_time_us": data["mean_time"],
                    "std_time_us": data["std_time"],
                    "n_seeds": len(data["raw_acc"]),
                })

            for test in sig_tests:
                all_sig_tests.append({"dataset": dataset, **test})

            for rec in per_class_records:
                all_per_class.append({"dataset": dataset, **rec})

        # ---------------------------------------------------------------------
        # Save top-level summary CSVs.
        # ---------------------------------------------------------------------
        results_dir = f"{project_root}/results"
        os.makedirs(results_dir, exist_ok=True)

        if all_results:
            df_results = pd.DataFrame(all_results)
            df_results.to_csv(
                f"{results_dir}/cv_summary.csv", index=False)
            print(f"\nResults saved to: {results_dir}/cv_summary.csv")

        if all_sig_tests:
            df_sig = pd.DataFrame(all_sig_tests)
            df_sig.to_csv(
                f"{results_dir}/significance_tests.csv", index=False)
            print(f"Significance tests saved to: "
                  f"{results_dir}/significance_tests.csv")

        if all_per_class:
            df_pc = pd.DataFrame(all_per_class)
            df_pc.to_csv(
                f"{results_dir}/per_class_metrics_all.csv", index=False)
            print(f"All per-class metrics saved to: "
                  f"{results_dir}/per_class_metrics_all.csv")

    finally:
        # Clean up all temp directories, even on error.
        print("\n[Cleanup] Removing temporary seed directories...")
        for seed, tmp in seed_roots.items():
            shutil.rmtree(tmp, ignore_errors=True)

    print("\n" + "=" * 72)
    print("MULTI-SEED CROSS-VALIDATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()