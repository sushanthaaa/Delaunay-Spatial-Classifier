#!/usr/bin/env python3
"""
Cross-Validation Benchmark for Delaunay Triangulation Classifier

Performs 10-fold stratified cross-validation comparing:
  - KNN (k=5)
  - SVM (RBF kernel)
  - Decision Tree (adaptive depth)
  - Random Forest (100 estimators)
  - Delaunay Classifier (C++ via subprocess)

Reports:
  - Mean ± std accuracy per algorithm per dataset
  - 95% confidence intervals
  - Paired t-test and Wilcoxon signed-rank statistical significance tests
  - Bonferroni-corrected p-values

Usage:
  python scripts/benchmark_cv.py                                    # All datasets
  python scripts/benchmark_cv.py --datasets moons,spiral,earthquake # Specific
  python scripts/benchmark_cv.py --folds 5                          # Custom folds

Outputs:
  results/cv_summary.csv          — Mean ± std accuracy for all algorithms
  results/significance_tests.csv  — Statistical significance test results
"""

import argparse
import os
import subprocess
import time
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

RANDOM_SEED = 42
N_FOLDS = 10

# Must match generate_datasets.py ALL_DATASETS
ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake',
    'wine', 'cancer', 'bloodmnist'
]


def load_data(path):
    """Load CSV data with x, y, label columns (no header)."""
    if not os.path.exists(path):
        print(f"  Warning: {path} not found, skipping.")
        return None, None
    df = pd.read_csv(path, header=None, names=['x', 'y', 'label'])
    return df[['x', 'y']].values, df['label'].values


def run_cpp_delaunay(train_X, train_y, test_X, cpp_exe, results_dir):
    """Run C++ Delaunay classifier via subprocess and return predictions + timing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        train_path = f.name
        for i in range(len(train_X)):
            f.write(f"{train_X[i,0]},{train_X[i,1]},{int(train_y[i])}\n")

    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        test_path = f.name
        for i in range(len(test_X)):
            f.write(f"{test_X[i,0]},{test_X[i,1]}\n")

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

    os.remove(train_path)
    os.remove(test_path)

    return predictions, avg_time_us


def run_cv_benchmark(dataset_name, root_dir, n_folds):
    """Run n-fold cross-validation on a single dataset."""
    train_csv = f"{root_dir}/data/train/{dataset_name}_train.csv"
    test_csv = f"{root_dir}/data/test/{dataset_name}_test_y.csv"
    cpp_exe = f"{root_dir}/build/main"
    results_dir = f"{root_dir}/results"

    X_train, y_train = load_data(train_csv)
    X_test, y_test = load_data(test_csv)

    if X_train is None or X_test is None:
        return None

    # Combine train + test for proper CV
    X = np.vstack([X_train, X_test])
    y = np.hstack([y_train, y_test])

    # Ensure enough samples per class for stratified CV
    unique, counts = np.unique(y, return_counts=True)
    min_count = counts.min()
    if min_count < n_folds:
        print(f"  Warning: class with only {min_count} samples, "
              f"reducing folds to {min_count}")
        n_folds = max(2, min_count)

    print(f"\n{'='*70}")
    print(f"Dataset: {dataset_name.upper()} ({len(X)} samples, "
          f"{len(unique)} classes, {n_folds}-fold CV)")
    print(f"{'='*70}")

    # Adaptive Decision Tree depth (matches C++ benchmark)
    adaptive_depth = min(20, max(5, int(2.0 * np.log2(len(X)))))

    classifiers = {
        "KNN (k=5)": KNeighborsClassifier(n_neighbors=5),
        "SVM (RBF)": SVC(kernel='rbf', random_state=RANDOM_SEED),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=adaptive_depth, random_state=RANDOM_SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_SEED),
    }

    results = {name: {"accuracy": [], "time_us": []} for name in classifiers}
    results["Delaunay (Ours)"] = {"accuracy": [], "time_us": []}

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True,
                          random_state=RANDOM_SEED)

    fold_num = 0
    for train_idx, test_idx in skf.split(X, y):
        fold_num += 1
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]

        print(f"  Fold {fold_num}/{n_folds}...", end=" ", flush=True)

        for name, clf in classifiers.items():
            clf.fit(X_train_fold, y_train_fold)

            start = time.perf_counter()
            y_pred = clf.predict(X_test_fold)
            end = time.perf_counter()

            acc = accuracy_score(y_test_fold, y_pred)
            avg_time = ((end - start) / len(X_test_fold)) * 1_000_000

            results[name]["accuracy"].append(acc)
            results[name]["time_us"].append(avg_time)

        # C++ Delaunay
        if os.path.exists(cpp_exe):
            preds, cpp_time = run_cpp_delaunay(
                X_train_fold, y_train_fold, X_test_fold, cpp_exe, results_dir)
            if preds is not None and len(preds) == len(y_test_fold):
                acc = accuracy_score(y_test_fold, preds)
                results["Delaunay (Ours)"]["accuracy"].append(acc)
                results["Delaunay (Ours)"]["time_us"].append(cpp_time)
        else:
            print(f"\n  Warning: C++ executable not found at {cpp_exe}")

        print("Done")

    return results


def compute_statistics(results):
    """Compute mean, std, and 95% CI for each algorithm."""
    stats_results = {}
    for name, data in results.items():
        accs = np.array(data["accuracy"])
        times = np.array(data["time_us"])

        if len(accs) == 0:
            continue

        stats_results[name] = {
            "mean_acc": np.mean(accs),
            "std_acc": np.std(accs, ddof=1) if len(accs) > 1 else 0,
            "ci_acc": 1.96 * np.std(accs, ddof=1) / np.sqrt(len(accs))
                      if len(accs) > 1 else 0,
            "mean_time": np.mean(times),
            "std_time": np.std(times, ddof=1) if len(times) > 1 else 0,
            "raw_acc": accs,
            "raw_time": times
        }

    return stats_results


def run_significance_tests(stats_results, n_datasets,
                           baseline="Delaunay (Ours)"):
    """Run paired t-test and Wilcoxon test comparing Delaunay vs each baseline."""
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

        # Wilcoxon signed-rank test
        try:
            diff = our_accs - other_accs
            if np.all(diff == 0):
                w_stat, w_pval = 0.0, 1.0
            else:
                w_stat, w_pval = stats.wilcoxon(our_accs, other_accs)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        # Bonferroni correction
        bonferroni_threshold = 0.05 / max(1, n_comparisons)
        significant = t_pval < bonferroni_threshold

        sig_tests.append({
            "baseline": name,
            "t_stat": t_stat,
            "t_pval": t_pval,
            "w_stat": w_stat,
            "w_pval": w_pval,
            "bonferroni_threshold": bonferroni_threshold,
            "significant": significant
        })

    return sig_tests


def print_results(dataset_name, stats_results, sig_tests, n_folds):
    """Print formatted results table."""
    print(f"\n{'='*90}")
    print(f"CROSS-VALIDATION RESULTS: {dataset_name.upper()} "
          f"({n_folds}-fold, seed={RANDOM_SEED})")
    print(f"{'-'*90}")
    print(f"{'Algorithm':<25} | {'Accuracy':<20} | {'Inference Time (us)':<25}")
    print(f"{'-'*90}")

    for name, data in stats_results.items():
        acc_str = f"{data['mean_acc']*100:.2f} +/- {data['std_acc']*100:.2f}%"
        time_str = f"{data['mean_time']:.4f} +/- {data['std_time']:.4f}"
        print(f"{name:<25} | {acc_str:<20} | {time_str:<25}")

    print(f"{'='*90}")

    if sig_tests:
        print(f"\nSTATISTICAL SIGNIFICANCE (vs Delaunay)")
        print(f"{'-'*80}")
        print(f"{'Baseline':<25} | {'t-test p':<15} | {'Wilcoxon p':<15} | "
              f"{'Bonf. thr.':<12} | {'Sig.'}")
        print(f"{'-'*80}")
        for test in sig_tests:
            sig_str = "YES" if test['significant'] else "no"
            print(f"{test['baseline']:<25} | {test['t_pval']:<15.6f} | "
                  f"{test['w_pval']:<15.6f} | {test['bonferroni_threshold']:<12.6f} | "
                  f"{sig_str}")
        print(f"{'-'*80}")


def main():
    parser = argparse.ArgumentParser(
        description="Cross-validation benchmark for Delaunay classifier")
    parser.add_argument('--datasets', default='all',
                        help='Comma-separated dataset names or "all"')
    parser.add_argument('--folds', type=int, default=10,
                        help='Number of CV folds (default: 10)')
    args = parser.parse_args()

    n_folds = args.folds

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.datasets == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    print("=" * 70)
    print("CROSS-VALIDATION BENCHMARK")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Folds: {n_folds}, Seed: {RANDOM_SEED}")
    print("=" * 70)

    all_results = []
    all_sig_tests = []

    for dataset in datasets:
        results = run_cv_benchmark(dataset, root, n_folds)
        if results is None:
            continue

        stats_results = compute_statistics(results)
        sig_tests = run_significance_tests(stats_results, len(datasets))

        print_results(dataset, stats_results, sig_tests, n_folds)

        for name, data in stats_results.items():
            all_results.append({
                "dataset": dataset,
                "algorithm": name,
                "mean_acc": data["mean_acc"],
                "std_acc": data["std_acc"],
                "ci_acc": data["ci_acc"],
                "mean_time_us": data["mean_time"],
                "std_time_us": data["std_time"]
            })

        for test in sig_tests:
            all_sig_tests.append({"dataset": dataset, **test})

    # Save to CSV
    results_dir = f"{root}/results"
    os.makedirs(results_dir, exist_ok=True)

    if all_results:
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(f"{results_dir}/cv_summary.csv", index=False)
        print(f"\nResults saved to: {results_dir}/cv_summary.csv")

    if all_sig_tests:
        df_sig = pd.DataFrame(all_sig_tests)
        df_sig.to_csv(f"{results_dir}/significance_tests.csv", index=False)
        print(f"Significance tests saved to: {results_dir}/significance_tests.csv")

    print("\n" + "=" * 70)
    print("CROSS-VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()