#!/usr/bin/env python3
"""
Ablation Study Wrapper for Delaunay Triangulation Classifier

Wraps the C++ ablation_bench executable to run ablation experiments
across all datasets and aggregate results.

Ablation components tested (by ablation_bench C++):
  A1: Full Pipeline           — classify() via 2D Buckets (baseline)
  A2: Without 2D Buckets Grid — classify_no_grid() via DT locate walk
  A3: Without Outlier Removal — classify() with outlier removal disabled
  A4: Nearest Vertex Only     — classify_nearest_vertex() (1-NN baseline)
  A6: Outlier Multiplier      — Sensitivity analysis (m = 1.5, 2.0, 3.0, 5.0, 10.0)

Dynamic ablation:
  D1: Full Dynamic             — insert/move/delete with bucket rebuild

Usage:
  python scripts/ablation_study.py                                    # All datasets
  python scripts/ablation_study.py --datasets moons,spiral,earthquake # Specific

Outputs:
  results/ablation_{dataset}.csv          — Static ablation per dataset
  results/ablation_dynamic_{dataset}.csv  — Dynamic ablation per dataset
  results/ablation_summary.csv            — Aggregated summary across datasets
"""

import argparse
import os
import subprocess
import pandas as pd

# Must match generate_datasets.py
ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake',
    'wine', 'cancer', 'bloodmnist'
]


def run_ablation(dataset_name, root_dir):
    """Run C++ ablation_bench for a single dataset."""
    train_csv = f"{root_dir}/data/train/{dataset_name}_train.csv"
    test_csv = f"{root_dir}/data/test/{dataset_name}_test_y.csv"
    ablation_exe = f"{root_dir}/build/ablation_bench"

    if not os.path.exists(train_csv):
        print(f"  [SKIP] {dataset_name}: training data not found")
        return None, None

    if not os.path.exists(test_csv):
        print(f"  [SKIP] {dataset_name}: test data not found")
        return None, None

    if not os.path.exists(ablation_exe):
        print(f"  [ERROR] ablation_bench not found at {ablation_exe}")
        print(f"  Build with: cd build && cmake .. && make ablation_bench")
        return None, None

    print(f"\n{'='*70}")
    print(f"ABLATION STUDY: {dataset_name.upper()}")
    print(f"{'='*70}")

    cmd = [ablation_exe, train_csv, test_csv, dataset_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    print(proc.stdout)
    if proc.stderr:
        print(f"  Warnings: {proc.stderr[:200]}")

    # Read generated CSV files
    static_csv = f"{root_dir}/results/ablation_{dataset_name}.csv"
    dynamic_csv = f"{root_dir}/results/ablation_dynamic_{dataset_name}.csv"

    static_df = None
    dynamic_df = None

    if os.path.exists(static_csv):
        static_df = pd.read_csv(static_csv)
        static_df['dataset'] = dataset_name
        print(f"  Static results: {static_csv}")
    else:
        print(f"  Warning: {static_csv} not generated")

    if os.path.exists(dynamic_csv):
        dynamic_df = pd.read_csv(dynamic_csv)
        dynamic_df['dataset'] = dataset_name
        print(f"  Dynamic results: {dynamic_csv}")
    else:
        print(f"  Warning: {dynamic_csv} not generated")

    return static_df, dynamic_df


def main():
    parser = argparse.ArgumentParser(
        description="Run ablation study across datasets")
    parser.add_argument('--datasets', default='all',
                        help='Comma-separated dataset names or "all"')
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))
    root = os.path.abspath(os.path.join(script_dir, ".."))

    if args.datasets == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    os.makedirs(f"{root}/results", exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY — Component Contribution Analysis")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Executable: build/ablation_bench")
    print()
    print("Components tested:")
    print("  A1: Full Pipeline (2D Buckets + Outlier + Decision Boundary)")
    print("  A2: Without 2D Buckets Grid (DT locate walk, no O(1) grid)")
    print("  A3: Without Outlier Removal (no Phase 1 cleanup)")
    print("  A4: Nearest Vertex Only (1-NN, no half-plane boundary)")
    print("  A6: Outlier Multiplier Sensitivity (m = 1.5 to 10.0)")
    print("  D1: Dynamic Operations (insert/move/delete with bucket rebuild)")
    print("=" * 70)

    all_static = []
    all_dynamic = []

    for dataset in datasets:
        static_df, dynamic_df = run_ablation(dataset, root)
        if static_df is not None:
            all_static.append(static_df)
        if dynamic_df is not None:
            all_dynamic.append(dynamic_df)

    # Aggregate and save summary
    if all_static:
        summary_df = pd.concat(all_static, ignore_index=True)
        summary_path = f"{root}/results/ablation_summary.csv"
        summary_df.to_csv(summary_path, index=False)
        print(f"\nAggregated static results: {summary_path}")

        # Print summary table
        print(f"\n{'='*90}")
        print("ABLATION SUMMARY — Accuracy Impact")
        print(f"{'='*90}")
        print(f"{'Dataset':<20} | {'Full Pipeline':<15} | {'No Grid':<15} | "
              f"{'No Outlier':<15} | {'1-NN Only':<15}")
        print(f"{'-'*90}")

        for dataset in datasets:
            ds_data = summary_df[summary_df['dataset'] == dataset]
            if ds_data.empty:
                continue

            full = ds_data[ds_data['variant'].str.contains('Full', na=False)]
            no_grid = ds_data[ds_data['variant'].str.contains('Without SRR', na=False)]
            no_outlier = ds_data[ds_data['variant'].str.contains('Without Outlier', na=False)]
            nn_only = ds_data[ds_data['variant'].str.contains('Nearest Vertex', na=False)]

            full_acc = f"{full['accuracy'].values[0]*100:.1f}%" if len(full) > 0 else "—"
            ng_acc = f"{no_grid['accuracy'].values[0]*100:.1f}%" if len(no_grid) > 0 else "—"
            no_acc = f"{no_outlier['accuracy'].values[0]*100:.1f}%" if len(no_outlier) > 0 else "—"
            nn_acc = f"{nn_only['accuracy'].values[0]*100:.1f}%" if len(nn_only) > 0 else "—"

            print(f"{dataset:<20} | {full_acc:<15} | {ng_acc:<15} | "
                  f"{no_acc:<15} | {nn_acc:<15}")

        print(f"{'='*90}")

    if all_dynamic:
        dyn_summary_df = pd.concat(all_dynamic, ignore_index=True)
        dyn_summary_path = f"{root}/results/ablation_dynamic_summary.csv"
        dyn_summary_df.to_csv(dyn_summary_path, index=False)
        print(f"Aggregated dynamic results: {dyn_summary_path}")

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()