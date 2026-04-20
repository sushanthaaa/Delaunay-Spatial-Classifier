#!/usr/bin/env python3
"""
Ablation Study Wrapper for Delaunay Triangulation Classifier

Wraps the C++ ablation_bench executable to run ablation experiments
across all datasets and aggregate results, with multi-seed averaging
for statistical robustness.

Ablation components tested (by ablation_bench C++):
  A1: Full Pipeline           — classify() via 2D Buckets (baseline)
  A2: Without 2D Buckets Grid — classify_no_grid() via DT locate walk
  A3: Without Outlier Removal — classify() with outlier removal disabled
  A4: Nearest Vertex Only     — classify_nearest_vertex() (1-NN baseline)
  A6: Outlier Multiplier      — Sensitivity analysis (m = 1.5, 2.0, 3.0, 5.0, 10.0)

Dynamic ablation:
  D1: Full Dynamic             — insert/move/delete with bucket rebuild

Usage:
  # Default: 12 datasets x 5 seeds, full paper run (~50-100 min)
  python scripts/ablation_study.py

  # Quick: single seed for development iteration
  python scripts/ablation_study.py --seeds 42

  # Subset of datasets
  python scripts/ablation_study.py --datasets moons,spiral,earthquake

  # Custom seeds
  python scripts/ablation_study.py --seeds 42,123,456

Outputs:
  results/ablation_summary.csv              — Cross-seed aggregated static
                                              (accuracy_mean ± _std, etc.)
  results/ablation_per_seed.csv             — Raw per-seed long-format data
  results/ablation_dynamic_summary.csv      — Cross-seed aggregated dynamic
  results/ablation_dynamic_per_seed.csv     — Raw per-seed dynamic data

"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import pandas as pd

# =============================================================================
# Configuration constants
# =============================================================================

# Module-level default seed. Matches generate_datasets.py and benchmark_cv.py.
DEFAULT_SEED = 42

# Cross-seed mean/std lets the paper report ablation effects with proper
# statistical robustness rather than single-seed point estimates.
DEFAULT_SEEDS = [42, 123, 456, 789, 1000]

# ablation study covers the same 12 datasets as benchmark_cv.py.
ALL_DATASETS = [
    'moons', 'circles', 'spiral', 'gaussian_quantiles', 'cassini',
    'checkerboard', 'blobs', 'earthquake', 'sfcrime',
    'wine', 'cancer', 'bloodmnist'
]


# =============================================================================
# Phase 1: Per-seed dataset regeneration
# =============================================================================
# Same pattern as benchmark_cv.py. We pre-generate all datasets for all seeds
# into separate temp directories so Phase 2 can iterate (dataset, seed) in any
# order without re-running expensive dataset generation. The data/cached/
# directory is copied into each temp dir so cache-first datasets (earthquake,
# sfcrime, bloodmnist) hit their offline-safe cache rather than re-fetching.

def regenerate_datasets_for_seeds(seeds, project_root):
    """Regenerate all datasets for each seed into separate temp directories.

    Returns a dict {seed: temp_dir_path} where each temp_dir contains a
    fresh data/train/, data/test/, and data/cached/ tree for that seed.
    """
    seed_roots = {}
    cached_src = os.path.join(project_root, "data", "cached")
    generator_script = os.path.join(
        project_root, "scripts", "generate_datasets.py")

    if not os.path.exists(generator_script):
        print(f"Error: generate_datasets.py not found at {generator_script}")
        sys.exit(1)

    print("[Phase 1/2] Regenerating datasets for all seeds...")
    for seed in seeds:
        temp_dir = tempfile.mkdtemp(prefix=f"ablation_seed_{seed}_")
        os.makedirs(os.path.join(temp_dir, "data"), exist_ok=True)

        # Copy data/cached/ into the temp dir so offline-safe cache hits work.
        if os.path.exists(cached_src):
            shutil.copytree(
                cached_src, os.path.join(temp_dir, "data", "cached"))

        # Run generate_datasets.py with --seed N --out_dir <temp_dir>.
        #
        # NOTE: generate_datasets.py expects --out_dir (underscore, not hyphen)
        # and treats the value as the PROJECT ROOT. It internally creates
        # {out_dir}/data/train/, {out_dir}/data/test/, and
        # {out_dir}/data/cached/ via create_output_dirs(). So we pass
        # temp_dir (the project-root-like path), NOT temp_dir/data.
        cmd = [
            sys.executable, generator_script,
            "--seed", str(seed),
            "--out_dir", temp_dir,
        ]
        print(f"  seed={seed}... ", end='', flush=True)
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            print(f"\n    Warning: generate_datasets.py exited with "
                  f"code {proc.returncode}")
            print(f"    stderr: {proc.stderr[-300:]}")
        else:
            print(f"-> {temp_dir}")

        seed_roots[seed] = temp_dir

    return seed_roots


def cleanup_seed_dirs(seed_roots):
    """Remove all temp directories created by regenerate_datasets_for_seeds."""
    for seed, path in seed_roots.items():
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
            except OSError as e:
                print(f"  Warning: could not remove {path}: {e}")


# =============================================================================
# Phase 2: Per-(dataset, seed) ablation runs
# =============================================================================

def run_ablation_for_seed(dataset_name, seed_root, project_root, seed):
    """Run C++ ablation_bench for one (dataset, seed) and read its CSV outputs.

    Returns (static_df, dynamic_df) — each with a 'seed' and 'dataset'
    column added, or (None, None) if the run failed.
    """
    train_csv = f"{seed_root}/data/train/{dataset_name}_train.csv"
    test_csv = f"{seed_root}/data/test/{dataset_name}_test_y.csv"
    ablation_exe = f"{project_root}/build/ablation_bench"
    results_dir = f"{project_root}/results"

    if not os.path.exists(train_csv):
        print(f"    [SKIP] {dataset_name} seed={seed}: training data not found")
        return None, None
    if not os.path.exists(test_csv):
        print(f"    [SKIP] {dataset_name} seed={seed}: test data not found")
        return None, None
    if not os.path.exists(ablation_exe):
        print(f"    [ERROR] ablation_bench not found at {ablation_exe}")
        print(f"    Build with: cd build && cmake .. && make ablation_bench")
        return None, None

    cmd = [ablation_exe, train_csv, test_csv, dataset_name]
    proc = subprocess.run(cmd, capture_output=True, text=True)

    if proc.returncode != 0:
        print(f"    [WARN] ablation_bench exited with code "
              f"{proc.returncode} for {dataset_name} seed={seed}")
        if proc.stderr:
            print(f"    stderr: {proc.stderr[-200:]}")

    # Read the per-dataset CSVs the C++ binary just wrote.
    static_csv = f"{results_dir}/ablation_{dataset_name}.csv"
    dynamic_csv = f"{results_dir}/ablation_dynamic_{dataset_name}.csv"

    static_df = None
    dynamic_df = None

    if os.path.exists(static_csv):
        try:
            static_df = pd.read_csv(static_csv)
            static_df['dataset'] = dataset_name
            static_df['seed'] = seed
        except Exception as e:
            print(f"    [WARN] Could not read {static_csv}: {e}")
    else:
        print(f"    [WARN] {static_csv} not generated by ablation_bench")

    if os.path.exists(dynamic_csv):
        try:
            dynamic_df = pd.read_csv(dynamic_csv)
            dynamic_df['dataset'] = dataset_name
            dynamic_df['seed'] = seed
        except Exception as e:
            print(f"    [WARN] Could not read {dynamic_csv}: {e}")
    else:
        # Dynamic CSV is optional; some datasets may not produce one.
        pass

    return static_df, dynamic_df


# =============================================================================
# Cross-seed aggregation
# =============================================================================
# rows (one per seed). We aggregate to mean/std across seeds for the
# headline table, and preserve raw per-seed data in a sidecar CSV.

def aggregate_static_across_seeds(per_seed_df):
    """Aggregate static ablation results across seeds.

    Input:  long-format DataFrame with one row per (dataset, variant, seed),
            columns: dataset, variant, accuracy, inference_us, train_ms,
                     notes, seed
    Output: aggregated DataFrame with one row per (dataset, variant),
            columns: dataset, variant, accuracy_mean, accuracy_std,
                     inference_us_mean, inference_us_std, train_ms_mean,
                     train_ms_std, n_seeds, notes
    """
    # Identify the numeric columns to aggregate. The C++ binary's static
    # ablation CSV emits these three timing/accuracy fields; everything
    # else (variant, notes, dataset, seed) is metadata.
    #
    # NOTE: ablation_bench emits the column as 'avg_inference_us' (not
    # 'inference_us'). We accept both spellings for forward compatibility
    # but normalize to 'inference_us' in the output for consistency with
    # benchmark_cv.py's column naming convention.
    if 'avg_inference_us' in per_seed_df.columns and 'inference_us' not in per_seed_df.columns:
        per_seed_df = per_seed_df.rename(columns={'avg_inference_us': 'inference_us'})

    numeric_cols = [c for c in ['accuracy', 'inference_us', 'train_ms']
                    if c in per_seed_df.columns]

    grouped = per_seed_df.groupby(['dataset', 'variant'], sort=False)

    rows = []
    for (dataset, variant), group in grouped:
        row = {'dataset': dataset, 'variant': variant}
        for col in numeric_cols:
            vals = group[col].astype(float).values
            row[f'{col}_mean'] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
            # ddof=1 for sample std; falls back to 0 with single seed.
            row[f'{col}_std'] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        row['n_seeds'] = len(group)
        # Notes are identical across seeds for a given variant; pick first.
        if 'notes' in group.columns:
            row['notes'] = group['notes'].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


def aggregate_dynamic_across_seeds(per_seed_df):
    """Aggregate dynamic ablation results across seeds.

    The C++ binary already emits within-seed mean/std for dynamic ops
    (e.g. insert_ns_mean, insert_ns_std from 1000 measurements per run).
    Here we compute the cross-seed mean and std of the within-seed means.

    The within-seed std reflects per-operation jitter inside one run; the
    cross-seed std reflects per-run variability. Both are useful but answer
    different questions, so we preserve them separately. The within-seed
    std is averaged across seeds and saved as <op>_within_std_mean for
    diagnostic comparison.
    """
    # The mean columns we'll cross-seed aggregate.
    mean_cols = [c for c in ['insert_ns_mean', 'move_ns_mean', 'delete_ns_mean']
                 if c in per_seed_df.columns]
    # The within-seed std columns we'll average for diagnostic context.
    std_cols = [c for c in ['insert_ns_std', 'move_ns_std', 'delete_ns_std']
                if c in per_seed_df.columns]

    grouped = per_seed_df.groupby(['dataset', 'variant'], sort=False)

    rows = []
    for (dataset, variant), group in grouped:
        row = {'dataset': dataset, 'variant': variant}
        for col in mean_cols:
            # col is e.g. "insert_ns_mean"; strip the "_mean" suffix for the
            # aggregated column name (we'll re-add appropriate suffixes).
            base = col[:-len('_mean')]  # "insert_ns"
            vals = group[col].astype(float).values
            # Cross-seed aggregation of the per-seed means
            row[f'{base}_mean'] = float(np.mean(vals)) if len(vals) > 0 else float('nan')
            row[f'{base}_cross_seed_std'] = (
                float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0)
        for col in std_cols:
            base = col[:-len('_std')]  # "insert_ns"
            vals = group[col].astype(float).values
            # Average within-seed std across seeds (diagnostic)
            row[f'{base}_within_std_mean'] = (
                float(np.mean(vals)) if len(vals) > 0 else float('nan'))
        row['n_seeds'] = len(group)
        if 'num_ops' in group.columns:
            row['num_ops'] = int(group['num_ops'].iloc[0])
        if 'notes' in group.columns:
            row['notes'] = group['notes'].iloc[0]
        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# Summary printing
# =============================================================================

def print_static_summary(aggregated_df, datasets):
    """Print a human-readable summary table of static ablation results."""
    if aggregated_df.empty:
        return

    print(f"\n{'='*100}")
    print("ABLATION SUMMARY — Cross-Seed Accuracy (mean +/- std)")
    print(f"{'='*100}")

    # Get the unique variants in the order the C++ binary emitted them
    # (groupby with sort=False preserves first-occurrence order).
    variants_seen = aggregated_df.drop_duplicates(
        'variant', keep='first')['variant'].tolist()

    # Header row
    header = f"{'Dataset':<20}"
    for v in variants_seen:
        # Truncate long variant names for column headers
        v_short = v if len(v) <= 22 else v[:19] + '...'
        header += f" | {v_short:<22}"
    print(header)
    print('-' * len(header))

    for dataset in datasets:
        ds_data = aggregated_df[aggregated_df['dataset'] == dataset]
        if ds_data.empty:
            continue

        row = f"{dataset:<20}"
        for v in variants_seen:
            variant_row = ds_data[ds_data['variant'] == v]
            if variant_row.empty:
                cell = "—"
            else:
                acc_mean = variant_row['accuracy_mean'].values[0]
                acc_std = variant_row['accuracy_std'].values[0]
                if np.isnan(acc_mean):
                    cell = "—"
                else:
                    cell = f"{acc_mean*100:5.1f} +/- {acc_std*100:.1f}%"
            row += f" | {cell:<22}"
        print(row)
    print('=' * len(header))


def print_dynamic_summary(aggregated_df, datasets):
    """Print a human-readable summary table of dynamic ablation results."""
    if aggregated_df.empty:
        return

    print(f"\n{'='*110}")
    print("DYNAMIC ABLATION SUMMARY — Cross-Seed Mean Operation Time (ns)")
    print(f"{'='*110}")
    print(f"{'Dataset':<20} | {'Variant':<28} | "
          f"{'Insert (ns)':<22} | {'Move (ns)':<22} | {'Delete (ns)':<22}")
    print('-' * 110)

    for dataset in datasets:
        ds_data = aggregated_df[aggregated_df['dataset'] == dataset]
        if ds_data.empty:
            continue
        for _, row in ds_data.iterrows():
            variant = row['variant'][:28]
            ins = (f"{row['insert_ns_mean']:8.0f} +/- "
                   f"{row['insert_ns_cross_seed_std']:6.0f}"
                   if not np.isnan(row.get('insert_ns_mean', float('nan')))
                   else "—")
            mov = (f"{row['move_ns_mean']:8.0f} +/- "
                   f"{row['move_ns_cross_seed_std']:6.0f}"
                   if not np.isnan(row.get('move_ns_mean', float('nan')))
                   else "—")
            dlt = (f"{row['delete_ns_mean']:8.0f} +/- "
                   f"{row['delete_ns_cross_seed_std']:6.0f}"
                   if not np.isnan(row.get('delete_ns_mean', float('nan')))
                   else "—")
            print(f"{dataset:<20} | {variant:<28} | "
                  f"{ins:<22} | {mov:<22} | {dlt:<22}")
    print('=' * 110)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run multi-seed ablation study across datasets")
    parser.add_argument(
        '--datasets', default='all',
        help='Comma-separated dataset names or "all" '
             f'(default: all 12 datasets)')
    parser.add_argument(
        '--seeds', default=None,
        help=f'Comma-separated seeds '
             f'(default: {",".join(str(s) for s in DEFAULT_SEEDS)})')
    args = parser.parse_args()

    # Parse seeds
    if args.seeds is None:
        seeds = DEFAULT_SEEDS
    else:
        try:
            seeds = [int(s.strip()) for s in args.seeds.split(',')]
        except ValueError:
            print("Error: --seeds must be comma-separated integers")
            sys.exit(1)
        if not seeds:
            print("Error: at least one seed required")
            sys.exit(1)

    # Parse datasets
    if args.datasets == 'all':
        datasets = ALL_DATASETS
    else:
        datasets = [d.strip() for d in args.datasets.split(',')]

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, ".."))
    results_dir = os.path.join(project_root, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 70)
    print("ABLATION STUDY — Multi-Seed Component Contribution Analysis")
    print("=" * 70)
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Seeds:    {seeds}")
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

    # Phase 1: regenerate datasets per seed.
    seed_roots = regenerate_datasets_for_seeds(seeds, project_root)

    try:
        # Phase 2: run ablation_bench per (dataset, seed) and accumulate.
        print(f"\n[Phase 2/2] Running ablation_bench per (dataset, seed)...")
        all_static = []
        all_dynamic = []

        for dataset in datasets:
            print(f"\n{'='*70}")
            print(f"DATASET: {dataset.upper()}")
            print(f"{'='*70}")

            for seed in seeds:
                print(f"  Seed {seed}... ", end='', flush=True)
                seed_root = seed_roots[seed]
                static_df, dynamic_df = run_ablation_for_seed(
                    dataset, seed_root, project_root, seed)

                if static_df is not None:
                    all_static.append(static_df)
                if dynamic_df is not None:
                    all_dynamic.append(dynamic_df)

                print("Done")

        # Aggregate and save.
        if all_static:
            per_seed_static = pd.concat(all_static, ignore_index=True)
            per_seed_path = f"{results_dir}/ablation_per_seed.csv"
            per_seed_static.to_csv(per_seed_path, index=False)
            print(f"\nRaw per-seed static results: {per_seed_path}")

            agg_static = aggregate_static_across_seeds(per_seed_static)
            summary_path = f"{results_dir}/ablation_summary.csv"
            agg_static.to_csv(summary_path, index=False)
            print(f"Aggregated static results:    {summary_path}")

            print_static_summary(agg_static, datasets)
        else:
            print("\nNo static results produced.")

        if all_dynamic:
            per_seed_dynamic = pd.concat(all_dynamic, ignore_index=True)
            dyn_per_seed_path = f"{results_dir}/ablation_dynamic_per_seed.csv"
            per_seed_dynamic.to_csv(dyn_per_seed_path, index=False)
            print(f"\nRaw per-seed dynamic results: {dyn_per_seed_path}")

            agg_dynamic = aggregate_dynamic_across_seeds(per_seed_dynamic)
            dyn_summary_path = f"{results_dir}/ablation_dynamic_summary.csv"
            agg_dynamic.to_csv(dyn_summary_path, index=False)
            print(f"Aggregated dynamic results:   {dyn_summary_path}")

            print_dynamic_summary(agg_dynamic, datasets)

    finally:
        # Always clean up temp dirs, even if we hit an error mid-loop.
        cleanup_seed_dirs(seed_roots)

    print("\n" + "=" * 70)
    print("ABLATION STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()