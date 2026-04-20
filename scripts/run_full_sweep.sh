#!/bin/bash
# =============================================================================
# Full Benchmark Sweep — Delaunay Triangulation Classifier
# =============================================================================
#
# Runs the complete reproduction pipeline end-to-end. Expected runtime:
# 100-160 minutes (2-2.5 hours) on MacBook Pro M3.
#
# Usage:
#   ./scripts/run_full_sweep.sh                  # Normal run
#   ./scripts/run_full_sweep.sh --skip-build     # Skip C++ rebuild (faster)
#   ./scripts/run_full_sweep.sh --skip-tests     # Skip unit tests (faster)
#
# Output:
#   results/full_sweep_<timestamp>.log  — Complete log of every step
#   results/full_sweep_<timestamp>.txt  — Per-step timing and status
#
# Design choices:
#   - Keeps going past errors (set +e) so you don't lose all results if
#     one step fails midway. Final summary shows which steps succeeded.
#   - Dataset generation runs FIRST so real-world datasets are cached
#     before multi-seed scripts try to regenerate them 5 times each.
#   - Single-seed C++ ablation_bench loop is INTENTIONALLY SKIPPED — it's
#     superseded by the multi-seed scripts/ablation_study.py .
#     Running both would waste 5-10 min and cause CSV overwrites.
#   - Figures run LAST, after all data CSVs exist.
#   - caffeinate prevents macOS from sleeping during the long run.
# =============================================================================

# --- Parse flags ---
SKIP_BUILD=0
SKIP_TESTS=0
for arg in "$@"; do
    case "$arg" in
        --skip-build) SKIP_BUILD=1 ;;
        --skip-tests) SKIP_TESTS=1 ;;
        --help|-h)
            sed -n '/^# Usage:/,/^# ===/p' "$0" | head -20
            exit 0
            ;;
        *) echo "Unknown flag: $arg"; exit 1 ;;
    esac
done

# --- Setup ---
cd "$(dirname "$0")/.." || exit 1
ROOT_DIR="$(pwd)"
mkdir -p results

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="results/full_sweep_${TIMESTAMP}.log"
SUMMARY_FILE="results/full_sweep_${TIMESTAMP}.txt"

# Arrays to track per-step status
STEP_NAMES=()
STEP_STATUS=()
STEP_SECONDS=()

# --- Helper: run a step, log it, track status, keep going on error ---
run_step() {
    local step_name="$1"
    shift
    local start_time=$(date +%s)

    echo "" | tee -a "$LOG_FILE"
    echo "=========================================================" | tee -a "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] START: $step_name" | tee -a "$LOG_FILE"
    echo "Command: $*" | tee -a "$LOG_FILE"
    echo "=========================================================" | tee -a "$LOG_FILE"

    # Run with stdout AND stderr appended to log; disable set -e
    set +e
    "$@" >> "$LOG_FILE" 2>&1
    local status=$?
    set +e

    local end_time=$(date +%s)
    local elapsed=$((end_time - start_time))
    local elapsed_str
    elapsed_str=$(printf '%dm %02ds' $((elapsed/60)) $((elapsed%60)))

    STEP_NAMES+=("$step_name")
    STEP_SECONDS+=("$elapsed")
    if [ $status -eq 0 ]; then
        STEP_STATUS+=("OK")
        echo "[$(date '+%H:%M:%S')] OK: $step_name (${elapsed_str})" | tee -a "$LOG_FILE"
    else
        STEP_STATUS+=("FAIL(exit=$status)")
        echo "[$(date '+%H:%M:%S')] FAIL: $step_name (exit=$status, ${elapsed_str})" | tee -a "$LOG_FILE"
        echo "  See $LOG_FILE for details" | tee -a "$LOG_FILE"
    fi
}

# --- Banner ---
SWEEP_START=$(date +%s)
{
    echo "============================================================="
    echo "DELAUNAY TRIANGULATION CLASSIFIER — FULL BENCHMARK SWEEP"
    echo "============================================================="
    echo "Start time:   $(date)"
    echo "Log file:     $LOG_FILE"
    echo "Summary file: $SUMMARY_FILE"
    echo "Working dir:  $ROOT_DIR"
    echo "Skip build:   $SKIP_BUILD"
    echo "Skip tests:   $SKIP_TESTS"
    echo ""
    echo "Estimated runtime: 100-160 minutes."
    echo "You can safely background this script and check results/ later."
    echo "============================================================="
} | tee -a "$LOG_FILE"

# --- Verify venv is active ---
if [ -z "$VIRTUAL_ENV" ]; then
    echo "WARNING: no Python venv detected. Continuing anyway."
    echo "If you hit ImportErrors, activate the venv first: source venv/bin/activate"
fi

# =============================================================================
# STEP 0: Build C++ binaries (skippable)
# =============================================================================
if [ $SKIP_BUILD -eq 0 ]; then
    run_step "0. C++ build" bash -c '
        mkdir -p build
        cd build
        # Detect macOS Homebrew CGAL location; harmless on Linux
        if [ -d /opt/homebrew/opt/cgal ]; then
            cmake .. -DCMAKE_BUILD_TYPE=Release \
                     -DCGAL_DIR=/opt/homebrew/opt/cgal/lib/cmake/CGAL \
                     -DCMAKE_PREFIX_PATH=/opt/homebrew
        else
            cmake .. -DCMAKE_BUILD_TYPE=Release
        fi
        make -j4
    '

    # Sanity check: the binaries we need
    for bin in main benchmark ablation_bench; do
        if [ ! -x "build/$bin" ]; then
            echo "ERROR: build/$bin not found after build. Aborting." | tee -a "$LOG_FILE"
            echo "ERROR: missing build/$bin" >> "$SUMMARY_FILE"
            exit 1
        fi
    done
else
    echo "[$(date '+%H:%M:%S')] SKIPPED: C++ build (--skip-build)" | tee -a "$LOG_FILE"
fi

# =============================================================================
# STEP 1: Generate all 12 datasets (seeds cached for downstream multi-seed)
# =============================================================================
run_step "1. Dataset generation (all 12)" \
    python scripts/generate_datasets.py

# =============================================================================
# STEP 2: C++ benchmarks per dataset (static + dynamic)
# =============================================================================
# Note: per-dataset C++ benchmark loop. Produces cpp_benchmark_<dataset>.csv
# which feeds the per-dataset speedup table in README. NOT the multi-seed
# headline numbers — those come from steps 4-5.
run_step "2. C++ benchmarks (static + dynamic, all 12 datasets)" \
    bash -c '
        for ds in moons circles spiral gaussian_quantiles cassini \
                  checkerboard blobs earthquake sfcrime wine cancer bloodmnist; do
            train="data/train/${ds}_train.csv"
            test="data/test/${ds}_test_y.csv"
            if [ -f "$train" ] && [ -f "$test" ]; then
                echo ""
                echo "--- cpp_benchmark: $ds ---"
                ./build/benchmark "$train" "$test" "$ds" || {
                    echo "WARN: benchmark failed on $ds, continuing"
                }
            else
                echo "SKIP: $ds (missing data files)"
            fi
        done
    '

# =============================================================================
# STEP 3: Scalability test (n=100 to n=1M)
# =============================================================================
# Feeds summary_scalability.png. Dominated by n=1M (~4-5 min alone).
run_step "3. Scalability test (n=100 to n=1M)" \
    python scripts/scalability_test.py

# =============================================================================
# STEP 4: Multi-seed CV benchmark (benchmark_cv.py)
# =============================================================================
# Multi-seed [42, 123, 456, 789, 1000] × 12 datasets. Produces:
#   - results/cv_summary.csv (paper-headline accuracy table)
#   - results/significance_tests.csv (Bonferroni-corrected paired tests)
#   - results/per_class_metrics.csv
#   - results/confusion_matrix_{dataset}_{algorithm}.csv 
run_step "4. Multi-seed CV benchmark (12 datasets x 5 seeds)" \
    python scripts/benchmark_cv.py

# =============================================================================
# STEP 5: Multi-seed ablation study (ablation_study.py)
# =============================================================================
# Multi-seed [42, 123, 456, 789, 1000] × 12 datasets. Produces:
#   - results/ablation_summary.csv (paper-headline ablation table)
#   - results/ablation_per_seed.csv
#   - results/ablation_dynamic_summary.csv (multi-seed dynamic ops)
#   - results/ablation_dynamic_per_seed.csv
# Intentionally supersedes the old single-seed ./build/ablation_bench loop.
run_step "5. Multi-seed ablation study (12 datasets x 5 seeds)" \
    python scripts/ablation_study.py

# =============================================================================
# STEP 6: Unit tests (including slow C++ integration)
# =============================================================================
if [ $SKIP_TESTS -eq 0 ]; then
    run_step "6. Unit tests (RUN_SLOW_TESTS=1)" \
        env RUN_SLOW_TESTS=1 python tests/test_classifier.py
else
    echo "[$(date '+%H:%M:%S')] SKIPPED: Unit tests (--skip-tests)" | tee -a "$LOG_FILE"
fi

# =============================================================================
# STEP 7: Generate all figures
# =============================================================================
# Must run LAST — consumes outputs from steps 3-5.
# --regenerate-bucket-stats to pick up any schema changes from step 2.
run_step "7. Figure generation (all figures + bucket stats refresh)" \
    python scripts/generate_figures.py --regenerate-bucket-stats

# =============================================================================
# Final summary
# =============================================================================
SWEEP_END=$(date +%s)
TOTAL_ELAPSED=$((SWEEP_END - SWEEP_START))
TOTAL_STR=$(printf '%dh %dm %ds' $((TOTAL_ELAPSED/3600)) $(((TOTAL_ELAPSED%3600)/60)) $((TOTAL_ELAPSED%60)))

# Count successes and failures BEFORE the tee subshell so the counts are
# available for the exit-code decision at the end of the script.
N_OK=0
N_FAIL=0
for status in "${STEP_STATUS[@]}"; do
    if [ "$status" = "OK" ]; then
        N_OK=$((N_OK + 1))
    else
        N_FAIL=$((N_FAIL + 1))
    fi
done

{
    echo ""
    echo "============================================================="
    echo "FULL SWEEP COMPLETE"
    echo "============================================================="
    # Portable timestamp formatting — works on BSD (macOS) and GNU (Linux).
    echo "Start time:  $(date -d "@$SWEEP_START" 2>/dev/null || \
                       date -r "$SWEEP_START" 2>/dev/null || \
                       echo "unknown (sweep start=$SWEEP_START)")"
    echo "End time:    $(date -d "@$SWEEP_END" 2>/dev/null || \
                       date -r "$SWEEP_END" 2>/dev/null || \
                       echo "unknown (sweep end=$SWEEP_END)")"
    echo "Total time:  $TOTAL_STR"
    echo ""
    echo "Per-step summary:"
    echo "-------------------------------------------------------------"
    printf "%-55s %-20s %s\n" "STEP" "STATUS" "TIME"
    echo "-------------------------------------------------------------"

    for i in "${!STEP_NAMES[@]}"; do
        local_name="${STEP_NAMES[$i]}"
        local_status="${STEP_STATUS[$i]}"
        local_secs="${STEP_SECONDS[$i]}"
        local_time=$(printf '%dm %02ds' $((local_secs/60)) $((local_secs%60)))
        printf "%-55s %-20s %s\n" "$local_name" "$local_status" "$local_time"
    done

    echo "-------------------------------------------------------------"
    echo "$N_OK steps succeeded, $N_FAIL failed."
    echo ""
    echo "Full log:    $LOG_FILE"
    echo ""
    echo "Key outputs to inspect:"
    echo "  - results/cv_summary.csv            (multi-seed accuracy)"
    echo "  - results/ablation_summary.csv      (multi-seed ablation)"
    echo "  - results/ablation_dynamic_summary.csv (multi-seed dynamic ops)"
    echo "  - results/scalability_train.csv     (training timing)"
    echo "  - results/scalability_inference.csv (O(1) validation)"
    echo "  - figures/summary_*.png             (figures)"
    echo "  - figures/confusion_matrices/       (per-dataset CM panels)"
    echo ""
    if [ "$N_FAIL" -eq 0 ]; then
        echo "ALL STEPS PASSED. Safe to commit results/ and figures/."
    else
        echo "SOME STEPS FAILED. Check $LOG_FILE for details before committing."
    fi
    echo "============================================================="
} | tee "$SUMMARY_FILE"

# Exit code for CI / callers
if [ "$N_FAIL" -eq 0 ]; then
    exit 0
else
    exit 1
fi