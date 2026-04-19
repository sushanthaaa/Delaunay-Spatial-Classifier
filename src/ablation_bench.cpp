/**
 * @file ablation_bench.cpp
 * @brief Ablation Study: Measures individual contribution of each component.
 *
 * Static ablation components:
 *   A1: Full Pipeline — classify() via 2D Buckets (baseline)
 *   A2: Without 2D Buckets Grid — classify_no_grid() via DT locate walk
 *   A3: Without Outlier Removal — classify() with Phase 1 disabled
 *   A4: Nearest Vertex Only — classify_nearest_vertex() (1-NN baseline)
 *   A5: Outlier Multiplier Sensitivity — m = {1.5, 2.0, 3.0, 5.0, 10.0}
 *
 * Dynamic ablation:
 *   D1: Full Dynamic — insert/move/delete with local bucket rebuild
 *
 * Fixes applied (Week 2 of the master action list):
 *               test_data.size(). Matches the bump applied to benchmark.cpp
 *               so the dynamic ablation's measurement count is consistent
 *               with the main benchmark.
 *               vectors and reports mean ± std (Bessel-corrected) for
 *               insert/move/delete. The CSV output and print function were
 *               extended to carry the std columns alongside the means.
 *               std::runtime_error so that file-open failures from the
 *               updated CSV loaders surface cleanly with a non-zero exit
 *               code instead of propagating uncaught.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <vector>

#include "../include/DelaunayClassifier.h"

// =============================================================================
// BENCHMARK CONFIGURATION CONSTANTS
// =============================================================================
//
// Named constants for measurement parameters. These control the harness, not
// the algorithm — reviewers should not mistake them for tuned hyperparameters.

namespace {

/// Target number of dynamic operations (insert/move/delete) to measure.
/// Capped at test_data.size() so small datasets still produce a valid run.
constexpr int TARGET_DYNAMIC_OPS = 1000;

/// Movement offset used in the dynamic ablation, expressed as a fraction of
/// the smaller data dimension. Benchmark parameter, not a model parameter.
/// Mirrors the same constant in benchmark.cpp and DelaunayClassifier.cpp.
constexpr double DYNAMIC_MOVE_OFFSET_FRACTION = 0.01;

} // anonymous namespace

// =============================================================================
// STATISTICAL HELPERS
// =============================================================================
//
// per-op timings. These helpers mirror the ones in benchmark.cpp so the two
// files report timing statistics in the same way (Bessel-corrected sample
// std, divides by n-1, returns 0 for vectors of size < 2).

/// Compute mean of a vector of doubles.
static double compute_mean(const std::vector<double> &values) {
  if (values.empty())
    return 0.0;
  double sum = 0.0;
  for (double v : values)
    sum += v;
  return sum / static_cast<double>(values.size());
}

/// Compute sample standard deviation (Bessel-corrected, divides by n-1).
/// Returns 0.0 for vectors of size < 2.
static double compute_std(const std::vector<double> &values, double mean) {
  if (values.size() < 2)
    return 0.0;
  double sum_sq = 0.0;
  for (double v : values) {
    double d = v - mean;
    sum_sq += d * d;
  }
  return std::sqrt(sum_sq / static_cast<double>(values.size() - 1));
}

// =============================================================================
// RESULT STRUCTURES
// =============================================================================

struct AblationResult {
  std::string variant;
  double accuracy;
  double avg_inference_us;
  double train_time_ms;
  std::string notes;
};

/**
 * @brief Dynamic ablation result with mean and standard deviation per phase.
 *
 * Now carries mean and std for insert/move/delete plus an explicit num_ops
 * count so external scripts can weight statistics by sample size if needed.
 */
struct DynamicAblationResult {
  std::string variant;
  double insert_ns_mean;
  double insert_ns_std;
  double move_ns_mean;
  double move_ns_std;
  double delete_ns_mean;
  double delete_ns_std;
  int num_ops;
  std::string notes;
};

// =============================================================================
// HELPERS
// =============================================================================

std::vector<std::tuple<float, float, int>>
load_csv(const std::string &filepath) {
  std::vector<std::tuple<float, float, int>> points;
  std::ifstream file(filepath);
  if (!file.is_open()) {
    std::cerr << "Error: Cannot open " << filepath << std::endl;
    return points;
  }
  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string val;
    float x, y;
    int label;
    std::getline(ss, val, ',');
    x = std::stof(val);
    std::getline(ss, val, ',');
    y = std::stof(val);
    std::getline(ss, val, ',');
    label = std::stoi(val);
    points.push_back({x, y, label});
  }
  return points;
}

struct DataRange {
  float x_min, x_max, y_min, y_max;
};

DataRange
compute_range(const std::vector<std::tuple<float, float, int>> &data) {
  DataRange r = {1e18f, -1e18f, 1e18f, -1e18f};
  for (const auto &pt : data) {
    float x = std::get<0>(pt), y = std::get<1>(pt);
    if (x < r.x_min)
      r.x_min = x;
    if (x > r.x_max)
      r.x_max = x;
    if (y < r.y_min)
      r.y_min = y;
    if (y > r.y_max)
      r.y_max = y;
  }
  return r;
}

// =============================================================================
// PRINT FUNCTIONS
// =============================================================================

void print_ablation_table(const std::vector<AblationResult> &results,
                          const std::string &title) {
  std::cout << "\n" << std::string(105, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(105, '-') << std::endl;
  printf("%-40s | %-10s | %-15s | %-12s | %-20s\n", "Variant", "Accuracy",
         "Inference (us)", "Train (ms)", "Notes");
  std::cout << std::string(105, '-') << std::endl;

  for (const auto &r : results) {
    printf("%-40s | %6.1f%%   | %12.4f    | %9.2f   | %s\n", r.variant.c_str(),
           r.accuracy * 100, r.avg_inference_us, r.train_time_ms,
           r.notes.c_str());
  }
  std::cout << std::string(105, '=') << std::endl;
}

/**
 *
 * Previously showed only means. Now shows "mean ± std" for each operation
 * type so reviewers can assess measurement stability. The num_ops column
 * shows how many operations contributed to each statistic.
 */
void print_dynamic_ablation_table(
    const std::vector<DynamicAblationResult> &results,
    const std::string &title) {
  std::cout << "\n" << std::string(140, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(140, '-') << std::endl;
  printf("%-40s | %-22s | %-22s | %-22s | %-6s | %s\n", "Variant",
         "Insert (ns) mean±std", "Move (ns) mean±std", "Delete (ns) mean±std",
         "N ops", "Notes");
  std::cout << std::string(140, '-') << std::endl;

  for (const auto &r : results) {
    printf("%-40s | %10.0f ± %-9.0f | %10.0f ± %-9.0f | %10.0f ± %-9.0f | %6d "
           "| %s\n",
           r.variant.c_str(), r.insert_ns_mean, r.insert_ns_std, r.move_ns_mean,
           r.move_ns_std, r.delete_ns_mean, r.delete_ns_std, r.num_ops,
           r.notes.c_str());
  }
  std::cout << std::string(140, '=') << std::endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: ./ablation_bench <train_csv> <test_csv> <dataset_name>"
              << std::endl;
    return 1;
  }

  std::string train_file = argv[1];
  std::string test_file = argv[2];
  std::string dataset_name = argv[3];

  // try/catch so that std::runtime_error from the updated CSV loaders
  // surfaces with a clean error message and non-zero exit code.
  try {
    auto train_data = load_csv(train_file);
    auto test_data = load_csv(test_file);

    if (train_data.empty() || test_data.empty()) {
      std::cerr << "Error loading data!" << std::endl;
      return 1;
    }

    DataRange range = compute_range(train_data);

    std::cout << "=== ABLATION STUDY: " << dataset_name << " ===" << std::endl;
    std::cout << "Training: " << train_data.size()
              << " points, Testing: " << test_data.size() << " points"
              << std::endl;
    std::cout << "Data range: x=[" << range.x_min << "," << range.x_max
              << "], y=[" << range.y_min << "," << range.y_max << "]"
              << std::endl;

    std::vector<AblationResult> static_ablation;
    std::vector<DynamicAblationResult> dynamic_ablation;

    // ============================================================
    // A1: FULL PIPELINE (2D Buckets + Outlier + Decision Boundary)
    // ============================================================
    {
      std::cout << "\n[A1] Full Pipeline..." << std::endl;
      DelaunayClassifier clf;
      clf.set_use_outlier_removal(true);
      clf.set_output_dir("results");

      auto train_start = std::chrono::high_resolution_clock::now();
      clf.train(train_file, 3);
      auto train_end = std::chrono::high_resolution_clock::now();
      double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            train_end - train_start)
                            .count() /
                        1000.0;

      int correct = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (const auto &pt : test_data) {
        int pred = clf.classify(std::get<0>(pt), std::get<1>(pt));
        if (pred == std::get<2>(pt))
          correct++;
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_us =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double avg_us = total_us / test_data.size();
      double accuracy = (double)correct / test_data.size();

      static_ablation.push_back({"Full Pipeline (Ours)", accuracy, avg_us,
                                 train_ms, "2D Buckets + Outlier + HalfPlane"});
    }

    // ============================================================
    // A2: Without 2D Buckets Grid (raw DT locate walk)
    // ============================================================
    {
      std::cout << "[A2] Without 2D Buckets Grid..." << std::endl;
      DelaunayClassifier clf;
      clf.set_use_outlier_removal(true);
      clf.set_output_dir("results");
      clf.train(train_file, 3);

      int correct = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (const auto &pt : test_data) {
        int pred = clf.classify_no_grid(std::get<0>(pt), std::get<1>(pt));
        if (pred == std::get<2>(pt))
          correct++;
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_us =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double avg_us = total_us / test_data.size();
      double accuracy = (double)correct / test_data.size();

      static_ablation.push_back({"Without 2D Buckets Grid", accuracy, avg_us, 0,
                                 "DT locate walk (no O(1) grid)"});
    }

    // ============================================================
    // A3: Without Outlier Removal
    // ============================================================
    {
      std::cout << "[A3] Without Outlier Removal..." << std::endl;
      DelaunayClassifier clf;
      clf.set_use_outlier_removal(false);
      clf.set_output_dir("results");

      auto train_start = std::chrono::high_resolution_clock::now();
      clf.train(train_file, 3);
      auto train_end = std::chrono::high_resolution_clock::now();
      double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                            train_end - train_start)
                            .count() /
                        1000.0;

      int correct = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (const auto &pt : test_data) {
        int pred = clf.classify(std::get<0>(pt), std::get<1>(pt));
        if (pred == std::get<2>(pt))
          correct++;
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_us =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double avg_us = total_us / test_data.size();
      double accuracy = (double)correct / test_data.size();

      static_ablation.push_back({"Without Outlier Removal", accuracy, avg_us,
                                 train_ms, "No Phase 1 cleanup"});
    }

    // ============================================================
    // A4: Nearest-Vertex Only (1-NN, no decision boundary)
    // ============================================================
    {
      std::cout << "[A4] Nearest Vertex Only (1-NN)..." << std::endl;
      DelaunayClassifier clf;
      clf.set_use_outlier_removal(true);
      clf.set_output_dir("results");
      clf.train(train_file, 3);

      int correct = 0;
      auto start = std::chrono::high_resolution_clock::now();
      for (const auto &pt : test_data) {
        int pred =
            clf.classify_nearest_vertex(std::get<0>(pt), std::get<1>(pt));
        if (pred == std::get<2>(pt))
          correct++;
      }
      auto end = std::chrono::high_resolution_clock::now();

      double total_us =
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count() /
          1000.0;
      double avg_us = total_us / test_data.size();
      double accuracy = (double)correct / test_data.size();

      static_ablation.push_back({"Nearest Vertex Only (1-NN)", accuracy, avg_us,
                                 0, "No half-plane boundary"});
    }

    // ============================================================
    // A5: Varying outlier multiplier sensitivity
    // ============================================================
    {
      std::cout << "[A5] Outlier Multiplier Sensitivity..." << std::endl;
      double multipliers[] = {1.5, 2.0, 3.0, 5.0, 10.0};

      for (double m : multipliers) {
        DelaunayClassifier clf;
        clf.set_use_outlier_removal(true);
        clf.set_connectivity_multiplier(m);
        clf.set_output_dir("results");
        clf.train(train_file, 3);

        int correct = 0;
        for (const auto &pt : test_data) {
          int pred = clf.classify(std::get<0>(pt), std::get<1>(pt));
          if (pred == std::get<2>(pt))
            correct++;
        }
        double accuracy = (double)correct / test_data.size();

        std::string label =
            "Outlier multiplier=" + std::to_string(m).substr(0, 4);
        std::string note = std::to_string(clf.num_vertices()) + " vertices";
        static_ablation.push_back({label, accuracy, 0, 0, note});
      }
    }

    print_ablation_table(static_ablation, "STATIC ABLATION: " + dataset_name);

    // ============================================================
    // DYNAMIC ABLATION
    // ============================================================
    std::cout << "\n--- DYNAMIC ABLATION ---" << std::endl;

    // Capped at test_data.size() so small datasets (e.g., wine with 36 test
    // points) still produce a valid run.
    const int NUM_OPS = std::min(TARGET_DYNAMIC_OPS, (int)test_data.size());
    std::cout << "Measuring " << NUM_OPS << " operations per phase"
              << std::endl;

    // Adaptive movement offset (named constant; mirrors benchmark.cpp)
    double range_x = range.x_max - range.x_min;
    double range_y = range.y_max - range.y_min;
    double move_offset = DYNAMIC_MOVE_OFFSET_FRACTION *
                         std::min((double)range_x, (double)range_y);

    // --------------------------------------------------------
    // D1: Full pipeline with local bucket rebuild
    // --------------------------------------------------------
    {
      std::cout << "[D1] Full Dynamic (with local bucket rebuild)..."
                << std::endl;
      DelaunayClassifier clf;
      clf.set_use_outlier_removal(true);
      clf.set_output_dir("results");
      clf.train(train_file, 3);

      // mean and std (instead of just summing into a single accumulator).
      std::vector<double> insert_times_ns;
      std::vector<double> move_times_ns;
      std::vector<double> delete_times_ns;
      insert_times_ns.reserve(NUM_OPS);
      move_times_ns.reserve(NUM_OPS);
      delete_times_ns.reserve(NUM_OPS);

      // --- INSERT phase ---
      std::vector<std::pair<float, float>> inserted;
      inserted.reserve(NUM_OPS);
      for (int i = 0; i < NUM_OPS; i++) {
        float x = std::get<0>(test_data[i]);
        float y = std::get<1>(test_data[i]);
        int label = std::get<2>(test_data[i]);

        auto s = std::chrono::high_resolution_clock::now();
        clf.insert_point(x, y, label);
        auto e = std::chrono::high_resolution_clock::now();
        insert_times_ns.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(e - s)
                .count()));
        inserted.push_back({x, y});
      }

      // --- MOVE phase (directly measured via move_point / Algorithm 3) ---
      std::vector<std::pair<float, float>> moved;
      moved.reserve(NUM_OPS);
      for (int i = 0; i < NUM_OPS; i++) {
        float ox = inserted[i].first, oy = inserted[i].second;
        float nx = ox + (float)move_offset, ny = oy + (float)move_offset;

        auto s = std::chrono::high_resolution_clock::now();
        clf.move_point(ox, oy, nx, ny);
        auto e = std::chrono::high_resolution_clock::now();
        move_times_ns.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(e - s)
                .count()));
        moved.push_back({nx, ny});
      }

      // --- DELETE phase ---
      for (int i = NUM_OPS - 1; i >= 0; i--) {
        float x = moved[i].first, y = moved[i].second;

        auto s = std::chrono::high_resolution_clock::now();
        clf.remove_point(x, y);
        auto e = std::chrono::high_resolution_clock::now();
        delete_times_ns.push_back(static_cast<double>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(e - s)
                .count()));
      }

      double insert_mean = compute_mean(insert_times_ns);
      double move_mean = compute_mean(move_times_ns);
      double delete_mean = compute_mean(delete_times_ns);

      DynamicAblationResult res;
      res.variant = "Full Dynamic (2D Buckets)";
      res.insert_ns_mean = insert_mean;
      res.insert_ns_std = compute_std(insert_times_ns, insert_mean);
      res.move_ns_mean = move_mean;
      res.move_ns_std = compute_std(move_times_ns, move_mean);
      res.delete_ns_mean = delete_mean;
      res.delete_ns_std = compute_std(delete_times_ns, delete_mean);
      res.num_ops = NUM_OPS;
      res.notes = "DT update + local bucket rebuild";
      dynamic_ablation.push_back(res);
    }

    print_dynamic_ablation_table(dynamic_ablation,
                                 "DYNAMIC ABLATION: " + dataset_name);

    // ============================================================
    // SAVE TO CSV
    // ============================================================
    std::string csv_file = "results/ablation_" + dataset_name + ".csv";
    std::ofstream csv(csv_file);
    csv << "variant,accuracy,avg_inference_us,train_ms,notes\n";
    for (const auto &r : static_ablation) {
      csv << r.variant << "," << r.accuracy << "," << r.avg_inference_us << ","
          << r.train_time_ms << "," << r.notes << "\n";
    }
    csv.close();

    // the num_ops count, mirroring the format used by benchmark.cpp's
    // cpp_benchmark_dynamic_<dataset>.csv output.
    std::string dyn_csv = "results/ablation_dynamic_" + dataset_name + ".csv";
    std::ofstream dcsv(dyn_csv);
    dcsv << "variant,insert_ns_mean,insert_ns_std,move_ns_mean,move_ns_std,"
            "delete_ns_mean,delete_ns_std,num_ops,notes\n";
    for (const auto &r : dynamic_ablation) {
      dcsv << r.variant << "," << r.insert_ns_mean << "," << r.insert_ns_std
           << "," << r.move_ns_mean << "," << r.move_ns_std << ","
           << r.delete_ns_mean << "," << r.delete_ns_std << "," << r.num_ops
           << "," << r.notes << "\n";
    }
    dcsv.close();

    std::cout << "\nResults saved to: " << csv_file << std::endl;
    std::cout << "Dynamic results saved to: " << dyn_csv << std::endl;

    return 0;

  } catch (const std::runtime_error &e) {
    std::cerr << "Runtime error during ablation benchmark: " << e.what()
              << std::endl;
    return 1;
  } catch (const std::exception &e) {
    std::cerr << "Unexpected error during ablation benchmark: " << e.what()
              << std::endl;
    return 1;
  }
}