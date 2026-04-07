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
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <vector>

#include "../include/DelaunayClassifier.h"

// =============================================================================
// HELPERS
// =============================================================================

struct AblationResult {
  std::string variant;
  double accuracy;
  double avg_inference_us;
  double train_time_ms;
  std::string notes;
};

struct DynamicAblationResult {
  std::string variant;
  double avg_insert_ns;
  double avg_move_ns;
  double avg_delete_ns;
  std::string notes;
};

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

void print_dynamic_ablation_table(
    const std::vector<DynamicAblationResult> &results,
    const std::string &title) {
  std::cout << "\n" << std::string(105, '=') << std::endl;
  std::cout << title << std::endl;
  std::cout << std::string(105, '-') << std::endl;
  printf("%-40s | %-15s | %-15s | %-15s | %-15s\n", "Variant", "Insert (ns)",
         "Move (ns)", "Delete (ns)", "Notes");
  std::cout << std::string(105, '-') << std::endl;

  for (const auto &r : results) {
    printf("%-40s | %12.0f    | %12.0f    | %12.0f    | %s\n",
           r.variant.c_str(), r.avg_insert_ns, r.avg_move_ns, r.avg_delete_ns,
           r.notes.c_str());
  }
  std::cout << std::string(105, '=') << std::endl;
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
      int pred = clf.classify_nearest_vertex(std::get<0>(pt), std::get<1>(pt));
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

  const int NUM_OPS = std::min(100, (int)test_data.size());

  // Adaptive movement offset: 1% of smaller data dimension
  double range_x = range.x_max - range.x_min;
  double range_y = range.y_max - range.y_min;
  double move_offset = 0.01 * std::min((double)range_x, (double)range_y);

  // D1: Full pipeline with local bucket rebuild
  {
    std::cout << "[D1] Full Dynamic (with local bucket rebuild)..."
              << std::endl;
    DelaunayClassifier clf;
    clf.set_use_outlier_removal(true);
    clf.set_output_dir("results");
    clf.train(train_file, 3);

    // INSERT
    double total_insert = 0;
    std::vector<std::pair<float, float>> inserted;
    for (int i = 0; i < NUM_OPS; i++) {
      float x = std::get<0>(test_data[i]);
      float y = std::get<1>(test_data[i]);
      int label = std::get<2>(test_data[i]);

      auto s = std::chrono::high_resolution_clock::now();
      clf.insert_point(x, y, label);
      auto e = std::chrono::high_resolution_clock::now();
      total_insert +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
      inserted.push_back({x, y});
    }

    // MOVE (directly measured via move_point / Algorithm 3)
    double total_move = 0;
    std::vector<std::pair<float, float>> moved;
    for (int i = 0; i < NUM_OPS; i++) {
      float ox = inserted[i].first, oy = inserted[i].second;
      float nx = ox + (float)move_offset, ny = oy + (float)move_offset;

      auto s = std::chrono::high_resolution_clock::now();
      clf.move_point(ox, oy, nx, ny);
      auto e = std::chrono::high_resolution_clock::now();
      total_move +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
      moved.push_back({nx, ny});
    }

    // DELETE
    double total_delete = 0;
    for (int i = NUM_OPS - 1; i >= 0; i--) {
      float x = moved[i].first, y = moved[i].second;

      auto s = std::chrono::high_resolution_clock::now();
      clf.remove_point(x, y);
      auto e = std::chrono::high_resolution_clock::now();
      total_delete +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(e - s).count();
    }

    dynamic_ablation.push_back({"Full Dynamic (2D Buckets)",
                                total_insert / NUM_OPS, total_move / NUM_OPS,
                                total_delete / NUM_OPS,
                                "DT update + local bucket rebuild"});
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

  std::string dyn_csv = "results/ablation_dynamic_" + dataset_name + ".csv";
  std::ofstream dcsv(dyn_csv);
  dcsv << "variant,avg_insert_ns,avg_move_ns,avg_delete_ns,notes\n";
  for (const auto &r : dynamic_ablation) {
    dcsv << r.variant << "," << r.avg_insert_ns << "," << r.avg_move_ns << ","
         << r.avg_delete_ns << "," << r.notes << "\n";
  }
  dcsv.close();

  std::cout << "\nResults saved to: " << csv_file << std::endl;
  std::cout << "Dynamic results saved to: " << dyn_csv << std::endl;

  return 0;
}