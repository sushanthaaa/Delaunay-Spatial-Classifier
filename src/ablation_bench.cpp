/**
 * @file ablation_bench.cpp
 * @brief Ablation Study Benchmark for IEEE Publication
 *
 * This benchmark quantifies the contribution of each component in our
 * classifier:
 *
 * EXPERIMENTAL CONDITIONS:
 *
 * 1. Full System (SRR + Decision Boundary)
 *    - Uses SRR grid for O(1) point location
 *    - Uses triangle-based classification with boundary interpolation
 *    - This is our complete proposed method
 *
 * 2. No SRR Grid (O(sqrt(n)) locate)
 *    - Disables SRR spatial index
 *    - Uses raw CGAL locate() which has O(sqrt(n)) expected complexity
 *    - Same accuracy, slower inference
 *    → Measures: SRR contribution to SPEED
 *
 * 3. Nearest Vertex Only (1-NN)
 *    - Ignores triangle structure entirely
 *    - Simply returns label of nearest vertex (like 1-NN)
 *    → Measures: Decision boundary contribution to ACCURACY
 *
 * EXPECTED RESULTS:
 * - SRR grid provides 1.5-2x speedup over raw CGAL locate
 * - Decision boundary provides 0-5% accuracy improvement over 1-NN
 *   (dataset dependent; more improvement on complex boundaries)
 *
 * Usage: ./ablation_bench <train.csv> <test.csv>
 *
 * @see DelaunayClassifier.h for ablation method signatures
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../include/DelaunayClassifier.h"

/**
 * @struct AblationResult
 * @brief Stores results from each ablation condition.
 */
struct AblationResult {
  std::string condition; ///< Name of the experimental condition
  double accuracy;       ///< Classification accuracy (0-1)
  double avg_time_us;    ///< Average inference time in microseconds
};

/**
 * @brief Load labeled test data from CSV file.
 *
 * Format: x,y,label (headerless)
 * Used to get ground truth labels for accuracy calculation.
 */
std::vector<std::tuple<float, float, int>>
load_labeled_csv(const std::string &filepath) {
  std::vector<std::tuple<float, float, int>> points;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    std::cerr << "Error: Cannot open " << filepath << std::endl;
    return points;
  }

  std::string line;
  while (std::getline(file, line)) {
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

int main(int argc, char *argv[]) {
  // --- Argument parsing ---
  if (argc < 3) {
    std::cout << "Usage: ./ablation_bench <train.csv> <test.csv>" << std::endl;
    std::cout << "\nThis benchmark tests the contribution of SRR grid and "
                 "decision boundary."
              << std::endl;
    return 1;
  }

  std::string train_file = argv[1];
  std::string test_file = argv[2];

  auto test_data = load_labeled_csv(test_file);
  if (test_data.empty()) {
    std::cerr << "Error loading test data!" << std::endl;
    return 1;
  }

  std::vector<AblationResult> results;

  // --- Train the classifier (shared across all conditions) ---
  // We train once and test with different classification methods
  DelaunayClassifier classifier;
  classifier.train(train_file, 3); // k=3 for outlier removal

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "ABLATION STUDY BENCHMARK" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // ==========================================================================
  // CONDITION 1: Full System (SRR + Decision Boundary)
  // ==========================================================================
  // This is our complete proposed method with all optimizations enabled.
  // Expected: Best speed (O(1)), competitive accuracy
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto &pt : test_data) {
      // Uses SRR hint + triangle classification
      int pred = classifier.classify_single(std::get<0>(pt), std::get<1>(pt));
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

    results.push_back({"Full System (SRR + Boundary)", accuracy, avg_us});
  }

  // ==========================================================================
  // CONDITION 2: No SRR Grid (raw CGAL locate, O(sqrt(n)))
  // ==========================================================================
  // Tests speed without our spatial indexing optimization.
  // Same accuracy expected, but slower due to O(sqrt(n)) point location.
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto &pt : test_data) {
      // Bypasses SRR grid, uses CGAL's default O(sqrt(n)) locate
      int pred =
          classifier.classify_single_no_srr(std::get<0>(pt), std::get<1>(pt));
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

    results.push_back({"No SRR Grid (O(sqrt(n)))", accuracy, avg_us});
  }

  // ==========================================================================
  // CONDITION 3: Nearest Vertex Only (1-NN baseline)
  // ==========================================================================
  // Tests accuracy without triangle-based decision boundaries.
  // Simply returns label of nearest Delaunay vertex (like 1-NN).
  // Faster than full system but potentially lower accuracy on complex
  // boundaries.
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();

    for (const auto &pt : test_data) {
      // Just find nearest vertex, ignore triangle structure
      int pred =
          classifier.classify_nearest_vertex(std::get<0>(pt), std::get<1>(pt));
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

    results.push_back({"Nearest Vertex Only", accuracy, avg_us});
  }

  // ==========================================================================
  // RESULTS OUTPUT
  // ==========================================================================
  std::cout << std::string(80, '-') << std::endl;
  printf("%-35s | %-12s | %-15s | %-10s\n", "Condition", "Accuracy",
         "Time (µs)", "Speedup");
  std::cout << std::string(80, '-') << std::endl;

  double baseline_time = results[0].avg_time_us;
  for (const auto &r : results) {
    // Speedup > 1 means this condition is SLOWER than baseline
    double speedup = r.avg_time_us / baseline_time;
    printf("%-35s | %6.2f%%     | %12.4f    | %.2fx\n", r.condition.c_str(),
           r.accuracy * 100, r.avg_time_us, speedup);
  }

  std::cout << std::string(80, '=') << std::endl;

  // ==========================================================================
  // ABLATION INSIGHTS (Automated Analysis)
  // ==========================================================================
  std::cout << "\nABLATION INSIGHTS:" << std::endl;
  std::cout << std::string(40, '-') << std::endl;

  // Calculate SRR grid contribution to speed
  // Compares Full System vs No SRR (same accuracy, different speed)
  double srr_speedup = results[1].avg_time_us / results[0].avg_time_us;
  printf("SRR Grid Speed Contribution: %.1fx faster with SRR\n", srr_speedup);

  // Calculate decision boundary contribution to accuracy
  // Compares Full System vs Nearest Vertex Only (different accuracy)
  double boundary_acc_gain = (results[0].accuracy - results[2].accuracy) * 100;
  printf(
      "Decision Boundary Accuracy Contribution: +%.2f%% over nearest vertex\n",
      boundary_acc_gain);

  std::cout << "\nInterpretation:" << std::endl;
  std::cout << "- SRR grid provides O(1) lookup vs O(sqrt(n)) raw CGAL"
            << std::endl;
  std::cout
      << "- Decision boundary helps on datasets with complex class interfaces"
      << std::endl;

  return 0;
}
