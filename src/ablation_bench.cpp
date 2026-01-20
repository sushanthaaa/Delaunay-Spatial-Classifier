/**
 * Ablation Benchmark for IEEE Publication
 *
 * Tests 3 configurations:
 * 1. Full System (SRR + Decision Boundary)
 * 2. No SRR Grid (bypass O(1) hint, use raw CGAL locate)
 * 3. Nearest Vertex Only (no boundary interpolation)
 *
 * Usage: ./ablation_bench <train.csv> <test.csv>
 */

#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

#include "../include/DelaunayClassifier.h"

struct AblationResult {
  std::string condition;
  double accuracy;
  double avg_time_us;
};

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
  if (argc < 3) {
    std::cout << "Usage: ./ablation_bench <train.csv> <test.csv>" << std::endl;
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

  // Train classifier
  DelaunayClassifier classifier;
  classifier.train(train_file, 3); // k=3 for outlier removal

  std::cout << "\n" << std::string(80, '=') << std::endl;
  std::cout << "ABLATION STUDY BENCHMARK" << std::endl;
  std::cout << std::string(80, '=') << std::endl;

  // Condition 1: Full System (SRR + Decision Boundary)
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
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

  // Condition 2: No SRR Grid (raw CGAL locate, O(sqrt(n)))
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
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

  // Condition 3: Nearest Vertex Only (no decision boundary)
  {
    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
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

  // Print results
  std::cout << std::string(80, '-') << std::endl;
  printf("%-35s | %-12s | %-15s | %-10s\n", "Condition", "Accuracy",
         "Time (µs)", "Speedup");
  std::cout << std::string(80, '-') << std::endl;

  double baseline_time = results[0].avg_time_us;
  for (const auto &r : results) {
    double speedup = r.avg_time_us / baseline_time; // >1 means slower
    printf("%-35s | %6.2f%%     | %12.4f    | %.2fx\n", r.condition.c_str(),
           r.accuracy * 100, r.avg_time_us, speedup);
  }

  std::cout << std::string(80, '=') << std::endl;

  // Print insights
  std::cout << "\nABLATION INSIGHTS:" << std::endl;
  std::cout << std::string(40, '-') << std::endl;

  // SRR contribution to speed
  double srr_speedup = results[1].avg_time_us / results[0].avg_time_us;
  printf("SRR Grid Speed Contribution: %.1fx faster with SRR\n", srr_speedup);

  // Decision boundary contribution to accuracy
  double boundary_acc_gain = (results[0].accuracy - results[2].accuracy) * 100;
  printf(
      "Decision Boundary Accuracy Contribution: +%.2f%% over nearest vertex\n",
      boundary_acc_gain);

  return 0;
}
