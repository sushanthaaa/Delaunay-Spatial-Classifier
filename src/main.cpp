/**
 * @file main.cpp
 * @brief Command-line interface for the Delaunay Triangulation Classifier.
 *
 * This is the main entry point for running classification experiments.
 * Supports three operational modes:
 *
 * 1. STATIC MODE: Standard train → predict workflow
 *    - Builds triangulation from training data
 *    - Classifies test points with timing
 *    - Exports visualization files
 *
 * 2. DYNAMIC MODE: Stress test for insert/move/delete operations
 *    - Tests O(1) amortized update performance
 *    - Logs per-operation timing in nanoseconds
 *
 * 3. VISUAL MODE: Generate snapshots for dynamic operation visualization
 *    - Shows triangulation state after each phase
 *    - Useful for figures in publications
 *
 * Usage Examples:
 *   ./main static data/train/wine_train.csv data/test/wine_test_X.csv results/
 *   ./main dynamic data/train/wine_train.csv data/train/wine_stream.csv
 * results/log.csv
 *   ./main visualize_dynamic data/train/wine_train.csv
 * data/train/wine_stream.csv results/
 */

#include "../include/DelaunayClassifier.h"
#include <iostream>
#include <string>
#include <sys/stat.h>

int main(int argc, char *argv[]) {
  // --- Argument Validation ---
  if (argc < 2) {
    std::cout << "Usage: ./main [mode] [args...]\n";
    std::cout
        << "  1. Static:  ./main static <train_csv> <test_csv> <output_dir>\n";
    std::cout
        << "  2. Dynamic: ./main dynamic <base_csv> <stream_csv> <log_file>\n";
    std::cout << "  3. Visual:  ./main visualize_dynamic <base_csv> "
                 "<stream_csv> <output_dir>\n";
    return 1;
  }

  std::string mode = argv[1];
  DelaunayClassifier classifier;

  // --- MODE 1: Static Classification ---
  // Standard workflow: train on labeled data, predict on test data
  // Outputs: predictions.csv, triangles.csv, boundaries.csv
  if (mode == "static" && argc == 5) {
    std::string out_dir = argv[4];

    // Train: builds DT + 2D Buckets grid - O(n log n)
    classifier.train(argv[2]);

    // Export visualization files for figures
    classifier.export_visualization(out_dir + "/triangles.csv",
                                    out_dir + "/boundaries.csv");

    // Predict with timing benchmark - O(1) per point
    classifier.predict_benchmark(argv[3], out_dir + "/predictions.csv");
  }
  // --- MODE 2: Dynamic Stress Test ---
  // Measures insert/move/delete timing for incremental update claims
  else if (mode == "dynamic" && argc == 5) {
    // Start with base model
    classifier.train(argv[2]);

    // Stream in points, log timing for each operation
    classifier.run_dynamic_stress_test(argv[3], argv[4]);
  }
  // --- MODE 3: Dynamic Visualization ---
  // Generates snapshots for publication figures
  else if (mode == "visualize_dynamic" && argc == 5) {
    std::string out_dir = argv[4];

    classifier.train(argv[2]);

    // Creates 3 sets of files: after insert, after move, after delete
    classifier.run_dynamic_visualization(argv[3], out_dir);
  }
  // --- Invalid arguments ---
  else {
    std::cerr << "Error: Invalid mode or argument count.\n";
    std::cerr << "Run ./main without arguments for usage help.\n";
    return 1;
  }

  return 0;
}