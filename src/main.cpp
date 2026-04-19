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
 *
 * Fixes applied (Week 2 of the master action list):
 *               so that std::runtime_error from the updated CSV loaders
 *               (formerly exit(1)) surfaces with a clean error message and
 *               a non-zero exit code instead of an uncaught exception
 *               crashing the process.
 */

#include "../include/DelaunayClassifier.h"
#include <chrono>
#include <exception>
#include <iomanip> // for std::setprecision
#include <iostream>
#include <stdexcept>
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

  // std::runtime_error thrown by the updated CSV loaders (or by any other
  // classifier call) is reported cleanly to stderr with a non-zero exit
  // code. Previously, file-open failures called exit(1) directly inside
  // load_labeled_csv() / load_unlabeled_csv(); now they throw, and we
  // catch them here at the top level.
  //
  // We catch std::runtime_error specifically to give a targeted message,
  // and std::exception as a fallback for any other unexpected failure
  // (e.g., std::bad_alloc, std::invalid_argument from std::stoi/std::stod
  // on a malformed CSV row).
  try {
    DelaunayClassifier classifier;

    // --- MODE 1: Static Classification ---
    // Standard workflow: train on labeled data, predict on test data
    // Outputs: predictions.csv, triangles.csv, boundaries.csv
    if (mode == "static" && argc == 5) {
      std::string out_dir = argv[4];

      // a "Training Time: X.XXXX ms" line that scalability_test.py parses.
      // This excludes subprocess startup and filesystem I/O from the
      // training-time measurement, which the Python-side wall-clock
      auto train_start = std::chrono::high_resolution_clock::now();
      classifier.train(argv[2]);
      auto train_end = std::chrono::high_resolution_clock::now();
      double train_time_ms =
          std::chrono::duration_cast<std::chrono::nanoseconds>(train_end -
                                                               train_start)
              .count() /
          1e6;
      std::cout << "Training Time: " << std::fixed << std::setprecision(4)
                << train_time_ms << " ms" << std::endl;

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

  } catch (const std::runtime_error &e) {
    // Most likely cause: a CSV file could not be opened, or a CSV row was
    // malformed and one of the std::sto* parsers threw. The exception
    // message from DelaunayClassifier::load_*_csv() includes the offending
    // file path, so the user can immediately see which file failed.
    std::cerr << "Runtime error: " << e.what() << std::endl;
    std::cerr << "Hint: check that all input CSV files exist and are readable,"
              << std::endl
              << "      and that each row has the expected x,y[,label] format."
              << std::endl;
    return 1;
  } catch (const std::exception &e) {
    // Catch-all for any other standard exception (std::bad_alloc,
    // std::out_of_range, std::invalid_argument from a malformed numeric
    // field, etc.). We still report cleanly rather than letting the
    // process crash with an uncaught exception.
    std::cerr << "Unexpected error: " << e.what() << std::endl;
    return 1;
  }
}