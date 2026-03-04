/**
 * @file benchmark.cpp
 * @brief Fair C++ Benchmark Suite for Delaunay Classifier vs. Baselines
 *
 * FIXES APPLIED:
 * #10: Direct move timing (not insert+delete approximation)
 * #15: Adaptive SVM gamma based on data variance
 * #16: Adaptive Decision Tree depth based on dataset size
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <flann/flann.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <svm.h>
#include <vector>

#include "../include/DelaunayClassifier.h"

// =============================================================================
// RESULT STRUCTURES
// =============================================================================

struct BenchmarkResult {
  std::string method;
  double accuracy;
  double avg_inference_us;
  double train_time_ms;
};

struct DynamicResult {
  std::string method;
  double insert_ns;
  double move_ns;
  double delete_ns;
};

// =============================================================================
// DATA LOADING
// =============================================================================

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

// =============================================================================
// HELPER: Compute data statistics for adaptive parameters
// =============================================================================

struct DataStats {
  double mean_x, mean_y;
  double var_x, var_y;
  double total_variance;
  float x_min, x_max, y_min, y_max;
};

DataStats
compute_data_stats(const std::vector<std::tuple<float, float, int>> &data) {
  DataStats stats = {};
  int n = static_cast<int>(data.size());
  if (n == 0)
    return stats;

  stats.x_min = stats.y_min = 1e18f;
  stats.x_max = stats.y_max = -1e18f;

  for (const auto &pt : data) {
    float x = std::get<0>(pt), y = std::get<1>(pt);
    stats.mean_x += x;
    stats.mean_y += y;
    if (x < stats.x_min)
      stats.x_min = x;
    if (x > stats.x_max)
      stats.x_max = x;
    if (y < stats.y_min)
      stats.y_min = y;
    if (y > stats.y_max)
      stats.y_max = y;
  }
  stats.mean_x /= n;
  stats.mean_y /= n;

  for (const auto &pt : data) {
    float x = std::get<0>(pt), y = std::get<1>(pt);
    stats.var_x += (x - stats.mean_x) * (x - stats.mean_x);
    stats.var_y += (y - stats.mean_y) * (y - stats.mean_y);
  }
  stats.var_x /= n;
  stats.var_y /= n;
  stats.total_variance = stats.var_x + stats.var_y;

  return stats;
}

// =============================================================================
// FLANN-based KNN classifier
// =============================================================================

class FlannKNN {
private:
  flann::Index<flann::L2<float>> *index_;
  flann::Matrix<float> *dataset_;
  std::vector<int> labels_;
  int k_;

public:
  FlannKNN(int k_neighbors = 5)
      : k_(k_neighbors), index_(nullptr), dataset_(nullptr) {}

  ~FlannKNN() {
    if (index_)
      delete index_;
    if (dataset_) {
      delete[] dataset_->ptr();
      delete dataset_;
    }
  }

  void fit(const std::vector<std::tuple<float, float, int>> &train_data) {
    int n = static_cast<int>(train_data.size());
    float *data = new float[n * 2];
    labels_.resize(n);

    for (int i = 0; i < n; i++) {
      data[i * 2] = std::get<0>(train_data[i]);
      data[i * 2 + 1] = std::get<1>(train_data[i]);
      labels_[i] = std::get<2>(train_data[i]);
    }

    dataset_ = new flann::Matrix<float>(data, n, 2);
    index_ = new flann::Index<flann::L2<float>>(*dataset_,
                                                flann::KDTreeIndexParams(4));
    index_->buildIndex();
  }

  int predict(float x, float y) {
    float query_data[2] = {x, y};
    flann::Matrix<float> query(query_data, 1, 2);

    std::vector<int> indices(k_);
    std::vector<float> dists(k_);
    flann::Matrix<int> indices_mat(indices.data(), 1, k_);
    flann::Matrix<float> dists_mat(dists.data(), 1, k_);

    index_->knnSearch(query, indices_mat, dists_mat, k_,
                      flann::SearchParams(128));

    std::map<int, int> votes;
    for (int i = 0; i < k_; i++) {
      if (indices[i] >= 0 && indices[i] < (int)labels_.size()) {
        votes[labels_[indices[i]]]++;
      }
    }

    int best_label = -1, best_count = 0;
    for (const auto &v : votes) {
      if (v.second > best_count) {
        best_count = v.second;
        best_label = v.first;
      }
    }
    return best_label;
  }
};

// =============================================================================
// LibSVM Wrapper — FIX #15: Adaptive gamma
// =============================================================================

class SVMClassifier {
private:
  svm_model *model_;
  svm_problem prob_;
  svm_parameter param_;
  std::vector<svm_node *> x_space_;

public:
  SVMClassifier() : model_(nullptr) {
    param_.svm_type = C_SVC;
    param_.kernel_type = RBF;
    param_.gamma = 0.5; // Will be overridden in fit()
    param_.C = 1.0;     // Will be overridden in fit()
    param_.cache_size = 100;
    param_.eps = 0.001;
    param_.shrinking = 1;
    param_.probability = 0;
    param_.nr_weight = 0;
    param_.weight_label = nullptr;
    param_.weight = nullptr;
  }

  ~SVMClassifier() {
    if (model_)
      svm_free_and_destroy_model(&model_);
    for (auto &x : x_space_)
      delete[] x;
  }

  void fit(const std::vector<std::tuple<float, float, int>> &train_data) {
    int n = static_cast<int>(train_data.size());

    // FIX #15: Adaptive gamma = 1 / (n_features * variance)
    // This is the standard scikit-learn 'scale' heuristic
    DataStats stats = compute_data_stats(train_data);
    double variance = stats.total_variance;
    if (variance > 1e-10) {
      param_.gamma = 1.0 / (2.0 * variance);
    } else {
      param_.gamma = 1.0; // Fallback for zero-variance data
    }

    // FIX #15: Also use a reasonable C value based on data scale
    // Common heuristic: C = 1.0 works well when data is normalized;
    // for unnormalized data, scale C by the reciprocal of variance
    if (variance > 1e-10) {
      param_.C = std::max(1.0, 1.0 / variance);
    } else {
      param_.C = 1.0;
    }

    prob_.l = n;
    prob_.y = new double[n];
    prob_.x = new svm_node *[n];

    for (int i = 0; i < n; i++) {
      prob_.y[i] = std::get<2>(train_data[i]);
      svm_node *x = new svm_node[3];
      x[0].index = 1;
      x[0].value = std::get<0>(train_data[i]);
      x[1].index = 2;
      x[1].value = std::get<1>(train_data[i]);
      x[2].index = -1;
      prob_.x[i] = x;
      x_space_.push_back(x);
    }

    // Suppress libsvm output
    svm_set_print_string_function([](const char *) {});
    model_ = svm_train(&prob_, &param_);

    delete[] prob_.y;
    delete[] prob_.x;
  }

  int predict(float x, float y) {
    svm_node query[3];
    query[0].index = 1;
    query[0].value = x;
    query[1].index = 2;
    query[1].value = y;
    query[2].index = -1;
    return (int)svm_predict(model_, query);
  }
};

// =============================================================================
// Decision Tree — FIX #16: Adaptive depth
// =============================================================================

struct DTNode {
  bool is_leaf;
  int label;
  int split_feature;
  float split_value;
  DTNode *left;
  DTNode *right;

  DTNode()
      : is_leaf(false), label(-1), split_feature(0), split_value(0),
        left(nullptr), right(nullptr) {}
  ~DTNode() {
    delete left;
    delete right;
  }
};

class DecisionTreeCpp {
private:
  DTNode *root_;
  int max_depth_;
  int min_samples_;

  float gini(const std::vector<int> &labels) {
    if (labels.empty())
      return 0;
    std::map<int, int> counts;
    for (int l : labels)
      counts[l]++;
    float impurity = 1.0f;
    for (const auto &c : counts) {
      float p = (float)c.second / (float)labels.size();
      impurity -= p * p;
    }
    return impurity;
  }

  DTNode *build(std::vector<std::tuple<float, float, int>> &data, int depth) {
    DTNode *node = new DTNode();

    std::map<int, int> label_counts;
    for (const auto &pt : data)
      label_counts[std::get<2>(pt)]++;

    // Leaf conditions: pure node, max depth reached, or too few samples
    if (label_counts.size() == 1 || depth >= max_depth_ ||
        (int)data.size() < min_samples_) {
      node->is_leaf = true;
      int best_label = -1, best_count = 0;
      for (const auto &lc : label_counts) {
        if (lc.second > best_count) {
          best_count = lc.second;
          best_label = lc.first;
        }
      }
      node->label = best_label;
      return node;
    }

    float best_gini = 1e9;
    int best_feature = 0;
    float best_value = 0;

    for (int feature = 0; feature < 2; feature++) {
      std::vector<float> values;
      for (const auto &pt : data) {
        values.push_back(feature == 0 ? std::get<0>(pt) : std::get<1>(pt));
      }
      std::sort(values.begin(), values.end());

      // Sample split candidates (every ~5% of sorted values)
      int step = std::max(1, (int)(values.size() / 20));
      for (size_t i = 1; i < values.size(); i += step) {
        float split = (values[i - 1] + values[i]) / 2;

        std::vector<int> left_labels, right_labels;
        for (const auto &pt : data) {
          float v = feature == 0 ? std::get<0>(pt) : std::get<1>(pt);
          if (v <= split)
            left_labels.push_back(std::get<2>(pt));
          else
            right_labels.push_back(std::get<2>(pt));
        }

        if (left_labels.empty() || right_labels.empty())
          continue;

        float weighted_gini =
            ((float)left_labels.size() * gini(left_labels) +
             (float)right_labels.size() * gini(right_labels)) /
            (float)data.size();

        if (weighted_gini < best_gini) {
          best_gini = weighted_gini;
          best_feature = feature;
          best_value = split;
        }
      }
    }

    std::vector<std::tuple<float, float, int>> left_data, right_data;
    for (const auto &pt : data) {
      float v = best_feature == 0 ? std::get<0>(pt) : std::get<1>(pt);
      if (v <= best_value)
        left_data.push_back(pt);
      else
        right_data.push_back(pt);
    }

    if (left_data.empty() || right_data.empty()) {
      node->is_leaf = true;
      int best_label = -1, best_count = 0;
      for (const auto &lc : label_counts) {
        if (lc.second > best_count) {
          best_count = lc.second;
          best_label = lc.first;
        }
      }
      node->label = best_label;
      return node;
    }

    node->split_feature = best_feature;
    node->split_value = best_value;
    node->left = build(left_data, depth + 1);
    node->right = build(right_data, depth + 1);

    return node;
  }

  int predict_node(DTNode *node, float x, float y) {
    if (node->is_leaf)
      return node->label;
    float v = node->split_feature == 0 ? x : y;
    if (v <= node->split_value)
      return predict_node(node->left, x, y);
    return predict_node(node->right, x, y);
  }

public:
  // FIX #16: Accept adaptive max_depth parameter
  DecisionTreeCpp(int max_depth = 10, int min_samples = 2)
      : root_(nullptr), max_depth_(max_depth), min_samples_(min_samples) {}

  ~DecisionTreeCpp() { delete root_; }

  void fit(std::vector<std::tuple<float, float, int>> train_data) {
    root_ = build(train_data, 0);
  }

  int predict(float x, float y) { return predict_node(root_, x, y); }
};

// =============================================================================
// PRINT RESULTS
// =============================================================================

void print_static_results(const std::vector<BenchmarkResult> &results,
                          const std::string &dataset_name) {
  std::cout << "\n" << std::string(95, '=') << std::endl;
  std::cout << "C++ STATIC BENCHMARK: " << dataset_name << std::endl;
  std::cout << std::string(95, '-') << std::endl;
  printf("%-30s | %-10s | %-15s | %-12s | %-10s\n", "Algorithm", "Accuracy",
         "Inference (us)", "Train (ms)", "Speedup");
  std::cout << std::string(95, '-') << std::endl;

  double baseline_time = results.empty() ? 1.0 : results[0].avg_inference_us;
  for (const auto &r : results) {
    double speedup =
        (r.avg_inference_us > 0) ? baseline_time / r.avg_inference_us : 0;
    printf("%-30s | %6.1f%%   | %12.4f    | %9.2f   | %7.1fx\n",
           r.method.c_str(), r.accuracy * 100, r.avg_inference_us,
           r.train_time_ms, speedup);
  }
  std::cout << std::string(95, '=') << std::endl;
}

void print_dynamic_results(const std::vector<DynamicResult> &results,
                           const std::string &dataset_name) {
  std::cout << "\n" << std::string(95, '=') << std::endl;
  std::cout << "C++ DYNAMIC BENCHMARK: " << dataset_name << std::endl;
  std::cout << std::string(95, '-') << std::endl;
  printf("%-30s | %-15s | %-15s | %-15s\n", "Algorithm", "Insert (ns)",
         "Move (ns)", "Delete (ns)");
  std::cout << std::string(95, '-') << std::endl;

  for (const auto &r : results) {
    if (r.insert_ns > 0) {
      printf("%-30s | %12.0f    | %12.0f    | %12.0f\n", r.method.c_str(),
             r.insert_ns, r.move_ns, r.delete_ns);
    } else {
      printf("%-30s | %15s | %15s | %15s\n", r.method.c_str(), "N/A", "N/A",
             "N/A");
    }
  }
  std::cout << std::string(95, '=') << std::endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char *argv[]) {
  if (argc < 4) {
    std::cout << "Usage: ./benchmark <train_csv> <test_csv> <dataset_name>"
              << std::endl;
    return 1;
  }

  std::string train_file = argv[1];
  std::string test_file = argv[2];
  std::string dataset_name = argv[3];

  auto train_data = load_labeled_csv(train_file);
  auto test_data = load_labeled_csv(test_file);

  if (train_data.empty() || test_data.empty()) {
    std::cerr << "Error loading data!" << std::endl;
    return 1;
  }

  std::cout << "Loaded " << train_data.size() << " training points, "
            << test_data.size() << " test points." << std::endl;

  // Compute data statistics for adaptive parameters
  DataStats stats = compute_data_stats(train_data);
  std::cout << "Data Stats: variance=" << stats.total_variance << ", range_x=["
            << stats.x_min << "," << stats.x_max << "]"
            << ", range_y=[" << stats.y_min << "," << stats.y_max << "]"
            << std::endl;

  std::vector<BenchmarkResult> static_results;
  std::vector<DynamicResult> dynamic_results;

  // ============================================
  // STATIC 1: FLANN C++ KNN (k=5)
  // ============================================
  {
    std::cout << "\nRunning FLANN C++ KNN (k=5)..." << std::endl;
    FlannKNN knn(5);

    auto train_start = std::chrono::high_resolution_clock::now();
    knn.fit(train_data);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          train_end - train_start)
                          .count() /
                      1000.0;

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
      int pred = knn.predict(std::get<0>(pt), std::get<1>(pt));
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

    static_results.push_back(
        {"FLANN C++ KNN (k=5)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // STATIC 2: LibSVM — FIX #15: Adaptive gamma
  // ============================================
  {
    std::cout << "Running LibSVM C++ (RBF, adaptive gamma="
              << 1.0 / (2.0 * std::max(stats.total_variance, 1e-10)) << ")..."
              << std::endl;
    SVMClassifier svm;

    auto train_start = std::chrono::high_resolution_clock::now();
    svm.fit(train_data);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          train_end - train_start)
                          .count() /
                      1000.0;

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
      int pred = svm.predict(std::get<0>(pt), std::get<1>(pt));
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

    static_results.push_back(
        {"LibSVM C++ (RBF, adaptive)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // STATIC 3: Decision Tree — FIX #16: Adaptive depth
  // ============================================
  {
    // FIX #16: depth = min(20, max(5, 2 * log2(n)))
    int adaptive_depth = std::min(
        20, std::max(5, (int)(2.0 * std::log2((double)train_data.size()))));
    std::cout << "Running C++ Decision Tree (adaptive depth=" << adaptive_depth
              << ")..." << std::endl;

    DecisionTreeCpp dt_clf(adaptive_depth, 2);

    auto train_start = std::chrono::high_resolution_clock::now();
    dt_clf.fit(train_data);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          train_end - train_start)
                          .count() /
                      1000.0;

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
      int pred = dt_clf.predict(std::get<0>(pt), std::get<1>(pt));
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

    static_results.push_back(
        {"C++ Decision Tree (adaptive)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // STATIC 4: Delaunay C++ (Ours)
  // ============================================
  {
    std::cout << "Running Delaunay C++ (Ours)..." << std::endl;
    DelaunayClassifier classifier;
    classifier.set_output_dir("results");

    auto train_start = std::chrono::high_resolution_clock::now();
    classifier.train(train_file, 3);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          train_end - train_start)
                          .count() /
                      1000.0;

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

    static_results.push_back(
        {"**Delaunay C++ (Ours)**", accuracy, avg_us, train_ms});
  }

  print_static_results(static_results, dataset_name);

  // ============================================
  // DYNAMIC BENCHMARKS
  // ============================================
  std::cout << "\n--- DYNAMIC BENCHMARKS ---" << std::endl;

  const int NUM_DYNAMIC_OPS = std::min(10, (int)test_data.size());

  // Dynamic: Decision Tree (requires full rebuild)
  {
    std::cout << "Running C++ Decision Tree (Rebuild)..." << std::endl;
    std::vector<std::tuple<float, float, int>> working_data = train_data;

    int adaptive_depth = std::min(
        20, std::max(5, (int)(2.0 * std::log2((double)train_data.size()))));

    double total_rebuild_ns = 0;
    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      working_data.push_back(test_data[i]);

      DecisionTreeCpp dt_new(adaptive_depth, 2);
      auto start = std::chrono::high_resolution_clock::now();
      dt_new.fit(working_data);
      auto end = std::chrono::high_resolution_clock::now();
      total_rebuild_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
    }

    double avg_rebuild = total_rebuild_ns / NUM_DYNAMIC_OPS;
    dynamic_results.push_back(
        {"C++ Decision Tree (Rebuild)", avg_rebuild, avg_rebuild, avg_rebuild});
  }

  // Dynamic: Delaunay — FIX #10: DIRECTLY measure insert, move, delete
  {
    std::cout << "Running Delaunay C++ (Incremental)..." << std::endl;

    DelaunayClassifier classifier;
    classifier.set_output_dir("results");
    classifier.train(train_file, 3);

    // FIX #10 & #17: Adaptive move offset based on data range
    double range_x = stats.x_max - stats.x_min;
    double range_y = stats.y_max - stats.y_min;
    double move_offset_x = 0.01 * range_x; // 1% of x range
    double move_offset_y = 0.01 * range_y; // 1% of y range

    // --- Measure INSERT times ---
    double total_insert_ns = 0;
    std::vector<std::tuple<float, float, int>> inserted_points;

    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      float x = std::get<0>(test_data[i]);
      float y = std::get<1>(test_data[i]);
      int label = std::get<2>(test_data[i]);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.insert_point(x, y, label);
      auto end = std::chrono::high_resolution_clock::now();
      total_insert_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      inserted_points.push_back({x, y, label});
    }

    // --- FIX #10: Measure MOVE times DIRECTLY using move_point() ---
    // Previously this was approximated as insert + delete time.
    // Now we call move_point() which implements Algorithm 3.
    double total_move_ns = 0;
    std::vector<std::tuple<float, float, int>> moved_points;

    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      float old_x = std::get<0>(inserted_points[i]);
      float old_y = std::get<1>(inserted_points[i]);
      float new_x = old_x + static_cast<float>(move_offset_x);
      float new_y = old_y + static_cast<float>(move_offset_y);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.move_point(old_x, old_y, new_x, new_y);
      auto end = std::chrono::high_resolution_clock::now();
      total_move_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
      moved_points.push_back({new_x, new_y, std::get<2>(inserted_points[i])});
    }

    // --- Measure DELETE times ---
    double total_delete_ns = 0;
    for (int i = NUM_DYNAMIC_OPS - 1; i >= 0; i--) {
      float x = std::get<0>(moved_points[i]);
      float y = std::get<1>(moved_points[i]);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.remove_point(x, y);
      auto end = std::chrono::high_resolution_clock::now();
      total_delete_ns +=
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count();
    }

    double avg_insert_ns = total_insert_ns / NUM_DYNAMIC_OPS;
    double avg_move_ns = total_move_ns / NUM_DYNAMIC_OPS;
    double avg_delete_ns = total_delete_ns / NUM_DYNAMIC_OPS;

    dynamic_results.push_back({"**Delaunay C++ (O(1) Update)**", avg_insert_ns,
                               avg_move_ns, avg_delete_ns});
  }

  print_dynamic_results(dynamic_results, dataset_name);

  // ============================================
  // SAVE RESULTS TO CSV
  // ============================================
  std::string output_file = "results/cpp_benchmark_" + dataset_name + ".csv";
  std::ofstream csv(output_file);
  csv << "method,accuracy,avg_inference_us,train_time_ms,speedup_vs_knn\n";
  double baseline =
      static_results.empty() ? 1.0 : static_results[0].avg_inference_us;
  for (const auto &r : static_results) {
    double speedup =
        (r.avg_inference_us > 0) ? baseline / r.avg_inference_us : 0;
    csv << r.method << "," << r.accuracy << "," << r.avg_inference_us << ","
        << r.train_time_ms << "," << speedup << "\n";
  }
  csv.close();
  std::cout << "\nResults saved to: " << output_file << std::endl;

  return 0;
}