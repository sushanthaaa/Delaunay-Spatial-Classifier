/**
 * @file benchmark.cpp
 * @brief Fair C++ Benchmark Suite for Delaunay Classifier vs. Baselines
 *
 * All implementations in C++ with -O3 optimization for fair comparison.
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <flann/flann.hpp>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <sstream>
#include <svm.h>
#include <vector>

#include "../include/DelaunayClassifier.h"

// =============================================================================
// BENCHMARK CONFIGURATION CONSTANTS
// =============================================================================

namespace {

/// Number of training points below which we fall back to a smaller CV fold
/// count to avoid degenerate folds. Empirically, 5-fold CV needs at least
/// ~5 * n_classes points per class; below this, 3-fold is safer.
constexpr int CV_MIN_POINTS_FOR_5FOLD = 100;

/// Number of cross-validation folds for hyperparameter selection (used by
/// SVM grid search, KNN k selection).
constexpr int CV_NUM_FOLDS = 5;

/// Fallback CV fold count for small datasets below CV_MIN_POINTS_FOR_5FOLD.
constexpr int CV_NUM_FOLDS_SMALL = 3;

/// Target number of dynamic operations (insert/move/delete) to measure.
/// Capped at test_data.size() per dataset so small datasets still work.
constexpr int TARGET_DYNAMIC_OPS = 1000;

/// 100 is the sklearn default and matches the conventional baseline.
constexpr int RF_NUM_TREES = 100;

/// Bootstrap sample size as a fraction of training data for each RF tree.
/// 1.0 matches sklearn's default bootstrap sampling with replacement.
constexpr double RF_BOOTSTRAP_FRACTION = 1.0;

/// Maximum depth clamp for decision trees (both standalone DT and RF trees).
/// Computed as min(DT_MAX_DEPTH_CAP, max(DT_MIN_DEPTH_CAP, 2*log2(n))).
constexpr int DT_MAX_DEPTH_CAP = 20;
constexpr int DT_MIN_DEPTH_CAP = 5;

/// Movement offset used in the dynamic benchmark, as a fraction of the
/// smaller data dimension. Benchmark parameter, not a model parameter.
constexpr double DYNAMIC_MOVE_OFFSET_FRACTION = 0.01;

/// Number of features for the 2D classifier. Hardcoded since this code
/// is 2D-only; if the algorithm is ever extended to 3D+, update this.
constexpr int N_FEATURES = 2;

// =============================================================================
// HYPERPARAMETER GRIDS FOR CV SEARCH
// =============================================================================

/// Covers the typical range from "very local" (k=1) through "smoothed" (k=15).
const std::vector<int> KNN_K_GRID = {1, 3, 5, 7, 9, 11, 15};

/// Logarithmic scale from under-regularized (C=100) to over-regularized
/// (C=0.1).
const std::vector<double> SVM_C_GRID = {0.1, 1.0, 10.0, 100.0};

/// Applied relative to the sklearn 'scale' default: 1/(n_features * variance).
/// Multipliers {0.1, 1.0, 10.0, 100.0} give four gamma values per dataset,
/// adapting to the data's variance while exploring the neighborhood of 'scale'.
const std::vector<double> SVM_GAMMA_MULTIPLIER_GRID = {0.1, 1.0, 10.0, 100.0};

} // anonymous namespace

// =============================================================================
// RESULT STRUCTURES
// =============================================================================

struct BenchmarkResult {
  std::string method;
  double accuracy;
  double avg_inference_us;
  double train_time_ms;
};

/// Dynamic benchmark result with mean and standard deviation per operation.
struct DynamicResult {
  std::string method;
  double insert_ns_mean;
  double insert_ns_std;
  double move_ns_mean;
  double move_ns_std;
  double delete_ns_mean;
  double delete_ns_std;
  int num_ops; ///< Actual number of operations measured (may be < TARGET)
};

// =============================================================================
// STATISTICAL HELPERS
// =============================================================================

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

/**
 * @brief Compute the sklearn 'scale' gamma heuristic for RBF SVM.
 *
 * This is the sklearn default for gamma='scale' (since v0.22):
 *   gamma = 1 / (n_features * X.var())
 *
 * 1/(n_features * variance) only because n_features=2 for this 2D classifier.
 * We make the formula explicit here so future 3D extensions don't silently
 * use the wrong value.
 */
static double sklearn_scale_gamma(double total_variance) {
  if (total_variance <= 1e-10)
    return 1.0;
  return 1.0 / (static_cast<double>(N_FEATURES) * total_variance);
}

// =============================================================================
// CROSS-VALIDATION SPLIT HELPER
// =============================================================================

static std::vector<std::pair<std::vector<int>, std::vector<int>>>
make_cv_folds(int n_samples, int n_folds, unsigned int seed = 42) {
  std::vector<int> indices(n_samples);
  std::iota(indices.begin(), indices.end(), 0);

  std::mt19937 rng(seed);
  std::shuffle(indices.begin(), indices.end(), rng);

  std::vector<std::pair<std::vector<int>, std::vector<int>>> folds(n_folds);

  for (int i = 0; i < n_samples; ++i) {
    int fold = i % n_folds;
    // This sample goes to fold's validation set; all other folds get it in
    // train
    for (int f = 0; f < n_folds; ++f) {
      if (f == fold) {
        folds[f].second.push_back(indices[i]);
      } else {
        folds[f].first.push_back(indices[i]);
      }
    }
  }
  return folds;
}

/// Select CV fold count based on dataset size.
static int select_cv_folds(int n_samples) {
  return (n_samples >= CV_MIN_POINTS_FOR_5FOLD) ? CV_NUM_FOLDS
                                                : CV_NUM_FOLDS_SMALL;
}

// =============================================================================
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

  int get_k() const { return k_; }

  void fit(const std::vector<std::tuple<float, float, int>> &train_data) {
    // Clean up any previous fit state
    if (index_) {
      delete index_;
      index_ = nullptr;
    }
    if (dataset_) {
      delete[] dataset_->ptr();
      delete dataset_;
      dataset_ = nullptr;
    }

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
    // Ensure k doesn't exceed the number of training points
    int effective_k = std::min(k_, static_cast<int>(labels_.size()));
    if (effective_k <= 0)
      return -1;

    float query_data[2] = {x, y};
    flann::Matrix<float> query(query_data, 1, 2);

    std::vector<int> indices(effective_k);
    std::vector<float> dists(effective_k);
    flann::Matrix<int> indices_mat(indices.data(), 1, effective_k);
    flann::Matrix<float> dists_mat(dists.data(), 1, effective_k);

    index_->knnSearch(query, indices_mat, dists_mat, effective_k,
                      flann::SearchParams(128));

    std::map<int, int> votes;
    for (int i = 0; i < effective_k; i++) {
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

/**
 *
 * Runs 5-fold (or 3-fold for small datasets) CV on the training set across
 * KNN_K_GRID and returns the k that maximizes mean validation accuracy.
 * Ties are broken by preferring smaller k (more local).
 */
static int
knn_cv_select_k(const std::vector<std::tuple<float, float, int>> &train_data) {
  int n = static_cast<int>(train_data.size());
  int n_folds = select_cv_folds(n);
  auto folds = make_cv_folds(n, n_folds);

  int best_k = KNN_K_GRID[0];
  double best_mean_acc = -1.0;

  for (int k : KNN_K_GRID) {
    // Skip k values larger than the smallest training fold
    bool k_too_large = false;
    for (const auto &fold : folds) {
      if (k > static_cast<int>(fold.first.size())) {
        k_too_large = true;
        break;
      }
    }
    if (k_too_large)
      continue;

    double total_acc = 0.0;
    for (const auto &fold : folds) {
      const auto &train_idx = fold.first;
      const auto &val_idx = fold.second;

      std::vector<std::tuple<float, float, int>> fold_train;
      fold_train.reserve(train_idx.size());
      for (int i : train_idx)
        fold_train.push_back(train_data[i]);

      FlannKNN knn(k);
      knn.fit(fold_train);

      int correct = 0;
      for (int i : val_idx) {
        int pred =
            knn.predict(std::get<0>(train_data[i]), std::get<1>(train_data[i]));
        if (pred == std::get<2>(train_data[i]))
          correct++;
      }
      total_acc +=
          static_cast<double>(correct) / static_cast<double>(val_idx.size());
    }
    double mean_acc = total_acc / static_cast<double>(folds.size());

    if (mean_acc > best_mean_acc) {
      best_mean_acc = mean_acc;
      best_k = k;
    }
  }

  return best_k;
}

// =============================================================================
// LibSVM Wrapper — with grid-search CV tuning
// =============================================================================

class SVMClassifier {
private:
  svm_model *model_;
  svm_problem prob_;
  svm_parameter param_;
  std::vector<svm_node *> x_space_;

  void init_default_params() {
    param_.svm_type = C_SVC;
    param_.kernel_type = RBF;
    param_.gamma = 0.5;
    param_.C = 1.0;
    param_.cache_size = 100;
    param_.eps = 0.001;
    param_.shrinking = 1;
    param_.probability = 0;
    param_.nr_weight = 0;
    param_.weight_label = nullptr;
    param_.weight = nullptr;
  }

  void free_model() {
    if (model_) {
      svm_free_and_destroy_model(&model_);
      model_ = nullptr;
    }
    for (auto &x : x_space_)
      delete[] x;
    x_space_.clear();
  }

public:
  SVMClassifier() : model_(nullptr) { init_default_params(); }

  ~SVMClassifier() { free_model(); }

  double get_C() const { return param_.C; }
  double get_gamma() const { return param_.gamma; }

  /**
   * @brief Fit SVM with explicit (C, gamma) hyperparameters.
   *
   * Used by the grid-search CV driver to test candidate hyperparameters
   * and by the final fit after CV has selected the best values.
   */
  void
  fit_with_params(const std::vector<std::tuple<float, float, int>> &train_data,
                  double C, double gamma) {
    free_model();
    init_default_params();
    param_.C = C;
    param_.gamma = gamma;

    int n = static_cast<int>(train_data.size());
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

/**
 *
 * Previously, SVM used hardcoded C = max(1, 1/var) and gamma = 1/(2*var),
 * which produced pathological results on some datasets (e.g., Spiral: 65%).
 *
 * This function:
 *   1. Computes the sklearn 'scale' gamma as a reference: 1/(n_feat * var)
 *   2. Explores a grid of C values and gamma values around that reference
 *   3. Runs k-fold CV on the training set, picking the (C, gamma) with
 *      the highest mean validation accuracy
 *   4. Returns the best (C, gamma) pair for use in the final fit
 */
static std::pair<double, double> svm_cv_select_params(
    const std::vector<std::tuple<float, float, int>> &train_data,
    const DataStats &stats) {
  int n = static_cast<int>(train_data.size());
  int n_folds = select_cv_folds(n);
  auto folds = make_cv_folds(n, n_folds);

  double scale_gamma = sklearn_scale_gamma(stats.total_variance);
  double best_C = 1.0;
  double best_gamma = scale_gamma;
  double best_mean_acc = -1.0;

  for (double C : SVM_C_GRID) {
    for (double gamma_mult : SVM_GAMMA_MULTIPLIER_GRID) {
      double gamma = scale_gamma * gamma_mult;

      double total_acc = 0.0;
      for (const auto &fold : folds) {
        const auto &train_idx = fold.first;
        const auto &val_idx = fold.second;

        std::vector<std::tuple<float, float, int>> fold_train;
        fold_train.reserve(train_idx.size());
        for (int i : train_idx)
          fold_train.push_back(train_data[i]);

        SVMClassifier svm;
        svm.fit_with_params(fold_train, C, gamma);

        int correct = 0;
        for (int i : val_idx) {
          int pred = svm.predict(std::get<0>(train_data[i]),
                                 std::get<1>(train_data[i]));
          if (pred == std::get<2>(train_data[i]))
            correct++;
        }
        total_acc +=
            static_cast<double>(correct) / static_cast<double>(val_idx.size());
      }
      double mean_acc = total_acc / static_cast<double>(folds.size());

      if (mean_acc > best_mean_acc) {
        best_mean_acc = mean_acc;
        best_C = C;
        best_gamma = gamma;
      }
    }
  }

  return {best_C, best_gamma};
}

// =============================================================================
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

    //
    // The previous implementation sampled every 20th value:
    //   int step = std::max(1, (int)(values.size() / 20));
    //   for (size_t i = 1; i < values.size(); i += step) { ... }
    //
    // This meant the DT only tried ~20 split points per feature regardless
    // of dataset size, weakening the baseline unfairly. sklearn's DT tries
    // every midpoint between distinct adjacent sorted values. We match that
    // here by iterating over all adjacent pairs and using midpoints where
    // values differ.
    for (int feature = 0; feature < 2; feature++) {
      // Extract and sort feature values with their labels
      std::vector<std::pair<float, int>> sorted_data;
      sorted_data.reserve(data.size());
      for (const auto &pt : data) {
        float v = (feature == 0) ? std::get<0>(pt) : std::get<1>(pt);
        sorted_data.push_back({v, std::get<2>(pt)});
      }
      std::sort(
          sorted_data.begin(), sorted_data.end(),
          [](const std::pair<float, int> &a, const std::pair<float, int> &b) {
            return a.first < b.first;
          });

      // Try every midpoint between distinct adjacent values
      for (size_t i = 1; i < sorted_data.size(); ++i) {
        // Skip duplicates: only consider splits where values differ
        if (sorted_data[i - 1].first == sorted_data[i].first)
          continue;

        float split = (sorted_data[i - 1].first + sorted_data[i].first) / 2.0f;

        std::vector<int> left_labels, right_labels;
        left_labels.reserve(i);
        right_labels.reserve(sorted_data.size() - i);
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
  DecisionTreeCpp(int max_depth = 10, int min_samples = 2)
      : root_(nullptr), max_depth_(max_depth), min_samples_(min_samples) {}

  ~DecisionTreeCpp() { delete root_; }

  void fit(std::vector<std::tuple<float, float, int>> train_data) {
    if (root_) {
      delete root_;
      root_ = nullptr;
    }
    root_ = build(train_data, 0);
  }

  int predict(float x, float y) { return predict_node(root_, x, y); }
};

class RandomForestCpp {
private:
  std::vector<DecisionTreeCpp *> trees_;
  int num_trees_;
  int max_depth_;
  int min_samples_;
  unsigned int seed_;

public:
  RandomForestCpp(int num_trees = RF_NUM_TREES, int max_depth = 10,
                  int min_samples = 2, unsigned int seed = 42)
      : num_trees_(num_trees), max_depth_(max_depth), min_samples_(min_samples),
        seed_(seed) {}

  ~RandomForestCpp() {
    for (auto *t : trees_)
      delete t;
    trees_.clear();
  }

  void fit(const std::vector<std::tuple<float, float, int>> &train_data) {
    // Clean up any previous trees
    for (auto *t : trees_)
      delete t;
    trees_.clear();
    trees_.reserve(num_trees_);

    int n = static_cast<int>(train_data.size());
    int bootstrap_size = static_cast<int>(n * RF_BOOTSTRAP_FRACTION);

    std::mt19937 rng(seed_);
    std::uniform_int_distribution<int> sampler(0, n - 1);

    for (int t = 0; t < num_trees_; ++t) {
      // Draw bootstrap sample with replacement
      std::vector<std::tuple<float, float, int>> bootstrap;
      bootstrap.reserve(bootstrap_size);
      for (int i = 0; i < bootstrap_size; ++i) {
        bootstrap.push_back(train_data[sampler(rng)]);
      }

      DecisionTreeCpp *tree = new DecisionTreeCpp(max_depth_, min_samples_);
      tree->fit(bootstrap);
      trees_.push_back(tree);
    }
  }

  int predict(float x, float y) {
    if (trees_.empty())
      return -1;

    std::map<int, int> votes;
    for (auto *tree : trees_) {
      int pred = tree->predict(x, y);
      votes[pred]++;
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
  std::cout << "\n" << std::string(120, '=') << std::endl;
  std::cout << "C++ DYNAMIC BENCHMARK: " << dataset_name << std::endl;
  std::cout << std::string(120, '-') << std::endl;
  printf("%-30s | %-22s | %-22s | %-22s | %-8s\n", "Algorithm",
         "Insert (ns) mean±std", "Move (ns) mean±std", "Delete (ns) mean±std",
         "N ops");
  std::cout << std::string(120, '-') << std::endl;

  for (const auto &r : results) {
    if (r.insert_ns_mean > 0) {
      printf(
          "%-30s | %10.0f ± %-9.0f | %10.0f ± %-9.0f | %10.0f ± %-9.0f | %8d\n",
          r.method.c_str(), r.insert_ns_mean, r.insert_ns_std, r.move_ns_mean,
          r.move_ns_std, r.delete_ns_mean, r.delete_ns_std, r.num_ops);
    } else {
      printf("%-30s | %22s | %22s | %22s | %8s\n", r.method.c_str(), "N/A",
             "N/A", "N/A", "N/A");
    }
  }
  std::cout << std::string(120, '=') << std::endl;
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

  DataStats stats = compute_data_stats(train_data);
  std::cout << "Data Stats: variance=" << stats.total_variance << ", range_x=["
            << stats.x_min << "," << stats.x_max << "]"
            << ", range_y=[" << stats.y_min << "," << stats.y_max << "]"
            << std::endl;

  std::vector<BenchmarkResult> static_results;
  std::vector<DynamicResult> dynamic_results;

  // Shared adaptive depth used by DT and RF trees
  int adaptive_depth =
      std::min(DT_MAX_DEPTH_CAP,
               std::max(DT_MIN_DEPTH_CAP,
                        (int)(2.0 * std::log2((double)train_data.size()))));

  // ============================================
  // ============================================
  {
    std::cout << "\n[1/5] Running FLANN C++ KNN (CV-tuned k)..." << std::endl;
    int best_k = knn_cv_select_k(train_data);
    std::cout << "      Selected k=" << best_k << " via "
              << select_cv_folds((int)train_data.size()) << "-fold CV"
              << std::endl;

    FlannKNN knn(best_k);

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

    std::string method_label =
        "FLANN C++ KNN (k=" + std::to_string(best_k) + ", CV)";
    static_results.push_back({method_label, accuracy, avg_us, train_ms});
  }

  // ============================================
  // STATIC 2: LibSVM — grid-search CV
  // ============================================
  {
    std::cout << "\n[2/5] Running LibSVM C++ (RBF, grid-search CV)..."
              << std::endl;
    double scale_gamma_ref = sklearn_scale_gamma(stats.total_variance);
    std::cout << "      Reference gamma (sklearn 'scale') = " << scale_gamma_ref
              << std::endl;

    std::pair<double, double> best_params =
        svm_cv_select_params(train_data, stats);
    double best_C = best_params.first;
    double best_gamma = best_params.second;
    std::cout << "      Selected C=" << best_C << ", gamma=" << best_gamma
              << " via " << select_cv_folds((int)train_data.size())
              << "-fold CV" << std::endl;

    SVMClassifier svm;

    auto train_start = std::chrono::high_resolution_clock::now();
    svm.fit_with_params(train_data, best_C, best_gamma);
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
        {"LibSVM C++ (RBF, CV-tuned)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // ============================================
  {
    std::cout << "\n[3/5] Running C++ Decision Tree (exhaustive splits, depth="
              << adaptive_depth << ")..." << std::endl;

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
        {"C++ Decision Tree (exhaustive)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // ============================================
  {
    std::cout << "\n[4/5] Running C++ Random Forest (" << RF_NUM_TREES
              << " trees, depth=" << adaptive_depth << ")..." << std::endl;

    RandomForestCpp rf(RF_NUM_TREES, adaptive_depth, 2, 42);

    auto train_start = std::chrono::high_resolution_clock::now();
    rf.fit(train_data);
    auto train_end = std::chrono::high_resolution_clock::now();
    double train_ms = std::chrono::duration_cast<std::chrono::microseconds>(
                          train_end - train_start)
                          .count() /
                      1000.0;

    int correct = 0;
    auto start = std::chrono::high_resolution_clock::now();
    for (const auto &pt : test_data) {
      int pred = rf.predict(std::get<0>(pt), std::get<1>(pt));
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
        {"C++ Random Forest (100 trees)", accuracy, avg_us, train_ms});
  }

  // ============================================
  // STATIC 5: Delaunay C++ (Ours)
  // ============================================
  {
    std::cout << "\n[5/5] Running Delaunay C++ (Ours)..." << std::endl;
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
      int pred = classifier.classify(std::get<0>(pt), std::get<1>(pt));
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

    // Report MULTI_PARTITIONED fallback rate for soundness transparency
    std::size_t fallback_count = classifier.get_multi_fallback_count();
    std::size_t query_count = classifier.get_total_query_count();
    if (query_count > 0) {
      double fallback_pct = 100.0 * static_cast<double>(fallback_count) /
                            static_cast<double>(query_count);
      std::cout << "  MULTI fallback rate: " << fallback_pct << "% ("
                << fallback_count << "/" << query_count << ")" << std::endl;
    }

    static_results.push_back(
        {"**Delaunay C++ (Ours)**", accuracy, avg_us, train_ms});
  }

  print_static_results(static_results, dataset_name);

  // ============================================
  // DYNAMIC BENCHMARKS
  // ============================================
  std::cout << "\n--- DYNAMIC BENCHMARKS ---" << std::endl;

  // Capped at test_data.size() so small datasets (e.g., wine with 36 test
  // points) still produce a valid benchmark.
  const int NUM_DYNAMIC_OPS =
      std::min(TARGET_DYNAMIC_OPS, (int)test_data.size());
  std::cout << "Measuring " << NUM_DYNAMIC_OPS << " operations per phase"
            << std::endl;

  // --------------------------------------------
  // Dynamic: Decision Tree (full rebuild per insertion)
  // --------------------------------------------
  {
    std::cout << "Running C++ Decision Tree (Rebuild)..." << std::endl;
    std::vector<std::tuple<float, float, int>> working_data = train_data;

    // mean and std. Previously only the sum was tracked.
    std::vector<double> rebuild_times_ns;
    rebuild_times_ns.reserve(NUM_DYNAMIC_OPS);

    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      working_data.push_back(test_data[i]);

      DecisionTreeCpp dt_new(adaptive_depth, 2);
      auto start = std::chrono::high_resolution_clock::now();
      dt_new.fit(working_data);
      auto end = std::chrono::high_resolution_clock::now();
      double elapsed_ns = static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count());
      rebuild_times_ns.push_back(elapsed_ns);
    }

    double mean = compute_mean(rebuild_times_ns);
    double stddev = compute_std(rebuild_times_ns, mean);

    // DT Rebuild has no distinct insert/move/delete phases — all three are
    // the same full-rebuild cost. Report the same mean±std across columns
    // for consistent visual alignment.
    DynamicResult res;
    res.method = "C++ Decision Tree (Rebuild)";
    res.insert_ns_mean = mean;
    res.insert_ns_std = stddev;
    res.move_ns_mean = mean;
    res.move_ns_std = stddev;
    res.delete_ns_mean = mean;
    res.delete_ns_std = stddev;
    res.num_ops = NUM_DYNAMIC_OPS;
    dynamic_results.push_back(res);
  }

  // --------------------------------------------
  // Dynamic: Delaunay — direct insert/move/delete instrumentation
  // --------------------------------------------
  {
    std::cout << "Running Delaunay C++ (Incremental)..." << std::endl;

    DelaunayClassifier classifier;
    classifier.set_output_dir("results");
    classifier.train(train_file, 3);

    // Adaptive move offset: fraction of data range
    double range_x = stats.x_max - stats.x_min;
    double range_y = stats.y_max - stats.y_min;
    double move_offset_x = DYNAMIC_MOVE_OFFSET_FRACTION * range_x;
    double move_offset_y = DYNAMIC_MOVE_OFFSET_FRACTION * range_y;

    std::vector<double> insert_times_ns;
    std::vector<double> move_times_ns;
    std::vector<double> delete_times_ns;
    insert_times_ns.reserve(NUM_DYNAMIC_OPS);
    move_times_ns.reserve(NUM_DYNAMIC_OPS);
    delete_times_ns.reserve(NUM_DYNAMIC_OPS);

    std::vector<std::tuple<float, float, int>> inserted_points;
    inserted_points.reserve(NUM_DYNAMIC_OPS);

    // --- INSERT phase ---
    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      float x = std::get<0>(test_data[i]);
      float y = std::get<1>(test_data[i]);
      int label = std::get<2>(test_data[i]);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.insert_point(x, y, label);
      auto end = std::chrono::high_resolution_clock::now();
      insert_times_ns.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()));
      inserted_points.push_back({x, y, label});
    }

    // --- MOVE phase ---
    std::vector<std::tuple<float, float, int>> moved_points;
    moved_points.reserve(NUM_DYNAMIC_OPS);
    for (int i = 0; i < NUM_DYNAMIC_OPS; i++) {
      float old_x = std::get<0>(inserted_points[i]);
      float old_y = std::get<1>(inserted_points[i]);
      float new_x = old_x + static_cast<float>(move_offset_x);
      float new_y = old_y + static_cast<float>(move_offset_y);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.move_point(old_x, old_y, new_x, new_y);
      auto end = std::chrono::high_resolution_clock::now();
      move_times_ns.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()));
      moved_points.push_back({new_x, new_y, std::get<2>(inserted_points[i])});
    }

    // --- DELETE phase ---
    for (int i = NUM_DYNAMIC_OPS - 1; i >= 0; i--) {
      float x = std::get<0>(moved_points[i]);
      float y = std::get<1>(moved_points[i]);

      auto start = std::chrono::high_resolution_clock::now();
      classifier.remove_point(x, y);
      auto end = std::chrono::high_resolution_clock::now();
      delete_times_ns.push_back(static_cast<double>(
          std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
              .count()));
    }

    double insert_mean = compute_mean(insert_times_ns);
    double move_mean = compute_mean(move_times_ns);
    double delete_mean = compute_mean(delete_times_ns);

    DynamicResult res;
    res.method = "**Delaunay C++ (O(1) Update)**";
    res.insert_ns_mean = insert_mean;
    res.insert_ns_std = compute_std(insert_times_ns, insert_mean);
    res.move_ns_mean = move_mean;
    res.move_ns_std = compute_std(move_times_ns, move_mean);
    res.delete_ns_mean = delete_mean;
    res.delete_ns_std = compute_std(delete_times_ns, delete_mean);
    res.num_ops = NUM_DYNAMIC_OPS;
    dynamic_results.push_back(res);
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

  // so external scripts (generate_figures.py) can consume the variance data.
  std::string dyn_output_file =
      "results/cpp_benchmark_dynamic_" + dataset_name + ".csv";
  std::ofstream dyn_csv(dyn_output_file);
  dyn_csv << "method,insert_ns_mean,insert_ns_std,move_ns_mean,move_ns_std,"
             "delete_ns_mean,delete_ns_std,num_ops\n";
  for (const auto &r : dynamic_results) {
    dyn_csv << r.method << "," << r.insert_ns_mean << "," << r.insert_ns_std
            << "," << r.move_ns_mean << "," << r.move_ns_std << ","
            << r.delete_ns_mean << "," << r.delete_ns_std << "," << r.num_ops
            << "\n";
  }
  dyn_csv.close();
  std::cout << "Dynamic results saved to: " << dyn_output_file << std::endl;

  return 0;
}