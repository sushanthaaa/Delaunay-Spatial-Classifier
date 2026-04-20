/**
 * @file DelaunayClassifier.cpp
 * @brief Implementation of the Delaunay Triangulation Classifier.
 *
 * Unified classification through 2D Buckets. All queries go through the
 * bucket grid — no separate SRR face-hint path.
 */

#include "../include/DelaunayClassifier.h"
#include <CGAL/centroid.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

// =============================================================================
// NAMED CONSTANTS
// =============================================================================
//
// All magic numerical values used by the classifier are declared here as
// named constants. Do NOT hardcode these values in function bodies — modify
// them here and they will propagate throughout the code.
//
// Each constant is documented with:
//   - Purpose (what it controls)
//   - Rationale (why this specific value)
//   - Sensitivity (whether the algorithm is robust to changes)
//   - Paper reference (which section or figure, if applicable)

namespace {

// -------- Geometric robustness constants --------

/// Relative padding (1%) added to the data bounding box before constructing
/// the 2D Buckets grid. Prevents boundary points from falling exactly on the
/// grid edge, which would cause floating-point ambiguity in bucket assignment.
/// RATIONALE: 1% is a standard margin in computational geometry; large enough
/// to absorb numerical error, small enough to not waste grid resolution.
/// SENSITIVITY: Robust in [0.001, 0.1]. Paper uses 0.01.
constexpr double BBOX_PADDING_FRACTION = 0.01;

/// Minimum absolute padding for degenerate cases where range_x or range_y
/// is near zero (e.g., all points collinear). Prevents division by zero
/// when computing step_x and step_y for the grid.
constexpr double MIN_BBOX_PADDING = 1e-6;

/// Multiplier for the Voronoi clipping bounding box, relative to the grid
/// extent. Unbounded Voronoi cells are extended to this bounding box before
/// clipping to individual buckets. Must be >1 to ensure all hull sites'
/// Voronoi cells fit inside.
/// RATIONALE: 2x grid extent ensures even the furthest hull site's Voronoi
/// cell (which can extend arbitrarily far) is captured.
constexpr double VORONOI_BBOX_MARGIN_MULTIPLIER = 2.0;

// -------- Dynamic benchmark constants (NOT model parameters) --------

/// Movement offset used by `run_dynamic_stress_test` and
/// `run_dynamic_visualization`, expressed as a fraction of the smaller data
/// dimension. This is a BENCHMARK parameter, not a model parameter — it
/// defines "small movement" for dynamic-update timing measurements.
/// The real classifier works for any movement magnitude.
/// RATIONALE: 1% represents a realistic "small perturbation" in streaming
/// data scenarios (e.g., sensor drift, GPS jitter). For large movements,
/// the move_point() function falls back to delete+insert.
constexpr double DYNAMIC_MOVE_OFFSET_FRACTION = 0.01;

// -------- Floating-point epsilon constants --------

/// Epsilon for detecting degenerate cases in cross-product tests (Case 2
/// half-plane classification). A value of 1e-15 is near machine epsilon for
/// double-precision floats (~2.22e-16) with safety margin.
constexpr double DEGENERATE_CROSS_EPSILON = 1e-15;

/// Epsilon for detecting degenerate (zero-length) segments in the
/// point-to-segment distance helper. 1e-20 squared is effectively "any
/// numerically representable non-zero length."
constexpr double DEGENERATE_SEGMENT_SQ_EPSILON = 1e-20;

/// Epsilon for line intersection determinant checks. Values below this
/// indicate near-parallel lines where intersection is numerically unreliable.
/// 1e-12 is a standard threshold for 2D geometric predicates.
constexpr double INTERSECTION_DET_EPSILON = 1e-12;

/// Epsilon for vector length normalization in direction computations.
/// Prevents division by zero when normalizing near-zero vectors.
constexpr double VECTOR_NORM_EPSILON = 1e-10;

} // anonymous namespace

// =============================================================================
// CONSTRUCTOR / DESTRUCTOR
// =============================================================================

DelaunayClassifier::DelaunayClassifier()
    : use_outlier_removal_(true), connectivity_multiplier_(3.0),
      output_dir_("") {}

DelaunayClassifier::~DelaunayClassifier() { grid_.clear(); }

// =============================================================================
// FILE I/O
// =============================================================================

std::vector<std::pair<Point, int>>
DelaunayClassifier::load_labeled_csv(const std::string &filepath) {
  std::vector<std::pair<Point, int>> points;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    throw std::runtime_error(
        "DelaunayClassifier::load_labeled_csv: Cannot open file: " + filepath);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string val;
    double x, y;
    int label;

    std::getline(ss, val, ',');
    x = std::stod(val);
    std::getline(ss, val, ',');
    y = std::stod(val);
    std::getline(ss, val, ',');
    label = std::stoi(val);

    points.push_back({Point(x, y), label});
  }
  return points;
}

std::vector<Point>
DelaunayClassifier::load_unlabeled_csv(const std::string &filepath) {
  std::vector<Point> points;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    throw std::runtime_error(
        "DelaunayClassifier::load_unlabeled_csv: Cannot open file: " +
        filepath);
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty())
      continue;
    std::stringstream ss(line);
    std::string val;
    double x, y;

    std::getline(ss, val, ',');
    x = std::stod(val);
    std::getline(ss, val, ',');
    y = std::stod(val);

    points.push_back(Point(x, y));
  }
  return points;
}

// =============================================================================
// (SRR face-hint grid removed — all classification via 2D Buckets)
// =============================================================================

// =============================================================================
// Adaptive outlier threshold with index-based arrays
// =============================================================================

double DelaunayClassifier::compute_median_edge_length(const Delaunay &temp_dt) {
  std::vector<double> edge_lengths;
  edge_lengths.reserve(temp_dt.number_of_vertices() * 3);

  for (auto e = temp_dt.finite_edges_begin(); e != temp_dt.finite_edges_end();
       ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);
    double dist_sq =
        CGAL::to_double(CGAL::squared_distance(v1->point(), v2->point()));
    edge_lengths.push_back(std::sqrt(dist_sq));
  }

  if (edge_lengths.empty())
    return 1.0;

  std::sort(edge_lengths.begin(), edge_lengths.end());
  return edge_lengths[edge_lengths.size() / 2];
}

std::vector<std::pair<Point, int>> DelaunayClassifier::remove_outliers(
    const std::vector<std::pair<Point, int>> &input_points, int k) {

  std::cout << "Phase 1: Detecting Outliers (Min Cluster Size k=" << k << ")..."
            << std::endl;

  if (input_points.size() < 3) {
    return input_points;
  }

  // Build temporary triangulation
  Delaunay temp_dt;
  temp_dt.insert(input_points.begin(), input_points.end());

  // Adaptive threshold based on median edge length
  double median_len = compute_median_edge_length(temp_dt);
  double threshold = median_len * connectivity_multiplier_;
  double threshold_sq = threshold * threshold;

  std::cout << "  Adaptive threshold: " << threshold
            << " (median edge=" << median_len
            << " × multiplier=" << connectivity_multiplier_ << ")" << std::endl;

  // Index-based data structures for O(1) access
  // Map CGAL vertices to indices
  int n_vertices = static_cast<int>(temp_dt.number_of_vertices());
  std::vector<Point> vertex_points(n_vertices);
  std::vector<int> vertex_labels(n_vertices);
  std::unordered_map<Vertex_handle, int> vh_to_idx;

  int idx = 0;
  for (auto v = temp_dt.finite_vertices_begin();
       v != temp_dt.finite_vertices_end(); ++v) {
    vh_to_idx[v] = idx;
    vertex_points[idx] = v->point();
    vertex_labels[idx] = v->info();
    idx++;
  }

  // Build adjacency list using indices
  std::vector<std::vector<int>> adj(n_vertices);

  for (auto e = temp_dt.finite_edges_begin(); e != temp_dt.finite_edges_end();
       ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);

    auto it1 = vh_to_idx.find(v1);
    auto it2 = vh_to_idx.find(v2);
    if (it1 == vh_to_idx.end() || it2 == vh_to_idx.end())
      continue;

    int i1 = it1->second;
    int i2 = it2->second;

    double dist_sq =
        CGAL::to_double(CGAL::squared_distance(v1->point(), v2->point()));

    // Only connect same-class vertices within adaptive threshold
    if (vertex_labels[i1] == vertex_labels[i2] && dist_sq < threshold_sq) {
      adj[i1].push_back(i2);
      adj[i2].push_back(i1);
    }
  }

  // Find connected components via DFS using index arrays
  std::vector<bool> visited(n_vertices, false);
  std::vector<std::pair<Point, int>> clean_points;
  int removed_count = 0;

  for (int start = 0; start < n_vertices; ++start) {
    if (visited[start])
      continue;

    // DFS from start
    std::vector<int> component;
    std::vector<int> stack = {start};
    visited[start] = true;

    while (!stack.empty()) {
      int curr = stack.back();
      stack.pop_back();
      component.push_back(curr);

      for (int neighbor : adj[curr]) {
        if (!visited[neighbor]) {
          visited[neighbor] = true;
          stack.push_back(neighbor);
        }
      }
    }

    // Keep if component >= k members
    if (static_cast<int>(component.size()) >= k) {
      for (int vi : component) {
        clean_points.push_back({vertex_points[vi], vertex_labels[vi]});
      }
    } else {
      removed_count += static_cast<int>(component.size());
    }
  }

  std::cout << "Phase 1 Complete: Removed " << removed_count << " outliers."
            << std::endl;
  return clean_points;
}

// =============================================================================
// CLASSIFICATION LOGIC — Half-plane decision boundary
// =============================================================================

/**
 * @brief Classify query point within a triangle using geometric decision
 * boundaries.
 *
 * Case 1 (all same class): return unanimous label
 * Case 2 (two distinct classes): half-plane test against the line connecting
 *         midpoints of the two cross-class edges
 * Case 3 (three distinct classes): nearest-vertex approximation of the
 *         Algorithm 4 centroid-to-midpoint partition. Exact for equilateral
 *         triangles; tight approximation for Delaunay meshes (which favor
 *         well-shaped triangles). See ablation A4 for accuracy impact.
 */
int DelaunayClassifier::classify_point_in_face(Face_handle f, Point p) const {
  int l0 = f->vertex(0)->info();
  int l1 = f->vertex(1)->info();
  int l2 = f->vertex(2)->info();

  Point p0 = f->vertex(0)->point();
  Point p1 = f->vertex(1)->point();
  Point p2 = f->vertex(2)->point();

  // CASE 1: All same class — O(1)
  if (l0 == l1 && l1 == l2)
    return l0;

  // Count distinct labels
  std::set<int> distinct_labels = {l0, l1, l2};

  if (distinct_labels.size() == 2) {
    // CASE 2: Two distinct classes
    // Identify the isolated vertex (the one with a unique label)
    // and the two vertices that share the majority label.
    //
    // The decision boundary is the line connecting the midpoints of the
    // two cross-class edges. We classify by checking which side of this
    // line the query point falls on.

    int isolated_idx = -1;
    int majority_label = -1;

    if (l0 == l1) {
      // v2 is isolated
      isolated_idx = 2;
      majority_label = l0;
    } else if (l0 == l2) {
      // v1 is isolated
      isolated_idx = 1;
      majority_label = l0;
    } else {
      // l1 == l2, v0 is isolated
      isolated_idx = 0;
      majority_label = l1;
    }

    int isolated_label = f->vertex(isolated_idx)->info();
    Point isolated_pt = f->vertex(isolated_idx)->point();

    // Find the two cross-class edges and their midpoints
    // The cross-class edges are the ones connecting the isolated vertex
    // to each of the majority vertices.
    Point maj_a, maj_b;
    if (isolated_idx == 0) {
      maj_a = p1;
      maj_b = p2;
    } else if (isolated_idx == 1) {
      maj_a = p0;
      maj_b = p2;
    } else {
      maj_a = p0;
      maj_b = p1;
    }

    // Midpoints of cross-class edges
    Point mid1((isolated_pt.x() + maj_a.x()) / 2.0,
               (isolated_pt.y() + maj_a.y()) / 2.0);
    Point mid2((isolated_pt.x() + maj_b.x()) / 2.0,
               (isolated_pt.y() + maj_b.y()) / 2.0);

    // The decision boundary is the line from mid1 to mid2.
    // We use the cross product to determine which side of this line
    // the query point p and the isolated vertex are on.
    //
    // cross = (mid2 - mid1) × (p - mid1)
    // If cross has the SAME sign as for the isolated vertex → isolated class
    // Otherwise → majority class

    double line_dx = mid2.x() - mid1.x();
    double line_dy = mid2.y() - mid1.y();

    double cross_query =
        line_dx * (p.y() - mid1.y()) - line_dy * (p.x() - mid1.x());
    double cross_isolated = line_dx * (isolated_pt.y() - mid1.y()) -
                            line_dy * (isolated_pt.x() - mid1.x());

    // Handle degenerate case where the boundary line is a point
    if (std::abs(cross_isolated) < DEGENERATE_CROSS_EPSILON) {
      // Degenerate: fall back to nearest vertex
      double d0 = CGAL::to_double(CGAL::squared_distance(p, p0));
      double d1 = CGAL::to_double(CGAL::squared_distance(p, p1));
      double d2 = CGAL::to_double(CGAL::squared_distance(p, p2));
      if (d0 <= d1 && d0 <= d2)
        return l0;
      if (d1 <= d0 && d1 <= d2)
        return l1;
      return l2;
    }

    // Same side as isolated vertex → isolated class
    // Different side → majority class
    if ((cross_query > 0) == (cross_isolated > 0)) {
      return isolated_label;
    } else {
      return majority_label;
    }
  }

  // CASE 3: Three distinct classes.
  //
  // We approximate this partition using nearest-vertex assignment. The two
  // partitions are:
  //   - EXACTLY equal for equilateral triangles (centroid = circumcenter)
  //   - APPROXIMATELY equal for near-equilateral triangles
  //   - DIFFERENT for highly obtuse or degenerate triangles
  //
  // Since Delaunay triangulation maximizes the minimum angle across all
  // triangulations of the same point set, its triangles are typically
  // well-shaped (close to equilateral), making nearest-vertex a tight
  // approximation in practice. Ablation A4 quantifies the accuracy gap
  // (~1% on typical datasets) between this approximation and the exact
  // centroid-to-midpoint partition.
  double d0 = CGAL::to_double(CGAL::squared_distance(p, p0));
  double d1 = CGAL::to_double(CGAL::squared_distance(p, p1));
  double d2 = CGAL::to_double(CGAL::squared_distance(p, p2));

  if (d0 <= d1 && d0 <= d2)
    return l0;
  if (d1 <= d0 && d1 <= d2)
    return l1;
  return l2;
}

// =============================================================================
// Point-to-segment distance for outside-hull classification
// =============================================================================

double DelaunayClassifier::squared_distance_point_to_segment(const Point &p,
                                                             const Point &a,
                                                             const Point &b) {
  double dx = b.x() - a.x();
  double dy = b.y() - a.y();
  double len_sq = dx * dx + dy * dy;

  if (len_sq < DEGENERATE_SEGMENT_SQ_EPSILON) {
    // Degenerate segment (a == b)
    return CGAL::to_double(CGAL::squared_distance(p, a));
  }

  // Project p onto line (a, b), clamped to [0, 1]
  double t = ((p.x() - a.x()) * dx + (p.y() - a.y()) * dy) / len_sq;
  t = std::max(0.0, std::min(1.0, t));

  double proj_x = a.x() + t * dx;
  double proj_y = a.y() + t * dy;
  double ddx = p.x() - proj_x;
  double ddy = p.y() - proj_y;
  return ddx * ddx + ddy * ddy;
}

// =============================================================================
// TRAIN
// =============================================================================

void DelaunayClassifier::train(const std::string &train_file, int outlier_k) {
  auto raw_points = load_labeled_csv(train_file);

  std::vector<std::pair<Point, int>> clean_points;
  if (use_outlier_removal_) {
    clean_points = remove_outliers(raw_points, outlier_k);
  } else {
    clean_points = raw_points;
  }

  // Export clean points if output directory is set
  if (!output_dir_.empty()) {
    std::ofstream out(output_dir_ + "/clean_points.csv");
    for (const auto &p : clean_points) {
      out << p.first.x() << "," << p.first.y() << "," << p.second << "\n";
    }
    out.close();
  }

  // Build Delaunay triangulation — O(n log n)
  dt_.clear();
  dt_.insert(clean_points.begin(), clean_points.end());
  std::cout << "Phase 2 Complete: Delaunay Mesh Built ("
            << dt_.number_of_vertices() << " vertices)." << std::endl;

  // Build 2D Buckets grid and validate
  build_2d_buckets();
  validate_all_buckets();
}

// =============================================================================
// PREDICT (Batch Benchmark)
// =============================================================================

void DelaunayClassifier::predict_benchmark(const std::string &test_file,
                                           const std::string &output_file) {
  // Auto-detect format: peek at first line to count columns
  std::ifstream peek_file(test_file);
  std::string first_line;
  std::getline(peek_file, first_line);
  peek_file.close();

  int num_commas = 0;
  for (char c : first_line) {
    if (c == ',')
      num_commas++;
  }
  bool has_labels = (num_commas >= 2); // x,y,label = 2 commas

  // Load data
  std::vector<Point> test_points;
  std::vector<int> true_labels;

  if (has_labels) {
    auto labeled = load_labeled_csv(test_file);
    test_points.reserve(labeled.size());
    true_labels.reserve(labeled.size());
    for (const auto &p : labeled) {
      test_points.push_back(p.first);
      true_labels.push_back(p.second);
    }
  } else {
    test_points = load_unlabeled_csv(test_file);
  }

  std::cout << "Starting Benchmark (2D Buckets Classification)..." << std::endl;

  std::vector<int> results;
  results.reserve(test_points.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto &p : test_points) {
    int pred = classify(p.x(), p.y());
    results.push_back(pred);
  }

  auto end = std::chrono::high_resolution_clock::now();
  // inference (typically 2-8 ns per point) is not truncated to 0.
  // The old integer-microsecond formula rounded 110 ns (Wine total) to
  // 0 us, then 0 / 36 = 0, producing the misleading "0 us" output.
  auto duration_ns =
      std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  double avg_time = test_points.empty()
                        ? 0.0
                        : static_cast<double>(duration_ns.count()) / 1000.0 /
                              static_cast<double>(test_points.size());

  int total = static_cast<int>(results.size());

  std::cout << "\n=== Classification Results ===" << std::endl;
  std::cout << "Total Points: " << total << std::endl;
  std::cout << "Avg Time Per Point:   " << avg_time << " us" << std::endl;

  if (has_labels && !true_labels.empty()) {
    int correct = 0;
    for (int i = 0; i < total; ++i) {
      if (results[i] == true_labels[i])
        correct++;
    }
    double accuracy = (double)correct / total * 100.0;
    std::cout << "Accuracy:             " << accuracy << "% (" << correct << "/"
              << total << ")" << std::endl;
  }

  // Report MULTI_PARTITIONED fallback rate for soundness transparency
  std::size_t fallback_count = get_multi_fallback_count();
  std::size_t query_count = get_total_query_count();
  if (query_count > 0) {
    double fallback_pct = 100.0 * static_cast<double>(fallback_count) /
                          static_cast<double>(query_count);
    std::cout << "MULTI fallback rate:  " << fallback_pct << "% ("
              << fallback_count << "/" << query_count << ")" << std::endl;
  }

  std::cout << "================================================" << std::endl;

  // Save predictions
  std::ofstream out(output_file);
  for (int p : results)
    out << p << "\n";
  out.close();
}

// =============================================================================
// UNIFIED CLASSIFICATION — 2D Buckets only
// =============================================================================

/**
 * @brief Classify a single query point via 2D Buckets (O(1) expected).
 *
 * ALGORITHM:
 *   1. Compute bucket index by arithmetic (O(1))
 *   2. Call bucket's classify_point() which handles HOMO/BI/MULTI
 *
 * OUT-OF-HULL QUERIES:
 *   For query points outside the training data's bounding box, the bucket
 *   index is clamped to the nearest edge bucket via get_bucket_index(). The
 *   query is then classified according to that edge bucket's metadata.
 *   This is equivalent to extrapolating the nearest training region outward,
 *   which is the standard behavior for spatial classifiers. No separate
 *   "extended boundary" computation is performed.
 *
 * SOUNDNESS:
 *   validate_all_buckets() guarantees every bucket returns a valid class
 *   label >= 0, so classify() never fails on trained models. Rare fallback
 *   cases (MULTI_PARTITIONED polygon gaps) are counted and exposed via
 *   get_multi_fallback_count().
 */
int DelaunayClassifier::classify(double x, double y) const {
  total_queries_++;

  if (grid_.buckets.empty()) {
    // Model not trained yet — return nearest vertex as safety
    Vertex_handle v = dt_.nearest_vertex(Point(x, y));
    return (v != Vertex_handle()) ? v->info() : -1;
  }

  int bucket_idx = grid_.get_bucket_index(x, y);
  bool fallback_fired = false;
  int result = grid_.buckets[bucket_idx].classify_point(x, y, &fallback_fired);
  if (fallback_fired) {
    multi_fallback_count_++;
  }
  return result;
}

/**
 * @brief Classify WITHOUT the 2D Buckets grid (for ablation study A2).
 * Uses raw DT locate + face-based decision boundary.
 */
int DelaunayClassifier::classify_no_grid(double x, double y) const {
  Point p(x, y);
  Face_handle f = dt_.locate(p);
  if (!dt_.is_infinite(f)) {
    return classify_point_in_face(f, p);
  }
  Vertex_handle v = dt_.nearest_vertex(p);
  return v->info();
}

/**
 * @brief Nearest-vertex classification only (1-NN, for ablation study A4).
 */
int DelaunayClassifier::classify_nearest_vertex(double x, double y) const {
  Point p(x, y);
  Vertex_handle v = dt_.nearest_vertex(p);
  return v->info();
}

// =============================================================================
// DYNAMIC OPERATIONS
// =============================================================================

/**
 * @brief Update local cells (3×3 neighborhood) after a point change.
 */
void DelaunayClassifier::update_local_cells(double x, double y) {
  if (grid_.rows == 0 || grid_.cols == 0)
    return;

  int c = static_cast<int>((x - grid_.min_x) / grid_.step_x);
  int r = static_cast<int>((y - grid_.min_y) / grid_.step_y);
  c = std::max(0, std::min(c, grid_.cols - 1));
  r = std::max(0, std::min(r, grid_.rows - 1));

  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      int nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < grid_.rows && nc >= 0 && nc < grid_.cols) {
        int idx = nr * grid_.cols + nc;
        rebuild_bucket(idx);
      }
    }
  }
}

void DelaunayClassifier::insert_point(double x, double y, int label) {
  Point p(x, y);
  Vertex_handle v = dt_.insert(p);
  v->info() = label;
  update_local_cells(x, y);
}

void DelaunayClassifier::remove_point(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt_.nearest_vertex(p);
  if (v != Vertex_handle()) {
    double vx = v->point().x();
    double vy = v->point().y();
    dt_.remove(v);
    update_local_cells(vx, vy);
  }
}

/**
 * @brief Algorithm 3: Move a point with same-star polygon check.
 * Case A: New position within same star polygon → local flip via
 * move_if_no_collision Case B: Different polygon → delete + re-insert
 */
void DelaunayClassifier::move_point(double old_x, double old_y, double new_x,
                                    double new_y) {
  Point old_p(old_x, old_y);
  Point new_p(new_x, new_y);

  Vertex_handle v = dt_.nearest_vertex(old_p);
  if (v == Vertex_handle())
    return;

  int label = v->info();

  // Check if new position is within the same star polygon
  bool same_star = false;
  Face_handle f_new = dt_.locate(new_p);

  if (!dt_.is_infinite(f_new)) {
    auto circ = dt_.incident_faces(v);
    auto start_circ = circ;
    if (circ != Face_handle()) {
      do {
        if (!dt_.is_infinite(circ) && circ == f_new) {
          same_star = true;
          break;
        }
        ++circ;
      } while (circ != start_circ);
    }
  }

  if (same_star) {
    // CASE A: Short movement within same star polygon
    Vertex_handle moved = dt_.move_if_no_collision(v, new_p);
    if (moved != v) {
      // Collision: fall back to delete + re-insert
      dt_.remove(v);
      Vertex_handle v_new = dt_.insert(new_p);
      v_new->info() = label;
    }
  } else {
    // CASE B: Long movement to different polygon
    dt_.remove(v);
    Vertex_handle v_new = dt_.insert(new_p);
    v_new->info() = label;
  }

  // Update affected cells for both old and new positions
  update_local_cells(old_x, old_y);
  update_local_cells(new_x, new_y);
}

// =============================================================================
// DYNAMIC STRESS TEST
// =============================================================================

void DelaunayClassifier::run_dynamic_stress_test(const std::string &stream_file,
                                                 const std::string &log_file) {
  auto stream_points = load_labeled_csv(stream_file);
  std::ofstream log(log_file);
  log << "operation,time_ns\n";

  std::cout << "Running Dynamic Algorithms 1, 2, 3 Stress Test..." << std::endl;

  // Adaptive movement offset based on data range
  double range_x = grid_.max_x - grid_.min_x;
  double range_y = grid_.max_y - grid_.min_y;
  double move_offset =
      DYNAMIC_MOVE_OFFSET_FRACTION * std::min(range_x, range_y);

  Vertex_handle hint_vertex = dt_.finite_vertices_begin();
  std::vector<std::pair<double, double>> inserted_coords;

  // --- INSERTION PHASE ---
  for (const auto &entry : stream_points) {
    double x = entry.first.x();
    double y = entry.first.y();
    int label = entry.second;

    auto start = std::chrono::high_resolution_clock::now();
    insert_point(x, y, label);
    auto end = std::chrono::high_resolution_clock::now();

    inserted_coords.push_back({x, y});
    log << "insert,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

  // --- MOVEMENT PHASE ---
  for (auto &coord : inserted_coords) {
    double old_x = coord.first;
    double old_y = coord.second;
    double new_x = old_x + move_offset;
    double new_y = old_y + move_offset;

    auto start = std::chrono::high_resolution_clock::now();
    move_point(old_x, old_y, new_x, new_y);
    auto end = std::chrono::high_resolution_clock::now();

    // Update stored coordinates for deletion phase
    coord.first = new_x;
    coord.second = new_y;

    log << "move,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

  // --- DELETION PHASE ---
  for (int i = static_cast<int>(inserted_coords.size()) - 1; i >= 0; --i) {
    double x = inserted_coords[i].first;
    double y = inserted_coords[i].second;

    auto start = std::chrono::high_resolution_clock::now();
    remove_point(x, y);
    auto end = std::chrono::high_resolution_clock::now();

    log << "delete,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

  log.close();
  std::cout << "Dynamic Benchmark Logs saved to " << log_file << std::endl;
}

// =============================================================================
// VISUALIZATION EXPORT
// =============================================================================

void DelaunayClassifier::export_visualization(const std::string &mesh_file,
                                              const std::string &boundary_file,
                                              const std::string &points_file) {
  std::ofstream triFile(mesh_file);
  for (auto e = dt_.finite_edges_begin(); e != dt_.finite_edges_end(); ++e) {
    auto s = dt_.segment(e);
    triFile << s.source().x() << "," << s.source().y() << "," << s.target().x()
            << "," << s.target().y() << "\n";
  }
  triFile.close();

  std::ofstream boundFile(boundary_file);
  for (auto f = dt_.finite_faces_begin(); f != dt_.finite_faces_end(); ++f) {
    auto v0 = f->vertex(0);
    auto v1 = f->vertex(1);
    auto v2 = f->vertex(2);

    int l0 = v0->info();
    int l1 = v1->info();
    int l2 = v2->info();

    Point m01 = CGAL::midpoint(v0->point(), v1->point());
    Point m12 = CGAL::midpoint(v1->point(), v2->point());
    Point m20 = CGAL::midpoint(v2->point(), v0->point());

    if (l0 != l1 && l1 != l2 && l0 != l2) {
      Point c = CGAL::centroid(v0->point(), v1->point(), v2->point());
      boundFile << c.x() << "," << c.y() << "," << m01.x() << "," << m01.y()
                << "\n";
      boundFile << c.x() << "," << c.y() << "," << m12.x() << "," << m12.y()
                << "\n";
      boundFile << c.x() << "," << c.y() << "," << m20.x() << "," << m20.y()
                << "\n";
    } else if (l0 != l1 || l1 != l2) {
      std::vector<Point> actives;
      if (l0 != l1)
        actives.push_back(m01);
      if (l1 != l2)
        actives.push_back(m12);
      if (l2 != l0)
        actives.push_back(m20);

      if (actives.size() == 2)
        boundFile << actives[0].x() << "," << actives[0].y() << ","
                  << actives[1].x() << "," << actives[1].y() << "\n";
    }
  }
  boundFile.close();

  if (!points_file.empty()) {
    std::ofstream pFile(points_file);
    for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
         ++v) {
      pFile << v->point().x() << "," << v->point().y() << "," << v->info()
            << "\n";
    }
    pFile.close();
  }
}

void DelaunayClassifier::run_dynamic_visualization(
    const std::string &stream_file, const std::string &out_dir) {
  auto stream_points = load_labeled_csv(stream_file);
  std::vector<std::pair<double, double>> inserted_coords;

  std::cout << "Generating Dynamic Visualization Snapshots..." << std::endl;

  // Adaptive offset
  double range_x = grid_.max_x - grid_.min_x;
  double range_y = grid_.max_y - grid_.min_y;
  double move_offset =
      DYNAMIC_MOVE_OFFSET_FRACTION * std::min(range_x, range_y);

  // 1. INSERTION
  for (const auto &entry : stream_points) {
    insert_point(entry.first.x(), entry.first.y(), entry.second);
    inserted_coords.push_back({entry.first.x(), entry.first.y()});
  }
  export_visualization(out_dir + "/dynamic_1_inserted_triangles.csv",
                       out_dir + "/dynamic_1_inserted_boundaries.csv",
                       out_dir + "/dynamic_1_inserted_points.csv");
  std::cout << "   - Snapshot 1: Insertion Complete" << std::endl;

  // 2. MOVEMENT
  for (auto &coord : inserted_coords) {
    move_point(coord.first, coord.second, coord.first + move_offset,
               coord.second + move_offset);
    coord.first += move_offset;
    coord.second += move_offset;
  }
  export_visualization(out_dir + "/dynamic_2_moved_triangles.csv",
                       out_dir + "/dynamic_2_moved_boundaries.csv",
                       out_dir + "/dynamic_2_moved_points.csv");
  std::cout << "   - Snapshot 2: Movement Complete" << std::endl;

  // 3. DELETION
  for (int i = static_cast<int>(inserted_coords.size()) - 1; i >= 0; --i) {
    remove_point(inserted_coords[i].first, inserted_coords[i].second);
  }
  export_visualization(out_dir + "/dynamic_3_deleted_triangles.csv",
                       out_dir + "/dynamic_3_deleted_boundaries.csv",
                       out_dir + "/dynamic_3_deleted_points.csv");
  std::cout << "   - Snapshot 3: Deletion Complete" << std::endl;
}

// =============================================================================
// 2D BUCKETS IMPLEMENTATION
// =============================================================================

bool Bucket2D::point_in_polygon(double x, double y,
                                const std::vector<Point> &polygon) {
  if (polygon.size() < 3)
    return false;

  bool inside = false;
  size_t n = polygon.size();

  for (size_t i = 0, j = n - 1; i < n; j = i++) {
    double xi = polygon[i].x(), yi = polygon[i].y();
    double xj = polygon[j].x(), yj = polygon[j].y();

    if (((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi)) {
      inside = !inside;
    }
  }

  return inside;
}

/**
 * @brief O(1) classification using bucket type metadata.
 *
 * Case A (HOMOGENEOUS):       return dominant_class — O(1)
 * Case B (BIPARTITIONED):     half-plane test — O(1)
 * Case C (MULTI_PARTITIONED): point-in-polygon — O(k), k bounded
 */
int Bucket2D::classify_point(double x, double y, bool *fallback_fired) const {
  switch (type) {
  case BUCKET_HOMOGENEOUS:
    return dominant_class;

  case BUCKET_BIPARTITIONED: {
    // O(1) half-plane test
    double dot = boundary_nx * x + boundary_ny * y + boundary_d;
    return (dot >= 0) ? class_positive : class_negative;
  }

  case BUCKET_MULTI_PARTITIONED: {
    // Check each polygon region — O(k) where k = O(1) under bounded density
    LL_Polygon *poly = polygons;
    while (poly != nullptr) {
      if (poly->inside_label == 1 && poly->vertices.size() >= 3) {
        if (point_in_polygon(x, y, poly->vertices)) {
          return poly->class_label;
        }
      }
      poly = poly->next;
    }
    // Fallback: no polygon contained the query point. This happens due to
    // floating-point gaps between clipped Voronoi polygons. We return the
    // dominant class as a soundness safety net. Caller is notified via
    // fallback_fired so the frequency can be reported alongside accuracy.
    if (fallback_fired != nullptr) {
      *fallback_fired = true;
    }
    return dominant_class;
  }

  default:
    return dominant_class;
  }
}

void Bucket2D::clear() {
  LL_Vertex *v = vertices;
  while (v != nullptr) {
    LL_Vertex *next = v->next;
    delete v;
    v = next;
  }
  vertices = nullptr;
  vertex_count = 0;

  LL_Edge *e = edges;
  while (e != nullptr) {
    LL_Edge *next = e->next;
    delete e;
    e = next;
  }
  edges = nullptr;
  edge_count = 0;

  LL_GridEdge *ge = grid_edges;
  while (ge != nullptr) {
    LL_GridEdge *next = ge->next;
    delete ge;
    ge = next;
  }
  grid_edges = nullptr;
  grid_edge_count = 0;

  LL_Polygon *p = polygons;
  while (p != nullptr) {
    LL_Polygon *next = p->next;
    delete p;
    p = next;
  }
  polygons = nullptr;
  polygon_count = 0;

  num_classes = 0;
  dominant_class = -1;
  type = BUCKET_HOMOGENEOUS;
  class_positive = class_negative = -1;
  boundary_nx = boundary_ny = boundary_d = 0;
}

int SRR_Grid_2D::get_bucket_index(double x, double y) const {
  int c = static_cast<int>((x - min_x) / step_x);
  int r = static_cast<int>((y - min_y) / step_y);
  c = std::max(0, std::min(c, cols - 1));
  r = std::max(0, std::min(r, rows - 1));
  return r * cols + c;
}

void SRR_Grid_2D::print_statistics() const {
  std::cout << "=== 2D Buckets Grid Statistics ===" << std::endl;
  std::cout << "Grid size: " << rows << " x " << cols << " = " << (rows * cols)
            << " buckets" << std::endl;
  std::cout << "Homogeneous (Case A):     " << single_class_buckets
            << std::endl;
  std::cout << "Bipartitioned (Case B):   " << bipartitioned_buckets
            << std::endl;
  std::cout << "Multi-partitioned (Case C): " << multi_class_buckets
            << std::endl;
  std::cout << "Total polygon regions:    " << total_polygons << std::endl;
  std::cout << "==================================" << std::endl;
}

// --- Geometric helpers ---

static bool segment_intersects_bucket(double x1, double y1, double x2,
                                      double y2, double bmin_x, double bmin_y,
                                      double bmax_x, double bmax_y) {
  auto inside = [&](double x, double y) {
    return x >= bmin_x && x <= bmax_x && y >= bmin_y && y <= bmax_y;
  };
  if (inside(x1, y1) || inside(x2, y2))
    return true;

  auto intersects_edge = [&](double ex1, double ey1, double ex2, double ey2) {
    double dx = x2 - x1, dy = y2 - y1;
    double edx = ex2 - ex1, edy = ey2 - ey1;
    double denom = dx * edy - dy * edx;
    if (std::abs(denom) < INTERSECTION_DET_EPSILON)
      return false;
    double t = ((ex1 - x1) * edy - (ey1 - y1) * edx) / denom;
    double s = ((ex1 - x1) * dy - (ey1 - y1) * dx) / denom;
    return t >= 0 && t <= 1 && s >= 0 && s <= 1;
  };

  return intersects_edge(bmin_x, bmin_y, bmax_x, bmin_y) ||
         intersects_edge(bmax_x, bmin_y, bmax_x, bmax_y) ||
         intersects_edge(bmax_x, bmax_y, bmin_x, bmax_y) ||
         intersects_edge(bmin_x, bmax_y, bmin_x, bmin_y);
}

static bool compute_intersection(double x1, double y1, double x2, double y2,
                                 double x3, double y3, double x4, double y4,
                                 double &ix, double &iy) {
  double dx1 = x2 - x1, dy1 = y2 - y1;
  double dx2 = x4 - x3, dy2 = y4 - y3;
  double denom = dx1 * dy2 - dy1 * dx2;
  if (std::abs(denom) < INTERSECTION_DET_EPSILON)
    return false;

  double t = ((x3 - x1) * dy2 - (y3 - y1) * dx2) / denom;
  double s = ((x3 - x1) * dy1 - (y3 - y1) * dx1) / denom;

  if (t >= 0 && t <= 1 && s >= 0 && s <= 1) {
    ix = x1 + t * dx1;
    iy = y1 + t * dy1;
    return true;
  }
  return false;
}

static std::vector<Point>
clip_polygon_to_bucket(const std::vector<Point> &polygon, double bmin_x,
                       double bmin_y, double bmax_x, double bmax_y) {
  if (polygon.empty())
    return {};

  std::vector<Point> result = polygon;

  auto clip_edge = [](const std::vector<Point> &input, double ex1, double ey1,
                      double ex2, double ey2) {
    std::vector<Point> output;
    if (input.empty())
      return output;

    auto is_inside = [&](const Point &p) {
      return (ex2 - ex1) * (p.y() - ey1) - (ey2 - ey1) * (p.x() - ex1) >= 0;
    };

    Point prev = input.back();
    bool prev_inside = is_inside(prev);

    for (const Point &curr : input) {
      bool curr_inside = is_inside(curr);

      if (curr_inside) {
        if (!prev_inside) {
          double ix, iy;
          if (compute_intersection(prev.x(), prev.y(), curr.x(), curr.y(), ex1,
                                   ey1, ex2, ey2, ix, iy)) {
            output.push_back(Point(ix, iy));
          }
        }
        output.push_back(curr);
      } else if (prev_inside) {
        double ix, iy;
        if (compute_intersection(prev.x(), prev.y(), curr.x(), curr.y(), ex1,
                                 ey1, ex2, ey2, ix, iy)) {
          output.push_back(Point(ix, iy));
        }
      }

      prev = curr;
      prev_inside = curr_inside;
    }
    return output;
  };

  result = clip_edge(result, bmin_x, bmin_y, bmin_x, bmax_y);
  result = clip_edge(result, bmin_x, bmax_y, bmax_x, bmax_y);
  result = clip_edge(result, bmax_x, bmax_y, bmax_x, bmin_y);
  result = clip_edge(result, bmax_x, bmin_y, bmin_x, bmin_y);

  return result;
}

static double compute_polygon_area(const std::vector<Point> &poly) {
  double area = 0;
  size_t n = poly.size();
  for (size_t i = 0; i < n; ++i) {
    size_t j = (i + 1) % n;
    area += poly[i].x() * poly[j].y();
    area -= poly[j].x() * poly[i].y();
  }
  return std::abs(area) / 2.0;
}

// =============================================================================
// Rebuild a single bucket's linked lists
// =============================================================================

void DelaunayClassifier::rebuild_bucket(int bucket_idx) {
  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(grid_.buckets.size()))
    return;

  Bucket2D &bucket = grid_.buckets[bucket_idx];
  bucket.clear();

  // Rebuild LL_V: find all vertices within this cell
  int vid = 0;
  for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
       ++v) {
    double vx = v->point().x();
    double vy = v->point().y();
    if (vx >= bucket.min_x && vx <= bucket.max_x && vy >= bucket.min_y &&
        vy <= bucket.max_y) {
      LL_Vertex *new_v = new LL_Vertex(v->point(), v->info(), vid, v);
      new_v->next = bucket.vertices;
      bucket.vertices = new_v;
      bucket.vertex_count++;
    }
    vid++;
  }

  // Rebuild LL_E: find edges intersecting this cell
  int eid = 0;
  for (auto e = dt_.finite_edges_begin(); e != dt_.finite_edges_end(); ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);
    Point ep1 = v1->point(), ep2 = v2->point();

    // Quick bounding box check first
    double ex_min = std::min(ep1.x(), ep2.x());
    double ex_max = std::max(ep1.x(), ep2.x());
    double ey_min = std::min(ep1.y(), ep2.y());
    double ey_max = std::max(ep1.y(), ep2.y());

    if (ex_max < bucket.min_x || ex_min > bucket.max_x ||
        ey_max < bucket.min_y || ey_min > bucket.max_y) {
      eid++;
      continue;
    }

    if (segment_intersects_bucket(ep1.x(), ep1.y(), ep2.x(), ep2.y(),
                                  bucket.min_x, bucket.min_y, bucket.max_x,
                                  bucket.max_y)) {
      LL_Edge *new_e = new LL_Edge(ep1, ep2, v1->info(), v2->info(), eid);
      new_e->next = bucket.edges;
      bucket.edges = new_e;
      bucket.edge_count++;
    }
    eid++;
  }

  // Determine bucket classification type
  std::set<int> classes;
  LL_Vertex *vit = bucket.vertices;
  while (vit) {
    classes.insert(vit->class_label);
    vit = vit->next;
  }

  // Also sample center
  if (classes.empty()) {
    double cx = (bucket.min_x + bucket.max_x) / 2;
    double cy = (bucket.min_y + bucket.max_y) / 2;
    Vertex_handle nv = dt_.nearest_vertex(Point(cx, cy));
    if (nv != Vertex_handle()) {
      classes.insert(nv->info());
    }
  }

  bucket.num_classes = static_cast<int>(classes.size());

  if (bucket.num_classes <= 1) {
    bucket.type = BUCKET_HOMOGENEOUS;
    bucket.dominant_class = classes.empty() ? 0 : *classes.begin();
  } else if (bucket.num_classes == 2) {
    bucket.type = BUCKET_BIPARTITIONED;

    // Find a boundary edge to compute half-plane
    LL_Edge *be = bucket.edges;
    bool found_boundary = false;
    while (be) {
      if (be->is_boundary) {
        // Compute half-plane from midpoint of boundary edge
        double mx = (be->p1.x() + be->p2.x()) / 2;
        double my = (be->p1.y() + be->p2.y()) / 2;
        double edge_dx = be->p2.x() - be->p1.x();
        double edge_dy = be->p2.y() - be->p1.y();

        // Normal to boundary edge
        bucket.boundary_nx = -edge_dy;
        bucket.boundary_ny = edge_dx;
        bucket.boundary_d =
            -(bucket.boundary_nx * mx + bucket.boundary_ny * my);

        // Determine which class is on which side
        double dot1 = bucket.boundary_nx * be->p1.x() +
                      bucket.boundary_ny * be->p1.y() + bucket.boundary_d;
        bucket.class_positive = (dot1 >= 0) ? be->class1 : be->class2;
        bucket.class_negative = (dot1 >= 0) ? be->class2 : be->class1;

        // Dominant class: sample center
        double cx = (bucket.min_x + bucket.max_x) / 2;
        double cy = (bucket.min_y + bucket.max_y) / 2;
        double dot_center = bucket.boundary_nx * cx + bucket.boundary_ny * cy +
                            bucket.boundary_d;
        bucket.dominant_class =
            (dot_center >= 0) ? bucket.class_positive : bucket.class_negative;

        found_boundary = true;
        break;
      }
      be = be->next;
    }

    if (!found_boundary) {
      // No boundary edge found, treat as homogeneous
      bucket.type = BUCKET_HOMOGENEOUS;
      bucket.dominant_class = classes.empty() ? 0 : *classes.begin();
    }
  } else {
    bucket.type = BUCKET_MULTI_PARTITIONED;
    // Dominant = most frequent class
    std::map<int, int> counts;
    LL_Vertex *cv = bucket.vertices;
    while (cv) {
      counts[cv->class_label]++;
      cv = cv->next;
    }
    int max_count = 0;
    for (auto &kv : counts) {
      if (kv.second > max_count) {
        max_count = kv.second;
        bucket.dominant_class = kv.first;
      }
    }
  }
}

// =============================================================================
// BUILD 2D BUCKETS
// =============================================================================

void DelaunayClassifier::build_2d_buckets() {
  if (dt_.number_of_vertices() == 0)
    return;

  std::cout << "Building 2D Buckets with full linked list structures..."
            << std::endl;

  int n = static_cast<int>(dt_.number_of_vertices());
  int k = std::max(2, static_cast<int>(std::ceil(std::sqrt(n))));

  // Compute bounding box with relative padding
  double data_min_x = 1e18, data_max_x = -1e18;
  double data_min_y = 1e18, data_max_y = -1e18;
  for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
       ++v) {
    double vx = v->point().x(), vy = v->point().y();
    data_min_x = std::min(data_min_x, vx);
    data_max_x = std::max(data_max_x, vx);
    data_min_y = std::min(data_min_y, vy);
    data_max_y = std::max(data_max_y, vy);
  }

  double range_x = data_max_x - data_min_x;
  double range_y = data_max_y - data_min_y;
  double pad_x = std::max(range_x * BBOX_PADDING_FRACTION, MIN_BBOX_PADDING);
  double pad_y = std::max(range_y * BBOX_PADDING_FRACTION, MIN_BBOX_PADDING);

  grid_.clear();
  grid_.rows = k;
  grid_.cols = k;
  grid_.min_x = data_min_x - pad_x;
  grid_.max_x = data_max_x + pad_x;
  grid_.min_y = data_min_y - pad_y;
  grid_.max_y = data_max_y + pad_y;
  grid_.step_x = (grid_.max_x - grid_.min_x) / k;
  grid_.step_y = (grid_.max_y - grid_.min_y) / k;
  grid_.single_class_buckets = 0;
  grid_.multi_class_buckets = 0;
  grid_.bipartitioned_buckets = 0;
  grid_.total_polygons = 0;

  grid_.buckets.resize(k * k);

  // Initialize bucket boundaries
  for (int r = 0; r < k; ++r) {
    for (int c = 0; c < k; ++c) {
      int idx = r * k + c;
      Bucket2D &bucket = grid_.buckets[idx];
      bucket.row = r;
      bucket.col = c;
      bucket.min_x = grid_.min_x + c * grid_.step_x;
      bucket.max_x = bucket.min_x + grid_.step_x;
      bucket.min_y = grid_.min_y + r * grid_.step_y;
      bucket.max_y = bucket.min_y + grid_.step_y;
    }
  }

  // =========================================================================
  // PHASE 1: Build LL_V (Vertices)
  // =========================================================================
  std::cout << "  Phase 1: Building LL_V (vertices)..." << std::endl;
  int vertex_id = 0;

  for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
       ++v) {
    Point p = v->point();
    int label = v->info();

    int bucket_idx = grid_.get_bucket_index(p.x(), p.y());
    Bucket2D &bucket = grid_.buckets[bucket_idx];

    // Store Vertex_handle for direct Voronoi cell lookup
    LL_Vertex *new_vertex = new LL_Vertex(p, label, vertex_id, v);
    new_vertex->next = bucket.vertices;
    bucket.vertices = new_vertex;
    bucket.vertex_count++;

    vertex_id++;
  }

  // =========================================================================
  // PHASE 2: Build LL_E (Edges) — O(n) via bounding-box enumeration
  // =========================================================================
  std::cout << "  Phase 2: Building LL_E (edges) [O(n) bounding-box method]..."
            << std::endl;
  int edge_id = 0;

  for (auto e = dt_.finite_edges_begin(); e != dt_.finite_edges_end(); ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);

    Point p1 = v1->point();
    Point p2 = v2->point();
    int class1 = v1->info();
    int class2 = v2->info();

    // Compute bounding box of edge, only check overlapping cells
    double ex_min = std::min(p1.x(), p2.x());
    double ex_max = std::max(p1.x(), p2.x());
    double ey_min = std::min(p1.y(), p2.y());
    double ey_max = std::max(p1.y(), p2.y());

    int col_start = std::max(0, (int)((ex_min - grid_.min_x) / grid_.step_x));
    int col_end =
        std::min(grid_.cols - 1, (int)((ex_max - grid_.min_x) / grid_.step_x));
    int row_start = std::max(0, (int)((ey_min - grid_.min_y) / grid_.step_y));
    int row_end =
        std::min(grid_.rows - 1, (int)((ey_max - grid_.min_y) / grid_.step_y));

    for (int r = row_start; r <= row_end; ++r) {
      for (int c = col_start; c <= col_end; ++c) {
        int idx = r * grid_.cols + c;
        Bucket2D &bucket = grid_.buckets[idx];

        if (segment_intersects_bucket(p1.x(), p1.y(), p2.x(), p2.y(),
                                      bucket.min_x, bucket.min_y, bucket.max_x,
                                      bucket.max_y)) {
          LL_Edge *new_edge = new LL_Edge(p1, p2, class1, class2, edge_id);
          new_edge->next = bucket.edges;
          bucket.edges = new_edge;
          bucket.edge_count++;
        }
      }
    }

    edge_id++;
  }

  // =========================================================================
  // PHASE 3: Build LL_GE (Grid Edge Intersections)
  // =========================================================================
  std::cout << "  Phase 3: Building LL_GE (grid edge intersections)..."
            << std::endl;

  for (int idx = 0; idx < k * k; ++idx) {
    Bucket2D &bucket = grid_.buckets[idx];

    LL_Edge *edge = bucket.edges;
    while (edge != nullptr) {
      if (edge->is_boundary) {
        struct EdgeDef {
          GridEdgeSide side;
          double x1, y1, x2, y2;
        };
        EdgeDef sides[] = {
            {GRID_LEFT, bucket.min_x, bucket.min_y, bucket.min_x, bucket.max_y},
            {GRID_BOTTOM, bucket.min_x, bucket.min_y, bucket.max_x,
             bucket.min_y},
            {GRID_RIGHT, bucket.max_x, bucket.min_y, bucket.max_x,
             bucket.max_y},
            {GRID_TOP, bucket.min_x, bucket.max_y, bucket.max_x, bucket.max_y}};

        for (const auto &side : sides) {
          double ix, iy;
          if (compute_intersection(edge->p1.x(), edge->p1.y(), edge->p2.x(),
                                   edge->p2.y(), side.x1, side.y1, side.x2,
                                   side.y2, ix, iy)) {
            LL_GridEdge *new_ge = new LL_GridEdge(side.side, Point(ix, iy),
                                                  edge->class1, edge->class2);
            new_ge->next = bucket.grid_edges;
            bucket.grid_edges = new_ge;
            bucket.grid_edge_count++;
          }
        }
      }
      edge = edge->next;
    }
  }

  // =========================================================================
  // PHASE 4: Build LL_Poly (Voronoi polygon regions)
  // =========================================================================
  std::cout << "  Phase 4: Building LL_Poly (Voronoi polygon regions)..."
            << std::endl;

  VoronoiDiagram vd(dt_);

  double bbox_margin =
      std::max(grid_.step_x, grid_.step_y) * VORONOI_BBOX_MARGIN_MULTIPLIER;
  double bbox_min_x = grid_.min_x - bbox_margin;
  double bbox_max_x = grid_.max_x + bbox_margin;
  double bbox_min_y = grid_.min_y - bbox_margin;
  double bbox_max_y = grid_.max_y + bbox_margin;

  int global_poly_id = 0;

  // Map Vertex_handle directly to Voronoi cell polygon
  std::unordered_map<Vertex_handle, std::vector<Point>> vertex_to_cell;

  // Compute centroid of all vertices for direction determination
  double cx_all = 0, cy_all = 0;
  int nv_all = 0;
  for (auto v = dt_.finite_vertices_begin(); v != dt_.finite_vertices_end();
       ++v) {
    cx_all += v->point().x();
    cy_all += v->point().y();
    nv_all++;
  }
  if (nv_all > 0) {
    cx_all /= nv_all;
    cy_all /= nv_all;
  }

  for (auto face_it = vd.faces_begin(); face_it != vd.faces_end(); ++face_it) {
    auto dual_vertex = face_it->dual();
    if (dual_vertex == nullptr)
      continue;

    Vertex_handle site_vertex = dual_vertex;
    std::vector<Point> cell_polygon;

    if (face_it->is_unbounded()) {
      // Correct unbounded Voronoi edge direction
      auto ccb_start = face_it->ccb();
      auto ccb_it = ccb_start;

      std::vector<Point> raw_vertices;
      do {
        if (ccb_it->has_source()) {
          auto src = ccb_it->source();
          raw_vertices.push_back(src->point());
        } else {
          // Unbounded Voronoi edge: extend outward using the perpendicular
          // bisector of the DUAL Delaunay edge.
          //
          // GEOMETRIC PRINCIPLE:
          //   A Voronoi edge between two sites s1 and s2 lies on the
          //   perpendicular bisector of the segment s1-s2. When this edge
          //   is unbounded, the ray extends perpendicular to s1-s2, pointing
          //   outward from the convex hull (away from the third site that
          //   would close the Delaunay triangle).
          //
          // CGAL API:
          //   In the Voronoi_diagram_2 adapter, halfedge->dual() returns the
          //   dual Delaunay edge. Its two endpoints are the sites s1 and s2
          //   defining this Voronoi edge. The current face's site (site_vertex)
          //   is ONE of them; the other is the twin face's dual vertex.
          if (!raw_vertices.empty()) {
            Point last_pt = raw_vertices.back();

            // Retrieve the two sites (s1, s2) that this Voronoi edge separates.
            // site_vertex is s1 (this Voronoi face's site).
            // s2 is the twin halfedge's face's dual vertex.
            Vertex_handle s1 = site_vertex;
            Vertex_handle s2;

            auto twin_he = ccb_it->twin();
            // The twin halfedge's face may be bounded (has a dual site) or
            // unbounded (infinite face). Only bounded faces have a valid dual.
            try {
              if (!twin_he->face()->is_unbounded()) {
                s2 = twin_he->face()->dual();
              }
            } catch (...) {
              // Some CGAL configurations may throw on invalid access;
              // s2 remains default-constructed (invalid handle).
            }

            double dx = 0, dy = 0;

            if (s2 != Vertex_handle()) {
              // Correct case: we have both sites. Compute perpendicular to
              // s1-s2.
              Point p1 = s1->point();
              Point p2 = s2->point();

              // Vector from s1 to s2
              double edge_dx = p2.x() - p1.x();
              double edge_dy = p2.y() - p1.y();
              double edge_len =
                  std::sqrt(edge_dx * edge_dx + edge_dy * edge_dy);

              if (edge_len > VECTOR_NORM_EPSILON) {
                // Perpendicular to s1-s2 (rotate 90 degrees)
                // Two candidates: (-edge_dy, edge_dx) and (edge_dy, -edge_dx)
                double perp_x = -edge_dy / edge_len;
                double perp_y = edge_dx / edge_len;

                // Orient outward: the correct direction is the one that points
                // AWAY from the third vertex of the Delaunay triangle opposite
                // this edge (if it exists). Equivalently, we pick the direction
                // that moves away from the midpoint of the convex hull's
                // interior.
                //
                // We check which perpendicular direction moves AWAY from the
                // data centroid (cx_all, cy_all). This is correct because the
                // unbounded ray of a hull site must exit the hull.
                double mid_x = (p1.x() + p2.x()) / 2.0;
                double mid_y = (p1.y() + p2.y()) / 2.0;
                double to_centroid_x = cx_all - mid_x;
                double to_centroid_y = cy_all - mid_y;

                // If (perp_x, perp_y) points toward the centroid, flip it.
                if (perp_x * to_centroid_x + perp_y * to_centroid_y > 0) {
                  perp_x = -perp_x;
                  perp_y = -perp_y;
                }

                dx = perp_x;
                dy = perp_y;
              }
            }

            // Fallback only if we couldn't retrieve s2 or the edge is
            // degenerate. This path should rarely execute on well-formed input.
            if (dx == 0 && dy == 0) {
              Point site_pt = site_vertex->point();
              double to_center_dx = cx_all - site_pt.x();
              double to_center_dy = cy_all - site_pt.y();
              double len = std::sqrt(to_center_dx * to_center_dx +
                                     to_center_dy * to_center_dy);
              if (len > VECTOR_NORM_EPSILON) {
                dx = -to_center_dx / len;
                dy = -to_center_dy / len;
              } else {
                dx =
                    1.0; // arbitrary direction for degenerate single-point case
                dy = 0.0;
              }
            }

            double t_max =
                std::max(bbox_max_x - bbox_min_x, bbox_max_y - bbox_min_y) * 2;
            raw_vertices.push_back(
                Point(last_pt.x() + dx * t_max, last_pt.y() + dy * t_max));
          }
        }
        ++ccb_it;
      } while (ccb_it != ccb_start);

      cell_polygon = clip_polygon_to_bucket(raw_vertices, bbox_min_x,
                                            bbox_min_y, bbox_max_x, bbox_max_y);
    } else {
      // Bounded face
      auto ccb_start = face_it->ccb();
      auto ccb_it = ccb_start;
      do {
        if (ccb_it->has_source()) {
          auto src = ccb_it->source();
          cell_polygon.push_back(src->point());
        }
        ++ccb_it;
      } while (ccb_it != ccb_start);
    }

    if (!cell_polygon.empty()) {
      vertex_to_cell[site_vertex] = cell_polygon;
    }
  }

  std::cout << "    Extracted " << vertex_to_cell.size() << " Voronoi cells"
            << std::endl;

  // Assign clipped Voronoi cells to buckets
  for (int idx = 0; idx < k * k; ++idx) {
    Bucket2D &bucket = grid_.buckets[idx];

    // Collect ALL clipped polygons per class
    std::map<int, std::vector<std::vector<Point>>> class_polygons;

    // Use stored Vertex_handle for direct O(1) lookup
    LL_Vertex *v = bucket.vertices;
    while (v != nullptr) {
      auto it = vertex_to_cell.find(v->vh);
      if (it != vertex_to_cell.end()) {
        std::vector<Point> clipped = clip_polygon_to_bucket(
            it->second, bucket.min_x, bucket.min_y, bucket.max_x, bucket.max_y);

        if (clipped.size() >= 3) {
          class_polygons[v->class_label].push_back(clipped);
        }
      }
      v = v->next;
    }

    // Handle empty buckets via nearest vertex sampling
    if (class_polygons.empty()) {
      double bx = (bucket.min_x + bucket.max_x) / 2;
      double by = (bucket.min_y + bucket.max_y) / 2;
      Point center_pt(bx, by);
      Vertex_handle nearest = dt_.nearest_vertex(center_pt);

      if (nearest != Vertex_handle()) {
        auto it = vertex_to_cell.find(nearest);
        if (it != vertex_to_cell.end()) {
          std::vector<Point> clipped =
              clip_polygon_to_bucket(it->second, bucket.min_x, bucket.min_y,
                                     bucket.max_x, bucket.max_y);
          if (clipped.size() >= 3) {
            class_polygons[nearest->info()].push_back(clipped);
          }
        }
      }
    }

    bucket.num_classes = static_cast<int>(class_polygons.size());

    if (bucket.num_classes == 0) {
      // Ultimate fallback
      double bx = (bucket.min_x + bucket.max_x) / 2;
      double by = (bucket.min_y + bucket.max_y) / 2;
      bucket.type = BUCKET_HOMOGENEOUS;
      bucket.num_classes = 1;
      Vertex_handle nv = dt_.nearest_vertex(Point(bx, by));
      bucket.dominant_class = (nv != Vertex_handle()) ? nv->info() : 0;
      grid_.single_class_buckets++;

      LL_Polygon *poly =
          new LL_Polygon(global_poly_id++, bucket.dominant_class);
      poly->inside_label = 1;
      poly->vertices = {
          Point(bucket.min_x, bucket.min_y), Point(bucket.max_x, bucket.min_y),
          Point(bucket.max_x, bucket.max_y), Point(bucket.min_x, bucket.max_y)};
      poly->area = grid_.step_x * grid_.step_y;
      bucket.polygons = poly;
      bucket.polygon_count = 1;
      grid_.total_polygons++;
      continue;
    }

    // Determine dominant class (largest total area)
    double max_area = 0;
    for (const auto &cp : class_polygons) {
      double total_area = 0;
      for (const auto &poly_verts : cp.second) {
        total_area += compute_polygon_area(poly_verts);
      }
      if (total_area > max_area) {
        max_area = total_area;
        bucket.dominant_class = cp.first;
      }
    }

    // Set bucket type
    if (bucket.num_classes == 1) {
      bucket.type = BUCKET_HOMOGENEOUS;
      grid_.single_class_buckets++;
    } else if (bucket.num_classes == 2) {
      bucket.type = BUCKET_BIPARTITIONED;
      grid_.bipartitioned_buckets++;

      // Compute half-plane from a boundary edge in this bucket
      LL_Edge *be = bucket.edges;
      bool found_boundary = false;
      while (be) {
        if (be->is_boundary) {
          double mx = (be->p1.x() + be->p2.x()) / 2;
          double my = (be->p1.y() + be->p2.y()) / 2;
          double edge_dx = be->p2.x() - be->p1.x();
          double edge_dy = be->p2.y() - be->p1.y();

          bucket.boundary_nx = -edge_dy;
          bucket.boundary_ny = edge_dx;
          bucket.boundary_d =
              -(bucket.boundary_nx * mx + bucket.boundary_ny * my);

          double dot1 = bucket.boundary_nx * be->p1.x() +
                        bucket.boundary_ny * be->p1.y() + bucket.boundary_d;
          bucket.class_positive = (dot1 >= 0) ? be->class1 : be->class2;
          bucket.class_negative = (dot1 >= 0) ? be->class2 : be->class1;

          found_boundary = true;
          break;
        }
        be = be->next;
      }

      if (!found_boundary) {
        bucket.type = BUCKET_HOMOGENEOUS;
        grid_.single_class_buckets++;
        grid_.bipartitioned_buckets--;
      }
    } else {
      bucket.type = BUCKET_MULTI_PARTITIONED;
      grid_.multi_class_buckets++;
    }

    // Create polygon regions for EACH class, storing ALL polygons
    LL_Polygon *prev_poly = nullptr;
    for (const auto &cp : class_polygons) {
      int cls = cp.first;

      for (const auto &poly_verts : cp.second) {
        if (poly_verts.size() < 3)
          continue;

        LL_Polygon *poly = new LL_Polygon(global_poly_id++, cls);
        poly->inside_label = 1;
        poly->vertices = poly_verts;
        poly->area = compute_polygon_area(poly_verts);

        if (prev_poly == nullptr) {
          bucket.polygons = poly;
        } else {
          prev_poly->next = poly;
        }
        prev_poly = poly;
        bucket.polygon_count++;
        grid_.total_polygons++;
      }
    }
  }

  grid_.print_statistics();
  std::cout << "2D Buckets construction complete." << std::endl;
}
// =============================================================================
// POST-BUILD VALIDATION
// =============================================================================

void DelaunayClassifier::validate_all_buckets() {
  for (int idx = 0; idx < static_cast<int>(grid_.buckets.size()); ++idx) {
    Bucket2D &bucket = grid_.buckets[idx];

    if (bucket.dominant_class < 0) {
      // This bucket has no assigned class — find one from nearest vertex
      double cx = (bucket.min_x + bucket.max_x) / 2.0;
      double cy = (bucket.min_y + bucket.max_y) / 2.0;
      Vertex_handle nv = dt_.nearest_vertex(Point(cx, cy));
      if (nv != Vertex_handle()) {
        bucket.dominant_class = nv->info();
        bucket.type = BUCKET_HOMOGENEOUS;
        bucket.num_classes = 1;
      }
    }

    // Ensure MULTI_PARTITIONED buckets have a valid dominant_class
    // as the final fallback for floating-point polygon gaps
    if (bucket.type == BUCKET_MULTI_PARTITIONED && bucket.dominant_class < 0 &&
        bucket.polygons != nullptr) {
      // Use the class of the polygon with the largest area
      double max_area = -1.0;
      LL_Polygon *poly = bucket.polygons;
      while (poly != nullptr) {
        if (poly->area > max_area) {
          max_area = poly->area;
          bucket.dominant_class = poly->class_label;
        }
        poly = poly->next;
      }
    }
  }
}

std::vector<std::pair<int, std::vector<Point>>>
DelaunayClassifier::compute_class_regions() const {
  std::vector<std::pair<int, std::vector<Point>>> regions;

  for (const auto &bucket : grid_.buckets) {
    LL_Polygon *poly = bucket.polygons;
    while (poly != nullptr) {
      if (poly->inside_label == 1 && !poly->vertices.empty()) {
        regions.push_back({poly->class_label, poly->vertices});
      }
      poly = poly->next;
    }
  }

  return regions;
}

// =============================================================================
// BUCKET OCCUPANCY INSTRUMENTATION
// =============================================================================
// Exposes per-bucket polygon and vertex counts for O(1) claim validation.
// Called by external tools (generate_figures.py) to produce histograms and
// summary statistics for the paper.

std::vector<int> DelaunayClassifier::get_bucket_polygon_counts() const {
  std::vector<int> counts;
  counts.reserve(grid_.buckets.size());
  for (const auto &bucket : grid_.buckets) {
    counts.push_back(bucket.polygon_count);
  }
  return counts;
}

std::vector<int> DelaunayClassifier::get_bucket_vertex_counts() const {
  std::vector<int> counts;
  counts.reserve(grid_.buckets.size());
  for (const auto &bucket : grid_.buckets) {
    counts.push_back(bucket.vertex_count);
  }
  return counts;
}

DelaunayClassifier::BucketOccupancyStats
DelaunayClassifier::get_bucket_occupancy_stats() const {
  BucketOccupancyStats stats{};
  stats.num_buckets = static_cast<int>(grid_.buckets.size());

  if (grid_.buckets.empty()) {
    return stats;
  }

  std::vector<int> poly_counts = get_bucket_polygon_counts();
  std::vector<int> vert_counts = get_bucket_vertex_counts();

  // Max values
  stats.max_polygons =
      *std::max_element(poly_counts.begin(), poly_counts.end());
  stats.max_vertices =
      *std::max_element(vert_counts.begin(), vert_counts.end());

  // Mean
  long long sum_polys = 0;
  int empty = 0;
  for (int c : poly_counts)
    sum_polys += c;
  for (int c : vert_counts) {
    if (c == 0)
      empty++;
  }
  stats.mean_polygons = static_cast<double>(sum_polys) / poly_counts.size();
  stats.empty_buckets = empty;

  // Median and 99th percentile (requires sorting)
  std::vector<int> sorted_polys = poly_counts;
  std::sort(sorted_polys.begin(), sorted_polys.end());
  stats.median_polygons = sorted_polys[sorted_polys.size() / 2];
  size_t p99_idx = static_cast<size_t>(sorted_polys.size() * 0.99);
  if (p99_idx >= sorted_polys.size())
    p99_idx = sorted_polys.size() - 1;
  stats.p99_polygons = sorted_polys[p99_idx];

  return stats;
}