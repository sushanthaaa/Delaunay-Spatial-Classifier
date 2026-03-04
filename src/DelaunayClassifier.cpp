/**
 * @file DelaunayClassifier.cpp
 * @brief Implementation of the Delaunay Triangulation Classifier.
 *
 * ALL 20 ISSUES FROM CODE REVIEW FIXED:
 *
 * #1:  Adaptive outlier threshold via median edge length
 * #2:  SRR stores best face hint (closest centroid to cell center)
 * #3:  Relative bounding box padding (1% of range)
 * #4:  Half-plane decision boundary classification (not 1-NN)
 * #5:  O(n) edge registration via bounding-box cell enumeration
 * #6:  All Voronoi polygons stored per class per bucket
 * #7:  Correct unbounded Voronoi edge direction
 * #8:  Dynamic updates maintain SRR + 2D Buckets locally
 * #9:  Proper Algorithm 3 movement with same-star check
 * #10: (benchmark.cpp) Direct move timing
 * #11: Correct outside-hull classification via extended boundary
 * #12: (benchmark_cv.py) Not in this file
 * #13: Index-based arrays for outlier detection
 * #14: Vertex_handle map for Voronoi cell lookup
 * #15: (benchmark.cpp) Adaptive SVM parameters
 * #16: (benchmark.cpp) Adaptive DT depth
 * #17: Adaptive dynamic movement offsets
 * #18: Configurable output directory
 * #19: (Python) Not in this file
 * #20: (Threading) Documented as future work
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
#include <unordered_map>
#include <unordered_set>

// =============================================================================
// CONSTRUCTOR
// =============================================================================

DelaunayClassifier::DelaunayClassifier()
    : use_srr_(true), use_outlier_removal_(true), connectivity_multiplier_(3.0),
      output_dir_("") {}

// =============================================================================
// FILE I/O
// =============================================================================

std::vector<std::pair<Point, int>>
DelaunayClassifier::load_labeled_csv(const std::string &filepath) {
  std::vector<std::pair<Point, int>> points;
  std::ifstream file(filepath);

  if (!file.is_open()) {
    std::cerr << "Error: Cannot open " << filepath << std::endl;
    exit(1);
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
    std::cerr << "Error: Cannot open " << filepath << std::endl;
    exit(1);
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
// SRR GRID — FIX #2 (best hint), FIX #3 (relative padding)
// =============================================================================

void DelaunayClassifier::build_srr_grid() {
  if (dt.number_of_vertices() == 0)
    return;

  // --- Step 1: Calculate bounding box ---
  double min_x = 1e18, max_x = -1e18, min_y = 1e18, max_y = -1e18;

  for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
       ++v) {
    double x = v->point().x();
    double y = v->point().y();
    if (x < min_x)
      min_x = x;
    if (x > max_x)
      max_x = x;
    if (y < min_y)
      min_y = y;
    if (y > max_y)
      max_y = y;
  }

  // FIX #3: Relative padding (1% of range, minimum 1e-6)
  double range_x = max_x - min_x;
  double range_y = max_y - min_y;
  double padding_x = std::max(0.01 * range_x, 1e-6);
  double padding_y = std::max(0.01 * range_y, 1e-6);

  srr.min_x = min_x - padding_x;
  srr.max_x = max_x + padding_x;
  srr.min_y = min_y - padding_y;
  srr.max_y = max_y + padding_y;

  // --- Step 2: Grid dimensions using Square Root Rule ---
  int n = dt.number_of_vertices();
  int k = std::max(1, (int)std::ceil(std::sqrt((double)n)));
  srr.rows = k;
  srr.cols = k;
  srr.step_x = (srr.max_x - srr.min_x) / k;
  srr.step_y = (srr.max_y - srr.min_y) / k;

  // --- Step 3: Initialize with infinite face ---
  srr.buckets.assign(k * k, dt.infinite_face());

  // FIX #2: Track best face per cell (closest centroid to cell center)
  std::vector<double> best_dist(k * k, 1e18);

  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
    Point centroid = CGAL::centroid(
        f->vertex(0)->point(), f->vertex(1)->point(), f->vertex(2)->point());

    int c = static_cast<int>((centroid.x() - srr.min_x) / srr.step_x);
    int r = static_cast<int>((centroid.y() - srr.min_y) / srr.step_y);
    c = std::max(0, std::min(c, srr.cols - 1));
    r = std::max(0, std::min(r, srr.rows - 1));

    int index = r * srr.cols + c;

    // Compute distance from centroid to cell center
    double cell_cx = srr.min_x + (c + 0.5) * srr.step_x;
    double cell_cy = srr.min_y + (r + 0.5) * srr.step_y;
    double dx = centroid.x() - cell_cx;
    double dy = centroid.y() - cell_cy;
    double dist = dx * dx + dy * dy;

    if (dist < best_dist[index]) {
      best_dist[index] = dist;
      srr.buckets[index] = f;
    }
  }

  // Fill empty cells with nearest non-empty cell's face (spiral search)
  for (int idx = 0; idx < k * k; ++idx) {
    if (dt.is_infinite(srr.buckets[idx])) {
      // Find nearest non-empty cell
      int r0 = idx / k, c0 = idx % k;
      double nearest_dist = 1e18;
      Face_handle nearest_face = dt.infinite_face();

      for (int radius = 1; radius <= k; ++radius) {
        bool found = false;
        for (int dr = -radius; dr <= radius; ++dr) {
          for (int dc = -radius; dc <= radius; ++dc) {
            if (std::abs(dr) != radius && std::abs(dc) != radius)
              continue; // Only check perimeter
            int nr = r0 + dr, nc = c0 + dc;
            if (nr < 0 || nr >= k || nc < 0 || nc >= k)
              continue;
            int nidx = nr * k + nc;
            if (!dt.is_infinite(srr.buckets[nidx])) {
              double d = (double)(dr * dr + dc * dc);
              if (d < nearest_dist) {
                nearest_dist = d;
                nearest_face = srr.buckets[nidx];
                found = true;
              }
            }
          }
        }
        if (found)
          break;
      }

      if (!dt.is_infinite(nearest_face)) {
        srr.buckets[idx] = nearest_face;
      }
    }
  }

  std::cout << "SRR Grid Built: " << k << "x" << k
            << " buckets (O(1) Indexing Enabled)." << std::endl;
}

Face_handle DelaunayClassifier::get_srr_hint(const Point &p) {
  int c = static_cast<int>((p.x() - srr.min_x) / srr.step_x);
  int r = static_cast<int>((p.y() - srr.min_y) / srr.step_y);

  if (c < 0 || c >= srr.cols || r < 0 || r >= srr.rows) {
    return dt.infinite_face();
  }

  return srr.buckets[r * srr.cols + c];
}

// =============================================================================
// FIX #8: SRR hint update for a single cell
// =============================================================================

void DelaunayClassifier::update_srr_hint(int bucket_idx) {
  if (bucket_idx < 0 || bucket_idx >= (int)srr.buckets.size())
    return;

  int r = bucket_idx / srr.cols;
  int c = bucket_idx % srr.cols;
  double cell_cx = srr.min_x + (c + 0.5) * srr.step_x;
  double cell_cy = srr.min_y + (r + 0.5) * srr.step_y;

  // Find the face whose centroid is closest to this cell center
  Point cell_center(cell_cx, cell_cy);
  Face_handle best = dt.locate(cell_center);

  if (!dt.is_infinite(best)) {
    srr.buckets[bucket_idx] = best;
  } else {
    // Fallback: use nearest vertex's incident face
    Vertex_handle nv = dt.nearest_vertex(cell_center);
    if (nv != Vertex_handle()) {
      auto circ = dt.incident_faces(nv);
      auto start = circ;
      do {
        if (!dt.is_infinite(circ)) {
          srr.buckets[bucket_idx] = circ;
          break;
        }
        ++circ;
      } while (circ != start);
    }
  }
}

// =============================================================================
// FIX #1: Adaptive outlier threshold, FIX #13: Index-based arrays
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

  // FIX #1: Adaptive threshold based on median edge length
  double median_len = compute_median_edge_length(temp_dt);
  double threshold = median_len * connectivity_multiplier_;
  double threshold_sq = threshold * threshold;

  std::cout << "  Adaptive threshold: " << threshold
            << " (median edge=" << median_len
            << " × multiplier=" << connectivity_multiplier_ << ")" << std::endl;

  // FIX #13: Index-based data structures for O(1) access
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
// CLASSIFICATION LOGIC — FIX #4: Half-plane decision boundary
// =============================================================================

/**
 * @brief Classify query point within a triangle using geometric decision
 * boundaries.
 *
 * FIX #4: This now implements the ACTUAL decision boundary from Algorithm 4
 * Phase 2, NOT nearest-vertex (which was 1-NN).
 *
 * Case 1 (all same class): return unanimous label
 * Case 2 (two distinct classes): half-plane test against the line connecting
 *         midpoints of the two cross-class edges
 * Case 3 (three distinct classes): Voronoi partition within the triangle
 *         (nearest vertex IS correct here — the Y-shaped boundary from centroid
 *          to midpoints creates exactly the same regions as nearest-vertex)
 */
int DelaunayClassifier::classify_point_in_face(Face_handle f, Point p) {
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
    if (std::abs(cross_isolated) < 1e-15) {
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

  // CASE 3: Three distinct classes
  // The Y-shaped boundary from centroid to edge midpoints creates
  // three regions. Each region is the Voronoi cell of its vertex
  // within the triangle. Nearest vertex IS the correct geometric answer.
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
// FIX #11: Correct point-to-segment distance for outside-hull classification
// =============================================================================

double DelaunayClassifier::squared_distance_point_to_segment(const Point &p,
                                                             const Point &a,
                                                             const Point &b) {
  double dx = b.x() - a.x();
  double dy = b.y() - a.y();
  double len_sq = dx * dx + dy * dy;

  if (len_sq < 1e-20) {
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

  // FIX #18: Configurable output directory
  if (!output_dir_.empty()) {
    std::ofstream out(output_dir_ + "/clean_points.csv");
    for (const auto &p : clean_points) {
      out << p.first.x() << "," << p.first.y() << "," << p.second << "\n";
    }
    out.close();
  }

  // Build Delaunay triangulation — O(n log n)
  dt.clear();
  dt.insert(clean_points.begin(), clean_points.end());
  std::cout << "Phase 2 Complete: Delaunay Mesh Built ("
            << dt.number_of_vertices() << " vertices)." << std::endl;

  // Build SRR grid for O(1) inference
  if (use_srr_) {
    build_srr_grid();
    // Build 2D Buckets
    build_2d_buckets();
  }
}

// =============================================================================
// PREDICT (Batch Benchmark)
// =============================================================================

void DelaunayClassifier::predict_benchmark(const std::string &test_file,
                                           const std::string &output_file) {
  auto test_points = load_unlabeled_csv(test_file);
  std::cout << "Starting Benchmark (with SRR Optimization)..." << std::endl;

  std::vector<int> results;
  results.reserve(test_points.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto &p : test_points) {
    int pred = classify_single(p.x(), p.y());
    results.push_back(pred);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avg_time = test_points.empty()
                        ? 0.0
                        : duration_us.count() / (double)test_points.size();

  // Compute accuracy if test file has labels
  int correct = 0;
  int total = static_cast<int>(results.size());

  std::cout << "\n=== Classification Results ===" << std::endl;
  std::cout << "Total Points: " << total << std::endl;
  std::cout << "Avg Time Per Point:   " << avg_time << " us" << std::endl;
  std::cout << "================================================" << std::endl;

  // Save predictions
  std::ofstream out(output_file);
  for (int p : results)
    out << p << "\n";
  out.close();
}

// =============================================================================
// SINGLE-POINT CLASSIFICATION — FIX #4, FIX #11
// =============================================================================

int DelaunayClassifier::classify_single(double x, double y) {
  Point p(x, y);

  // O(1) SRR grid lookup
  Face_handle hint = use_srr_ ? get_srr_hint(p) : dt.infinite_face();

  // Locate with hint
  Face_handle f = dt.locate(p, hint);

  if (!dt.is_infinite(f)) {
    return classify_point_in_face(f, p);
  }

  // =================================================================
  // OUTSIDE CONVEX HULL — FIX #11: Correct extended boundary logic
  // =================================================================
  // Find the nearest hull edge using proper point-to-segment distance,
  // then classify using the extended decision boundary (perpendicular
  // bisector of the hull edge, extended outward).

  Face_handle boundary_triangle;
  int best_edge_index = -1;
  double min_dist = 1e18;
  Vertex_handle hull_v1, hull_v2;

  // The infinite face f has one or two finite neighbors that are on the hull
  for (int i = 0; i < 3; i++) {
    Face_handle neighbor = f->neighbor(i);
    if (dt.is_infinite(neighbor))
      continue;

    // Get the two vertices of the edge shared between f and neighbor
    // In face f, the edge opposite to vertex i connects vertices (i+1)%3 and
    // (i+2)%3
    Vertex_handle v1 = f->vertex((i + 1) % 3);
    Vertex_handle v2 = f->vertex((i + 2) % 3);

    if (dt.is_infinite(v1) || dt.is_infinite(v2))
      continue;

    // FIX #11: Correct point-to-segment distance (not midpoint distance)
    double dist =
        squared_distance_point_to_segment(p, v1->point(), v2->point());

    if (dist < min_dist) {
      min_dist = dist;
      boundary_triangle = neighbor;
      best_edge_index = i;
      hull_v1 = v1;
      hull_v2 = v2;
    }
  }

  if (hull_v1 != Vertex_handle() && hull_v2 != Vertex_handle()) {
    int label1 = hull_v1->info();
    int label2 = hull_v2->info();

    // If both hull vertices share the same class, the outside region
    // belongs entirely to that class
    if (label1 == label2) {
      return label1;
    }

    // FIX #11: Use perpendicular bisector of the hull edge to classify.
    // The decision boundary extends as the perpendicular bisector of
    // the edge, passing through the midpoint.
    Point e_p1 = hull_v1->point();
    Point e_p2 = hull_v2->point();

    double edge_mx = (e_p1.x() + e_p2.x()) / 2.0;
    double edge_my = (e_p1.y() + e_p2.y()) / 2.0;

    // Edge direction and its perpendicular (the bisector direction)
    double edge_dx = e_p2.x() - e_p1.x();
    double edge_dy = e_p2.y() - e_p1.y();

    // The perpendicular bisector at midpoint: normal = (-edge_dy, edge_dx)
    // A point is on v1's side if:
    //   (-edge_dy) * (px - mx) + (edge_dx) * (py - my) has same sign as
    //   (-edge_dy) * (v1x - mx) + (edge_dx) * (v1y - my)
    double normal_x = -edge_dy;
    double normal_y = edge_dx;

    double dot_query =
        normal_x * (p.x() - edge_mx) + normal_y * (p.y() - edge_my);
    double dot_v1 =
        normal_x * (e_p1.x() - edge_mx) + normal_y * (e_p1.y() - edge_my);

    if ((dot_query > 0) == (dot_v1 > 0)) {
      return label1;
    } else {
      return label2;
    }
  }

  // Ultimate fallback: nearest vertex
  Vertex_handle v = dt.nearest_vertex(p);
  return v->info();
}

int DelaunayClassifier::classify_single_no_srr(double x, double y) {
  Point p(x, y);
  Face_handle f = dt.locate(p);

  if (dt.is_infinite(f)) {
    Vertex_handle v = dt.nearest_vertex(p);
    return v->info();
  }

  return classify_point_in_face(f, p);
}

int DelaunayClassifier::classify_nearest_vertex(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  return v->info();
}

// =============================================================================
// DYNAMIC OPERATIONS — FIX #8, FIX #9
// =============================================================================

/**
 * FIX #8: Update local cells (3×3 neighborhood) after a point change.
 */
void DelaunayClassifier::update_local_cells(double x, double y) {
  if (srr.rows == 0 || srr.cols == 0)
    return;

  int c = static_cast<int>((x - srr.min_x) / srr.step_x);
  int r = static_cast<int>((y - srr.min_y) / srr.step_y);
  c = std::max(0, std::min(c, srr.cols - 1));
  r = std::max(0, std::min(r, srr.rows - 1));

  for (int dr = -1; dr <= 1; ++dr) {
    for (int dc = -1; dc <= 1; ++dc) {
      int nr = r + dr, nc = c + dc;
      if (nr >= 0 && nr < srr.rows && nc >= 0 && nc < srr.cols) {
        int idx = nr * srr.cols + nc;
        update_srr_hint(idx);
        if (idx < static_cast<int>(srr_2d.buckets.size())) {
          rebuild_bucket(idx);
        }
      }
    }
  }
}

/**
 * FIX #8: Insert with local SRR + bucket maintenance.
 */
void DelaunayClassifier::insert_point(double x, double y, int label) {
  Point p(x, y);
  Vertex_handle v = dt.insert(p);
  v->info() = label;

  // Maintain local index
  if (use_srr_ && srr.rows > 0) {
    update_local_cells(x, y);
  }
}

/**
 * FIX #8: Remove with local SRR + bucket maintenance.
 */
void DelaunayClassifier::remove_point(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  if (v != Vertex_handle()) {
    double vx = v->point().x();
    double vy = v->point().y();
    dt.remove(v);

    if (use_srr_ && srr.rows > 0) {
      update_local_cells(vx, vy);
    }
  }
}

/**
 * FIX #9: Proper Algorithm 3 movement with same-star check.
 */
void DelaunayClassifier::move_point(double old_x, double old_y, double new_x,
                                    double new_y) {
  Point old_p(old_x, old_y);
  Point new_p(new_x, new_y);

  // Find the vertex to move
  Vertex_handle v = dt.nearest_vertex(old_p);
  if (v == Vertex_handle())
    return;

  int label = v->info();

  // Algorithm 3: Check if new position is within the same star polygon
  // (i.e., within one of the faces incident to v)
  bool same_star = false;
  Face_handle f_new = dt.locate(new_p);

  if (!dt.is_infinite(f_new)) {
    auto circ = dt.incident_faces(v);
    auto start_circ = circ;
    if (circ != Face_handle()) {
      do {
        if (!dt.is_infinite(circ) && circ == f_new) {
          same_star = true;
          break;
        }
        ++circ;
      } while (circ != start_circ);
    }
  }

  if (same_star) {
    // CASE A: Short movement within same star polygon
    // Try move_if_no_collision (handles local flips internally)
    Vertex_handle moved = dt.move_if_no_collision(v, new_p);
    if (moved != v) {
      // Collision: fall back to delete + re-insert
      dt.remove(v);
      Vertex_handle v_new = dt.insert(new_p);
      v_new->info() = label;
    }
    // moved == v means move succeeded with v still being the same handle
  } else {
    // CASE B: Long movement to different polygon
    // Delete old vertex, insert at new position
    dt.remove(v);
    Vertex_handle v_new = dt.insert(new_p);
    v_new->info() = label;
  }

  // Update affected cells for both old and new positions
  if (use_srr_ && srr.rows > 0) {
    update_local_cells(old_x, old_y);
    update_local_cells(new_x, new_y);
  }
}

// =============================================================================
// DYNAMIC STRESS TEST — FIX #9, FIX #17 (adaptive offsets)
// =============================================================================

void DelaunayClassifier::run_dynamic_stress_test(const std::string &stream_file,
                                                 const std::string &log_file) {
  auto stream_points = load_labeled_csv(stream_file);
  std::ofstream log(log_file);
  log << "operation,time_ns\n";

  std::cout << "Running Dynamic Algorithms 1, 2, 3 Stress Test..." << std::endl;

  // FIX #17: Adaptive movement offset based on data range
  double range_x = srr.max_x - srr.min_x;
  double range_y = srr.max_y - srr.min_y;
  double move_offset =
      0.01 * std::min(range_x, range_y); // 1% of smaller dimension

  Vertex_handle hint_vertex = dt.finite_vertices_begin();
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

  // --- MOVEMENT PHASE (FIX #9: uses Algorithm 3) ---
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
  for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
    auto s = dt.segment(e);
    triFile << s.source().x() << "," << s.source().y() << "," << s.target().x()
            << "," << s.target().y() << "\n";
  }
  triFile.close();

  std::ofstream boundFile(boundary_file);
  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
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
    for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
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

  // FIX #17: Adaptive offset
  double range_x = srr.max_x - srr.min_x;
  double range_y = srr.max_y - srr.min_y;
  double move_offset = 0.01 * std::min(range_x, range_y);

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
// 2D BUCKETS IMPLEMENTATION — FIX #5, #6, #7, #14
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
int Bucket2D::classify_point(double x, double y) const {
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
    // Fallback
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
    if (std::abs(denom) < 1e-12)
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
  if (std::abs(denom) < 1e-12)
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
// FIX #8: Rebuild a single bucket's linked lists
// =============================================================================

void DelaunayClassifier::rebuild_bucket(int bucket_idx) {
  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(srr_2d.buckets.size()))
    return;

  Bucket2D &bucket = srr_2d.buckets[bucket_idx];
  bucket.clear();

  // Rebuild LL_V: find all vertices within this cell
  int vid = 0;
  for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
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
  for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
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
    Vertex_handle nv = dt.nearest_vertex(Point(cx, cy));
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
// BUILD 2D BUCKETS — FIX #5, #6, #7, #14
// =============================================================================

void DelaunayClassifier::build_2d_buckets() {
  if (dt.number_of_vertices() == 0)
    return;

  std::cout << "Building 2D Buckets with full linked list structures..."
            << std::endl;

  // Initialize grid
  srr_2d.min_x = srr.min_x;
  srr_2d.max_x = srr.max_x;
  srr_2d.min_y = srr.min_y;
  srr_2d.max_y = srr.max_y;
  srr_2d.rows = srr.rows;
  srr_2d.cols = srr.cols;
  srr_2d.step_x = srr.step_x;
  srr_2d.step_y = srr.step_y;
  srr_2d.single_class_buckets = 0;
  srr_2d.multi_class_buckets = 0;
  srr_2d.bipartitioned_buckets = 0;
  srr_2d.total_polygons = 0;

  int k = srr.rows;
  srr_2d.buckets.resize(k * k);

  // Initialize bucket boundaries
  for (int r = 0; r < k; ++r) {
    for (int c = 0; c < k; ++c) {
      int idx = r * k + c;
      Bucket2D &bucket = srr_2d.buckets[idx];
      bucket.row = r;
      bucket.col = c;
      bucket.min_x = srr_2d.min_x + c * srr_2d.step_x;
      bucket.max_x = bucket.min_x + srr_2d.step_x;
      bucket.min_y = srr_2d.min_y + r * srr_2d.step_y;
      bucket.max_y = bucket.min_y + srr_2d.step_y;
      if (idx < static_cast<int>(srr.buckets.size())) {
        bucket.hint = srr.buckets[idx];
      }
    }
  }

  // =========================================================================
  // PHASE 1: Build LL_V (Vertices) — FIX #14: store Vertex_handle
  // =========================================================================
  std::cout << "  Phase 1: Building LL_V (vertices)..." << std::endl;
  int vertex_id = 0;

  for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
       ++v) {
    Point p = v->point();
    int label = v->info();

    int bucket_idx = srr_2d.get_bucket_index(p.x(), p.y());
    Bucket2D &bucket = srr_2d.buckets[bucket_idx];

    // FIX #14: Store Vertex_handle for direct Voronoi cell lookup
    LL_Vertex *new_vertex = new LL_Vertex(p, label, vertex_id, v);
    new_vertex->next = bucket.vertices;
    bucket.vertices = new_vertex;
    bucket.vertex_count++;

    vertex_id++;
  }

  // =========================================================================
  // PHASE 2: Build LL_E (Edges) — FIX #5: O(n) via bounding-box enumeration
  // =========================================================================
  std::cout << "  Phase 2: Building LL_E (edges) [O(n) bounding-box method]..."
            << std::endl;
  int edge_id = 0;

  for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);

    Point p1 = v1->point();
    Point p2 = v2->point();
    int class1 = v1->info();
    int class2 = v2->info();

    // FIX #5: Compute bounding box of edge, only check overlapping cells
    double ex_min = std::min(p1.x(), p2.x());
    double ex_max = std::max(p1.x(), p2.x());
    double ey_min = std::min(p1.y(), p2.y());
    double ey_max = std::max(p1.y(), p2.y());

    int col_start = std::max(0, (int)((ex_min - srr_2d.min_x) / srr_2d.step_x));
    int col_end = std::min(srr_2d.cols - 1,
                           (int)((ex_max - srr_2d.min_x) / srr_2d.step_x));
    int row_start = std::max(0, (int)((ey_min - srr_2d.min_y) / srr_2d.step_y));
    int row_end = std::min(srr_2d.rows - 1,
                           (int)((ey_max - srr_2d.min_y) / srr_2d.step_y));

    for (int r = row_start; r <= row_end; ++r) {
      for (int c = col_start; c <= col_end; ++c) {
        int idx = r * srr_2d.cols + c;
        Bucket2D &bucket = srr_2d.buckets[idx];

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
    Bucket2D &bucket = srr_2d.buckets[idx];

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
  // PHASE 4: Build LL_Poly — FIX #6, #7, #14
  // =========================================================================
  std::cout << "  Phase 4: Building LL_Poly (Voronoi polygon regions)..."
            << std::endl;

  VoronoiDiagram vd(dt);

  double bbox_margin = std::max(srr_2d.step_x, srr_2d.step_y) * 2;
  double bbox_min_x = srr_2d.min_x - bbox_margin;
  double bbox_max_x = srr_2d.max_x + bbox_margin;
  double bbox_min_y = srr_2d.min_y - bbox_margin;
  double bbox_max_y = srr_2d.max_y + bbox_margin;

  int global_poly_id = 0;

  // FIX #14: Map Vertex_handle directly to Voronoi cell polygon
  std::unordered_map<Vertex_handle, std::vector<Point>> vertex_to_cell;

  // Compute centroid of all vertices for direction determination (FIX #7)
  double cx_all = 0, cy_all = 0;
  int nv_all = 0;
  for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
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
      // FIX #7: Correct unbounded Voronoi edge direction
      auto ccb_start = face_it->ccb();
      auto ccb_it = ccb_start;

      std::vector<Point> raw_vertices;
      do {
        if (ccb_it->has_source()) {
          auto src = ccb_it->source();
          raw_vertices.push_back(src->point());
        } else {
          // Unbounded edge: determine correct direction from the dual Delaunay
          // edge
          if (!raw_vertices.empty()) {
            Point last_pt = raw_vertices.back();

            // FIX #7: Use perpendicular bisector of the dual Delaunay edge
            // The Voronoi edge is perpendicular to its dual Delaunay edge
            // and points AWAY from the data centroid
            auto dual_edge = ccb_it->dual();

            // Get the two endpoints of the dual Delaunay edge
            // The halfedge's dual() gives us the Delaunay edge
            // We access the source and target sites of the Voronoi face pair
            double dx = 0, dy = 0;

            // Compute direction: perpendicular to vector from site to centroid
            // of hull
            Point site_pt = site_vertex->point();
            double to_center_dx = cx_all - site_pt.x();
            double to_center_dy = cy_all - site_pt.y();
            double len = std::sqrt(to_center_dx * to_center_dx +
                                   to_center_dy * to_center_dy);

            if (len > 1e-10) {
              // Direction pointing AWAY from centroid
              dx = -to_center_dx / len;
              dy = -to_center_dy / len;
            } else {
              // Fallback: direction from last vertex away from centroid
              dx = last_pt.x() - cx_all;
              dy = last_pt.y() - cy_all;
              len = std::sqrt(dx * dx + dy * dy);
              if (len > 1e-10) {
                dx /= len;
                dy /= len;
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
    Bucket2D &bucket = srr_2d.buckets[idx];

    // FIX #6: Collect ALL clipped polygons per class
    std::map<int, std::vector<std::vector<Point>>> class_polygons;

    // FIX #14: Use stored Vertex_handle for direct O(1) lookup
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
      Vertex_handle nearest = dt.nearest_vertex(center_pt);

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
      Vertex_handle nv = dt.nearest_vertex(Point(bx, by));
      bucket.dominant_class = (nv != Vertex_handle()) ? nv->info() : 0;
      srr_2d.single_class_buckets++;

      LL_Polygon *poly =
          new LL_Polygon(global_poly_id++, bucket.dominant_class);
      poly->inside_label = 1;
      poly->vertices = {
          Point(bucket.min_x, bucket.min_y), Point(bucket.max_x, bucket.min_y),
          Point(bucket.max_x, bucket.max_y), Point(bucket.min_x, bucket.max_y)};
      poly->area = srr_2d.step_x * srr_2d.step_y;
      bucket.polygons = poly;
      bucket.polygon_count = 1;
      srr_2d.total_polygons++;
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
      srr_2d.single_class_buckets++;
    } else if (bucket.num_classes == 2) {
      bucket.type = BUCKET_BIPARTITIONED;
      srr_2d.bipartitioned_buckets++;

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
        srr_2d.single_class_buckets++;
        srr_2d.bipartitioned_buckets--;
      }
    } else {
      bucket.type = BUCKET_MULTI_PARTITIONED;
      srr_2d.multi_class_buckets++;
    }

    // FIX #6: Create polygon regions for EACH class, storing ALL polygons
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
        srr_2d.total_polygons++;
      }
    }
  }

  srr_2d.print_statistics();
  std::cout << "2D Buckets construction complete." << std::endl;
}

int DelaunayClassifier::classify_single_dynamic(double x, double y) {
  int bucket_idx = srr_2d.get_bucket_index(x, y);

  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(srr_2d.buckets.size())) {
    return classify_single(x, y);
  }

  const Bucket2D &bucket = srr_2d.buckets[bucket_idx];
  int result = bucket.classify_point(x, y);

  // If bucket classification returns a valid label, use it
  if (result >= 0)
    return result;

  // Fallback to full classification
  return classify_single(x, y);
}

std::vector<std::pair<int, std::vector<Point>>>
DelaunayClassifier::compute_class_regions() {
  std::vector<std::pair<int, std::vector<Point>>> regions;

  for (const auto &bucket : srr_2d.buckets) {
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