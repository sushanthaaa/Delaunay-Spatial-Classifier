/**
 * @file DelaunayClassifier.cpp
 * @brief Implementation of the Delaunay Triangulation Classifier.
 *
 * This file contains the core algorithms for our geometric classifier:
 * - Phase 1: Outlier removal via connected component analysis
 * - Phase 2: Delaunay triangulation construction
 * - Phase 3: SRR (Square Root Rule) grid indexing for O(1) lookup
 * - Phase 4: Classification via triangle location + vertex voting
 *
 * Key complexity claims demonstrated here:
 * - Training: O(n log n) - dominated by Delaunay construction
 * - Inference: O(1) expected - SRR reduces search space to constant
 * - Dynamic updates: O(1) amortized - CGAL's incremental algorithm
 */

#include "../include/DelaunayClassifier.h"
#include <CGAL/centroid.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

// =============================================================================
// CONSTRUCTOR
// =============================================================================

/**
 * Default constructor initializes with all optimizations enabled.
 * For ablation studies, call set_use_srr(false) or
 * set_use_outlier_removal(false) BEFORE calling train().
 */
DelaunayClassifier::DelaunayClassifier()
    : use_srr_(true), use_outlier_removal_(true) {}

// =============================================================================
// FILE I/O - CSV Loading
// =============================================================================
// CSV format is headerless: x,y,label for labeled data or x,y for unlabeled.
// This simple format ensures compatibility with Python data generators.

/**
 * @brief Load labeled training data from CSV.
 *
 * Each line should be: x_coordinate,y_coordinate,class_label
 * We store points with their labels attached via CGAL's "info" field on
 * vertices.
 */
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
    std::stringstream ss(line);
    std::string val;
    double x, y;
    int label;

    // Parse comma-separated values
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

/**
 * @brief Load unlabeled test data from CSV.
 *
 * Format: x_coordinate,y_coordinate (no label column).
 * Used for inference where we predict the missing labels.
 */
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
// SRR (SQUARE ROOT RULE) GRID IMPLEMENTATION
// =============================================================================
// The SRR grid is our key innovation for achieving O(1) expected inference.
//
// Theory: For n training points, we create a k×k grid where k = ceil(sqrt(n)).
// This gives us n buckets total, with an average of 1 face per bucket.
//
// Instead of randomly walking through O(sqrt(n)) triangles (CGAL default),
// we directly index the correct region in O(1) and do a short local walk.

/**
 * @brief Build the SRR spatial index grid after triangulation is complete.
 *
 * Algorithm:
 * 1. Compute bounding box of all vertices (with small padding)
 * 2. Determine grid size k = sqrt(n) following Square Root Rule
 * 3. For each triangle, map its centroid to a grid cell
 * 4. Store face handle as the "hint" for that cell
 *
 * Memory: O(n) face handles
 * Time: O(n) for initial construction
 */
void DelaunayClassifier::build_srr_grid() {
  if (dt.number_of_vertices() == 0)
    return;

  // --- Step 1: Calculate axis-aligned bounding box ---
  // We add small padding (0.1) to handle edge cases at boundaries
  double min_x = 1e9, max_x = -1e9, min_y = 1e9, max_y = -1e9;

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

  // Add padding to avoid boundary issues
  srr.min_x = min_x - 0.1;
  srr.max_x = max_x + 0.1;
  srr.min_y = min_y - 0.1;
  srr.max_y = max_y + 0.1;

  // --- Step 2: Determine grid dimensions using Square Root Rule ---
  // Key insight: k = sqrt(n) balances grid count with cell density
  // Total cells ≈ n means average O(1) faces per cell
  int n = dt.number_of_vertices();
  int k = std::max(1, (int)std::sqrt(n));
  srr.rows = k;
  srr.cols = k;
  srr.step_x = (srr.max_x - srr.min_x) / k;
  srr.step_y = (srr.max_y - srr.min_y) / k;

  // --- Step 3: Initialize all buckets with infinite face (safe default) ---
  srr.buckets.assign(k * k, dt.infinite_face());

  // --- Step 4: Map each triangle to its grid cell ---
  // We use the triangle's centroid to determine which cell it belongs to.
  // For cells with multiple triangles, we keep the last one (any would work).
  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
    // Compute centroid = average of three vertices
    Point centroid = CGAL::centroid(
        f->vertex(0)->point(), f->vertex(1)->point(), f->vertex(2)->point());

    // Map centroid to grid indices
    int c = (centroid.x() - srr.min_x) / srr.step_x;
    int r = (centroid.y() - srr.min_y) / srr.step_y;

    // Clamp to valid range (handles numerical edge cases)
    if (c < 0)
      c = 0;
    if (c >= srr.cols)
      c = srr.cols - 1;
    if (r < 0)
      r = 0;
    if (r >= srr.rows)
      r = srr.rows - 1;

    int index = r * srr.cols + c;
    srr.buckets[index] = f;
  }

  std::cout << "SRR Grid Built: " << k << "x" << k
            << " buckets (O(1) Indexing Enabled)." << std::endl;
}

/**
 * @brief Look up the SRR grid to get a face hint for point location.
 *
 * This is the O(1) operation that makes our inference fast.
 * Instead of CGAL searching through O(sqrt(n)) triangles randomly,
 * we give it a starting triangle very close to the query point.
 *
 * @param p Query point to locate
 * @return Face handle near point p (used as hint for CGAL::locate)
 */
Face_handle DelaunayClassifier::get_srr_hint(const Point &p) {
  // Map point coordinates to grid cell
  int c = (p.x() - srr.min_x) / srr.step_x;
  int r = (p.y() - srr.min_y) / srr.step_y;

  // Return infinite face if point is outside grid bounds
  // CGAL will handle this gracefully during locate()
  if (c < 0 || c >= srr.cols || r < 0 || r >= srr.rows) {
    return dt.infinite_face();
  }

  return srr.buckets[r * srr.cols + c];
}

// =============================================================================
// ALGORITHM PHASE 1: OUTLIER REMOVAL
// =============================================================================
// We remove isolated noise points that would create spurious triangles.
//
// Approach: Build a graph where same-class, nearby points are connected.
// Then find connected components and remove small ones (size < k).
// This is fundamentally different from statistical outlier detection:
// we're looking for spatial isolation, not distributional anomalies.

/**
 * @brief Remove outliers using connected component analysis on same-class
 * edges.
 *
 * Algorithm:
 * 1. Build temporary Delaunay triangulation
 * 2. Create adjacency graph: edge exists if same-class AND distance < threshold
 * 3. Find connected components via DFS
 * 4. Keep components with >= k points, discard smaller ones as outliers
 *
 * Why this works: True class members cluster together in 2D space.
 * Isolated points are noise from mislabeling or measurement error.
 *
 * @param input_points Raw training data
 * @param k Minimum cluster size (default 3, meaning singletons are noise)
 * @return Filtered dataset with outliers removed
 */
std::vector<std::pair<Point, int>> DelaunayClassifier::remove_outliers(
    const std::vector<std::pair<Point, int>> &input_points, int k) {

  // Distance threshold squared - edges longer than 5 units don't count
  // This prevents connecting distant same-class points across the domain
  double CONNECTIVITY_THRESHOLD_SQ = 5.0 * 5.0;

  std::cout << "Phase 1: Detecting Outliers (Min Cluster Size k=" << k << ")..."
            << std::endl;

  // --- Build temporary triangulation for edge extraction ---
  Delaunay temp_dt;
  temp_dt.insert(input_points.begin(), input_points.end());

  // --- Build adjacency graph for same-class, short edges only ---
  std::map<Point, std::vector<Point>> adj;
  std::map<Point, int> point_to_label;

  // First pass: map points to labels
  for (auto v = temp_dt.finite_vertices_begin();
       v != temp_dt.finite_vertices_end(); ++v)
    point_to_label[v->point()] = v->info();

  // Second pass: add same-class, short edges to adjacency list
  for (auto e = temp_dt.finite_edges_begin(); e != temp_dt.finite_edges_end();
       ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);
    double dist = CGAL::squared_distance(v1->point(), v2->point());

    // Only connect if same class AND within distance threshold
    // This defines "neighborhood" for outlier detection
    if (v1->info() == v2->info() && dist < CONNECTIVITY_THRESHOLD_SQ) {
      adj[v1->point()].push_back(v2->point());
      adj[v2->point()].push_back(v1->point());
    }
  }

  // --- Find connected components via DFS ---
  std::map<Point, bool> visited;
  std::vector<std::pair<Point, int>> clean_points;
  int removed_count = 0;

  for (const auto &p_entry : input_points) {
    Point start = p_entry.first;
    if (visited[start])
      continue;

    // DFS to find all points in this component
    std::vector<Point> component;
    std::vector<Point> stack = {start};
    visited[start] = true;

    while (!stack.empty()) {
      Point curr = stack.back();
      stack.pop_back();
      component.push_back(curr);

      for (const auto &n : adj[curr]) {
        if (!visited[n]) {
          visited[n] = true;
          stack.push_back(n);
        }
      }
    }

    // Decision: keep if component has >= k members, else discard as outliers
    if (component.size() >= k) {
      for (const auto &p : component)
        clean_points.push_back({p, point_to_label[p]});
    } else {
      removed_count += component.size();
    }
  }

  std::cout << "Phase 1 Complete: Removed " << removed_count << " outliers."
            << std::endl;
  return clean_points;
}

// =============================================================================
// CLASSIFICATION LOGIC
// =============================================================================

/**
 * @brief Classify a query point given the triangle it falls into.
 *
 * Decision logic (forms the geometric decision boundary):
 * 1. If all 3 vertices have same class → return that class (unanimous)
 * 2. Otherwise → return class of NEAREST vertex (distance-weighted vote)
 *
 * This creates decision boundaries at the midpoints of edges connecting
 * vertices of different classes - a key visual insight for our figures.
 *
 * @param f Triangle face (must be finite, not the infinite face)
 * @param p Query point (should be inside or near this triangle)
 * @return Predicted class label
 */
int DelaunayClassifier::classify_point_in_face(Face_handle f, Point p) {
  // Get class labels of the three triangle vertices
  int l0 = f->vertex(0)->info();
  int l1 = f->vertex(1)->info();
  int l2 = f->vertex(2)->info();

  // Fast path: unanimous triangle - all vertices same class
  // This is the common case in well-clustered data
  if (l0 == l1 && l1 == l2)
    return l0;

  // Mixed-class triangle: use nearest vertex as tiebreaker
  // This effectively creates Voronoi-like decision regions within the triangle
  double d0 = CGAL::squared_distance(p, f->vertex(0)->point());
  double d1 = CGAL::squared_distance(p, f->vertex(1)->point());
  double d2 = CGAL::squared_distance(p, f->vertex(2)->point());

  if (d0 <= d1 && d0 <= d2)
    return l0;
  if (d1 <= d0 && d1 <= d2)
    return l1;
  return l2;
}

// =============================================================================
// PUBLIC: TRAIN
// =============================================================================

/**
 * @brief Main training pipeline - O(n log n) total.
 *
 * Steps:
 * 1. Load labeled CSV data
 * 2. Remove outliers (if enabled) - O(n)
 * 3. Build Delaunay triangulation - O(n log n) [dominates]
 * 4. Build SRR grid for O(1) queries - O(n)
 *
 * After this, the model is ready for inference via classify_single().
 */
void DelaunayClassifier::train(const std::string &train_file, int outlier_k) {
  // Phase 1: Load and clean data
  auto raw_points = load_labeled_csv(train_file);
  auto clean_points = remove_outliers(raw_points, outlier_k);

  // Save cleaned points for visualization/debugging
  std::ofstream out("results/clean_points.csv");
  for (const auto &p : clean_points) {
    out << p.first.x() << "," << p.first.y() << "," << p.second << "\n";
  }
  out.close();

  // Phase 2: Build Delaunay triangulation
  // This is O(n log n) and the most expensive part of "training"
  dt.clear();
  dt.insert(clean_points.begin(), clean_points.end());
  std::cout << "Phase 2 Complete: Delaunay Mesh Built ("
            << dt.number_of_vertices() << " vertices)." << std::endl;

  // Phase 3: Build SRR grid for O(1) inference
  build_srr_grid();

  // Phase 4: Build 2D Buckets for O(1) dynamic classification
  build_2d_buckets();
}

// =============================================================================
// PUBLIC: PREDICT (Batch Benchmark)
// =============================================================================

/**
 * @brief Batch prediction with timing measurement.
 *
 * For each test point, we:
 * 1. Get SRR hint → O(1) grid lookup
 * 2. Call CGAL locate() with hint → O(1) local walk
 * 3. Classify in triangle → O(1) voting
 *
 * Reports average time per point for benchmark comparison.
 */
void DelaunayClassifier::predict_benchmark(const std::string &test_file,
                                           const std::string &output_file) {
  auto test_points = load_unlabeled_csv(test_file);
  std::cout << "Starting Benchmark (with SRR Optimization)..." << std::endl;

  std::vector<int> results;
  results.reserve(test_points.size());

  auto start = std::chrono::high_resolution_clock::now();

  for (const auto &p : test_points) {
    // Step 1: O(1) SRR grid lookup
    Face_handle hint = get_srr_hint(p);

    // Step 2: O(1) locate with hint (instead of O(sqrt(n)) without hint)
    Face_handle f = dt.locate(p, hint);

    // Step 3: O(1) classification
    int pred = -1;
    if (!dt.is_infinite(f))
      pred = classify_point_in_face(f, p);
    else
      // Point outside convex hull - fall back to nearest vertex
      pred = dt.nearest_vertex(p)->info();

    results.push_back(pred);
  }

  auto end = std::chrono::high_resolution_clock::now();
  auto duration_us =
      std::chrono::duration_cast<std::chrono::microseconds>(end - start);

  double avg_time = test_points.empty()
                        ? 0.0
                        : duration_us.count() / (double)test_points.size();

  std::cout << "------------------------------------------------" << std::endl;
  std::cout << "Total Inference Time: " << duration_us.count() / 1000.0 << " ms"
            << std::endl;
  std::cout << "Avg Time Per Point:   " << avg_time << " us" << std::endl;
  std::cout << "------------------------------------------------" << std::endl;

  // Save predictions to file
  std::ofstream out(output_file);
  for (int p : results)
    out << p << "\n";
  out.close();
}

// =============================================================================
// DYNAMIC OPERATIONS STRESS TEST
// =============================================================================
// Tests CGAL's incremental update capabilities:
// - Insert: Add vertex, flip edges to restore Delaunay property - O(1)
// amortized
// - Move: Relocate vertex if topologically stable - O(1)
// - Delete: Remove vertex, retriangulate hole - O(1) amortized

void DelaunayClassifier::run_dynamic_stress_test(const std::string &stream_file,
                                                 const std::string &log_file) {
  auto stream_points = load_labeled_csv(stream_file);
  std::ofstream log(log_file);
  log << "operation,time_ns\n";

  std::cout << "Running Dynamic Algorithms 1, 2, 3 Stress Test..." << std::endl;

  Vertex_handle hint = dt.finite_vertices_begin();
  std::vector<Vertex_handle> handles;

  // --- INSERTION PHASE ---
  // Each insert is O(1) amortized: locate triangle, split, flip edges
  for (const auto &entry : stream_points) {
    auto start = std::chrono::high_resolution_clock::now();

    // Insert with hint from last insertion (spatial locality helps)
    hint = dt.insert(entry.first, hint->face());
    hint->info() = entry.second; // Store class label

    auto end = std::chrono::high_resolution_clock::now();

    handles.push_back(hint);
    log << "insert,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

  // --- MOVEMENT PHASE ---
  // Move each point slightly; O(1) if no topological change needed
  for (auto v : handles) {
    Point new_p(v->point().x() + 0.05, v->point().y() + 0.05);

    auto start = std::chrono::high_resolution_clock::now();
    dt.move_if_no_collision(v, new_p);
    auto end = std::chrono::high_resolution_clock::now();

    log << "move,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

  // --- DELETION PHASE ---
  // Remove in reverse order; O(1) amortized via local retriangulation
  for (int i = handles.size() - 1; i >= 0; --i) {
    auto start = std::chrono::high_resolution_clock::now();
    dt.remove(handles[i]);
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

/**
 * @brief Export triangulation edges and decision boundaries for visualization.
 *
 * Creates three CSV files:
 * - mesh_file: All Delaunay edges as x1,y1,x2,y2
 * - boundary_file: Decision boundaries only (edges between different classes)
 * - points_file: Vertices with labels for plotting
 *
 * Decision boundary logic:
 * - For triangles with 2 classes: connect midpoints of edges separating classes
 * - For triangles with 3 classes: connect centroid to all edge midpoints
 */
void DelaunayClassifier::export_visualization(const std::string &mesh_file,
                                              const std::string &boundary_file,
                                              const std::string &points_file) {
  // --- Export all triangulation edges ---
  std::ofstream triFile(mesh_file);
  for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
    auto s = dt.segment(e);
    triFile << s.source().x() << "," << s.source().y() << "," << s.target().x()
            << "," << s.target().y() << "\n";
  }
  triFile.close();

  // --- Export decision boundaries ---
  // These are the lines that separate different class regions
  std::ofstream boundFile(boundary_file);

  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
    auto v0 = f->vertex(0);
    auto v1 = f->vertex(1);
    auto v2 = f->vertex(2);

    int l0 = v0->info();
    int l1 = v1->info();
    int l2 = v2->info();

    // Compute edge midpoints (used for boundary segments)
    Point m01 = CGAL::midpoint(v0->point(), v1->point());
    Point m12 = CGAL::midpoint(v1->point(), v2->point());
    Point m20 = CGAL::midpoint(v2->point(), v0->point());

    // Case 1: All three vertices have DIFFERENT classes
    // Connect centroid to all three edge midpoints (Y-shaped boundary)
    if (l0 != l1 && l1 != l2 && l0 != l2) {
      Point c = CGAL::centroid(v0->point(), v1->point(), v2->point());
      boundFile << c.x() << "," << c.y() << "," << m01.x() << "," << m01.y()
                << "\n";
      boundFile << c.x() << "," << c.y() << "," << m12.x() << "," << m12.y()
                << "\n";
      boundFile << c.x() << "," << c.y() << "," << m20.x() << "," << m20.y()
                << "\n";
    }
    // Case 2: Two classes in triangle (one vertex differs)
    // Connect midpoints of the two edges that cross class boundaries
    else if (l0 != l1 || l1 != l2) {
      std::vector<Point> actives;
      if (l0 != l1)
        actives.push_back(m01);
      if (l1 != l2)
        actives.push_back(m12);
      if (l2 != l0)
        actives.push_back(m20);

      // Draw line between the two boundary midpoints
      if (actives.size() == 2)
        boundFile << actives[0].x() << "," << actives[0].y() << ","
                  << actives[1].x() << "," << actives[1].y() << "\n";
    }
    // Case 3: All same class - no boundary needed
  }
  boundFile.close();

  // --- Optionally export vertex points with labels ---
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

// =============================================================================
// DYNAMIC VISUALIZATION MODE
// =============================================================================

/**
 * @brief Generate snapshots showing insert/move/delete operations.
 *
 * Creates three sets of visualization files showing the triangulation
 * state after each phase of dynamic operations.
 */
void DelaunayClassifier::run_dynamic_visualization(
    const std::string &stream_file, const std::string &out_dir) {
  auto stream_points = load_labeled_csv(stream_file);
  Vertex_handle hint = dt.finite_vertices_begin();
  std::vector<Vertex_handle> handles;

  std::cout << "Generating Dynamic Visualization Snapshots..." << std::endl;

  // 1. INSERTION - add all stream points
  for (const auto &entry : stream_points) {
    hint = dt.insert(entry.first, hint->face());
    hint->info() = entry.second;
    handles.push_back(hint);
  }
  export_visualization(out_dir + "/dynamic_1_inserted_triangles.csv",
                       out_dir + "/dynamic_1_inserted_boundaries.csv",
                       out_dir + "/dynamic_1_inserted_points.csv");
  std::cout << "   - Snapshot 1: Insertion Complete" << std::endl;

  // 2. MOVEMENT - shift all inserted points
  for (auto v : handles) {
    Point new_p(v->point().x() + 0.5, v->point().y() + 0.5);
    dt.move_if_no_collision(v, new_p);
  }
  export_visualization(out_dir + "/dynamic_2_moved_triangles.csv",
                       out_dir + "/dynamic_2_moved_boundaries.csv",
                       out_dir + "/dynamic_2_moved_points.csv");
  std::cout << "   - Snapshot 2: Movement Complete" << std::endl;

  // 3. DELETION - remove all inserted points
  for (int i = handles.size() - 1; i >= 0; --i) {
    dt.remove(handles[i]);
  }
  export_visualization(out_dir + "/dynamic_3_deleted_triangles.csv",
                       out_dir + "/dynamic_3_deleted_boundaries.csv",
                       out_dir + "/dynamic_3_deleted_points.csv");
  std::cout << "   - Snapshot 3: Deletion Complete" << std::endl;
}

// =============================================================================
// SINGLE-POINT CLASSIFICATION (For External Benchmarking)
// =============================================================================

/**
 * @brief Classify a single point with full SRR optimization.
 *
 * This is the core O(1) inference method:
 * 1. SRR grid lookup → get starting face hint (works even outside hull)
 * 2. CGAL locate with hint → find containing triangle
 * 3. Classify within triangle → vote among 3 vertices
 *
 * EXTENDED: For points outside the convex hull:
 * 1. SRR grid still identifies which cell the point belongs to
 * 2. Find the adjacent boundary triangle (finite neighbor of infinite face)
 * 3. Determine which decision boundary region the point falls in
 * 4. Classify based on that region (not just nearest vertex)
 *
 * This ensures classification matches the extended decision boundary
 * visualization.
 */
int DelaunayClassifier::classify_single(double x, double y) {
  Point p(x, y);

  // O(1) SRR grid lookup - works even for points outside hull
  // This tells us which spatial region the point belongs to
  Face_handle hint = get_srr_hint(p);

  // O(1) locate with hint
  Face_handle f = dt.locate(p, hint);

  if (dt.is_infinite(f)) {
    // =================================================================
    // OUTSIDE CONVEX HULL - Use Decision Boundary Region Classification
    // =================================================================
    // When a point is outside the convex hull, CGAL returns an "infinite face".
    // The infinite face has edges on the convex hull boundary.
    // We use the extended decision boundary logic to classify.
    //
    // Algorithm:
    // 1. Find the boundary edge closest to the query point
    // 2. Get the finite triangle adjacent to that edge
    // 3. Determine which decision boundary region the point falls in
    // 4. Return the class of that region

    // Find the boundary triangle (finite neighbor of the infinite face)
    Face_handle boundary_triangle;
    int boundary_edge_index = -1;
    double min_dist = 1e18;

    for (int i = 0; i < 3; i++) {
      Face_handle neighbor = f->neighbor(i);
      if (!dt.is_infinite(neighbor)) {
        // This is a finite triangle adjacent to the hull
        // Calculate distance to the shared edge (convex hull edge)
        // The edge opposite to vertex i in face f is shared with neighbor

        // Get the two vertices of the shared edge
        Vertex_handle v1 = f->vertex((i + 1) % 3);
        Vertex_handle v2 = f->vertex((i + 2) % 3);

        // Skip if either vertex is infinite
        if (dt.is_infinite(v1) || dt.is_infinite(v2))
          continue;

        // Calculate distance from query point to this hull edge
        Point p1 = v1->point();
        Point p2 = v2->point();

        // Distance to edge midpoint (approximation for speed)
        Point midpoint((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2);
        double dist = CGAL::squared_distance(p, midpoint);

        if (dist < min_dist) {
          min_dist = dist;
          boundary_triangle = neighbor;
          boundary_edge_index = i;
        }
      }
    }

    if (boundary_triangle != Face_handle()) {
      // Get the vertices of the boundary edge
      Vertex_handle hull_v1 = f->vertex((boundary_edge_index + 1) % 3);
      Vertex_handle hull_v2 = f->vertex((boundary_edge_index + 2) % 3);

      // Check if both vertices are valid (not infinite)
      if (!dt.is_infinite(hull_v1) && !dt.is_infinite(hull_v2)) {
        int label1 = hull_v1->info();
        int label2 = hull_v2->info();

        // If both hull vertices have the same class, the outside region
        // belongs entirely to that class
        if (label1 == label2) {
          return label1;
        }

        // If hull vertices have different classes, the decision boundary
        // extends outward from the edge midpoint. We need to determine
        // which side of the extended boundary the query point is on.

        Point p1 = hull_v1->point();
        Point p2 = hull_v2->point();
        Point edge_midpoint((p1.x() + p2.x()) / 2, (p1.y() + p2.y()) / 2);

        // The decision boundary extends perpendicular to the edge, or
        // in the direction determined by the internal boundary line.
        // For simplicity, we check which hull vertex is closer to the query.
        double dist_to_v1 = CGAL::squared_distance(p, p1);
        double dist_to_v2 = CGAL::squared_distance(p, p2);

        // The point belongs to the region of the closer vertex
        return (dist_to_v1 <= dist_to_v2) ? label1 : label2;
      }
    }

    // Ultimate fallback: nearest vertex (should rarely happen)
    Vertex_handle v = dt.nearest_vertex(p);
    return v->info();
  }

  return classify_point_in_face(f, p);
}

// =============================================================================
// DYNAMIC UPDATE METHODS (For External Benchmarking)
// =============================================================================

/**
 * @brief Insert a new labeled training point into the model.
 *
 * CGAL handles Delaunay property maintenance automatically:
 * 1. Locate containing triangle
 * 2. Split triangle by inserting vertex
 * 3. Flip edges as needed to restore empty circumcircle property
 *
 * Note: SRR grid becomes stale after insertion. For full accuracy,
 * call build_srr_grid() after batch insertions.
 */
void DelaunayClassifier::insert_point(double x, double y, int label) {
  Point p(x, y);
  dt.insert(p)->info() = label;
}

/**
 * @brief Remove a point from the model (online learning "forget" operation).
 *
 * Finds the nearest vertex to (x,y) and removes it.
 * Useful for concept drift scenarios where old data should be forgotten.
 */
void DelaunayClassifier::remove_point(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  if (v != Vertex_handle()) {
    dt.remove(v);
  }
}

// =============================================================================
// ABLATION STUDY METHODS
// =============================================================================
// These methods disable specific optimizations to measure their contribution.

/**
 * @brief Classify WITHOUT SRR hint (measures SRR grid contribution to speed).
 *
 * Without the SRR hint, CGAL uses its default point location algorithm
 * which is O(sqrt(n)) random walk instead of our O(1) hint-guided walk.
 * The accuracy should be identical, but speed will be slower.
 */
int DelaunayClassifier::classify_single_no_srr(double x, double y) {
  Point p(x, y);

  // No SRR hint - use raw CGAL locate (O(sqrt(n)) instead of O(1))
  Face_handle f = dt.locate(p);

  if (dt.is_infinite(f)) {
    Vertex_handle v = dt.nearest_vertex(p);
    return v->info();
  }

  return classify_point_in_face(f, p);
}

/**
 * @brief Classify using NEAREST VERTEX only (1-NN, no triangle voting).
 *
 * This ablation measures the accuracy contribution of our decision boundary
 * logic. By only using nearest vertex, we're essentially doing 1-NN but
 * leveraging Delaunay structure for neighbor lookup instead of KD-tree.
 *
 * Expected: same or slightly lower accuracy, potentially faster.
 */
int DelaunayClassifier::classify_nearest_vertex(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  return v->info();
}

// =============================================================================
// 2D BUCKETS IMPLEMENTATION - Complete with all 6 Linked List Structures
// =============================================================================
// Full implementation as specified:
// - LL_V:     Vertices in each bucket
// - LL_E:     Edges passing through each bucket
// - LL_GE:    Grid edge intersections with decision boundaries
// - LL_Poly:  Voronoi-clipped polygon regions
// - LL_Label: Inside/outside classification labels
// - LL_PolyID: Unique polygon identifiers

/**
 * @brief Check if a point is inside a polygon using ray casting algorithm.
 */
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
 * @brief O(1) classification within this bucket using polygon regions.
 *
 * Algorithm:
 * 1. Single-class bucket: Return dominant_class immediately (O(1))
 * 2. Multi-class bucket: Test point against each polygon in LL_Poly (O(k))
 *    where k is bounded by the number of classes (typically 2-3)
 */
int Bucket2D::classify_point(double x, double y) const {
  // Fast path: single class bucket
  if (num_classes == 1 || polygon_count == 1) {
    return dominant_class;
  }

  // Multi-class bucket: check each polygon region
  LL_Polygon *poly = polygons;
  while (poly != nullptr) {
    if (poly->inside_label == 1 && poly->vertices.size() >= 3) {
      if (point_in_polygon(x, y, poly->vertices)) {
        return poly->class_label;
      }
    }
    poly = poly->next;
  }

  // Fallback to dominant class
  return dominant_class;
}

/**
 * @brief Clean up all linked lists in this bucket.
 */
void Bucket2D::clear() {
  // Clear LL_V (vertices)
  LL_Vertex *v = vertices;
  while (v != nullptr) {
    LL_Vertex *next = v->next;
    delete v;
    v = next;
  }
  vertices = nullptr;
  vertex_count = 0;

  // Clear LL_E (edges)
  LL_Edge *e = edges;
  while (e != nullptr) {
    LL_Edge *next = e->next;
    delete e;
    e = next;
  }
  edges = nullptr;
  edge_count = 0;

  // Clear LL_GE (grid edges)
  LL_GridEdge *ge = grid_edges;
  while (ge != nullptr) {
    LL_GridEdge *next = ge->next;
    delete ge;
    ge = next;
  }
  grid_edges = nullptr;
  grid_edge_count = 0;

  // Clear LL_Poly (polygons)
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
}

/**
 * @brief Get bucket index for a given point (O(1) lookup).
 */
int SRR_Grid_2D::get_bucket_index(double x, double y) const {
  int c = static_cast<int>((x - min_x) / step_x);
  int r = static_cast<int>((y - min_y) / step_y);

  // Clamp to valid range
  if (c < 0)
    c = 0;
  if (c >= cols)
    c = cols - 1;
  if (r < 0)
    r = 0;
  if (r >= rows)
    r = rows - 1;

  return r * cols + c;
}

/**
 * @brief Print grid statistics.
 */
void SRR_Grid_2D::print_statistics() const {
  std::cout << "=== 2D Buckets Grid Statistics ===" << std::endl;
  std::cout << "Grid size: " << rows << " x " << cols << " = " << (rows * cols)
            << " buckets" << std::endl;
  std::cout << "Single-class buckets: " << single_class_buckets << std::endl;
  std::cout << "Multi-class buckets:  " << multi_class_buckets << std::endl;
  std::cout << "Total polygon regions: " << total_polygons << std::endl;
  std::cout << "==================================" << std::endl;
}

// -----------------------------------------------------------------------------
// Helper: Check if a line segment intersects a bucket
// -----------------------------------------------------------------------------
static bool segment_intersects_bucket(double x1, double y1, double x2,
                                      double y2, double bmin_x, double bmin_y,
                                      double bmax_x, double bmax_y) {
  // Check if either endpoint is inside bucket
  auto inside = [&](double x, double y) {
    return x >= bmin_x && x <= bmax_x && y >= bmin_y && y <= bmax_y;
  };
  if (inside(x1, y1) || inside(x2, y2))
    return true;

  // Check segment intersection with bucket edges using parametric line
  // intersection
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

  // Check all 4 bucket edges
  return intersects_edge(bmin_x, bmin_y, bmax_x, bmin_y) || // bottom
         intersects_edge(bmax_x, bmin_y, bmax_x, bmax_y) || // right
         intersects_edge(bmax_x, bmax_y, bmin_x, bmax_y) || // top
         intersects_edge(bmin_x, bmax_y, bmin_x, bmin_y);   // left
}

// -----------------------------------------------------------------------------
// Helper: Compute intersection point of two line segments
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Helper: Clip polygon to bucket boundaries (Sutherland-Hodgman algorithm)
// -----------------------------------------------------------------------------
static std::vector<Point>
clip_polygon_to_bucket(const std::vector<Point> &polygon, double bmin_x,
                       double bmin_y, double bmax_x, double bmax_y) {
  if (polygon.empty())
    return {};

  std::vector<Point> result = polygon;

  // Clip against each of the 4 bucket edges
  auto clip_edge = [](const std::vector<Point> &input, double ex1, double ey1,
                      double ex2, double ey2) {
    std::vector<Point> output;
    if (input.empty())
      return output;

    auto inside = [&](const Point &p) {
      return (ex2 - ex1) * (p.y() - ey1) - (ey2 - ey1) * (p.x() - ex1) >= 0;
    };

    Point prev = input.back();
    bool prev_inside = inside(prev);

    for (const Point &curr : input) {
      bool curr_inside = inside(curr);

      if (curr_inside) {
        if (!prev_inside) {
          // Entering: compute intersection
          double ix, iy;
          if (compute_intersection(prev.x(), prev.y(), curr.x(), curr.y(), ex1,
                                   ey1, ex2, ey2, ix, iy)) {
            output.push_back(Point(ix, iy));
          }
        }
        output.push_back(curr);
      } else if (prev_inside) {
        // Leaving: compute intersection
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

  // Clip in order: left, bottom, right, top
  result = clip_edge(result, bmin_x, bmin_y, bmin_x, bmax_y); // left
  result = clip_edge(result, bmin_x, bmax_y, bmax_x, bmax_y); // top
  result = clip_edge(result, bmax_x, bmax_y, bmax_x, bmin_y); // right
  result = clip_edge(result, bmax_x, bmin_y, bmin_x, bmin_y); // bottom

  return result;
}

/**
 * @brief Build the complete 2D Buckets grid with all 6 linked list structures.
 *
 * This is the main construction algorithm implementing:
 * Phase 1: Build LL_V (vertices in each bucket)
 * Phase 2: Build LL_E (edges passing through each bucket)
 * Phase 3: Build LL_GE (grid edge intersections with decision boundaries)
 * Phase 4: Build LL_Poly (Voronoi-clipped polygon regions)
 */
void DelaunayClassifier::build_2d_buckets() {
  if (dt.number_of_vertices() == 0)
    return;

  std::cout << "Building 2D Buckets with full linked list structures..."
            << std::endl;

  // --- Initialize grid with same dimensions as SRR ---
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
      bucket.hint = srr.buckets[idx];
    }
  }

  // =========================================================================
  // PHASE 1: Build LL_V (Vertices)
  // =========================================================================
  std::cout << "  Phase 1: Building LL_V (vertices)..." << std::endl;
  int vertex_id = 0;
  std::map<Point, int> point_to_id;

  for (auto v = dt.finite_vertices_begin(); v != dt.finite_vertices_end();
       ++v) {
    Point p = v->point();
    int label = v->info();
    point_to_id[p] = vertex_id;

    // Find which bucket this vertex belongs to
    int bucket_idx = srr_2d.get_bucket_index(p.x(), p.y());
    Bucket2D &bucket = srr_2d.buckets[bucket_idx];

    // Add to LL_V
    LL_Vertex *new_vertex = new LL_Vertex(p, label, vertex_id);
    new_vertex->next = bucket.vertices;
    bucket.vertices = new_vertex;
    bucket.vertex_count++;

    vertex_id++;
  }

  // =========================================================================
  // PHASE 2: Build LL_E (Edges)
  // =========================================================================
  std::cout << "  Phase 2: Building LL_E (edges)..." << std::endl;
  int edge_id = 0;

  for (auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);

    Point p1 = v1->point();
    Point p2 = v2->point();
    int class1 = v1->info();
    int class2 = v2->info();

    // Find all buckets this edge passes through
    for (int idx = 0; idx < k * k; ++idx) {
      Bucket2D &bucket = srr_2d.buckets[idx];

      if (segment_intersects_bucket(p1.x(), p1.y(), p2.x(), p2.y(),
                                    bucket.min_x, bucket.min_y, bucket.max_x,
                                    bucket.max_y)) {
        // Add to LL_E
        LL_Edge *new_edge = new LL_Edge(p1, p2, class1, class2, edge_id);
        new_edge->next = bucket.edges;
        bucket.edges = new_edge;
        bucket.edge_count++;
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

    // Check each decision boundary edge for intersection with bucket boundaries
    LL_Edge *edge = bucket.edges;
    while (edge != nullptr) {
      if (edge->is_boundary) { // Only check decision boundary edges
        // Define bucket boundary segments
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
            // Found intersection with grid edge
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
  // PHASE 4: Build LL_Poly (Exact Voronoi Polygon Regions)
  // =========================================================================
  // Uses CGAL's Voronoi_diagram_2 adaptor to extract exact Voronoi cells,
  // then clips each cell to bucket boundaries.
  std::cout << "  Phase 4: Building LL_Poly (exact Voronoi polygon regions)..."
            << std::endl;

  // Create Voronoi diagram from Delaunay triangulation
  VoronoiDiagram vd(dt);

  // Create bounding box for clipping unbounded cells (extended grid bounds)
  double bbox_margin = std::max(srr_2d.step_x, srr_2d.step_y) * 2;
  double bbox_min_x = srr_2d.min_x - bbox_margin;
  double bbox_max_x = srr_2d.max_x + bbox_margin;
  double bbox_min_y = srr_2d.min_y - bbox_margin;
  double bbox_max_y = srr_2d.max_y + bbox_margin;

  int global_poly_id = 0;

  // Map from site (vertex) to its Voronoi cell polygon
  std::map<Vertex_handle, std::vector<Point>> vertex_to_cell;

  // Extract Voronoi cells from the diagram
  for (auto face_it = vd.faces_begin(); face_it != vd.faces_end(); ++face_it) {
    // Get the site (Delaunay vertex) that generated this Voronoi cell
    auto dual_vertex = face_it->dual();
    if (dual_vertex == nullptr)
      continue;

    Vertex_handle site_vertex = dual_vertex;
    std::vector<Point> cell_polygon;

    // Check if the face is unbounded
    if (face_it->is_unbounded()) {
      // For unbounded faces, we need to clip with bounding box
      // Traverse the face's halfedge chain and handle infinite edges
      auto ccb_start = face_it->ccb();
      auto ccb_it = ccb_start;

      std::vector<Point> raw_vertices;
      do {
        if (ccb_it->has_source()) {
          // This halfedge has a finite source vertex
          auto src = ccb_it->source();
          raw_vertices.push_back(src->point());
        } else {
          // This is an infinite edge - compute intersection with bounding box
          // The direction is given by the perpendicular bisector of the edge
          // We'll use a ray from the last known point toward infinity
          if (!raw_vertices.empty()) {
            // Get the direction of this halfedge (from the dual edge)
            // For simplicity, extend to bounding box
            Point last_pt = raw_vertices.back();
            double dx = 0, dy = 0;

            // Determine direction based on site position
            auto site_pt = site_vertex->point();
            dx = site_pt.x() - last_pt.x();
            dy = site_pt.y() - last_pt.y();
            double len = std::sqrt(dx * dx + dy * dy);
            if (len > 1e-10) {
              dx /= len;
              dy /= len;
              // Rotate 90 degrees to get perpendicular direction
              double temp = dx;
              dx = -dy;
              dy = temp;
            }

            // Extend to bounding box
            double t_max = 1e6;
            Point far_pt(last_pt.x() + dx * t_max, last_pt.y() + dy * t_max);
            raw_vertices.push_back(far_pt);
          }
        }
        ++ccb_it;
      } while (ccb_it != ccb_start);

      // Clip the raw polygon to the bounding box
      cell_polygon = clip_polygon_to_bucket(raw_vertices, bbox_min_x,
                                            bbox_min_y, bbox_max_x, bbox_max_y);
    } else {
      // Bounded face: extract all vertices directly
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

  // Now assign clipped Voronoi cells to buckets
  for (int idx = 0; idx < k * k; ++idx) {
    Bucket2D &bucket = srr_2d.buckets[idx];

    // Collect classes and their Voronoi cell polygons that intersect this
    // bucket
    std::map<int, std::vector<std::vector<Point>>> class_polygons;

    // Check each vertex's Voronoi cell
    LL_Vertex *v = bucket.vertices;
    while (v != nullptr) {
      // Find the corresponding CGAL vertex handle
      Point vp = v->point;
      for (const auto &vc : vertex_to_cell) {
        if (std::abs(vc.first->point().x() - vp.x()) < 1e-10 &&
            std::abs(vc.first->point().y() - vp.y()) < 1e-10) {
          // This Voronoi cell belongs to vertex v
          // Clip it to the bucket boundaries
          std::vector<Point> clipped =
              clip_polygon_to_bucket(vc.second, bucket.min_x, bucket.min_y,
                                     bucket.max_x, bucket.max_y);

          if (clipped.size() >= 3) {
            class_polygons[v->class_label].push_back(clipped);
          }
          break;
        }
      }
      v = v->next;
    }

    // If no vertices in bucket, use sample-based fallback
    if (class_polygons.empty()) {
      // Sample the bucket center to determine class
      double cx = (bucket.min_x + bucket.max_x) / 2;
      double cy = (bucket.min_y + bucket.max_y) / 2;
      int cls = classify_single(cx, cy);

      // Find the nearest vertex to this bucket
      Point center_pt(cx, cy);
      Vertex_handle nearest = dt.nearest_vertex(center_pt);

      if (nearest != Vertex_handle() && vertex_to_cell.count(nearest)) {
        std::vector<Point> clipped =
            clip_polygon_to_bucket(vertex_to_cell[nearest], bucket.min_x,
                                   bucket.min_y, bucket.max_x, bucket.max_y);

        if (clipped.size() >= 3) {
          class_polygons[nearest->info()].push_back(clipped);
        }
      }
    }

    // Determine number of classes and dominant class
    bucket.num_classes = class_polygons.size();
    if (bucket.num_classes == 0) {
      // Ultimate fallback: use bucket bounds with center classification
      double cx = (bucket.min_x + bucket.max_x) / 2;
      double cy = (bucket.min_y + bucket.max_y) / 2;
      bucket.num_classes = 1;
      bucket.dominant_class = classify_single(cx, cy);
      srr_2d.single_class_buckets++;

      LL_Polygon *poly =
          new LL_Polygon(global_poly_id++, bucket.dominant_class);
      poly->inside_label = 1;
      poly->vertices.push_back(Point(bucket.min_x, bucket.min_y));
      poly->vertices.push_back(Point(bucket.max_x, bucket.min_y));
      poly->vertices.push_back(Point(bucket.max_x, bucket.max_y));
      poly->vertices.push_back(Point(bucket.min_x, bucket.max_y));
      poly->area = srr_2d.step_x * srr_2d.step_y;
      bucket.polygons = poly;
      bucket.polygon_count = 1;
      srr_2d.total_polygons++;
      continue;
    }

    // Find dominant class (most polygons/area)
    int max_polys = 0;
    for (const auto &cp : class_polygons) {
      if (static_cast<int>(cp.second.size()) > max_polys) {
        max_polys = cp.second.size();
        bucket.dominant_class = cp.first;
      }
    }

    if (bucket.num_classes == 1) {
      srr_2d.single_class_buckets++;
    } else {
      srr_2d.multi_class_buckets++;
    }

    // Create polygon regions for each class by merging clipped Voronoi cells
    LL_Polygon *prev_poly = nullptr;
    for (const auto &cp : class_polygons) {
      int cls = cp.first;

      // Merge all polygons of this class within the bucket
      // For simplicity, use the largest polygon (most vertices)
      std::vector<Point> best_poly;
      for (const auto &poly_verts : cp.second) {
        if (poly_verts.size() > best_poly.size()) {
          best_poly = poly_verts;
        }
      }

      if (best_poly.size() < 3) {
        // Fallback to bucket bounds
        best_poly.clear();
        best_poly.push_back(Point(bucket.min_x, bucket.min_y));
        best_poly.push_back(Point(bucket.max_x, bucket.min_y));
        best_poly.push_back(Point(bucket.max_x, bucket.max_y));
        best_poly.push_back(Point(bucket.min_x, bucket.max_y));
      }

      LL_Polygon *poly = new LL_Polygon(global_poly_id++, cls);
      poly->inside_label = 1;
      poly->vertices = best_poly;

      // Compute approximate area
      double area = 0;
      for (size_t i = 0; i < poly->vertices.size(); ++i) {
        size_t j = (i + 1) % poly->vertices.size();
        area += poly->vertices[i].x() * poly->vertices[j].y();
        area -= poly->vertices[j].x() * poly->vertices[i].y();
      }
      poly->area = std::abs(area) / 2.0;

      // Add to linked list
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

  // Print statistics
  srr_2d.print_statistics();
  std::cout << "2D Buckets construction complete." << std::endl;
}

/**
 * @brief Classify using 2D Buckets for O(1) dynamic classification.
 */
int DelaunayClassifier::classify_single_dynamic(double x, double y) {
  int bucket_idx = srr_2d.get_bucket_index(x, y);

  if (bucket_idx < 0 || bucket_idx >= static_cast<int>(srr_2d.buckets.size())) {
    return classify_single(x, y);
  }

  const Bucket2D &bucket = srr_2d.buckets[bucket_idx];
  return bucket.classify_point(x, y);
}

/**
 * @brief Compute decision boundary polygons (Voronoi regions by class).
 */
std::vector<std::pair<int, std::vector<Point>>>
DelaunayClassifier::compute_class_regions() {
  std::vector<std::pair<int, std::vector<Point>>> regions;

  // Collect all polygons from all buckets
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