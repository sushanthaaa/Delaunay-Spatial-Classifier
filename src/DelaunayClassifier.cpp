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
 * 1. SRR grid lookup → get starting face hint
 * 2. CGAL locate with hint → find containing triangle
 * 3. Classify within triangle → vote among 3 vertices
 *
 * For points outside the convex hull, we fall back to nearest vertex.
 */
int DelaunayClassifier::classify_single(double x, double y) {
  Point p(x, y);

  // O(1) grid lookup
  Face_handle hint = get_srr_hint(p);

  // O(1) locate with hint
  Face_handle f = dt.locate(p, hint);

  if (dt.is_infinite(f)) {
    // Point is outside the convex hull - use nearest vertex
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