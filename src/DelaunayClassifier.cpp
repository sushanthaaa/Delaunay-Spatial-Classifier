#include "../include/DelaunayClassifier.h"
#include <CGAL/centroid.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>

DelaunayClassifier::DelaunayClassifier()
    : use_srr_(true), use_outlier_removal_(true) {}

// --- FILE IO ---
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

// --- SRR IMPLEMENTATION ---
void DelaunayClassifier::build_srr_grid() {
  if (dt.number_of_vertices() == 0)
    return;

  // 1. Calculate Bounding Box
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

  srr.min_x = min_x - 0.1;
  srr.max_x = max_x + 0.1;
  srr.min_y = min_y - 0.1;
  srr.max_y = max_y + 0.1;

  // 2. Square Root Rule: Grid Size = sqrt(N) * sqrt(N)
  int n = dt.number_of_vertices();
  int k = std::max(1, (int)std::sqrt(n));
  srr.rows = k;
  srr.cols = k;
  srr.step_x = (srr.max_x - srr.min_x) / k;
  srr.step_y = (srr.max_y - srr.min_y) / k;

  // 3. Initialize Buckets (Size = k*k)
  // Initialize with a default handle (infinite face)
  srr.buckets.assign(k * k, dt.infinite_face());

  // 4. Fill Buckets (Map Triangles to Grid)
  // We map every finite face's centroid to a bucket.
  // This gives us a "Local Hint" for every region in space.
  for (auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
    Point centroid = CGAL::centroid(
        f->vertex(0)->point(), f->vertex(1)->point(), f->vertex(2)->point());

    int c = (centroid.x() - srr.min_x) / srr.step_x;
    int r = (centroid.y() - srr.min_y) / srr.step_y;

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

Face_handle DelaunayClassifier::get_srr_hint(const Point &p) {
  int c = (p.x() - srr.min_x) / srr.step_x;
  int r = (p.y() - srr.min_y) / srr.step_y;

  if (c < 0 || c >= srr.cols || r < 0 || r >= srr.rows) {
    return dt.infinite_face();
  }

  return srr.buckets[r * srr.cols + c];
}

// --- ALGORITHM 4, PHASE 1: OUTLIER REMOVAL ---
std::vector<std::pair<Point, int>> DelaunayClassifier::remove_outliers(
    const std::vector<std::pair<Point, int>> &input_points, int k) {
  double CONNECTIVITY_THRESHOLD_SQ = 5.0 * 5.0;
  std::cout << "Phase 1: Detecting Outliers (Min Cluster Size k=" << k << ")..."
            << std::endl;
  Delaunay temp_dt;
  temp_dt.insert(input_points.begin(), input_points.end());

  std::map<Point, std::vector<Point>> adj;
  std::map<Point, int> point_to_label;

  for (auto v = temp_dt.finite_vertices_begin();
       v != temp_dt.finite_vertices_end(); ++v)
    point_to_label[v->point()] = v->info();

  for (auto e = temp_dt.finite_edges_begin(); e != temp_dt.finite_edges_end();
       ++e) {
    auto v1 = e->first->vertex((e->second + 1) % 3);
    auto v2 = e->first->vertex((e->second + 2) % 3);
    double dist = CGAL::squared_distance(v1->point(), v2->point());

    if (v1->info() == v2->info() && dist < CONNECTIVITY_THRESHOLD_SQ) {
      adj[v1->point()].push_back(v2->point());
      adj[v2->point()].push_back(v1->point());
    }
  }

  std::map<Point, bool> visited;
  std::vector<std::pair<Point, int>> clean_points;
  int removed_count = 0;

  for (const auto &p_entry : input_points) {
    Point start = p_entry.first;
    if (visited[start])
      continue;

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

int DelaunayClassifier::classify_point_in_face(Face_handle f, Point p) {
  int l0 = f->vertex(0)->info();
  int l1 = f->vertex(1)->info();
  int l2 = f->vertex(2)->info();
  if (l0 == l1 && l1 == l2)
    return l0;
  double d0 = CGAL::squared_distance(p, f->vertex(0)->point());
  double d1 = CGAL::squared_distance(p, f->vertex(1)->point());
  double d2 = CGAL::squared_distance(p, f->vertex(2)->point());
  if (d0 <= d1 && d0 <= d2)
    return l0;
  if (d1 <= d0 && d1 <= d2)
    return l1;
  return l2;
}

// --- PUBLIC: TRAIN ---
void DelaunayClassifier::train(const std::string &train_file, int outlier_k) {
  auto raw_points = load_labeled_csv(train_file);
  auto clean_points = remove_outliers(raw_points, outlier_k);

  std::ofstream out("results/clean_points.csv");
  for (const auto &p : clean_points) {
    out << p.first.x() << "," << p.first.y() << "," << p.second << "\n";
  }
  out.close();

  dt.clear();
  dt.insert(clean_points.begin(), clean_points.end());
  std::cout << "Phase 2 Complete: Delaunay Mesh Built ("
            << dt.number_of_vertices() << " vertices)." << std::endl;

  build_srr_grid();
}

// --- PUBLIC: PREDICT ---
void DelaunayClassifier::predict_benchmark(const std::string &test_file,
                                           const std::string &output_file) {
  auto test_points = load_unlabeled_csv(test_file);
  std::cout << "Starting Benchmark (with SRR Optimization)..." << std::endl;

  std::vector<int> results;
  results.reserve(test_points.size());

  auto start = std::chrono::high_resolution_clock::now();
  for (const auto &p : test_points) {
    Face_handle hint = get_srr_hint(p);

    Face_handle f = dt.locate(p, hint);

    int pred = -1;
    if (!dt.is_infinite(f))
      pred = classify_point_in_face(f, p);
    else
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

  std::ofstream out(output_file);
  for (int p : results)
    out << p << "\n";
  out.close();
}

void DelaunayClassifier::run_dynamic_stress_test(const std::string &stream_file,
                                                 const std::string &log_file) {
  auto stream_points = load_labeled_csv(stream_file);
  std::ofstream log(log_file);
  log << "operation,time_ns\n";

  std::cout << "Running Dynamic Algorithms 1, 2, 3 Stress Test..." << std::endl;

  Vertex_handle hint = dt.finite_vertices_begin();
  std::vector<Vertex_handle> handles;

  for (const auto &entry : stream_points) {
    auto start = std::chrono::high_resolution_clock::now();
    hint = dt.insert(entry.first, hint->face());
    hint->info() = entry.second;
    auto end = std::chrono::high_resolution_clock::now();

    handles.push_back(hint);
    log << "insert,"
        << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start)
               .count()
        << "\n";
  }

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

// --- DYNAMIC VISUALIZATION MODE ---
void DelaunayClassifier::run_dynamic_visualization(
    const std::string &stream_file, const std::string &out_dir) {
  auto stream_points = load_labeled_csv(stream_file);
  Vertex_handle hint = dt.finite_vertices_begin();
  std::vector<Vertex_handle> handles;

  std::cout << "Generating Dynamic Visualization Snapshots..." << std::endl;

  // 1. INSERTION
  for (const auto &entry : stream_points) {
    hint = dt.insert(entry.first, hint->face());
    hint->info() = entry.second;
    handles.push_back(hint);
  }
  export_visualization(out_dir + "/dynamic_1_inserted_triangles.csv",
                       out_dir + "/dynamic_1_inserted_boundaries.csv",
                       out_dir + "/dynamic_1_inserted_points.csv");
  std::cout << "   - Snapshot 1: Insertion Complete" << std::endl;

  // 2. MOVEMENT
  for (auto v : handles) {
    Point new_p(v->point().x() + 0.5, v->point().y() + 0.5);
    dt.move_if_no_collision(v, new_p);
  }
  export_visualization(out_dir + "/dynamic_2_moved_triangles.csv",
                       out_dir + "/dynamic_2_moved_boundaries.csv",
                       out_dir + "/dynamic_2_moved_points.csv");
  std::cout << "   - Snapshot 2: Movement Complete" << std::endl;

  // 3. DELETION
  for (int i = handles.size() - 1; i >= 0; --i) {
    dt.remove(handles[i]);
  }
  export_visualization(out_dir + "/dynamic_3_deleted_triangles.csv",
                       out_dir + "/dynamic_3_deleted_boundaries.csv",
                       out_dir + "/dynamic_3_deleted_points.csv");
  std::cout << "   - Snapshot 3: Deletion Complete" << std::endl;
}

// --- SINGLE POINT CLASSIFICATION (for external benchmarking) ---
int DelaunayClassifier::classify_single(double x, double y) {
  Point p(x, y);
  Face_handle hint = get_srr_hint(p);
  Face_handle f = dt.locate(p, hint);
  if (dt.is_infinite(f)) {
    // Point is outside the mesh, find nearest vertex
    Vertex_handle v = dt.nearest_vertex(p);
    return v->info();
  }
  return classify_point_in_face(f, p);
}

// --- DYNAMIC UPDATE METHODS (for external benchmarking) ---
void DelaunayClassifier::insert_point(double x, double y, int label) {
  Point p(x, y);
  dt.insert(p)->info() = label;
}

void DelaunayClassifier::remove_point(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  if (v != Vertex_handle()) {
    dt.remove(v);
  }
}

// --- ABLATION STUDY METHODS ---

// Classify WITHOUT SRR hint (measures SRR grid contribution to speed)
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

// Classify using NEAREST VERTEX only (no boundary interpolation)
// Measures contribution of decision boundary logic to accuracy
int DelaunayClassifier::classify_nearest_vertex(double x, double y) {
  Point p(x, y);
  Vertex_handle v = dt.nearest_vertex(p);
  return v->info();
}