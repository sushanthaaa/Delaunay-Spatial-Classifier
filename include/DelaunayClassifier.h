/**
 * @file DelaunayClassifier.h
 * @brief Delaunay Triangulation-based Spatial Classifier with O(1) Inference
 *
 * This classifier uses Delaunay Triangulation for 2D point classification,
 * achieving O(1) expected inference time via the Square Root Rule (SRR) grid.
 * Key innovation: no traditional "training phase" — triangulation IS the model.
 *
 * Main Features:
 * - O(1) average classification via SRR spatial indexing
 * - O(1) amortized dynamic updates (insert/delete points)
 * - Built-in outlier detection using connected component analysis
 * - Geometric decision boundaries (Voronoi-like regions)
 *
 * @author Research Project
 * @see https://www.cgal.org for CGAL Delaunay triangulation documentation
 */

#ifndef DELAUNAY_CLASSIFIER_H
#define DELAUNAY_CLASSIFIER_H

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <string>
#include <vector>

// =============================================================================
// CGAL Type Definitions
// =============================================================================
// We use CGAL's Exact_predicates_inexact_constructions kernel (EPIC) for a
// balance between numerical robustness and performance. The "info" field on
// each vertex stores the class label (int).

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K>
    Vb; // int = class label
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Face_handle Face_handle;

/**
 * @struct SRR_Grid
 * @brief Square Root Rule (SRR) spatial index for O(1) triangle lookup.
 *
 * The key insight from our lab proposal: partitioning 2D space into a k×k grid
 * where k = ceil(sqrt(n)) allows O(1) average-case point location instead of
 * O(sqrt(n)) for raw CGAL locate().
 *
 * Each bucket stores a "hint" Face_handle to start the point-location walk.
 * This dramatically reduces the search space from O(n) triangles to O(1).
 *
 * Trade-off Analysis:
 * - Grid size k = sqrt(n): Total buckets = n, average points per bucket = 1
 * - Memory: O(n) face handles (acceptable for most datasets)
 * - Query: O(1) bucket lookup + O(1) local walk ≈ O(1) total
 */
struct SRR_Grid {
  int rows, cols;                    ///< Grid dimensions (k × k where k ≈ √n)
  double min_x, max_x, min_y, max_y; ///< Bounding box of all training points
  double step_x, step_y;             ///< Cell width and height

  std::vector<Face_handle> buckets; ///< Face hints for each grid cell

  SRR_Grid() : rows(0), cols(0) {}

  void clear() {
    buckets.clear();
    rows = cols = 0;
  }
};

/**
 * @class DelaunayClassifier
 * @brief Main classifier using Delaunay Triangulation with SRR optimization.
 *
 * This class implements a novel geometric classifier where the Delaunay
 * triangulation itself acts as the model. Classification works by:
 * 1. Locating the triangle containing the query point (O(1) with SRR)
 * 2. Voting among the 3 triangle vertices based on their class labels
 *
 * Unlike KNN or SVM which have separate training and inference phases,
 * here "training" is simply building the triangulation O(n log n) and
 * "inference" is point location + voting O(1).
 */
class DelaunayClassifier {
private:
  Delaunay dt;  ///< CGAL Delaunay triangulation storing points + labels
  SRR_Grid srr; ///< Spatial index for O(1) lookup

  // Ablation configuration flags (set before train() to disable features)
  bool use_srr_; ///< Enable SRR grid for O(1) lookup (default: true)
  bool use_outlier_removal_; ///< Enable outlier removal preprocessing (default:
                             ///< true)

  /**
   * @brief Classify a point within a specific triangle using majority vote.
   *
   * For triangles with all 3 vertices sharing the same label, returns that
   * label. For mixed-class triangles, returns the label of the nearest vertex.
   * This effectively creates decision boundaries at the midpoints of
   * cross-class edges.
   *
   * @param f Triangle face handle (must not be infinite)
   * @param p Query point (should be inside or near face f)
   * @return Predicted class label
   */
  int classify_point_in_face(Face_handle f, Point p);

  /**
   * @brief Remove outliers using connected component analysis.
   *
   * Algorithm (Phase 1 from our methodology):
   * 1. Build temporary Delaunay triangulation
   * 2. For each edge, add to adjacency graph if same-class AND short enough
   * 3. Find connected components via DFS
   * 4. Keep only components with >= k points (removes isolated noise)
   *
   * @param input_points Raw training data
   * @param k Minimum cluster size to keep (default: 3)
   * @return Filtered points with outliers removed
   */
  std::vector<std::pair<Point, int>>
  remove_outliers(const std::vector<std::pair<Point, int>> &input_points,
                  int k);

  // File I/O helpers (CSV format: x,y,label or x,y)
  std::vector<std::pair<Point, int>>
  load_labeled_csv(const std::string &filepath);
  std::vector<Point> load_unlabeled_csv(const std::string &filepath);

  /**
   * @brief Build the SRR spatial index grid.
   *
   * Called after constructing the Delaunay triangulation. Creates a k×k grid
   * where k = sqrt(n) and populates each cell with a representative face.
   */
  void build_srr_grid();

  /**
   * @brief Get a face hint for point location from the SRR grid.
   *
   * Maps point (x,y) to grid cell and returns the stored face handle.
   * This hint dramatically speeds up CGAL's locate() function.
   *
   * @param p Query point
   * @return Face handle to use as starting point for walk
   */
  Face_handle get_srr_hint(const Point &p);

public:
  /**
   * @brief Default constructor. Enables SRR and outlier removal by default.
   */
  DelaunayClassifier();

  // --- Ablation Study Configuration ---
  // Call these BEFORE train() to disable specific optimizations
  void set_use_srr(bool use) { use_srr_ = use; }
  void set_use_outlier_removal(bool use) { use_outlier_removal_ = use; }

  /**
   * @brief Train the classifier by building Delaunay triangulation.
   *
   * Pipeline:
   * 1. Load labeled CSV file
   * 2. Remove outliers (if enabled) using connected component analysis
   * 3. Build Delaunay triangulation - O(n log n)
   * 4. Build SRR grid for O(1) queries
   *
   * @param train_file Path to training CSV (format: x,y,label)
   * @param outlier_k Minimum cluster size for outlier removal (default: 3)
   */
  void train(const std::string &train_file, int outlier_k = 3);

  /**
   * @brief Batch prediction with timing benchmark.
   *
   * For each test point:
   * 1. SRR grid lookup → O(1) bucket access
   * 2. CGAL locate with hint → O(1) walk
   * 3. Vertex voting → O(1)
   *
   * @param test_file Path to test CSV (format: x,y)
   * @param output_file Path to save predictions
   */
  void predict_benchmark(const std::string &test_file,
                         const std::string &output_file);

  /**
   * @brief Run dynamic operations stress test (insert/move/delete).
   *
   * Tests CGAL's incremental update performance:
   * - Insert: O(1) amortized via point location + local reconstruction
   * - Move: O(1) if no topological change needed
   * - Delete: O(1) amortized via local filling
   *
   * @param stream_file CSV of points to insert/move/delete
   * @param log_file Output log with per-operation timing (nanoseconds)
   */
  void run_dynamic_stress_test(const std::string &stream_file,
                               const std::string &log_file);

  /**
   * @brief Export triangulation for visualization.
   *
   * Outputs:
   * - mesh_file: All Delaunay edges as line segments
   * - boundary_file: Decision boundary segments (midpoints of cross-class
   * edges)
   * - points_file: Vertex coordinates with labels (optional)
   */
  void export_visualization(const std::string &mesh_file,
                            const std::string &boundary_file,
                            const std::string &points_file = "");

  /**
   * @brief Generate visualization snapshots for dynamic operations.
   */
  void run_dynamic_visualization(const std::string &stream_file,
                                 const std::string &out_dir);

  // --- Single-Point Classification (for external benchmarking) ---

  /**
   * @brief Classify a single point with full SRR optimization.
   *
   * This is the primary inference method. Uses SRR hint for O(1) expected time.
   * Fallback to nearest vertex if point lies outside the convex hull.
   *
   * @param x X-coordinate
   * @param y Y-coordinate
   * @return Predicted class label
   */
  int classify_single(double x, double y);

  /**
   * @brief Classify WITHOUT SRR hint (for ablation study).
   *
   * Measures the speed contribution of SRR. Without it, CGAL uses O(sqrt(n))
   * random walk instead of O(1) hint-guided walk.
   */
  int classify_single_no_srr(double x, double y);

  /**
   * @brief Classify using nearest vertex only (1-NN, no triangle voting).
   *
   * For ablation: measures accuracy contribution of decision boundary logic.
   * This is essentially 1-NN but using Delaunay structure instead of KD-tree.
   */
  int classify_nearest_vertex(double x, double y);

  // --- Dynamic Update Methods (for external benchmarking) ---

  /**
   * @brief Insert a new labeled training point into the model.
   *
   * O(1) amortized time. CGAL handles Delaunay property maintenance
   * via local edge flips. Note: SRR grid is NOT rebuilt (stale until retrain).
   *
   * @param x X-coordinate
   * @param y Y-coordinate
   * @param label Class label for the new point
   */
  void insert_point(double x, double y, int label);

  /**
   * @brief Remove a point from the model (finds nearest vertex and removes).
   *
   * O(1) amortized time. Use for online learning scenarios where training
   * data needs to be "forgotten" (e.g., data privacy, concept drift).
   *
   * @param x X-coordinate (approximate)
   * @param y Y-coordinate (approximate)
   */
  void remove_point(double x, double y);
};

#endif // DELAUNAY_CLASSIFIER_H