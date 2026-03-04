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
 * - O(1) amortized dynamic updates (insert/delete/move points)
 * - Built-in outlier detection using connected component analysis
 * - Geometric decision boundaries (half-plane tests within triangles)
 * - Full 2D Buckets with per-cell classification metadata
 *
 * Fixes Applied (IEEE Submission Quality):
 * - #1:  Adaptive outlier threshold (no hardcoded distance)
 * - #2:  SRR stores best face hint per cell (closest centroid to cell center)
 * - #3:  Relative bounding box padding (scale-independent)
 * - #4:  True half-plane decision boundary classification (not 1-NN)
 * - #5:  O(n) edge registration using bounding-box cell enumeration
 * - #6:  All Voronoi polygons stored per class per bucket (not just largest)
 * - #7:  Correct unbounded Voronoi edge direction via perpendicular bisector
 * - #8:  Dynamic updates maintain SRR grid and 2D Buckets locally
 * - #9:  Proper Algorithm 3 movement (same-star check)
 * - #10: Direct move timing in benchmarks
 * - #11: Correct outside-hull classification via extended decision boundary
 * - #13: Index-based arrays for outlier detection (O(1) access, not O(log n))
 * - #14: Direct Vertex_handle map for Voronoi cell lookup
 * - #18: Configurable output directory (no hardcoded path)
 *
 * @author Research Project — Asia University
 * @see https://www.cgal.org for CGAL Delaunay triangulation documentation
 */

#ifndef DELAUNAY_CLASSIFIER_H
#define DELAUNAY_CLASSIFIER_H

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <string>
#include <unordered_map>
#include <vector>

// =============================================================================
// CGAL Type Definitions
// =============================================================================

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Face_handle Face_handle;

// Voronoi diagram types for exact cell extraction
typedef CGAL::Delaunay_triangulation_adaptation_traits_2<Delaunay> VD_AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<
    Delaunay>
    VD_AP;
typedef CGAL::Voronoi_diagram_2<Delaunay, VD_AT, VD_AP> VoronoiDiagram;

// =============================================================================
// SRR Grid (Square Root Rule) — Fixed: best hint per cell, relative padding
// =============================================================================

/**
 * @struct SRR_Grid
 * @brief Square Root Rule spatial index for O(1) triangle lookup.
 *
 * FIX #2: Stores the face whose centroid is closest to the cell center,
 *         ensuring a high-quality hint for point location walks.
 * FIX #3: Uses relative bounding box padding (1% of range) instead of
 *         hardcoded 0.1.
 */
struct SRR_Grid {
  int rows, cols;
  double min_x, max_x, min_y, max_y;
  double step_x, step_y;

  std::vector<Face_handle> buckets; ///< Best face hint per cell

  SRR_Grid()
      : rows(0), cols(0), min_x(0), max_x(0), min_y(0), max_y(0), step_x(0),
        step_y(0) {}

  void clear() {
    buckets.clear();
    rows = cols = 0;
  }
};

// =============================================================================
// 2D BUCKETS DATA STRUCTURES — Full Implementation, All 6 Linked Lists
// =============================================================================

/**
 * @struct LL_Vertex
 * @brief Linked list node for vertices within a bucket (LL_V).
 * FIX #14: Stores Vertex_handle for direct O(1) Voronoi cell lookup.
 */
struct LL_Vertex {
  Point point;
  int class_label;
  int vertex_id;
  Vertex_handle vh; ///< FIX #14: Direct handle to CGAL vertex
  LL_Vertex *next;

  LL_Vertex() : class_label(-1), vertex_id(-1), vh(), next(nullptr) {}
  LL_Vertex(Point p, int label, int id, Vertex_handle v)
      : point(p), class_label(label), vertex_id(id), vh(v), next(nullptr) {}
};

/**
 * @struct LL_Edge
 * @brief Linked list node for edges passing through a bucket (LL_E).
 */
struct LL_Edge {
  Point p1, p2;
  int class1, class2;
  bool is_boundary;
  int edge_id;
  LL_Edge *next;

  LL_Edge()
      : class1(-1), class2(-1), is_boundary(false), edge_id(-1), next(nullptr) {
  }
  LL_Edge(Point a, Point b, int c1, int c2, int id)
      : p1(a), p2(b), class1(c1), class2(c2), is_boundary(c1 != c2),
        edge_id(id), next(nullptr) {}
};

/**
 * @enum GridEdgeSide
 * @brief Identifies which side of a bucket boundary.
 */
enum GridEdgeSide {
  GRID_LEFT = 0,
  GRID_BOTTOM = 1,
  GRID_RIGHT = 2,
  GRID_TOP = 3
};

/**
 * @struct LL_GridEdge
 * @brief Linked list node for grid edge intersections (LL_GE).
 */
struct LL_GridEdge {
  GridEdgeSide side;
  Point intersection;
  int class_inside;
  int class_outside;
  LL_GridEdge *next;

  LL_GridEdge()
      : side(GRID_LEFT), class_inside(-1), class_outside(-1), next(nullptr) {}
  LL_GridEdge(GridEdgeSide s, Point p, int ci, int co)
      : side(s), intersection(p), class_inside(ci), class_outside(co),
        next(nullptr) {}
};

/**
 * @struct LL_Polygon
 * @brief Linked list node for polygon regions (LL_Poly + LL_Label + LL_PolyID).
 * FIX #6: ALL clipped Voronoi polygons are stored, not just the largest.
 */
struct LL_Polygon {
  int poly_id;
  int class_label;
  int inside_label;
  std::vector<Point> vertices;
  double area;
  LL_Polygon *next;

  LL_Polygon()
      : poly_id(-1), class_label(-1), inside_label(1), area(0), next(nullptr) {}
  LL_Polygon(int id, int label)
      : poly_id(id), class_label(label), inside_label(1), area(0),
        next(nullptr) {}
};

/**
 * @enum BucketType
 * @brief Classification type for fast-path decisions.
 *
 * From the advisor's per-cell classification metadata concept:
 * - HOMOGENEOUS:     Single class, O(1) direct return
 * - BIPARTITIONED:   Two classes, O(1) half-plane test
 * - MULTI_PARTITIONED: 3+ classes, O(k) point-in-polygon (k bounded by O(1))
 */
enum BucketType {
  BUCKET_HOMOGENEOUS = 0,
  BUCKET_BIPARTITIONED = 1,
  BUCKET_MULTI_PARTITIONED = 2
};

/**
 * @struct Bucket2D
 * @brief Complete 2D Bucket with all 6 linked list data structures.
 */
struct Bucket2D {
  double min_x, max_x, min_y, max_y;
  int row, col;

  // LL_V: Linked List of Vertices
  LL_Vertex *vertices;
  int vertex_count;

  // LL_E: Linked List of Edges
  LL_Edge *edges;
  int edge_count;

  // LL_GE: Linked List of Grid Edge intersections
  LL_GridEdge *grid_edges;
  int grid_edge_count;

  // LL_Poly + LL_Label + LL_PolyID: Polygon regions
  LL_Polygon *polygons;
  int polygon_count;

  // Classification metadata
  BucketType type;
  int num_classes;
  int dominant_class;
  Face_handle hint;

  // BIPARTITIONED fast-path: half-plane boundary
  // The boundary line is defined by normal (nx, ny) and offset d:
  // A point (x,y) is on the positive side if nx*x + ny*y + d > 0
  double boundary_nx, boundary_ny, boundary_d;
  int class_positive; ///< Class on the positive side of the half-plane
  int class_negative; ///< Class on the negative side

  Bucket2D()
      : min_x(0), max_x(0), min_y(0), max_y(0), row(0), col(0),
        vertices(nullptr), vertex_count(0), edges(nullptr), edge_count(0),
        grid_edges(nullptr), grid_edge_count(0), polygons(nullptr),
        polygon_count(0), type(BUCKET_HOMOGENEOUS), num_classes(0),
        dominant_class(-1), hint(), boundary_nx(0), boundary_ny(0),
        boundary_d(0), class_positive(-1), class_negative(-1) {}

  /**
   * @brief O(1) classification using bucket type metadata.
   *
   * Case A (HOMOGENEOUS):       return dominant_class — O(1)
   * Case B (BIPARTITIONED):     half-plane test — O(1)
   * Case C (MULTI_PARTITIONED): point-in-polygon — O(k), k = O(1) bounded
   */
  int classify_point(double x, double y) const;

  static bool point_in_polygon(double x, double y,
                               const std::vector<Point> &polygon);

  void clear();

  bool contains(double x, double y) const {
    return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
  }
};

/**
 * @struct SRR_Grid_2D
 * @brief Complete 2D Buckets grid for O(1) dynamic classification.
 */
struct SRR_Grid_2D {
  int rows, cols;
  double min_x, max_x, min_y, max_y;
  double step_x, step_y;

  std::vector<Bucket2D> buckets;

  int single_class_buckets;
  int multi_class_buckets;
  int bipartitioned_buckets;
  int total_polygons;

  SRR_Grid_2D()
      : rows(0), cols(0), min_x(0), max_x(0), min_y(0), max_y(0), step_x(0),
        step_y(0), single_class_buckets(0), multi_class_buckets(0),
        bipartitioned_buckets(0), total_polygons(0) {}

  void clear() {
    for (auto &bucket : buckets) {
      bucket.clear();
    }
    buckets.clear();
    rows = cols = 0;
    single_class_buckets = multi_class_buckets = bipartitioned_buckets =
        total_polygons = 0;
  }

  int get_bucket_index(double x, double y) const;

  Bucket2D *get_bucket(int row, int col) {
    if (row < 0 || row >= rows || col < 0 || col >= cols)
      return nullptr;
    return &buckets[row * cols + col];
  }

  const Bucket2D *get_bucket(int row, int col) const {
    if (row < 0 || row >= rows || col < 0 || col >= cols)
      return nullptr;
    return &buckets[row * cols + col];
  }

  void print_statistics() const;
};

// =============================================================================
// MAIN CLASSIFIER CLASS
// =============================================================================

/**
 * @class DelaunayClassifier
 * @brief Main classifier using Delaunay Triangulation with SRR optimization.
 *
 * All 20 identified issues from code review have been addressed.
 */
class DelaunayClassifier {
private:
  Delaunay dt;
  SRR_Grid srr;
  SRR_Grid_2D srr_2d;

  // Configuration flags
  bool use_srr_;
  bool use_outlier_removal_;

  // FIX #1: Adaptive outlier connectivity multiplier (default 3.0)
  double connectivity_multiplier_;

  // FIX #18: Configurable output directory
  std::string output_dir_;

  // --- Internal methods ---

  /**
   * @brief Classify a point within a triangle using decision boundary logic.
   *
   * FIX #4: Uses half-plane test for Case 2 (two classes),
   * NOT nearest-vertex. This matches Algorithm 4 Phase 2.
   *
   * Case 1 (all same class): return that class
   * Case 2 (two classes): half-plane test against midpoint boundary line
   * Case 3 (three classes): Voronoi partition within triangle (nearest vertex)
   */
  int classify_point_in_face(Face_handle f, Point p);

  /**
   * @brief Remove outliers using connected component analysis.
   *
   * FIX #1:  Uses adaptive distance threshold based on median edge length.
   * FIX #13: Uses index-based arrays instead of std::map<Point> for O(1)
   * access.
   */
  std::vector<std::pair<Point, int>>
  remove_outliers(const std::vector<std::pair<Point, int>> &input_points,
                  int k);

  /**
   * @brief Compute the median edge length of a Delaunay triangulation.
   * Used for adaptive outlier threshold (FIX #1).
   */
  double compute_median_edge_length(const Delaunay &temp_dt);

  // File I/O helpers
  std::vector<std::pair<Point, int>>
  load_labeled_csv(const std::string &filepath);
  std::vector<Point> load_unlabeled_csv(const std::string &filepath);

  /**
   * @brief Build the SRR spatial index grid.
   *
   * FIX #2: Stores the face whose centroid is closest to each cell center.
   * FIX #3: Uses relative padding (1% of range, minimum 1e-6).
   */
  void build_srr_grid();

  /**
   * @brief Get a face hint for point location from the SRR grid.
   */
  Face_handle get_srr_hint(const Point &p);

  /**
   * @brief Rebuild a single bucket's linked lists after dynamic update.
   * FIX #8: Enables local-only bucket maintenance after insert/delete/move.
   */
  void rebuild_bucket(int bucket_idx);

  /**
   * @brief Update SRR face hint for a single cell.
   * FIX #8: Maintains hint validity after dynamic operations.
   */
  void update_srr_hint(int bucket_idx);

  /**
   * @brief Update all affected cells around a point after a dynamic operation.
   * FIX #8: Rebuilds the 3x3 neighborhood of the cell containing (x,y).
   */
  void update_local_cells(double x, double y);

  /**
   * @brief Compute the closest point on a segment to a query point.
   * FIX #11: Correct geometry for outside-hull classification.
   * @return Squared distance from p to closest point on segment (a, b).
   */
  double squared_distance_point_to_segment(const Point &p, const Point &a,
                                           const Point &b);

public:
  DelaunayClassifier();

  // --- Configuration ---
  void set_use_srr(bool use) { use_srr_ = use; }
  void set_use_outlier_removal(bool use) { use_outlier_removal_ = use; }
  void set_connectivity_multiplier(double m) { connectivity_multiplier_ = m; }
  void set_output_dir(const std::string &dir) { output_dir_ = dir; }

  /**
   * @brief Train the classifier.
   * @param train_file Path to training CSV (format: x,y,label)
   * @param outlier_k Minimum cluster size for outlier removal (default: 3)
   */
  void train(const std::string &train_file, int outlier_k = 3);

  void predict_benchmark(const std::string &test_file,
                         const std::string &output_file);

  void run_dynamic_stress_test(const std::string &stream_file,
                               const std::string &log_file);

  void export_visualization(const std::string &mesh_file,
                            const std::string &boundary_file,
                            const std::string &points_file = "");

  void run_dynamic_visualization(const std::string &stream_file,
                                 const std::string &out_dir);

  // --- Single-Point Classification ---

  /**
   * @brief Classify a single point with full SRR optimization.
   * FIX #4:  Uses half-plane decision boundary, not 1-NN.
   * FIX #11: Correct outside-hull classification via extended boundary.
   */
  int classify_single(double x, double y);

  /**
   * @brief Classify WITHOUT SRR hint (for ablation study).
   */
  int classify_single_no_srr(double x, double y);

  /**
   * @brief Classify using nearest vertex only (1-NN, for ablation).
   */
  int classify_nearest_vertex(double x, double y);

  // --- Dynamic Update Methods ---

  /**
   * @brief Insert a new labeled training point.
   * FIX #8: Updates SRR grid and 2D Buckets locally.
   */
  void insert_point(double x, double y, int label);

  /**
   * @brief Remove a point from the model.
   * FIX #8: Updates SRR grid and 2D Buckets locally.
   */
  void remove_point(double x, double y);

  /**
   * @brief Move a point using Algorithm 3 logic.
   * FIX #9: Implements proper same-star check.
   *         Case A: within same star polygon → local flip
   *         Case B: different polygon → delete + re-insert
   * FIX #8: Updates SRR grid and 2D Buckets locally.
   */
  void move_point(double old_x, double old_y, double new_x, double new_y);

  // --- 2D Buckets Methods ---

  /**
   * @brief Build the enhanced 2D Buckets grid.
   *
   * FIX #5:  O(n) edge registration via bounding-box cell enumeration.
   * FIX #6:  Stores ALL clipped Voronoi polygons per class per bucket.
   * FIX #7:  Correct unbounded Voronoi edge direction.
   * FIX #14: Uses Vertex_handle map for O(1) Voronoi cell lookup.
   */
  void build_2d_buckets();

  /**
   * @brief Classify using 2D Buckets for O(1) dynamic classification.
   */
  int classify_single_dynamic(double x, double y);

  /**
   * @brief Compute decision boundary polygons (Voronoi regions by class).
   */
  std::vector<std::pair<int, std::vector<Point>>> compute_class_regions();

  // --- Accessors for benchmarks ---
  int num_vertices() const { return static_cast<int>(dt.number_of_vertices()); }
  double get_srr_step_x() const { return srr.step_x; }
  double get_srr_step_y() const { return srr.step_y; }
  double get_data_range_x() const { return srr.max_x - srr.min_x; }
  double get_data_range_y() const { return srr.max_y - srr.min_y; }
};

#endif // DELAUNAY_CLASSIFIER_H