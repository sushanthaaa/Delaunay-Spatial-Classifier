/**
 * @file DelaunayClassifier.h
 * @brief Delaunay Triangulation-based Spatial Classifier with O(1) Inference
 *
 * This classifier uses Delaunay Triangulation (DT) for 2D point classification.
 * The DT defines geometric decision boundaries during training; the SRR grid
 * with 2D Buckets precomputes and caches these boundaries, enabling O(1)
 * inference without accessing the triangulation at query time.
 *
 * Pipeline:
 *   Phase 1 — Outlier removal via connected component analysis on DT edges
 *   Phase 2 — Delaunay Triangulation construction (the model)
 *   Phase 3 — SRR grid (ceil(sqrt(n)) x ceil(sqrt(n))) with 2D Buckets
 *   Phase 4 — O(1) classification via precomputed bucket metadata
 *
 * Key properties:
 *   - O(n log n) training (DT construction dominates)
 *   - O(1) expected classification via 2D Buckets under the SRR uniform-density
 *     assumption (see get_bucket_occupancy_stats() for empirical validation)
 *   - O(1) amortized dynamic updates (insert/delete/move with local rebuild)
 *
 */

#ifndef DELAUNAY_CLASSIFIER_H
#define DELAUNAY_CLASSIFIER_H

#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_policies_2.h>
#include <CGAL/Delaunay_triangulation_adaptation_traits_2.h>
#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <CGAL/Voronoi_diagram_2.h>
#include <cstddef>
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

typedef CGAL::Delaunay_triangulation_adaptation_traits_2<Delaunay> VD_AT;
typedef CGAL::Delaunay_triangulation_caching_degeneracy_removal_policy_2<
    Delaunay>
    VD_AP;
typedef CGAL::Voronoi_diagram_2<Delaunay, VD_AT, VD_AP> VoronoiDiagram;

// =============================================================================
// 2D Buckets — Linked List Data Structures
// =============================================================================
//
// The 2D Buckets data structure uses FOUR linked list types to store
// precomputed geometric information within each bucket cell:
//
//   LL_V   — Vertices (training points falling in the bucket)
//   LL_E   — Edges (DT edges passing through the bucket)
//   LL_GE  — Grid edge intersections (where boundary edges cross bucket walls)
//   LL_Poly — Polygons (clipped Voronoi cells, one per class region)

/// Linked list node storing a vertex within a bucket (LL_V).
struct LL_Vertex {
  Point point;
  int class_label;
  int vertex_id;
  Vertex_handle vh;
  LL_Vertex *next;

  LL_Vertex() : class_label(-1), vertex_id(-1), vh(), next(nullptr) {}
  LL_Vertex(Point p, int label, int id, Vertex_handle v)
      : point(p), class_label(label), vertex_id(id), vh(v), next(nullptr) {}
};

/// Linked list node storing an edge passing through a bucket (LL_E).
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

enum GridEdgeSide {
  GRID_LEFT = 0,
  GRID_BOTTOM = 1,
  GRID_RIGHT = 2,
  GRID_TOP = 3
};

/// Linked list node for grid edge intersections (LL_GE).
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

/// Linked list node for clipped Voronoi polygon regions (LL_Poly).
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

// =============================================================================
// Bucket Classification Types
// =============================================================================

/**
 * @enum BucketType
 * @brief Determines the O(1) classification strategy for each bucket.
 *
 * - HOMOGENEOUS:       Single class occupies the entire bucket → direct return
 * - BIPARTITIONED:     Two classes separated by a line → half-plane dot product
 * - MULTI_PARTITIONED: Three+ classes → point-in-polygon against Voronoi cells
 */
enum BucketType {
  BUCKET_HOMOGENEOUS = 0,
  BUCKET_BIPARTITIONED = 1,
  BUCKET_MULTI_PARTITIONED = 2
};

// =============================================================================
// Bucket2D — A single cell of the SRR grid
// =============================================================================

/**
 * @struct Bucket2D
 * @brief Contains precomputed classification metadata derived from the
 *        Delaunay Triangulation's Voronoi dual, clipped to bucket boundaries.
 *
 * Stores four linked list structures (LL_V, LL_E, LL_GE, LL_Poly) and
 * fast-path classification data. After construction, every bucket is
 * guaranteed to return a valid class label for any query point within its
 * bounds — no fallback to the DT is needed.
 */
struct Bucket2D {
  double min_x, max_x, min_y, max_y;
  int row, col;

  LL_Vertex *vertices;
  int vertex_count;

  LL_Edge *edges;
  int edge_count;

  LL_GridEdge *grid_edges;
  int grid_edge_count;

  LL_Polygon *polygons;
  int polygon_count;

  BucketType type;
  int num_classes;
  int dominant_class;

  /// BIPARTITIONED fast-path: half-plane boundary (nx*x + ny*y + d = 0)
  double boundary_nx, boundary_ny, boundary_d;
  int class_positive; ///< Class where nx*x + ny*y + d >= 0
  int class_negative; ///< Class where nx*x + ny*y + d < 0

  Bucket2D()
      : min_x(0), max_x(0), min_y(0), max_y(0), row(0), col(0),
        vertices(nullptr), vertex_count(0), edges(nullptr), edge_count(0),
        grid_edges(nullptr), grid_edge_count(0), polygons(nullptr),
        polygon_count(0), type(BUCKET_HOMOGENEOUS), num_classes(0),
        dominant_class(-1), boundary_nx(0), boundary_ny(0), boundary_d(0),
        class_positive(-1), class_negative(-1) {}

  /**
   * @brief Classify a query point using precomputed bucket metadata.
   *
   * HOMOGENEOUS:       return dominant_class — O(1)
   * BIPARTITIONED:     half-plane dot product test — O(1)
   * MULTI_PARTITIONED: point-in-polygon over clipped Voronoi cells — O(k),
   *                    where k is bounded by a constant under SRR density.
   *
   * Always returns a valid class label >= 0 (dominant_class as final fallback
   * for MULTI_PARTITIONED buckets when floating-point gaps exist).
   *
   * @param x Query x-coordinate.
   * @param y Query y-coordinate.
   * @param fallback_fired Optional output: set to true if a MULTI_PARTITIONED
   *                       bucket fell through to dominant_class because no
   *                       polygon contained the query point. Used for
   *                       instrumentation and soundness reporting. Pass
   *                       nullptr if you don't need the information.
   * @return Predicted class label (always >= 0).
   */
  int classify_point(double x, double y, bool *fallback_fired = nullptr) const;

  /// Ray-casting point-in-polygon test.
  static bool point_in_polygon(double x, double y,
                               const std::vector<Point> &polygon);

  /// Free all linked list memory.
  void clear();

  bool contains(double x, double y) const {
    return x >= min_x && x <= max_x && y >= min_y && y <= max_y;
  }
};

// =============================================================================
// SRR_Grid_2D — The complete Square Root Rule grid
// =============================================================================

/**
 * @struct SRR_Grid_2D
 * @brief ceil(sqrt(n)) x ceil(sqrt(n)) grid of Bucket2D cells.
 *
 * This is the sole inference structure. After construction from the DT,
 * all classification queries go through this grid with O(1) bucket lookup
 * followed by O(1) bucket-level classification. No DT access at query time.
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
    for (auto &bucket : buckets)
      bucket.clear();
    buckets.clear();
    rows = cols = 0;
    single_class_buckets = multi_class_buckets = bipartitioned_buckets =
        total_polygons = 0;
  }

  /// O(1) bucket index from coordinates (clamped to valid range).
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
// DelaunayClassifier — Main Classifier
// =============================================================================

/**
 * @class DelaunayClassifier
 * @brief Spatial classifier using Delaunay Triangulation with O(1) inference.
 *
 * The Delaunay Triangulation defines geometric decision boundaries during
 * training. The SRR grid with 2D Buckets precomputes these boundaries into
 * a spatial index, enabling O(1) expected classification (under the SRR
 * uniform-density assumption) without DT access at query time. Dynamic
 * updates (insert/delete/move) modify the DT and locally rebuild only the
 * affected buckets.
 *
 * FEATURES:
 *   - O(n log n) training (DT construction dominates)
 *   - O(1) expected inference via 2D Buckets
 *   - O(1) amortized dynamic updates (insert/delete/move)
 *   - Geometric decision boundaries following data topology (not axis-aligned)
 *   - Full interpretability: every decision region is a visible Voronoi cell
 *   - Boundary-clamped behavior for out-of-hull queries (edge bucket lookup)
 *   - Built-in instrumentation for O(1) claim validation and soundness
 *     reporting (see get_bucket_occupancy_stats, get_multi_fallback_count)
 */
class DelaunayClassifier {
private:
  Delaunay dt_;
  SRR_Grid_2D grid_;

  bool use_outlier_removal_;

  /// Multiplier applied to the median edge length of the temporary DT to
  /// produce the outlier-removal threshold. Components of same-class edges
  /// below (median * multiplier) in length are kept; longer edges are ignored
  /// when building the connectivity graph. This acts as a 3σ-like adaptive
  /// threshold that scales with data density — no manual tuning per dataset.
  ///
  /// DEFAULT: 3.0 (chosen via ablation A5 across {1.5, 2.0, 3.0, 5.0, 10.0};
  /// 3.0 maximizes mean accuracy across the 11-dataset benchmark).
  ///
  /// SENSITIVITY: Robust in [2.0, 5.0]. Values < 1.5 are too aggressive
  /// (removes legitimate points). Values > 10.0 are too permissive (fails
  /// to remove real outliers).
  double connectivity_multiplier_;

  std::string output_dir_;

  mutable std::size_t total_queries_ = 0;
  mutable std::size_t multi_fallback_count_ = 0;

  // --- Training internals (operate on DT, not used at query time) ---

  /// Classify a point within a triangle using decision boundary geometry.
  /// Case 1: all same class → unanimous label.
  /// Case 2: two classes → half-plane test via cross-class edge midpoints.
  /// Case 3: three classes → nearest-vertex approximation of the Algorithm 4
  ///         centroid-to-midpoint partition (exact for equilateral triangles).
  int classify_point_in_face(Face_handle f, Point p) const;

  /// Remove outliers via connected component analysis on same-class DT edges.
  std::vector<std::pair<Point, int>>
  remove_outliers(const std::vector<std::pair<Point, int>> &input_points,
                  int k);

  /// Compute median edge length of a Delaunay triangulation.
  double compute_median_edge_length(const Delaunay &temp_dt);

  /// Load CSV with format: x,y,label
  /// @throws std::runtime_error if the file cannot be opened.
  std::vector<std::pair<Point, int>>
  load_labeled_csv(const std::string &filepath);

  /// Load CSV with format: x,y (unlabeled query points)
  /// @throws std::runtime_error if the file cannot be opened.
  std::vector<Point> load_unlabeled_csv(const std::string &filepath);

  // --- 2D Buckets construction ---

  /// Build the complete 2D Buckets grid from the current DT.
  void build_2d_buckets();

  /// Rebuild a single bucket after a dynamic update.
  void rebuild_bucket(int bucket_idx);

  /// Rebuild the 3x3 neighborhood of buckets around point (x,y).
  void update_local_cells(double x, double y);

  /// Post-construction validation: ensure every bucket has dominant_class >= 0.
  void validate_all_buckets();

  // --- Geometry helpers ---

  double squared_distance_point_to_segment(const Point &p, const Point &a,
                                           const Point &b);

public:
  DelaunayClassifier();
  ~DelaunayClassifier();

  // --- Configuration ---
  void set_use_outlier_removal(bool use) { use_outlier_removal_ = use; }

  /**
   * @brief Set the outlier connectivity multiplier.
   *
   * @param m The multiplier applied to the temporary DT's median edge length
   *          to produce the outlier threshold. See documentation on the
   *          connectivity_multiplier_ member for rationale. Default: 3.0.
   *
   * NOTE: This is exposed primarily for the ablation A5 sensitivity sweep.
   * End users should rarely need to change it — the default works across
   * the full benchmark suite without per-dataset tuning.
   */
  void set_connectivity_multiplier(double m) { connectivity_multiplier_ = m; }

  void set_output_dir(const std::string &dir) { output_dir_ = dir; }

  /**
   * @brief Train: outlier removal → DT construction → 2D Buckets build.
   * @param train_file CSV file (x,y,label format, no header).
   * @param outlier_k  Minimum cluster size for outlier removal (default: 3).
   * @throws std::runtime_error if train_file cannot be opened.
   */
  void train(const std::string &train_file, int outlier_k = 3);

  /**
   * @brief Classify a single point in O(1) via the 2D Buckets grid.
   *
   * This is the sole classification entry point for all queries (static and
   * after dynamic updates). The bucket index is computed by arithmetic, then
   * the bucket's precomputed metadata determines the class. No DT access.
   *
   * OUT-OF-HULL QUERIES: Points outside the training bounding box are
   * clamped to the nearest edge bucket (see get_bucket_index implementation).
   * This is equivalent to extrapolating the nearest training region outward.
   *
   * INSTRUMENTATION: Every call increments total_queries_. If a
   * MULTI_PARTITIONED bucket falls through to dominant_class (floating-point
   * polygon gaps), multi_fallback_count_ is incremented. Access these via
   * get_total_query_count() and get_multi_fallback_count().
   */
  int classify(double x, double y) const;

  /// Batch classification with timing benchmark.
  /// @throws std::runtime_error if test_file cannot be opened.
  void predict_benchmark(const std::string &test_file,
                         const std::string &output_file);

  // --- Dynamic Updates ---

  /// Insert a new labeled point and locally rebuild affected buckets.
  void insert_point(double x, double y, int label);

  /// Remove the nearest point to (x,y) and locally rebuild affected buckets.
  void remove_point(double x, double y);

  /// Move a point: local flips if within same star, else delete + re-insert.
  void move_point(double old_x, double old_y, double new_x, double new_y);

  // --- Ablation study variants (for benchmarking comparison only) ---

  /// Classify by walking the DT from an arbitrary start face (no grid).
  int classify_no_grid(double x, double y) const;

  /// Classify by nearest-vertex only (1-NN baseline).
  int classify_nearest_vertex(double x, double y) const;

  // --- Visualization & stress testing ---

  void export_visualization(const std::string &mesh_file,
                            const std::string &boundary_file,
                            const std::string &points_file = "");
  void run_dynamic_stress_test(const std::string &stream_file,
                               const std::string &log_file);
  void run_dynamic_visualization(const std::string &stream_file,
                                 const std::string &out_dir);
  std::vector<std::pair<int, std::vector<Point>>> compute_class_regions() const;

  // --- Accessors ---
  int num_vertices() const {
    return static_cast<int>(dt_.number_of_vertices());
  }
  const SRR_Grid_2D &grid() const { return grid_; }

  // ---------------------------------------------------------------------------
  // Instrumentation for O(1) claim validation and soundness reporting
  // ---------------------------------------------------------------------------

  /**
   * @brief Return the distribution of polygon counts per bucket.
   *
   * Used to empirically validate the O(1) inference claim. Under the SRR
   * uniform-density assumption, each bucket should contain a bounded
   * constant number of polygons. A histogram of this distribution
   * (generated by scripts/generate_figures.py) provides visual evidence.
   *
   * @return Vector of polygon counts, one per bucket. Length == rows*cols.
   */
  std::vector<int> get_bucket_polygon_counts() const;

  /**
   * @brief Return the distribution of vertex counts per bucket.
   *
   * Complementary metric to polygon counts. Vertex count is the raw
   * "how many training points fall in this bucket" measurement, while
   * polygon count reflects the post-Voronoi-clipping complexity.
   *
   * @return Vector of vertex counts, one per bucket. Length == rows*cols.
   */
  std::vector<int> get_bucket_vertex_counts() const;

  /**
   * @brief Summary statistics about bucket occupancy.
   *
   * Computes max, mean, median, and 99th percentile of the polygon-count
   * distribution. The max value is the empirical worst-case k for this
   * dataset — a bounded max is evidence that classification is O(1) in
   * practice for this data distribution.
   */
  struct BucketOccupancyStats {
    int num_buckets;
    int max_polygons;
    int max_vertices;
    double mean_polygons;
    double median_polygons;
    double p99_polygons;
    int empty_buckets; ///< Buckets with zero training vertices
  };

  /**
   * @brief Return aggregated bucket occupancy statistics.
   *
   * @return BucketOccupancyStats struct with max/mean/median/p99 counts.
   */
  BucketOccupancyStats get_bucket_occupancy_stats() const;

  /**
   * @brief Total number of queries handled since the last reset.
   *
   * Incremented by every call to classify(). Call reset_query_counters()
   * before a benchmark run to get clean counts for that run.
   */
  std::size_t get_total_query_count() const { return total_queries_; }

  /**
   * @brief Number of times a MULTI_PARTITIONED bucket fell through to
   *        dominant_class because no polygon contained the query point.
   *
   * A nonzero value indicates floating-point gaps between clipped Voronoi
   * polygons. High counts suggest polygon clipping artifacts that should be
   * investigated. The ratio (fallback_count / total_queries) should be
   * reported alongside accuracy for full soundness transparency.
   */
  std::size_t get_multi_fallback_count() const { return multi_fallback_count_; }

  /**
   * @brief Reset query and fallback counters to zero.
   *
   * Call before a benchmark run to get clean counts for that run.
   * Marked const because the counters are mutable (updated by const
   * classify() calls).
   */
  void reset_query_counters() const {
    total_queries_ = 0;
    multi_fallback_count_ = 0;
  }
};

#endif // DELAUNAY_CLASSIFIER_H