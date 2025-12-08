#ifndef DELAUNAY_CLASSIFIER_H
#define DELAUNAY_CLASSIFIER_H

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <vector>
#include <string>

// --- CGAL Typedefs ---
typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Face_handle Face_handle;

class DelaunayClassifier {
private:
    Delaunay dt;

    // Phase 4 Helper: Classify a point inside a specific triangle
    int classify_point_in_face(Face_handle f, Point p);

    // Algorithm 4, Phase 1: Detect and remove outliers
    std::vector<std::pair<Point, int>> remove_outliers(
        const std::vector<std::pair<Point, int>>& input_points, int k);

    // Helpers to load CSVs
    std::vector<std::pair<Point, int>> load_labeled_csv(const std::string& filepath);
    std::vector<Point> load_unlabeled_csv(const std::string& filepath);

public:
    DelaunayClassifier();

    // --- MAIN WORKFLOW (Algorithm 4) ---
    // 1. Load Data -> 2. Remove Outliers (Phase 1) -> 3. Build Mesh (Phase 2)
    void train(const std::string& train_file, int outlier_k = 3);

    // --- CLASSIFICATION (Algorithm 4, Phase 4) ---
    // Predicts classes for a test file and logs inference speed
    void predict_benchmark(const std::string& test_file, const std::string& output_file);

    // --- DYNAMIC UPDATES (Algorithms 1, 2, 3) ---
    // Runs the "Stream" benchmark: Insertion, Movement, Deletion
    void run_dynamic_stress_test(const std::string& stream_file, const std::string& log_file);

    // --- VISUALIZATION EXPORT ---
    // Saves the Mesh and Decision Boundaries (Phase 2 & 3) for Python plotting
    void export_visualization(const std::string& mesh_file, const std::string& boundary_file);
};

#endif