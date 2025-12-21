#ifndef DELAUNAY_CLASSIFIER_H
#define DELAUNAY_CLASSIFIER_H

#include <CGAL/Exact_predicates_inexact_constructions_kernel.h>
#include <CGAL/Delaunay_triangulation_2.h>
#include <CGAL/Triangulation_vertex_base_with_info_2.h>
#include <vector>
#include <string>

typedef CGAL::Exact_predicates_inexact_constructions_kernel K;
typedef CGAL::Triangulation_vertex_base_with_info_2<int, K> Vb;
typedef CGAL::Triangulation_data_structure_2<Vb> Tds;
typedef CGAL::Delaunay_triangulation_2<K, Tds> Delaunay;
typedef Delaunay::Point Point;
typedef Delaunay::Vertex_handle Vertex_handle;
typedef Delaunay::Face_handle Face_handle;

struct SRR_Grid {
    int rows, cols;
    double min_x, max_x, min_y, max_y;
    double step_x, step_y;
    
    std::vector<Face_handle> buckets;

    SRR_Grid() : rows(0), cols(0) {}

    void clear() {
        buckets.clear();
        rows = cols = 0;
    }
};

class DelaunayClassifier {
private:
    Delaunay dt;
    SRR_Grid srr;

    int classify_point_in_face(Face_handle f, Point p);

    std::vector<std::pair<Point, int>> remove_outliers(
        const std::vector<std::pair<Point, int>>& input_points, int k);

    std::vector<std::pair<Point, int>> load_labeled_csv(const std::string& filepath);
    std::vector<Point> load_unlabeled_csv(const std::string& filepath);

    void build_srr_grid(); 
    Face_handle get_srr_hint(const Point& p); 

public:
    DelaunayClassifier();

    void train(const std::string& train_file, int outlier_k = 3);
    void predict_benchmark(const std::string& test_file, const std::string& output_file);
    void run_dynamic_stress_test(const std::string& stream_file, const std::string& log_file);
    void export_visualization(const std::string& mesh_file, const std::string& boundary_file, const std::string& points_file = "");
    void run_dynamic_visualization(const std::string& stream_file, const std::string& out_dir);
};

#endif