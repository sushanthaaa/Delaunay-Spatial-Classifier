#include "../include/DelaunayClassifier.h"
#include <CGAL/centroid.h>
#include <fstream>
#include <sstream>
#include <iostream>
#include <chrono>
#include <map>

DelaunayClassifier::DelaunayClassifier() {}

// --- FILE IO ---
std::vector<std::pair<Point, int>> DelaunayClassifier::load_labeled_csv(const std::string& filepath) {
    std::vector<std::pair<Point, int>> points;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "❌ Error: Cannot open " << filepath << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val; double x, y; int label;
        std::getline(ss, val, ','); x = std::stod(val);
        std::getline(ss, val, ','); y = std::stod(val);
        std::getline(ss, val, ','); label = std::stoi(val);
        points.push_back({Point(x, y), label});
    }
    return points;
}

std::vector<Point> DelaunayClassifier::load_unlabeled_csv(const std::string& filepath) {
    std::vector<Point> points;
    std::ifstream file(filepath);
    if (!file.is_open()) {
        std::cerr << "❌ Error: Cannot open " << filepath << std::endl;
        exit(1);
    }
    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string val; double x, y;
        std::getline(ss, val, ','); x = std::stod(val);
        std::getline(ss, val, ','); y = std::stod(val);
        points.push_back(Point(x, y));
    }
    return points;
}

// --- ALGORITHM 4, PHASE 1: OUTLIER REMOVAL ---
std::vector<std::pair<Point, int>> DelaunayClassifier::remove_outliers(
    const std::vector<std::pair<Point, int>>& input_points, int k) 
{
    std::cout << "🔍 Phase 1: Detecting Outliers (Min Cluster Size k=" << k << ")..." << std::endl;
    Delaunay temp_dt;
    temp_dt.insert(input_points.begin(), input_points.end());

    std::map<Point, std::vector<Point>> adj;
    std::map<Point, int> point_to_label;
    
    // Map points to labels
    for(auto v = temp_dt.finite_vertices_begin(); v != temp_dt.finite_vertices_end(); ++v) 
        point_to_label[v->point()] = v->info();

    // Build Adjacency Graph based on edges
    for(auto e = temp_dt.finite_edges_begin(); e != temp_dt.finite_edges_end(); ++e) {
        auto v1 = e->first->vertex((e->second + 1) % 3);
        auto v2 = e->first->vertex((e->second + 2) % 3);
        double dist = CGAL::squared_distance(v1->point(), v2->point());
        
        // Logic: Connected if SAME CLASS and CLOSE ENOUGH
        if (v1->info() == v2->info() && dist < 5.0) { 
            adj[v1->point()].push_back(v2->point());
            adj[v2->point()].push_back(v1->point());
        }
    }

    // DFS to find components
    std::map<Point, bool> visited;
    std::vector<std::pair<Point, int>> clean_points;
    int removed_count = 0;

    for(const auto& p_entry : input_points) {
        Point start = p_entry.first;
        if (visited[start]) continue;

        std::vector<Point> component;
        std::vector<Point> stack = {start};
        visited[start] = true;

        while (!stack.empty()) {
            Point curr = stack.back(); stack.pop_back();
            component.push_back(curr);
            for (const auto& n : adj[curr]) {
                if (!visited[n]) { visited[n] = true; stack.push_back(n); }
            }
        }

        // Filter small components
        if (component.size() >= k) {
            for (const auto& p : component) clean_points.push_back({p, point_to_label[p]});
        } else {
            removed_count += component.size();
        }
    }
    std::cout << "✅ Phase 1 Complete: Removed " << removed_count << " outliers." << std::endl;
    return clean_points;
}

// --- ALGORITHM 4, PHASE 4: CLASSIFICATION LOGIC ---
int DelaunayClassifier::classify_point_in_face(Face_handle f, Point p) {
    int l0 = f->vertex(0)->info();
    int l1 = f->vertex(1)->info();
    int l2 = f->vertex(2)->info();
    
    // Case 1: Uniform
    if (l0 == l1 && l1 == l2) return l0;
    
    // Case 2 & 3: Mixed -> Nearest Neighbor inside triangle
    double d0 = CGAL::squared_distance(p, f->vertex(0)->point());
    double d1 = CGAL::squared_distance(p, f->vertex(1)->point());
    double d2 = CGAL::squared_distance(p, f->vertex(2)->point());
    
    if (d0 <= d1 && d0 <= d2) return l0;
    if (d1 <= d0 && d1 <= d2) return l1;
    return l2;
}

// --- PUBLIC: TRAIN (Phase 1 + 2) ---
void DelaunayClassifier::train(const std::string& train_file, int outlier_k) {
    auto raw_points = load_labeled_csv(train_file);
    auto clean_points = remove_outliers(raw_points, outlier_k);
    
    dt.clear();
    dt.insert(clean_points.begin(), clean_points.end());
    std::cout << "✅ Phase 2 Complete: Delaunay Mesh Built (" << dt.number_of_vertices() << " vertices)." << std::endl;
}

// --- PUBLIC: PREDICT (Phase 4 + 5) ---
void DelaunayClassifier::predict_benchmark(const std::string& test_file, const std::string& output_file) {
    auto test_points = load_unlabeled_csv(test_file);
    std::cout << "🚀 Starting Benchmark on " << test_points.size() << " points..." << std::endl;

    std::vector<int> results;
    results.reserve(test_points.size());

    auto start = std::chrono::high_resolution_clock::now();
    for(const auto& p : test_points) {
        // ALGORITHM 4, STEP 6: Point Location
        Face_handle f = dt.locate(p); 
        int pred = -1;
        if (!dt.is_infinite(f)) pred = classify_point_in_face(f, p);
        else pred = dt.nearest_vertex(p)->info();
        results.push_back(pred);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration_us = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    double avg_time = duration_us.count() / (double)test_points.size();
    
    std::cout << "------------------------------------------------" << std::endl;
    std::cout << "⏱️  Total Inference Time: " << duration_us.count() / 1000.0 << " ms" << std::endl;
    std::cout << "🚀 Avg Time Per Point:   " << avg_time << " us" << std::endl;
    std::cout << "------------------------------------------------" << std::endl;

    // Save Predictions
    std::ofstream out(output_file);
    for(int p : results) out << p << "\n";
    out.close();
}

// --- PUBLIC: DYNAMIC STRESS TEST (Alg 1, 2, 3) ---
void DelaunayClassifier::run_dynamic_stress_test(const std::string& stream_file, const std::string& log_file) {
    auto stream_points = load_labeled_csv(stream_file);
    std::ofstream log(log_file);
    log << "operation,time_ns\n";
    
    std::cout << "🚀 Running Dynamic Algorithms 1, 2, 3 Stress Test..." << std::endl;
    
    Vertex_handle hint = dt.finite_vertices_begin();
    std::vector<Vertex_handle> handles;

    // 1. INSERTION (Algorithm 1)
    for(const auto& entry : stream_points) {
        auto start = std::chrono::high_resolution_clock::now();
        hint = dt.insert(entry.first, hint->face()); 
        hint->info() = entry.second;
        auto end = std::chrono::high_resolution_clock::now();
        
        handles.push_back(hint);
        log << "insert," << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";
    }

    // 2. MOVEMENT (Algorithm 3)
    for(auto v : handles) {
        Point new_p(v->point().x() + 0.05, v->point().y() + 0.05);
        auto start = std::chrono::high_resolution_clock::now();
        dt.move_if_no_collision(v, new_p);
        auto end = std::chrono::high_resolution_clock::now();
        log << "move," << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";
    }

    // 3. DELETION (Algorithm 2)
    for(int i = handles.size() - 1; i >= 0; --i) {
        auto start = std::chrono::high_resolution_clock::now();
        dt.remove(handles[i]);
        auto end = std::chrono::high_resolution_clock::now();
        log << "delete," << std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count() << "\n";
    }
    
    log.close();
    std::cout << "✅ Dynamic Benchmark Logs saved to " << log_file << std::endl;
}

// --- PUBLIC: EXPORT VISUALIZATION (Phase 2 & 3) ---
void DelaunayClassifier::export_visualization(const std::string& mesh_file, const std::string& boundary_file) {
    // Export Triangles
    std::ofstream triFile(mesh_file);
    for(auto e = dt.finite_edges_begin(); e != dt.finite_edges_end(); ++e) {
        auto s = dt.segment(e);
        triFile << s.source().x() << "," << s.source().y() << "," << s.target().x() << "," << s.target().y() << "\n";
    }
    triFile.close();

    // Export Boundaries (Case 2 & 3 Logic)
    std::ofstream boundFile(boundary_file);
    for(auto f = dt.finite_faces_begin(); f != dt.finite_faces_end(); ++f) {
        auto v0 = f->vertex(0); auto v1 = f->vertex(1); auto v2 = f->vertex(2);
        int l0 = v0->info(); int l1 = v1->info(); int l2 = v2->info();
        Point m01 = CGAL::midpoint(v0->point(), v1->point());
        Point m12 = CGAL::midpoint(v1->point(), v2->point());
        Point m20 = CGAL::midpoint(v2->point(), v0->point());

        if (l0 != l1 && l1 != l2 && l0 != l2) { // Case 3
            Point c = CGAL::centroid(v0->point(), v1->point(), v2->point());
            boundFile << c.x() << "," << c.y() << "," << m01.x() << "," << m01.y() << "\n";
            boundFile << c.x() << "," << c.y() << "," << m12.x() << "," << m12.y() << "\n";
            boundFile << c.x() << "," << c.y() << "," << m20.x() << "," << m20.y() << "\n";
        } else if (l0 != l1 || l1 != l2) { // Case 2
            std::vector<Point> actives;
            if (l0 != l1) actives.push_back(m01);
            if (l1 != l2) actives.push_back(m12);
            if (l2 != l0) actives.push_back(m20);
            if (actives.size() == 2) boundFile << actives[0].x() << "," << actives[0].y() << "," << actives[1].x() << "," << actives[1].y() << "\n";
        }
    }
    boundFile.close();
}