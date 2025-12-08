#include "../include/DelaunayClassifier.h"
#include <iostream>
#include <string>

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cout << "Usage: ./main [mode] [args...]\n";
        std::cout << "  1. Static:  ./main static <train_csv> <test_csv> <output_dir>\n";
        std::cout << "  2. Dynamic: ./main dynamic <base_csv> <stream_csv> <log_file_path>\n";
        return 1;
    }

    std::string mode = argv[1];
    DelaunayClassifier classifier;

    if (mode == "static" && argc == 5) {
        // Static Mode: Arg 4 is a DIRECTORY
        std::string out_dir = argv[4];
        classifier.train(argv[2]); 
        classifier.export_visualization(out_dir + "/triangles.csv", out_dir + "/boundaries.csv");
        classifier.predict_benchmark(argv[3], out_dir + "/predictions.csv");
    } 
    else if (mode == "dynamic" && argc == 5) {
        // Dynamic Mode: Arg 4 is a FILE PATH (Fixed)
        // We use argv[4] directly so we can have unique logs like 'wine_dynamic_logs.csv'
        classifier.train(argv[2]); 
        classifier.run_dynamic_stress_test(argv[3], argv[4]);
    }
    
    return 0;
}