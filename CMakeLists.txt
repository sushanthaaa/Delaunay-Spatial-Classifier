cmake_minimum_required(VERSION 3.10)
project(DelaunayClassifier)

find_package(CGAL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
pkg_check_modules(LZ4 REQUIRED liblz4)

# LibSVM (installed via Homebrew, no pkg-config)
set(LIBSVM_INCLUDE_DIRS "/opt/homebrew/include")
set(LIBSVM_LIBRARY_DIRS "/opt/homebrew/lib")
set(LIBSVM_LIBRARIES "svm")

include_directories(include ${FLANN_INCLUDE_DIRS} ${LIBSVM_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS} ${LZ4_LIBRARY_DIRS} ${LIBSVM_LIBRARY_DIRS})

# Main executable
add_executable(main src/main.cpp src/DelaunayClassifier.cpp)
target_link_libraries(main CGAL::CGAL)

# C++ benchmark (Delaunay vs FLANN KNN vs SVM vs Decision Tree)
add_executable(benchmark src/benchmark.cpp src/DelaunayClassifier.cpp)
target_link_libraries(benchmark CGAL::CGAL ${FLANN_LIBRARIES} ${LZ4_LIBRARIES} ${LIBSVM_LIBRARIES})

# Ablation benchmark (Full System vs No SRR vs Nearest Vertex)
add_executable(ablation_bench src/ablation_bench.cpp src/DelaunayClassifier.cpp)
target_link_libraries(ablation_bench CGAL::CGAL)