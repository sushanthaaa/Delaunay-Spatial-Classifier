cmake_minimum_required(VERSION 3.10)
project(DelaunayClassifier)

find_package(CGAL REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
pkg_check_modules(LZ4 REQUIRED liblz4)

# LibSVM — try pkg-config first, fall back to common install locations
pkg_check_modules(LIBSVM libsvm)
if(NOT LIBSVM_FOUND)
  # Fallback: search standard paths (Homebrew, system, vcpkg)
  find_path(LIBSVM_INCLUDE_DIR svm.h
    HINTS /opt/homebrew/include /usr/local/include /usr/include
  )
  find_library(LIBSVM_LIB svm
    HINTS /opt/homebrew/lib /usr/local/lib /usr/lib
  )
  if(LIBSVM_INCLUDE_DIR AND LIBSVM_LIB)
    set(LIBSVM_INCLUDE_DIRS ${LIBSVM_INCLUDE_DIR})
    set(LIBSVM_LIBRARIES ${LIBSVM_LIB})
    message(STATUS "LibSVM found at: ${LIBSVM_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR "LibSVM not found. Install via: brew install libsvm / apt install libsvm-dev")
  endif()
endif()

include_directories(include ${FLANN_INCLUDE_DIRS} ${LIBSVM_INCLUDE_DIRS})
link_directories(${FLANN_LIBRARY_DIRS} ${LZ4_LIBRARY_DIRS})

# Main executable
add_executable(main src/main.cpp src/DelaunayClassifier.cpp)
target_link_libraries(main CGAL::CGAL)

# C++ benchmark (Delaunay vs FLANN KNN vs SVM vs Decision Tree)
add_executable(benchmark src/benchmark.cpp src/DelaunayClassifier.cpp)
target_link_libraries(benchmark CGAL::CGAL ${FLANN_LIBRARIES} ${LZ4_LIBRARIES} ${LIBSVM_LIBRARIES})

# Ablation benchmark (component contribution analysis)
add_executable(ablation_bench src/ablation_bench.cpp src/DelaunayClassifier.cpp)
target_link_libraries(ablation_bench CGAL::CGAL)