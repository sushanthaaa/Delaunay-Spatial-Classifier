# ==============================================================================
# Delaunay Triangulation Classifier — Build Configuration
# ==============================================================================
#
# Toolchain requirements:
#   - CMake >= 3.10
#   - A C++17-capable compiler (GCC 7+, Clang 5+, Apple Clang 9+, MSVC 19.14+)
#   - CGAL >= 5.0 (tested with 5.x and 6.1.1 on macOS, 5.x on Ubuntu)
#   - FLANN >= 1.9
#   - LibSVM >= 3.25
#   - LZ4 (any recent version; a FLANN transitive dependency)
# ==============================================================================

cmake_minimum_required(VERSION 3.10)
project(DelaunayClassifier LANGUAGES CXX)

# ------------------------------------------------------------------------------
# C++ standard: enforce C++17
# ------------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# ------------------------------------------------------------------------------
# Default build type: Release
# ------------------------------------------------------------------------------
# If the user did not pass -DCMAKE_BUILD_TYPE=..., we default to Release so the
# benchmark numbers match the paper. Override with
#     cmake .. -DCMAKE_BUILD_TYPE=Debug
# for development / debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  message(STATUS
    "No CMAKE_BUILD_TYPE specified; defaulting to Release. "
    "Pass -DCMAKE_BUILD_TYPE=Debug to override.")
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
    "Choose the build type: Debug, Release, RelWithDebInfo, MinSizeRel"
    FORCE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS
    "Debug" "Release" "RelWithDebInfo" "MinSizeRel")
endif()

# ------------------------------------------------------------------------------
# Dependencies
# ------------------------------------------------------------------------------
# CGAL 5.0+ is required. Versions before 5.0 use a different CMake integration
# pattern and will not work with find_package(CGAL ...) as written here.
find_package(CGAL REQUIRED)

find_package(PkgConfig REQUIRED)
pkg_check_modules(FLANN REQUIRED flann)
pkg_check_modules(LZ4   REQUIRED liblz4)

# LibSVM: try pkg-config first, fall back to common install locations.
# The SVM library is named "svm" on most systems (libsvm.so / libsvm.dylib /
# svm.lib), and the header is svm.h.
pkg_check_modules(LIBSVM libsvm)
if(NOT LIBSVM_FOUND)
  find_path(LIBSVM_INCLUDE_DIR svm.h
    HINTS
      /opt/homebrew/include        # Apple Silicon Homebrew
      /usr/local/include           # Intel macOS Homebrew, or custom installs
      /usr/include                 # Linux system package
  )
  find_library(LIBSVM_LIB svm
    HINTS
      /opt/homebrew/lib
      /usr/local/lib
      /usr/lib
      /usr/lib/x86_64-linux-gnu    # Debian/Ubuntu multi-arch path
  )
  if(LIBSVM_INCLUDE_DIR AND LIBSVM_LIB)
    set(LIBSVM_INCLUDE_DIRS ${LIBSVM_INCLUDE_DIR})
    set(LIBSVM_LIBRARIES    ${LIBSVM_LIB})
    message(STATUS "LibSVM found at: ${LIBSVM_INCLUDE_DIR}")
  else()
    message(FATAL_ERROR
      "LibSVM not found. Install via one of:\n"
      "  macOS:    brew install libsvm\n"
      "  Debian:   sudo apt-get install libsvm-dev\n"
      "  Windows:  vcpkg install libsvm:x64-windows")
  endif()
endif()

# ------------------------------------------------------------------------------
# Include / link paths
# ------------------------------------------------------------------------------
include_directories(
  include
  ${FLANN_INCLUDE_DIRS}
  ${LIBSVM_INCLUDE_DIRS}
)
link_directories(
  ${FLANN_LIBRARY_DIRS}
  ${LZ4_LIBRARY_DIRS}
)

# ------------------------------------------------------------------------------
# Targets
# ------------------------------------------------------------------------------

# Core CLI — static mode, dynamic stress test, dynamic visualization.
add_executable(main
  src/main.cpp
  src/DelaunayClassifier.cpp
)
target_link_libraries(main PRIVATE CGAL::CGAL)

# C++ benchmark: FLANN KNN, LibSVM, DT, RF vs. Delaunay.
add_executable(benchmark
  src/benchmark.cpp
  src/DelaunayClassifier.cpp
)
target_link_libraries(benchmark PRIVATE
  CGAL::CGAL
  ${FLANN_LIBRARIES}
  ${LZ4_LIBRARIES}
  ${LIBSVM_LIBRARIES}
)

# Ablation benchmark — component-contribution analysis + dynamic ops.
add_executable(ablation_bench
  src/ablation_bench.cpp
  src/DelaunayClassifier.cpp
)
target_link_libraries(ablation_bench PRIVATE CGAL::CGAL)

# ------------------------------------------------------------------------------
# Optional: print a summary at configure time
# ------------------------------------------------------------------------------
message(STATUS "")
message(STATUS "=== Delaunay Classifier build configuration ===")
message(STATUS "  Build type:      ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ standard:    ${CMAKE_CXX_STANDARD}")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "  CGAL version:    ${CGAL_VERSION}")
message(STATUS "  FLANN found:     ${FLANN_FOUND} (${FLANN_VERSION})")
message(STATUS "  LZ4 found:       ${LZ4_FOUND} (${LZ4_VERSION})")
message(STATUS "================================================")
message(STATUS "")