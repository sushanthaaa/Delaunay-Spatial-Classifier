cmake_minimum_required(VERSION 3.10)
project(DelaunayClassifier)

find_package(CGAL REQUIRED)

include_directories(include)

add_executable(main src/main.cpp src/DelaunayClassifier.cpp)
target_link_libraries(main CGAL::CGAL)