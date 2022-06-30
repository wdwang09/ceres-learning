cmake_minimum_required(VERSION 3.0)
project(ceres_learning LANGUAGES CXX)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif ()

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Ceres REQUIRED)

add_executable(helloworld src/helloworld.cpp)
target_link_libraries(helloworld Ceres::ceres)

add_executable(helloworld_numeric_diff src/helloworld_numeric_diff.cpp)
target_link_libraries(helloworld_numeric_diff Ceres::ceres)

add_executable(helloworld_analytic_diff src/helloworld_analytic_diff.cpp)
target_link_libraries(helloworld_analytic_diff Ceres::ceres)

add_executable(powell src/powell.cpp)
target_link_libraries(powell Ceres::ceres)

add_executable(curve_fitting src/curve_fitting.cpp)
target_link_libraries(curve_fitting Ceres::ceres)
