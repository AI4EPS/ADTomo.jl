cmake_minimum_required(VERSION 3.5)

# C++11 required for tensorflow
set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")

include_directories(/home/weiqiang/.local/miniconda3/include/eigen3)
add_executable(main Eikonal3D.cpp)