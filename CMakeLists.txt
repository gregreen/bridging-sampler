cmake_minimum_required(VERSION 3.0)
project(bridging_sampler_project)

# Find source files
file(GLOB_RECURSE sources src/*.cpp)

add_executable(bridging_sampler ${sources})

target_compile_options(bridging_sampler PUBLIC -std=c++14 -O2 -march=native)
