cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (rect_intersects)

add_definitions("-std=c++11")
add_definitions("-fPIC")
# add_definitions("-g")

# set(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS} -fopenmp")

find_package(CUDA)  # cuda
# set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11; -g -G)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -std=c++11)
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")

# own include dirs
include_directories("include")

add_executable(gen_testcases src/gen_testcases.cpp)
# target_link_libraries(gen_testcases)

cuda_add_executable(test_rroi src/RROIAlign_cuda.cu src/test_rroi.cpp)
target_link_libraries(test_rroi ${CUDA_LIBRARIES})
