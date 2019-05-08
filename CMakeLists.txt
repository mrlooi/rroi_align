cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (rect_intersects)

add_definitions("-std=c++11")
add_definitions("-fPIC")

# set(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS} -fopenmp")

find_package(CUDA)  # cuda

list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
list(APPEND CUDA_NVCC_FLAGS "--compiler-options -fPIC")
list(APPEND CUDA_NVCC_FLAGS "--ptxas-options=-v")
# list(APPEND CUDA_NVCC_FLAGS "-maxrregcount=64")

set(CUDA_NVCC_FLAGS_DEBUG   "${CUDA_NVCC_FLAGS_DEBUG};-G;-g")
set(CUDA_NVCC_FLAGS_RELEASE "${CUDA_NVCC_FLAGS_RELEASE};-O3;-DNDEBUG;--use_fast_math")

# own include dirs
include_directories("include")

add_executable(gen_testcases test/gen_testcases.cpp)
# target_link_libraries(gen_testcases)

cuda_add_executable(test_rroi src/RROIAlign_cuda.cu src/rroi_align.cu src/RROIPool_cuda.cu src/rroi_pool.cu test/test_rroi.cpp)
target_link_libraries(test_rroi ${CUDA_LIBRARIES})
