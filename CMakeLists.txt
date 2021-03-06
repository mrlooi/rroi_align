cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project (rect_intersects)

add_definitions("-std=c++11")
add_definitions("-fPIC")

set(CMAKE_CXX_FLAGS "-w ${CMAKE_CXX_FLAGS} -fopenmp")

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

cuda_add_executable(test_rroi
  src/RROIAlign_cuda.cu src/rroi_align.cu
  src/RROIPool_cuda.cu src/rroi_pool.cu
  src/vincent_rroi_align.cu src/bp_rroi_align.cu
  src/rotate_nms_cuda.cu
  src/nms.cu
  test/test_rroi.cpp)
target_link_libraries(test_rroi ${CUDA_LIBRARIES})

cuda_add_executable(sample_test src/vincent_rroi_align.cu test/sample_test.cpp)
target_link_libraries(sample_test ${CUDA_LIBRARIES})

cuda_add_executable(rotate_nms_test src/rotate_nms_cuda.cu test/rotate_nms_test.cpp)
target_link_libraries(rotate_nms_test ${CUDA_LIBRARIES})


# SET PYTORCH STUFF
#set(PYTORCH_HOME "/usr/local/lib/python3.6/site-packages/torch/lib")
#include_directories("${PYTORCH_HOME}/include")
#include_directories("${PYTORCH_HOME}/include/TH")

#cuda_add_executable(torch_test src/vincent_rroi_align.cu test/torch_test.cpp)
#target_link_libraries(torch_test ${CUDA_LIBRARIES})



find_package(OpenCV REQUIRED)  # opencv 
add_executable(rotate_mask_iou test/rotate_mask_iou.cpp)
target_link_libraries(rotate_mask_iou ${OpenCV_LIBS})
