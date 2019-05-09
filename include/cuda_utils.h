#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>
#include <memory>


#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


#define CUDA_CHECK(call) { \
  cudaError_t err; \
  if ((err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
    exit(1); \
  } \
}

template <typename T>
struct device_ptr_deleter {
  void operator()(T* ptr) {
    CUDA_CHECK(cudaFree(ptr));
  }
};

template <typename T>
struct host_ptr_deleter {
  void operator()(T* ptr) {
    CUDA_CHECK(cudaFreeHost(ptr));
  }
};

template <typename T>
using unique_ptr_device = std::unique_ptr<T[], device_ptr_deleter<T>>;

template <typename T>
using unique_ptr_host = std::unique_ptr<T[], host_ptr_deleter<T>>;

#endif /* CUDA_UTILS_H */
