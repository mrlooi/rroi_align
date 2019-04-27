#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#include <cuda_runtime.h>
#include <cstdio>

#define CUDA_CHECK(call) { \
  cudaError_t err; \
  if ((err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
    exit(1); \
  } \
}

#endif /* CUDA_UTILS_H */
