#ifndef CUDA_TIMER_H
#define CUDA_TIMER_H
#include <cuda_runtime.h>

struct CUDATimer
{
  cudaEvent_t start_;
  cudaEvent_t stop_;

  CUDATimer()
  {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  ~CUDATimer()
  {
    cudaEventDestroy(start_);
    cudaEventDestroy(stop_);
  }

  void start()
  {
    cudaEventRecord(start_, 0);
  }

  void stop()
  {
    cudaEventRecord(stop_, 0);
  }

  float elapsed()
  {
    float elapsed;
    cudaEventSynchronize(stop_);
    cudaEventElapsedTime(&elapsed, start_, stop_);
    return elapsed;
  }
};

#endif /* CUDA_TIMER_H */
