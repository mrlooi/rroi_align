#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>

#include "RROIAlign_cuda.h"
#include "rroi_align.h"
#include "RROIPool_cuda.h"
#include "rroi_pool.h"

#include "cuda_timer.h"
#include "cuda_utils.h"


void test_correctness(float* golden_data, float* output_data, int data_size, const float elapsed = 0.0001f)
{
  bool is_correct = true;
  for (auto i = 0; i < data_size; i++) {
    if (fabs(golden_data[i] - output_data[i]) > elapsed) {
      is_correct = false;
      std::cout << "FAILED: " << i << " output: " << output_data[i] << " golden: " << golden_data[i] << std::endl;
      break;
    }
  }

  if (is_correct) {
    std::cout << "PASSED!" << std::endl;
  }
}

int main()
{
  std::string in_filename = "testcase";
  std::fstream fin(in_filename, std::ios::in);

  int batch_size;
  int num_rois;
  int channels;
  int height;
  int width;
  int pooled_height;
  int pooled_width;
  float spatial_scale;
  fin >> batch_size
      >> num_rois
      >> channels
      >> height
      >> width
      >> pooled_height
      >> pooled_width
      >> spatial_scale;

  unique_ptr_host<float> bottom_data_h(nullptr);
  unique_ptr_device<float> bottom_data_d(nullptr);
  auto bottom_data_size = batch_size * channels * height * width;
  CUDA_CHECK(cudaMallocHost((void **) &bottom_data_h, bottom_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &bottom_data_d, bottom_data_size * sizeof(float)));

  // Initialize bottom_data_h
  for (auto i = 0; i < bottom_data_size; i++) {
    fin >> bottom_data_h[i];
  }
  CUDA_CHECK(cudaMemcpy(bottom_data_d.get(), bottom_data_h.get(), bottom_data_size, cudaMemcpyHostToDevice));

  unique_ptr_host<float> rois_h(nullptr);
  unique_ptr_device<float> rois_d(nullptr);
  auto rois_size = num_rois * 6;
  CUDA_CHECK(cudaMallocHost((void **) &rois_h, rois_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &rois_d, rois_size * sizeof(float)));

  // Initialize rois_h
  for (auto i = 0; i < num_rois * 6; i++) {
    fin >> rois_h[i];
  }
  CUDA_CHECK(cudaMemcpy(rois_d.get(), rois_h.get(), rois_size, cudaMemcpyHostToDevice));

  fin.close();

  unique_ptr_host<float> top_data_golden_h(nullptr);
  unique_ptr_device<float> top_data_golden_d(nullptr);
  unique_ptr_host<float> top_data_h(nullptr);
  unique_ptr_device<float> top_data_d(nullptr);
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  CUDA_CHECK(cudaMallocHost((void **) &top_data_golden_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_golden_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &top_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_d, top_data_size * sizeof(float)));

  CUDATimer timer;
  auto write_output = [channels, pooled_height, pooled_width, top_data_size](const std::string& filename, const float* top_data_h) {
    std::fstream fout(filename, std::ios::out);
    for (auto i = 0; i < top_data_size; i++) {
      fout << top_data_h[i] << " ";
      if ((i+1) % (pooled_width * pooled_height) == 0) {
        fout << std::endl;
      }
    }
  };

  // Use golden function
  timer.start();
  RROIAlign_forward_golden(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get(),
      top_data_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward_golden: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_golden_h.get(), top_data_golden_d.get(), top_data_size, cudaMemcpyDeviceToHost));
  write_output("golden", top_data_golden_h.get());

  // CUDA_CHECK(cudaMemset(top_data_d.get(), 0, top_data_size));
  // std::memset(top_data_h.get(), 0, top_data_size);
  // CUDA_CHECK(cudaDeviceSynchronize());

  // Test RROIAlign_forward
  timer.start();
  RROIAlign_forward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get(),
      top_data_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_h.get(), top_data_d.get(), top_data_size, cudaMemcpyDeviceToHost));
  write_output("output", top_data_h.get());

  test_correctness(top_data_golden_h.get(), top_data_h.get(), top_data_size, 1e-4);

  ////////////////////////////////////////////////////////////////////////////////
  // RROI pooling
  ////////////////////////////////////////////////////////////////////////////////

  unique_ptr_host<float> top_pool_data_golden_h(nullptr);
  unique_ptr_device<float> top_pool_data_golden_d(nullptr);
  unique_ptr_host<float> top_pool_data_h(nullptr);
  unique_ptr_device<float> top_pool_data_d(nullptr);
  CUDA_CHECK(cudaMallocHost((void **) &top_pool_data_golden_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_pool_data_golden_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &top_pool_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_pool_data_d, top_data_size * sizeof(float)));

  // Use golden function
  timer.start();
  RROIPool_forward_golden(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get(),
      top_pool_data_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIPool_forward_golden: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_pool_data_golden_h.get(), top_pool_data_golden_d.get(), top_data_size, cudaMemcpyDeviceToHost));
  write_output("pool-golden", top_pool_data_golden_h.get());

  timer.start();
  RROIPool_forward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get(),
      top_pool_data_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIPool_forward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_pool_data_h.get(), top_pool_data_d.get(), top_data_size, cudaMemcpyDeviceToHost));
  write_output("pool", top_pool_data_h.get());

  test_correctness(top_pool_data_golden_h.get(), top_pool_data_h.get(), top_data_size, 1e-10);

  return 0;
}
