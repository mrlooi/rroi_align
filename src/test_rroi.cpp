#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>

#include "RROIAlign_cuda.h"
#include "rroi_align.h"
#include "cuda_timer.h"
#include "cuda_utils.h"

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

  float* bottom_data_h;
  float* bottom_data_d;
  auto bottom_data_size = batch_size * channels * height * width;
  CUDA_CHECK(cudaMallocHost(&bottom_data_h, bottom_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&bottom_data_d, bottom_data_size * sizeof(float)));

  // Initialize bottom_data_h
  for (auto i = 0; i < bottom_data_size; i++) {
    fin >> bottom_data_h[i];
  }
  CUDA_CHECK(cudaMemcpy(bottom_data_d, bottom_data_h, bottom_data_size, cudaMemcpyHostToDevice));

  float* rois_h;
  float* rois_d;
  auto rois_size = num_rois * 6;
  CUDA_CHECK(cudaMallocHost(&rois_h, rois_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&rois_d, rois_size * sizeof(float)));

  // Initialize rois_h
  for (auto i = 0; i < num_rois * 6; i++) {
    fin >> rois_h[i];
  }
  CUDA_CHECK(cudaMemcpy(rois_d, rois_h, rois_size, cudaMemcpyHostToDevice));

  fin.close();

  float* top_data_h;
  float* top_data_d;
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  CUDA_CHECK(cudaMallocHost(&top_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&top_data_d, top_data_size * sizeof(float)));

  CUDATimer timer;
  auto write_output = [top_data_h, top_data_size](const std::string& filename) {
    std::fstream fout(filename, std::ios::out);
    for (auto i = 0; i < top_data_size; i++) {
      fout << top_data_h[i] << " ";
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
      bottom_data_d,
      rois_d,
      top_data_d,
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward_golden: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_h, top_data_d, top_data_size, cudaMemcpyDeviceToHost));
  write_output("golden");

  // Test RROIAlign_forward
  CUDA_CHECK(cudaMemset(top_data_d, 0, top_data_size));
  std::memset(top_data_h, 0, top_data_size);

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
      bottom_data_d,
      rois_d,
      top_data_d,
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_h, top_data_d, top_data_size, cudaMemcpyDeviceToHost));
  write_output("output");

  CUDA_CHECK(cudaFreeHost(bottom_data_h));
  CUDA_CHECK(cudaFree(bottom_data_d));

  CUDA_CHECK(cudaFreeHost(rois_h));
  CUDA_CHECK(cudaFree(rois_d));

  CUDA_CHECK(cudaFreeHost(top_data_h));
  CUDA_CHECK(cudaFree(top_data_d));

  return 0;
}
