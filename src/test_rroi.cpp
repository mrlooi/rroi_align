#include <iostream>
#include <fstream>
#include <cuda_runtime.h>

#include "RROIAlign_cuda.h"
#include "cuda_timer.h"

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
  // cudaMallocHost((void **) &bottom_data_h, bottom_data_size * sizeof(float));
  // cudaMalloc((void **) &bottom_data_d, bottom_data_size * sizeof(float));
  cudaMallocHost(&bottom_data_h, bottom_data_size * sizeof(float));
  cudaMalloc(&bottom_data_d, bottom_data_size * sizeof(float));

  // Initialize bottom_data_h
  for (auto i = 0; i < bottom_data_size; i++) {
    fin >> bottom_data_h[i];
  }
  cudaMemcpy(bottom_data_d, bottom_data_h, bottom_data_size, cudaMemcpyHostToDevice);

  float* rois_h;
  float* rois_d;
  auto rois_size = num_rois * 6;
  // cudaMallocHost((void **) &rois_h, rois_size * sizeof(float));
  // cudaMalloc((void **) &rois_d, rois_size * sizeof(float));
  cudaMallocHost(&rois_h, rois_size * sizeof(float));
  cudaMalloc(&rois_d, rois_size * sizeof(float));

  // Initialize rois_h
  for (auto i = 0; i < num_rois * 6; i++) {
    fin >> rois_h[i];
  }
  cudaMemcpy(rois_d, rois_h, rois_size, cudaMemcpyHostToDevice);

  fin.close();

  float* top_data_h;
  float* top_data_d;
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  // cudaMallocHost((void **) &top_data_h, top_data_size * sizeof(float));
  // cudaMalloc((void **) &top_data_d, top_data_size * sizeof(float));
  cudaMallocHost(&top_data_h, top_data_size * sizeof(float));
  cudaMalloc(&top_data_d, top_data_size * sizeof(float));

  CUDATimer timer;

  timer.start();
  RROIAlign_forward_cuda(
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
      top_data_d
      );
  cudaDeviceSynchronize();
  timer.stop();
  std::cout << "RROIAlign_forward_cuda: " << timer.elapsed() << std::endl;

  cudaMemcpy(top_data_h, top_data_d, top_data_size, cudaMemcpyDeviceToHost);

  std::string out_filename = "output";
  std::fstream fout(out_filename, std::ios::out);
  for (auto i = 0; i < top_data_size; i++) {
    fout << top_data_h[i] << " ";
  }

  cudaFreeHost(bottom_data_h);
  cudaFree(bottom_data_d);

  cudaFreeHost(rois_h);
  cudaFree(rois_d);

  cudaFreeHost(top_data_h);
  cudaFree(top_data_d);

  return 0;
}
