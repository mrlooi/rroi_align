#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cstring>
#include <cmath>
#include <vector>

#include "cuda_timer.h"
#include "cuda_utils.h"

#include "rroi.h"
#include "rotate_nms.h"

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

void write_output(const std::string& filename, const float* top_data_h, const int top_data_size, const int channels, const int pooled_height, const int pooled_width)
{
  std::fstream fout(filename, std::ios::out);
  for (auto i = 0; i < top_data_size; i++) {
    fout << top_data_h[i] << " ";
    if ((i+1) % (pooled_width * pooled_height) == 0) {
      fout << std::endl;
    }
  }
}

void test_RROIAlign_forward(
  int batch_size,
  int num_rois,
  int channels,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  float spatial_scale,
  float* bottom_data_d,
  float* rois_d
    )
{
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;

  unique_ptr_host<float> top_data_golden_h(nullptr);
  unique_ptr_device<float> top_data_golden_d(nullptr);
  unique_ptr_host<float> top_data_h(nullptr);
  unique_ptr_device<float> top_data_d(nullptr);
  CUDA_CHECK(cudaMallocHost((void **) &top_data_golden_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_golden_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &top_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_d, top_data_size * sizeof(float)));

  CUDATimer timer;

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
      top_data_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward_golden: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_golden_h.get(), top_data_golden_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("rroi_align.golden", top_data_golden_h.get(), top_data_size, channels, pooled_height, pooled_width);

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
      bottom_data_d,
      rois_d,
      top_data_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIAlign_forward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_h.get(), top_data_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("rroi_align.output", top_data_h.get(), top_data_size, channels, pooled_height, pooled_width);

  test_correctness(top_data_golden_h.get(), top_data_h.get(), top_data_size, 1e-4);
}

void test_RROIPool_forward(
  int batch_size,
  int num_rois,
  int channels,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  float spatial_scale,
  float* bottom_data_d,
  float* rois_d
  )
{
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;

  unique_ptr_host<float> top_pool_data_golden_h(nullptr);
  unique_ptr_device<float> top_pool_data_golden_d(nullptr);
  unique_ptr_host<float> top_pool_data_h(nullptr);
  unique_ptr_device<float> top_pool_data_d(nullptr);

  CUDA_CHECK(cudaMallocHost((void **) &top_pool_data_golden_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_pool_data_golden_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &top_pool_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_pool_data_d, top_data_size * sizeof(float)));

  CUDATimer timer;

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
      bottom_data_d,
      rois_d,
      top_pool_data_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIPool_forward_golden: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_pool_data_golden_h.get(), top_pool_data_golden_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("rroi_pool.golden", top_pool_data_golden_h.get(), top_data_size, channels, pooled_height, pooled_width);

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
      bottom_data_d,
      rois_d,
      top_pool_data_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "RROIPool_forward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_pool_data_h.get(), top_pool_data_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("rroi_pool.output", top_pool_data_h.get(), top_data_size, channels, pooled_height, pooled_width);

  test_correctness(top_pool_data_golden_h.get(), top_pool_data_h.get(), top_data_size, 1e-10);
}

// binlinear interpolation version of RROI align
void test_bp_rroi_align_forward(
  int batch_size,
  int num_rois,
  int channels,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  float spatial_scale,
  float* bottom_data_d,
  float* rois_d
  )
{
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;

  unique_ptr_host<float> top_data_golden_h(nullptr);
  unique_ptr_device<float> top_data_golden_d(nullptr);
  unique_ptr_host<float> top_data_h(nullptr);
  unique_ptr_device<float> top_data_d(nullptr);
  CUDA_CHECK(cudaMallocHost((void **) &top_data_golden_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_golden_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &top_data_h, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &top_data_d, top_data_size * sizeof(float)));

  CUDATimer timer;

  // Use golden function
  timer.start();
  vincent_rroi_align(
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
      top_data_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "vincent_rroi_align: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_golden_h.get(), top_data_golden_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("bp_rroi_align.golden", top_data_golden_h.get(), top_data_size, channels, pooled_height, pooled_width);

  timer.start();
  bp_rroi_align_forward(
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
      top_data_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "bp_rroi_align: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(top_data_h.get(), top_data_d.get(), top_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("bp_rroi_align.output", top_data_h.get(), top_data_size, channels, pooled_height, pooled_width);

  test_correctness(top_data_golden_h.get(), top_data_h.get(), top_data_size, 1e-10);
}

// binlinear interpolation version of RROI align
void test_bp_rroi_align_backward(
  int batch_size,
  int num_rois,
  int channels,
  int height,
  int width,
  int pooled_height,
  int pooled_width,
  float spatial_scale,
  float* top_diff_d,
  float* rois_d
  )
{
  auto bottom_data_size = batch_size * channels * height * width;

  unique_ptr_host<float> bottom_diff_golden_h(nullptr);
  unique_ptr_device<float> bottom_diff_golden_d(nullptr);
  unique_ptr_host<float> bottom_diff_h(nullptr);
  unique_ptr_device<float> bottom_diff_d(nullptr);
  CUDA_CHECK(cudaMallocHost((void **) &bottom_diff_golden_h, bottom_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &bottom_diff_golden_d, bottom_data_size * sizeof(float)));
  CUDA_CHECK(cudaMallocHost((void **) &bottom_diff_h, bottom_data_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &bottom_diff_d, bottom_data_size * sizeof(float)));

  CUDATimer timer;

  // Use golden function
  timer.start();
  vincent_rroi_align_backward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      top_diff_d,
      rois_d,
      bottom_diff_golden_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "vincent_rroi_align_backward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(bottom_diff_golden_h.get(), bottom_diff_golden_d.get(), bottom_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("bp_rroi_align_backward.golden", bottom_diff_golden_h.get(), bottom_data_size, channels, height, width);

  timer.start();
  bp_rroi_align_backward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      top_diff_d,
      rois_d,
      bottom_diff_d.get(),
      0
      );
  CUDA_CHECK(cudaDeviceSynchronize());
  timer.stop();
  std::cout << "bp_rroi_align_backward: " << timer.elapsed() << std::endl;

  CUDA_CHECK(cudaMemcpy(bottom_diff_h.get(), bottom_diff_d.get(), bottom_data_size * sizeof(float), cudaMemcpyDeviceToHost));
  write_output("bp_rroi_align_backward.output", bottom_diff_h.get(), bottom_data_size, channels, height, width);

  test_correctness(bottom_diff_golden_h.get(), bottom_diff_h.get(), bottom_data_size, 1e-4);
}

void test_rroi(std::string& in_filename)
{
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
  CUDA_CHECK(cudaMemcpy(bottom_data_d.get(), bottom_data_h.get(), bottom_data_size * sizeof(float), cudaMemcpyHostToDevice));

  unique_ptr_host<float> rois_h(nullptr);
  unique_ptr_device<float> rois_d(nullptr);
  auto rois_size = num_rois * 6;
  CUDA_CHECK(cudaMallocHost((void **) &rois_h, rois_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &rois_d, rois_size * sizeof(float)));

  // Initialize rois_h
  for (auto i = 0; i < num_rois * 6; i++) {
    fin >> rois_h[i];
  }
  CUDA_CHECK(cudaMemcpy(rois_d.get(), rois_h.get(), rois_size * sizeof(float), cudaMemcpyHostToDevice));

  fin.close();

#if 0
  test_RROIAlign_forward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get()
      );
#endif

#if 0
  test_RROIPool_forward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get()
      );
#endif

  test_bp_rroi_align_forward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      bottom_data_d.get(),
      rois_d.get()
      );

  ////////////////////////////////////////////////////////////////////////////////
  // backward phrase
  ////////////////////////////////////////////////////////////////////////////////
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  std::vector<float> top_diff(top_data_size, 1);

  unique_ptr_device<float> top_diff_d(nullptr);
  CUDA_CHECK(cudaMalloc((void **) &top_diff_d, top_data_size * sizeof(float)));
  CUDA_CHECK(cudaMemcpy(top_diff_d.get(), &top_diff[0], top_data_size * sizeof(float), cudaMemcpyHostToDevice));

  test_bp_rroi_align_backward(
      batch_size,
      num_rois,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      spatial_scale,
      top_diff_d.get(),
      rois_d.get()
      );
}

void test_nms(std::string& in_filename)
{
  std::fstream fin(in_filename, std::ios::in);

  float nms_thresh;
  int max_output;
  int height;
  int width;
  int num_rois;

  fin >> nms_thresh
      >> max_output
      >> height
      >> width
      >> num_rois;

  unique_ptr_host<float> rois_h(nullptr);
  unique_ptr_device<float> rois_d(nullptr);
  auto rois_size = num_rois * 5;
  CUDA_CHECK(cudaMallocHost((void **) &rois_h, rois_size * sizeof(float)));
  CUDA_CHECK(cudaMalloc((void **) &rois_d, rois_size * sizeof(float)));

  for (auto i = 0; i < rois_size; i++) {
    fin >> rois_h[i];
  }
  CUDA_CHECK(cudaMemcpy(rois_d.get(), rois_h.get(), rois_size * sizeof(float), cudaMemcpyHostToDevice));

#if 1
  {
    unique_ptr_device<int64_t> out_keep(nullptr);

    CUDATimer timer;
    timer.start();
    int num_to_keep = rotated_nms_golden(
        rois_d.get(),
        out_keep.get(),
        num_rois,
        nms_thresh,
        max_output
        );
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();

    std::cout << "rotated_nms_golden: " << timer.elapsed() << std::endl;
    std::cout << "golden num_to_keep: " << num_to_keep << std::endl;
  }
#endif

#if 1
  {
    unique_ptr_device<int64_t> out_keep(nullptr);
    CUDA_CHECK(cudaMalloc((void **) &out_keep, num_rois * sizeof(int64_t)));

    CUDATimer timer;
    timer.start();
    int num_to_keep = rotated_nms(
        rois_d.get(),
        out_keep.get(),
        num_rois,
        max_output,
        nms_thresh
        );
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.stop();

    std::cout << "rotated_nms: " << timer.elapsed() << std::endl;
    std::cout << "num_to_keep: " << num_to_keep << std::endl;
  }
#endif
}

int main()
{
  std::string filename = "testcase";
  // test_rroi(filename);

  filename = "nms_testcase";
  test_nms(filename);

  return 0;
}
