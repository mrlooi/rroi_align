#include "rroi.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda.h>

#include "rroi_helper.h"
#include "rotate_rect_ops.h"
#include "cuda_utils.h"

#if 1
// NOTE: only cache one roi_pool_pt in the shared memory
template <typename T>
__global__ void compute_weight(
    T* __restrict__ top_data,
    const T* __restrict__ bottom_data,
    const T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  extern __shared__ T roi_pool_pts_shared[];

  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      int roi_batch_ind = rois_offset[0];
      T rbox_area = get_rotated_bounding_box_area(spatial_scale, rois_offset[4], rois_offset[3], pooled_height, pooled_width);

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      // int roi_pool_idx_shared = threadIdx.x * threadIdx.y / (pooled_width * pooled_height * channels);
      // int roi_pool_idx_shared = threadIdx.x;
      // int roi_pool_offset_shared = 8 * roi_pool_idx_shared;
      int roi_pool_offset_shared = 0;
      // if (threadIdx.x == 0) {
      //   for (int k = 0; k < 8; k++) {
      //     roi_pool_pts_shared[pooled_height*pooled_width*roi_pool_idx_shared + k] = roi_pool_pts[k * roi_pool_pt_num + roi_pool_idx];
      //   }
      // }
      if (threadIdx.x == 0 && c < 8) {
        roi_pool_pts_shared[roi_pool_offset_shared + c] = roi_pool_pts[c * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      int left, top, right, bottom;
      get_rotated_bounding_box(left, top, right, bottom, roi_pool_pts_shared + roi_pool_offset_shared, width, height);

      const T* bottom_data_offset = bottom_data + (roi_batch_ind * channels + c) * height * width;
      T output_val = 0.0;
      for (int hh = top; hh < bottom+1; ++hh) {
        for (int ww = left; ww < right+1; ++ww) {
          // T pixel_rect_vertices[8] = {ww+0.0f, hh+0.0f, ww+1.0f, hh+0.0f, ww+1.0f, hh+1.0f, ww+0.0f, hh+1.0f};
          // T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts_shared + roi_pool_offset_shared);

          T inter_area = itersect_area_rbox_aabox(
              roi_pool_pts_shared + roi_pool_offset_shared,
              rbox_area,
              ww + 0.f,
              ww + 1.f,
              hh + 0.f,
              hh + 1.f
              );

          T px_weight = inter_area / rbox_area;
          output_val += px_weight * bottom_data_offset[hh * width + ww];
        }
      }
      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = output_val;
    }
  }
}
#endif

#if 0
// NOTE: cache multiple roi_pool_pts in the shared mem
template <typename T>
__global__ void compute_weight(
    T* __restrict__ top_data,
    const T* __restrict__ bottom_data,
    const T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  extern __shared__ T roi_pool_pts_shared[];

  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      int roi_batch_ind = rois_offset[0];
      T rbox_area = get_rotated_bounding_box_area(spatial_scale, rois_offset[4], rois_offset[3], pooled_height, pooled_width);

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      int roi_pool_offset_shared = 8 * threadIdx.x;
      if (threadIdx.y < 8) {
        roi_pool_pts_shared[roi_pool_offset_shared + threadIdx.y] = roi_pool_pts[threadIdx.y * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      int left, top, right, bottom;
      get_rotated_bounding_box(left, top, right, bottom, roi_pool_pts_shared + roi_pool_offset_shared, width, height);

      const T* bottom_data_offset = bottom_data + (roi_batch_ind * channels + c) * height * width;
      T output_val = 0.0;
      for (int hh = top; hh < bottom+1; ++hh) {
        for (int ww = left; ww < right+1; ++ww) {
          T pixel_rect_vertices[8] = {ww+0.0f, hh+0.0f, ww+1.0f, hh+0.0f, ww+1.0f, hh+1.0f, ww+0.0f, hh+1.0f};
          T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts_shared + roi_pool_offset_shared);
          T px_weight = inter_area / rbox_area;
          output_val += px_weight * bottom_data_offset[hh * width + ww];
        }
      }
      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = output_val;
    }
  }
}
#endif

// local memory version
template <typename T>
__global__ void compute_weight_local(
    T* __restrict__ top_data,
    const T* __restrict__ bottom_data,
    const T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      int roi_batch_ind = rois_offset[0];
      T rbox_area = get_rotated_bounding_box_area(spatial_scale, rois_offset[4], rois_offset[3], pooled_height, pooled_width);

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      T roi_pool_pts_local[8];
      for (int k = 0; k < 8; k++) {
        roi_pool_pts_local[k] = roi_pool_pts[k * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      int left, top, right, bottom;
      get_rotated_bounding_box(left, top, right, bottom, roi_pool_pts_local, width, height);

      const T* bottom_data_offset = bottom_data + (roi_batch_ind * channels + c) * height * width;
      T output_val = 0.0;
      for (int hh = top; hh < bottom+1; ++hh) {
        for (int ww = left; ww < right+1; ++ww) {
          // T pixel_rect_vertices[8] = {ww+0.0f, hh+0.0f, ww+1.0f, hh+0.0f, ww+1.0f, hh+1.0f, ww+0.0f, hh+1.0f};
          // T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts_local);

          T inter_area = itersect_area_rbox_aabox(
              roi_pool_pts_local,
              rbox_area,
              ww + 0.f,
              ww + 1.f,
              hh + 0.f,
              hh + 1.f
              );

          T px_weight = inter_area / rbox_area;
          output_val += px_weight * bottom_data_offset[hh * width + ww];
        }
      }
      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = output_val;
    }
  }
}

void RROIAlign_forward(
    int batch_size,
    int num_rois,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    float* bottom_data_d,
    float* rois_d,
    float* top_data_d,
    cudaStream_t stream
    )
{
#if 0
  unique_ptr_device<float> transfrom_matrix_d(nullptr);
  CUDA_CHECK(cudaMalloc((void **) &transfrom_matrix_d, 6 * num_rois * sizeof(float)));
  {
    int thread_num = std::min(num_rois, 1024);
    int block_num = static_cast<int>(std::ceil(num_rois * 1.0 / thread_num));
    compute_all_transform_matrix<float><<<block_num, thread_num, 0, stream>>>(
        transfrom_matrix_d.get(),
        rois_d,
        spatial_scale,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  unique_ptr_device<float> roi_pool_pts_d(nullptr);
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));
  {
    int thread_num = std::min(roi_pool_pt_num, 1024);
    int block_num = static_cast<int>(std::ceil(roi_pool_pt_num * 1.0 / thread_num));
    compute_roi_pool_pts_coalesced<float><<<block_num, thread_num, 0, stream>>>(
        roi_pool_pts_d.get(),
        transfrom_matrix_d.get(),
        roi_pool_pt_num,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif

#if 0
  unique_ptr_device<float> roi_pool_pts_d(nullptr);
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));
  {
    dim3 block(pooled_height, pooled_width);
    dim3 grid(num_rois);
    compute_roi_pool_pts_shared<float><<<grid, block, 0, stream>>>(
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        roi_pool_pt_num,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  unique_ptr_device<float> roi_pool_pts_d(nullptr);
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));
  {
    dim3 block(pooled_height, pooled_width);
    dim3 grid(num_rois);
    compute_roi_pool_pts_local<float><<<grid, block, 0, stream>>>(
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        roi_pool_pt_num,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif

#if 0
  {
    cudaDeviceProp deviceProperties;
    int gpu_id = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, gpu_id));

    // int thread_num_x = std::min(pooled_width * pooled_height, 1024);
    int max_thread_num = 512;
    int thread_num_y = std::min(channels, max_thread_num);
    int thread_num_x = max_thread_num / thread_num_y;
    // int block_num_x = static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x));
    int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), deviceProperties.maxGridSize[0]);
    int block_num_y = static_cast<int>(std::ceil(channels * 1.0 / thread_num_y));
    dim3 block(thread_num_x, thread_num_y);
    dim3 grid(block_num_x, block_num_y);
    compute_weight_local<float><<<grid, block, 0, stream>>>(
        top_data_d,
        bottom_data_d,
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#else
  {
    cudaDeviceProp deviceProperties;
    int gpu_id = 0;
    CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, gpu_id));

    int max_thread_num = 512;
    // int thread_num_x = std::min(max_thread_num / 8, pooled_width);
    // int thread_num_y = std::min(max_thread_num / thread_num_x, channels);
    int thread_num_y = std::min(channels, max_thread_num);
    // int thread_num_x = max_thread_num / thread_num_y;
    int thread_num_x = 1;
    int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), deviceProperties.maxGridSize[0]);
    int block_num_y = static_cast<int>(std::ceil(channels * 1.0 / thread_num_y));
    dim3 block(thread_num_x, thread_num_y);
    dim3 grid(block_num_x, block_num_y);
    size_t shared_mem_size = 8 * thread_num_x * sizeof(float);
    compute_weight<float><<<grid, block, shared_mem_size, stream>>>(
        top_data_d,
        bottom_data_d,
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        num_rois,
        channels,
        height,
        width,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
#endif
}
