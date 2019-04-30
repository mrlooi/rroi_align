#include "rroi_align.h"

#include <iostream>
#include <cmath>
#include <algorithm>
#include <cuda.h>

#include "rotate_rect_ops.h"
#include "cuda_utils.h"

template <typename T>
__device__ void compute_transform_matrix(
    T* __restrict__ matrix,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int pooled_height, const int pooled_width)
{
  T cx = rois[1] * spatial_scale;
  T cy = rois[2] * spatial_scale;
  // Force malformed ROIs to be 1x1
  T w = max(rois[3] * spatial_scale, T(1));
  T h = max(rois[4] * spatial_scale, T(1));
  T angle = deg2rad(rois[5]);

  // TransformPrepare
  T dx = -pooled_width / 2.0;
  T dy = -pooled_height / 2.0;
  T Sx = w / pooled_width;
  T Sy = h / pooled_height;
  T Alpha = cos(angle);
  T Beta = -sin(angle);
  T Dx = cx;
  T Dy = cy;

  matrix[0] = Alpha*Sx;
  matrix[1] = Beta*Sy;
  matrix[2] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
  matrix[3] = -Beta*Sx;
  matrix[4] = Alpha*Sy;
  matrix[5] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;
}

template <typename T>
__global__ void compute_all_transform_matrix(
    T* __restrict__ matrix,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int num_rois,
    const int pooled_height, const int pooled_width)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois; i += blockDim.x * gridDim.x) {
    int step = i * 6;
    T cx = rois[step + 1] * spatial_scale;
    T cy = rois[step + 2] * spatial_scale;
    // Force malformed ROIs to be 1x1
    T w = max(rois[step + 3] * spatial_scale, T(1));
    T h = max(rois[step + 4] * spatial_scale, T(1));
    T angle = deg2rad(rois[step + 5]);

    // TransformPrepare
    T dx = -pooled_width / 2.0;
    T dy = -pooled_height / 2.0;
    T Sx = w / pooled_width;
    T Sy = h / pooled_height;
    T Alpha = cos(angle);
    T Beta = -sin(angle);
    T Dx = cx;
    T Dy = cy;

    matrix[0 * num_rois + i] = Alpha*Sx;
    matrix[1 * num_rois + i] = Beta*Sy;
    matrix[2 * num_rois + i] = Alpha*Sx*dx+Beta*Sy*dy+Dx;
    matrix[3 * num_rois + i] = -Beta*Sx;
    matrix[4 * num_rois + i] = Alpha*Sy;
    matrix[5 * num_rois + i] = -Beta*Sx*dx+Alpha*Sy*dy+Dy;
  }
}

template <typename T>
__global__ void compute_roi_pool_pts_coalesced(
    T* __restrict__ roi_pool_pts,
    const T* __restrict__ matrix,
    const int roi_pool_pt_num,
    const int num_rois,
    const int pooled_height, const int pooled_width)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int n = i / pooled_width / pooled_height;

    // ORDER IN CLOCKWISE OR ANTI-CLOCKWISE
    // (0,1),(0,0),(1,0),(1,1)
    // out_pts[0] = M[0][0]*pw+M[0][1]*(ph+1)+M[0][2];
    // out_pts[1] = M[1][0]*pw+M[1][1]*(ph+1)+M[1][2];
    // out_pts[2] = M[0][0]*pw+M[0][1]*ph+M[0][2];
    // out_pts[3] = M[1][0]*pw+M[1][1]*ph+M[1][2];
    // out_pts[4] = M[0][0]*(pw+1)+M[0][1]*ph+M[0][2];
    // out_pts[5] = M[1][0]*(pw+1)+M[1][1]*ph+M[1][2];
    // out_pts[6] = M[0][0]*(pw+1)+M[0][1]*(ph+1)+M[0][2];
    // out_pts[7] = M[1][0]*(pw+1)+M[1][1]*(ph+1)+M[1][2];
    // TODO: cache matrix in shared mem
    roi_pool_pts[roi_pool_pt_num * 0 + i] = matrix[0 * num_rois + n]*pw     + matrix[1 * num_rois + n]*(ph+1) + matrix[2 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 1 + i] = matrix[3 * num_rois + n]*pw     + matrix[4 * num_rois + n]*(ph+1) + matrix[5 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 2 + i] = matrix[0 * num_rois + n]*pw     + matrix[1 * num_rois + n]*ph     + matrix[2 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 3 + i] = matrix[3 * num_rois + n]*pw     + matrix[4 * num_rois + n]*ph     + matrix[5 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 4 + i] = matrix[0 * num_rois + n]*(pw+1) + matrix[1 * num_rois + n]*ph     + matrix[2 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 5 + i] = matrix[3 * num_rois + n]*(pw+1) + matrix[4 * num_rois + n]*ph     + matrix[5 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 6 + i] = matrix[0 * num_rois + n]*(pw+1) + matrix[1 * num_rois + n]*(ph+1) + matrix[2 * num_rois + n];
    roi_pool_pts[roi_pool_pt_num * 7 + i] = matrix[3 * num_rois + n]*(pw+1) + matrix[4 * num_rois + n]*(ph+1) + matrix[5 * num_rois + n];
  }
}

// shared memory version
template <typename T>
__global__ void compute_roi_pool_pts_shared(
    T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int roi_pool_pt_num,
    const int num_rois,
    const int pooled_height, const int pooled_width)
{
  __shared__ T matrix[6];

  // int idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * (pooled_height * pooled_width) + threadIdx.x * pooled_width + threadIdx.y;
  if (idx >= num_rois * pooled_height * pooled_width) {
    return;
  }

  int pw = threadIdx.y;
  int ph = threadIdx.x;
  int n = blockIdx.x;

  if (threadIdx.x * pooled_width + threadIdx.y == 0) {
    compute_transform_matrix(
        matrix,
        rois + n*6,
        spatial_scale,
        pooled_height,
        pooled_width);
    __threadfence_block();
  }
  __syncthreads();

  // ORDER IN CLOCKWISE OR ANTI-CLOCKWISE
  // (0,1),(0,0),(1,0),(1,1)
  roi_pool_pts[roi_pool_pt_num * 0 + idx] = matrix[0]*pw     + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 1 + idx] = matrix[3]*pw     + matrix[4]*(ph+1) + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 2 + idx] = matrix[0]*pw     + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 3 + idx] = matrix[3]*pw     + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 4 + idx] = matrix[0]*(pw+1) + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 5 + idx] = matrix[3]*(pw+1) + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 6 + idx] = matrix[0]*(pw+1) + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 7 + idx] = matrix[3]*(pw+1) + matrix[4]*(ph+1) + matrix[5];
}

// local memory version
template <typename T>
__global__ void compute_roi_pool_pts_local(
    T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int roi_pool_pt_num,
    const int num_rois,
    const int pooled_height, const int pooled_width)
{
  T matrix[6];

  // int idx = blockIdx.x * (blockDim.x * blockDim.y) + threadIdx.x * blockDim.y + threadIdx.y;
  int idx = blockIdx.x * (pooled_height * pooled_width) + threadIdx.x * pooled_width + threadIdx.y;
  if (idx >= num_rois * pooled_height * pooled_width) {
    return;
  }

  int pw = threadIdx.y;
  int ph = threadIdx.x;
  int n = blockIdx.x;

  compute_transform_matrix(
      matrix,
      rois + n*6,
      spatial_scale,
      pooled_height,
      pooled_width);

  // ORDER IN CLOCKWISE OR ANTI-CLOCKWISE
  // (0,1),(0,0),(1,0),(1,1)
  roi_pool_pts[roi_pool_pt_num * 0 + idx] = matrix[0]*pw     + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 1 + idx] = matrix[3]*pw     + matrix[4]*(ph+1) + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 2 + idx] = matrix[0]*pw     + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 3 + idx] = matrix[3]*pw     + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 4 + idx] = matrix[0]*(pw+1) + matrix[1]*ph     + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 5 + idx] = matrix[3]*(pw+1) + matrix[4]*ph     + matrix[5];
  roi_pool_pts[roi_pool_pt_num * 6 + idx] = matrix[0]*(pw+1) + matrix[1]*(ph+1) + matrix[2];
  roi_pool_pts[roi_pool_pt_num * 7 + idx] = matrix[3]*(pw+1) + matrix[4]*(ph+1) + matrix[5];
}

template <typename T>
__device__ inline void get_rotated_bounding_box_interleaved(
    int& left, int& top,
    int& right, int& bottom,
    const T* pts,
    const int pt_num,
    const int idx,
    const int width, const int height)
{
  left   = int(max(min(min(pts[0*pt_num+idx], pts[2*pt_num+idx]), min(pts[4*pt_num+idx], pts[6*pt_num+idx])), T(0)));
  top    = int(max(min(min(pts[1*pt_num+idx], pts[3*pt_num+idx]), min(pts[5*pt_num+idx], pts[7*pt_num+idx])), T(0)));
  right  = int(min(max(max(pts[0*pt_num+idx], pts[2*pt_num+idx]), max(pts[4*pt_num+idx], pts[6*pt_num+idx])) + 1, T(width - 1)));
  bottom = int(min(max(max(pts[1*pt_num+idx], pts[3*pt_num+idx]), max(pts[5*pt_num+idx], pts[7*pt_num+idx])) + 1, T(height - 1)));
}

template <typename T>
__device__ inline void get_rotated_bounding_box(
    int& left, int& top,
    int& right, int& bottom,
    const T* pts,
    const int width, const int height)
{
  left   = int(max(min(min(pts[0], pts[2]), min(pts[4], pts[6])), T(0.0)));
  top    = int(max(min(min(pts[1], pts[3]), min(pts[5], pts[7])), T(0.0)));
  right  = int(min(max(max(pts[0], pts[2]), max(pts[4], pts[6])) + 1, T(width - 1.0)));
  bottom = int(min(max(max(pts[1], pts[3]), max(pts[5], pts[7])) + 1, T(height - 1.0)));
}

template <typename T>
__device__ inline T get_rotated_bounding_box_area(
    const float spatial_scale,
    const T roi_height, const T roi_width,
    const int pooled_height, const int pooled_width)
{
  return roi_width * spatial_scale / pooled_width * roi_height * spatial_scale / pooled_height;  // area = w * h
}

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
          T pixel_rect_vertices[8] = {ww+0.0f, hh+0.0f, ww+1.0f, hh+0.0f, ww+1.0f, hh+1.0f, ww+0.0f, hh+1.0f};
          T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts_local);
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
