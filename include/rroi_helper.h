#ifndef RROI_HELPER_H
#define RROI_HELPER_H

#include <cuda.h>
#include "rotate_rect_ops.h"

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

#endif /* RROI_HELPER_H */
