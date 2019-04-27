#include "rroi_align.h"

#include <cmath>
#include <algorithm>
#include <cuda.h>

#include "rotate_rect_ops.h"
#include "cuda_utils.h"

template <typename T>
__global__ void compute_transform_matrix(
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

template <typename T>
__device__ inline void get_rotated_bounding_box(
    int& left, int& top,
    int& right, int& bottom,
    const T* pts,
    const int pt_num,
    const int idx,
    const int width, const int height)
{
  left   = int(max(min(min(pts[0*pt_num+idx], pts[2*pt_num+idx]), min(pts[4*pt_num+idx], pts[6*pt_num+idx])), T(0.0)));
  top    = int(max(min(min(pts[1*pt_num+idx], pts[3*pt_num+idx]), min(pts[5*pt_num+idx], pts[7*pt_num+idx])), T(0.0)));
  right  = int(min(max(max(pts[0*pt_num+idx], pts[2*pt_num+idx]), max(pts[4*pt_num+idx], pts[6*pt_num+idx])) + 1, T(width - 1.0)));
  bottom = int(min(max(max(pts[1*pt_num+idx], pts[3*pt_num+idx]), max(pts[5*pt_num+idx], pts[7*pt_num+idx])) + 1, T(height - 1.0)));
}

template <typename T>
__device__ inline T get_rotated_bounding_box_area(
    const float spatial_scale,
    const T roi_height, const T roi_width,
    const int pooled_height, const int pooled_width)
{
  return roi_width * spatial_scale / pooled_width * roi_height * spatial_scale / pooled_height;  // area = w * h
}

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
  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * channels * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    // int pw = i % pooled_width;
    // int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    int left, top, right, bottom;
    get_rotated_bounding_box(left, top, right, bottom, roi_pool_pts, roi_pool_pt_num, n, width, height);

    int roi_batch_ind = (rois + n * 6)[0];  // ROI: batch_ind, xc, yc, w, h, angle
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;
    T rbbox_area = get_rotated_bounding_box_area(spatial_scale, offset_bottom_data[4], offset_bottom_data[3], pooled_height, pooled_width);

    T output_val = 0.0;
    for (int hh = top; hh < bottom+1; ++hh) {
      for (int ww = left; ww < right+1; ++ww) {
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};

        T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts);
        T px_weight = inter_area / rbbox_area;
        output_val += px_weight * offset_bottom_data[hh * width + ww];
      }
    }
    top_data[i] = output_val;
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
  int top_data_size = num_rois * channels * pooled_height * pooled_width * sizeof(float);

  float* transfrom_matrix_d;
  CUDA_CHECK(cudaMalloc(&transfrom_matrix_d, 6 * num_rois * sizeof(float)));
  {
    int grid = std::min(num_rois, 1024);
    int block = static_cast<int>(std::ceil(num_rois * 1.0 / grid));
    compute_transform_matrix<float><<<grid, block, 0, stream>>>(transfrom_matrix_d,
        rois_d,
        spatial_scale,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  float* roi_pool_pts_d;
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
  CUDA_CHECK(cudaMalloc(&roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));
  {
    int grid = std::min(roi_pool_pt_num, 1024);
    int block = static_cast<int>(std::ceil(roi_pool_pt_num * 1.0 / grid));
    compute_roi_pool_pts_coalesced<float><<<grid, block, 0, stream>>>(roi_pool_pts_d,
        transfrom_matrix_d,
        roi_pool_pt_num,
        num_rois,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

  {
    int grid = std::min(top_data_size, 1024);
    int block = static_cast<int>(std::ceil(top_data_size * 1.0 / grid));
    compute_weight<float><<<grid, block, 0, stream>>>(
        top_data_d,
        bottom_data_d,
        roi_pool_pts_d,
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

  CUDA_CHECK(cudaFree(roi_pool_pts_d));
  CUDA_CHECK(cudaFree(transfrom_matrix_d));
}
