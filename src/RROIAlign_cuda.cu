#include <cstdio>
#include <algorithm>
#include <cuda.h>

#include "RROIAlign_cuda.h"
#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ inline void get_rotated_rect_bounding_box(const T* pts, int& leftMost, int& topMost,
  int& rightMost, int& bottomMost, const int width, const int height)
{
  leftMost = int(max(min(min(pts[0], pts[2]), min(pts[4], pts[6])), 0.0));
  topMost = int(max(min(min(pts[1], pts[3]), min(pts[5], pts[7])), 0.0));
  rightMost = int(min(max(max(pts[0], pts[2]), max(pts[4], pts[6])) + 1, width - 1.0));
  bottomMost = int(min(max(max(pts[1], pts[3]), max(pts[5], pts[7])) + 1, height - 1.0));
}

template <typename T>
__global__ void RRoIAlignFForward(const int nthreads, const T* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);
    T P_area = offset_bottom_rois[3] * spatial_scale / pooled_width * offset_bottom_rois[4] * spatial_scale / pooled_height;  // area = w * h

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    T output_val = 0.0;
    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        // T pixel_rect[5] = {ww+0.5f,hh+0.5f,1,1,0};
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};

        T inter_area = computeRectInterArea(pixel_rect_vertices, P);
        T px_weight = inter_area / P_area;
        output_val += px_weight * offset_bottom_data[hh * width + ww];
      }
    }
    top_data[index] = output_val;
  }
}

template <typename T>
__global__ void RRoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const float spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    T* bottom_diff,
    const T* bottom_rois)
{
  CUDA_1D_KERNEL_LOOP(index, nthreads)
  {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);
    T P_area = offset_bottom_rois[3] * spatial_scale / pooled_width * offset_bottom_rois[4] * spatial_scale / pooled_height;  // area = w * h

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        // T pixel_rect[5] = {ww+0.5f,hh+0.5f,1,1,0};
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};

        T inter_area = computeRectInterArea(pixel_rect_vertices, P);
        T px_weight = inter_area / P_area;
        atomicAdd(offset_bottom_diff + hh * width + ww, px_weight * top_diff_this_bin);
      }
    }

  } // CUDA_1D_KERNEL_LOOP
} // RRoIAlignBackward

void RROIAlign_forward_golden(
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
  auto top_data_size = num_rois * channels * pooled_height * pooled_width * sizeof(float);
  dim3 grid(std::min(static_cast<long>(std::ceil(top_data_size * 1.0 / 512L)), 4096L));
  dim3 block(512);
  RRoIAlignFForward<float><<<grid, block, 0, stream>>>(
      top_data_size,
      bottom_data_d,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      rois_d,
      top_data_d
      );
}
