#include "rroi.h"

#include <cuda.h>
#include <stdio.h>

#include "rotate_rect_ops.h"
#include "cuda_utils.h"


template <typename T>
__device__ T bilinear_interpolate(const T* bottom_data,
    const int height, const int width,
    T y, T x,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    return 0;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  int y_low = (int) y;
  int x_low = (int) x;
  int y_high;
  int x_high;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;
  // do bilinear interpolation
  T v1 = bottom_data[y_low * width + x_low];
  T v2 = bottom_data[y_low * width + x_high];
  T v3 = bottom_data[y_high * width + x_low];
  T v4 = bottom_data[y_high * width + x_high];
  T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  return val;
}



template <typename T>
__device__ void bilinear_interpolate_gradient(
    const int height, const int width,
    T y, T x,
    T & w1, T & w2, T & w3, T & w4,
    int & x_low, int & x_high, int & y_low, int & y_high,
    const int index /* index for debug only*/) {

  // deal with cases that inverse elements are out of feature map boundary
  if (y < -1.0 || y > height || x < -1.0 || x > width) {
    //empty
    w1 = w2 = w3 = w4 = 0.;
    x_low = x_high = y_low = y_high = -1;
    return;
  }

  if (y <= 0) y = 0;
  if (x <= 0) x = 0;

  y_low = (int) y;
  x_low = (int) x;

  if (y_low >= height - 1) {
    y_high = y_low = height - 1;
    y = (T) y_low;
  } else {
    y_high = y_low + 1;
  }

  if (x_low >= width - 1) {
    x_high = x_low = width - 1;
    x = (T) x_low;
  } else {
    x_high = x_low + 1;
  }

  T ly = y - y_low;
  T lx = x - x_low;
  T hy = 1. - ly, hx = 1. - lx;

  // reference in forward
  // T v1 = bottom_data[y_low * width + x_low];
  // T v2 = bottom_data[y_low * width + x_high];
  // T v3 = bottom_data[y_high * width + x_low];
  // T v4 = bottom_data[y_high * width + x_high];
  // T val = (w1 * v1 + w2 * v2 + w3 * v3 + w4 * v4);

  w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

  return;
}


template <typename T>
__global__ void RRoIAlignForward(const int nthreads, const T* bottom_data,
    const T spatial_scale, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois, T* top_data) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;


    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    // Force malformed ROIs to be 1x1
    T roi_width = max(offset_bottom_rois[3] * spatial_scale, (T)1.);
    T roi_height = max(offset_bottom_rois[4] * spatial_scale, (T)1.);

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T mw = 1.0 / roi_bin_grid_w;
    const T mh = 1.0 / roi_bin_grid_h;

    // compute pool points
    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    // compute line params
    T line_params[4];
    for (int i = 0; i < 2; ++i)
    {
        line_params[i * 2] = P[((i + 1) * 2) % 8] - P[i * 2];
        line_params[i * 2 + 1] = P[((i + 1) * 2) % 8 + 1] - P[i * 2 + 1];
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    T output_val = 0.;
    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = P[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
        const T y = P[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

        T val = bilinear_interpolate(offset_bottom_data, height, width, y, x, index);
        output_val += val;
//        printf("%.2f\n", val);
      }
    }

    output_val /= count;

    top_data[index] = output_val;
  }
}


template <typename T>
__global__ void RRoIAlignBackwardFeature(const int nthreads, const T* top_diff,
    const int num_rois, const T spatial_scale,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width,
    const int sampling_ratio,
    T* bottom_diff,
    const T* bottom_rois) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    // Force malformed ROIs to be 1x1
    T roi_width = max(offset_bottom_rois[3] * spatial_scale, (T)1.);
    T roi_height = max(offset_bottom_rois[4] * spatial_scale, (T)1.);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    const T mw = 1.0 / roi_bin_grid_w;
    const T mh = 1.0 / roi_bin_grid_h;

    T* offset_bottom_diff = bottom_diff + (roi_batch_ind * channels + c) * height * width;

    int top_offset    = (n * channels + c) * pooled_height * pooled_width;
    const T* offset_top_diff = top_diff + top_offset;
    const T top_diff_this_bin = offset_top_diff[ph * pooled_width + pw];

    // compute pool points
    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    // compute line params
    T line_params[4];
    for (int i = 0; i < 2; ++i)
    {
        line_params[i * 2] = P[((i + 1) * 2) % 8] - P[i * 2];
        line_params[i * 2 + 1] = P[((i + 1) * 2) % 8 + 1] - P[i * 2 + 1];
    }

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
    {
      for (int ix = 0; ix < roi_bin_grid_w; ix ++)
      {
        const T x = P[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
        const T y = P[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

        T w1, w2, w3, w4;
        int x_low, x_high, y_low, y_high;

        bilinear_interpolate_gradient(height, width, y, x,
            w1, w2, w3, w4,
            x_low, x_high, y_low, y_high,
            index);

        T g1 = top_diff_this_bin * w1 / count;
        T g2 = top_diff_this_bin * w2 / count;
        T g3 = top_diff_this_bin * w3 / count;
        T g4 = top_diff_this_bin * w4 / count;

        if (x_low >= 0 && x_high >= 0 && y_low >= 0 && y_high >= 0)
        {
          atomicAdd(offset_bottom_diff + y_low * width + x_low, static_cast<T>(g1));
          atomicAdd(offset_bottom_diff + y_low * width + x_high, static_cast<T>(g2));
          atomicAdd(offset_bottom_diff + y_high * width + x_low, static_cast<T>(g3));
          atomicAdd(offset_bottom_diff + y_high * width + x_high, static_cast<T>(g4));
        } // if
      } // ix
    } // iy
  } // CUDA_1D_KERNEL_LOOP
} // RoIAlignBackward




void vincent_rroi_align(
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
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  dim3 grid(std::min(static_cast<long>(std::ceil(top_data_size * 1.0 / 512L)), 4096L));
  dim3 block(512);
  int sampling_ratio = 0; // default
  RRoIAlignForward<float><<<grid, block, 0, stream>>>(
      top_data_size,
      bottom_data_d,
      spatial_scale,
      channels,
      height,
      width,
      pooled_height,
      pooled_width,
      sampling_ratio, // NEW
      rois_d,
      top_data_d
      );
}

void vincent_rroi_align_backward(
    int batch_size,
    int num_rois,
    int channels,
    int height,
    int width,
    int pooled_height,
    int pooled_width,
    float spatial_scale,
    const float* top_diff_d,
    const float* rois_d,
    float* bottom_diff_d,
    cudaStream_t stream
    )
{
  
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  dim3 grid(std::min(static_cast<long>(std::ceil(top_data_size * 1.0 / 512L)), 4096L));
  dim3 block(512);
  int sampling_ratio = 0; // default
  RRoIAlignBackwardFeature<float><<<grid, block, 0, stream>>>(
         top_data_size,
         top_diff_d,
         num_rois,
         spatial_scale,
         channels,
         height,
         width,
         pooled_height,
         pooled_width,
         sampling_ratio,
         bottom_diff_d,
         rois_d
    );

}
