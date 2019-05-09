#include "rroi.h"

#include <cuda.h>

#include "rroi_helper.h"
#include "cuda_utils.h"

// NOTE: only cache one roi_pool_pt in the shared memory
template <typename T>
__global__ void pool(
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

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < roi_pool_pt_num; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      int roi_batch_ind = rois_offset[0];
      T rbox_area = get_rotated_bounding_box_area(spatial_scale, rois_offset[4], rois_offset[3], pooled_height, pooled_width);

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      int roi_pool_idx_shared = threadIdx.y;
      if (roi_pool_idx_shared < 8) {
        roi_pool_pts_shared[roi_pool_idx_shared] = roi_pool_pts[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      int left, top, right, bottom;
      get_rotated_bounding_box(left, top, right, bottom, roi_pool_pts_shared, width, height);

      T* P = roi_pool_pts_shared;
      T AB[2];
      AB[0] = P[0] - P[2];
      AB[1] = P[1] - P[3];
      T ABAB = AB[0]*AB[0] +AB[1]*AB[1];
      T AC[2];
      AC[0] = P[4] - P[2];
      AC[1] = P[5] - P[3];
      T ACAC = AC[0]*AC[0] + AC[1]*AC[1];

      const T* bottom_data_offset = bottom_data + (roi_batch_ind * channels + c) * height * width;
      T maxval = 0;
      // int maxidx = -1;

      for (int hh = top; hh < bottom+1; ++hh) {
        for (int ww = left; ww < right+1; ++ww) {
          T AP[2];
          AP[0] = ww - P[2];
          AP[1] = hh - P[3];
          T ABAP = AB[0]*AP[0] + AB[1]*AP[1];
          T ACAP = AC[0]*AP[0] + AC[1]*AP[1];
          if (ABAP >= 1e-3 && (ABAB - ABAP) > -1e-3 && ACAP >= 1e-3 && (ACAC - ACAP) > -1e-3) {
            int bottom_index = hh * width + ww;
            if (bottom_data_offset[bottom_index] > maxval) {
              maxval = bottom_data_offset[bottom_index];
              // maxidx = bottom_index;
            }
          }
        }
      }

      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = maxval;
    }
  }
}

// NOTE: only cache one roi_pool_pt in the shared memory
template <typename T>
__global__ void pool(
    T* __restrict__ top_data,
    const T* __restrict__ bottom_data,
    const T* __restrict__ roi_pool_pts,
    int* __restrict__ roi_pool_region,
    T* __restrict__ roi_pool_tmp,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  __shared__ T roi_pool_pts_shared[8];
  __shared__ int roi_pool_region_shared[4];
  __shared__ T roi_pool_tmp_shared[6];

  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < roi_pool_pt_num; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      int roi_batch_ind = rois_offset[0];
      T rbox_area = get_rotated_bounding_box_area(spatial_scale, rois_offset[4], rois_offset[3], pooled_height, pooled_width);

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      int roi_pool_idx_shared = threadIdx.y;
      if (roi_pool_idx_shared < 8) {
        roi_pool_pts_shared[roi_pool_idx_shared] = roi_pool_pts[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      if (roi_pool_idx_shared < 4) {
        roi_pool_region_shared[roi_pool_idx_shared] = roi_pool_region[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      if (roi_pool_idx_shared < 6) {
        roi_pool_tmp_shared[roi_pool_idx_shared] = roi_pool_tmp[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      const T* bottom_data_offset = bottom_data + (roi_batch_ind * channels + c) * height * width;
      T maxval = 0;
      // int maxidx = -1;

      // for (int hh = top; hh < bottom+1; ++hh) {
      for (int hh = roi_pool_region_shared[1]; hh < roi_pool_region_shared[3]+1; ++hh) {
        // for (int ww = left; ww < right+1; ++ww) {
        for (int ww = roi_pool_region_shared[0]; ww < roi_pool_region_shared[2]+1; ++ww) {
          T AP[2];
          AP[0] = ww - roi_pool_pts_shared[2];
          AP[1] = hh - roi_pool_pts_shared[3];
          // T ABAP = AB[0]*AP[0] + AB[1]*AP[1];
          T ABAP = roi_pool_tmp_shared[0]*AP[0] + roi_pool_tmp_shared[1]*AP[1];
          // T ACAP = AC[0]*AP[0] + AC[1]*AP[1];
          T ACAP = roi_pool_tmp_shared[3]*AP[0] + roi_pool_tmp_shared[4]*AP[1];
          // if (ABAP >= 1e-3 && (ABAB - ABAP) > -1e-3 && ACAP >= 1e-3 && (ACAC - ACAP) > -1e-3) {
          if (ABAP >= 1e-3 && (roi_pool_tmp_shared[2] - ABAP) > -1e-3 && ACAP >= 1e-3 && (roi_pool_tmp_shared[5] - ACAP) > -1e-3) {
            int bottom_index = hh * width + ww;
            if (bottom_data_offset[bottom_index] > maxval) {
              maxval = bottom_data_offset[bottom_index];
              // maxidx = bottom_index;
            }
          }
        }
      }

      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = maxval;
    }
  }
}

#if 0
void RROIPool_forward(
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
    pool<float><<<grid, block, shared_mem_size, stream>>>(
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
}
#else
void RROIPool_forward(
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
  int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  unique_ptr_device<float> roi_pool_pts_d(nullptr);
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_pts_d, 8 * roi_pool_pt_num * sizeof(float)));

  unique_ptr_device<int> roi_pool_region_d(nullptr);
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_region_d, 4 * roi_pool_pt_num * sizeof(int)));

  unique_ptr_device<float> roi_pool_tmp_d(nullptr);
  CUDA_CHECK(cudaMalloc((void **) &roi_pool_tmp_d, 6 * roi_pool_pt_num * sizeof(float)));

  {
    dim3 block(pooled_height, pooled_width);
    dim3 grid(num_rois);
    compute_rroi_pool_temp<float><<<grid, block, 0, stream>>>(
        roi_pool_pts_d.get(),
        roi_pool_region_d.get(),
        roi_pool_tmp_d.get(),
        rois_d,
        spatial_scale,
        roi_pool_pt_num,
        num_rois,
        height,
        width,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }

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
    pool<float><<<grid, block, shared_mem_size, stream>>>(
        top_data_d,
        bottom_data_d,
        roi_pool_pts_d.get(),
        roi_pool_region_d.get(),
        roi_pool_tmp_d.get(),
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
}
#endif
