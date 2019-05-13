#include "rroi.h"

#include <cuda.h>

#include "rroi_helper.h"
#include "rotate_rect_ops.h"
#include "cuda_utils.h"

const int TILE_DIM = 32;

#if 0
// Traspose matrix bottom_data
template <typename T>
__global__ void compute_bottom_data_coalesced(
    T* __restrict__ bottom_data_coalesced,
    const T* __restrict__ bottom_data,
    const int batch_size,
    const int channels,
    const int height,
    const int width
    )
{
  const int num_columns = height * width;
  const int num_rows = batch_size * channels;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row;
  int column;

  do {
    row = by * TILE_DIM + threadIdx.y;
    do {
      column = bx * TILE_DIM + threadIdx.x;

      if (row < num_rows && column < num_columns) {
        bottom_data_coalesced[column * num_rows + row] = bottom_data[row * num_columns + column];
      }

      bx += gridDim.x;
    } while (column < num_columns);
    by += gridDim.y;
  } while (row < num_rows);
}
#endif

#if 0
// Traspose matrix bottom_data using shared memory
// General case
template <typename T>
__global__ void compute_bottom_data_coalesced(
    T* __restrict__ bottom_data_coalesced,
    const T* __restrict__ bottom_data,
    const int batch_size,
    const int channels,
    const int height,
    const int width
    )
{
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  const int num_columns = height * width;
  const int num_rows = batch_size * channels;

  int bx = blockIdx.x;
  int by = blockIdx.y;
  int row;
  int column;

  do {
    row = by * TILE_DIM + threadIdx.y;
    do {
      column = bx * TILE_DIM + threadIdx.x;

      if (row < num_rows && column < num_columns) {
        tile[threadIdx.y][threadIdx.x] = bottom_data[row * num_columns + column];
      }
      __syncthreads();

      int transpose_column = by * TILE_DIM + threadIdx.x;
      int transpose_row = bx * TILE_DIM + threadIdx.y;
      if (transpose_row < num_columns && transpose_column < num_rows) {
        bottom_data_coalesced[transpose_row * num_rows + transpose_column] = tile[threadIdx.x][threadIdx.y];
      }

      bx += gridDim.x;
    } while (column < num_columns);
    by += gridDim.y;
  } while (row < num_rows);
}
#endif

#if 1
// Traspose matrix bottom_data using shared memory
template <typename T>
__global__ void compute_bottom_data_coalesced(
    T* __restrict__ bottom_data_coalesced,
    const T* __restrict__ bottom_data,
    const int batch_size,
    const int channels,
    const int height,
    const int width
    )
{
  __shared__ T tile[TILE_DIM][TILE_DIM+1];

  const int num_columns = height * width;
  const int num_rows = batch_size * channels;

  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int column = blockIdx.x * TILE_DIM + threadIdx.x;

  if (row < num_rows && column < num_columns) {
    tile[threadIdx.y][threadIdx.x] = bottom_data[row * num_columns + column];
  }
  __syncthreads();

  int transpose_column = blockIdx.y * TILE_DIM + threadIdx.x;
  int transpose_row = blockIdx.x * TILE_DIM + threadIdx.y;
  if (transpose_row < num_columns && transpose_column < num_rows) {
    bottom_data_coalesced[transpose_row * num_rows + transpose_column] = tile[threadIdx.x][threadIdx.y];
  }
}
#endif

template <typename T>
__global__ void bp_rroi_align_kernel(
    T* __restrict__ top_data,
    const T* __restrict__ bottom_data_coalesced,
    const T* __restrict__ roi_pool_pts,
    const T* __restrict__ rois,
    const float spatial_scale,
    const int sampling_ratio,
    const int num_rois,
    const int batch_size, const int channels,
    const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  __shared__ T roi_pool_pts_shared[8];
  __shared__ T line_params[4];
  __shared__ T rois_shared[6];

  const int roi_pool_pt_num = num_rois * pooled_height * pooled_width;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < roi_pool_pt_num; i += blockDim.x * gridDim.x) {
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (c < channels) {
      int pw = i % pooled_width;
      int ph = (i / pooled_width) % pooled_height;
      int n = i / pooled_width / pooled_height;

      const T* rois_offset = rois + n * 6;  // batch_ind, xc, yc, w, h, angle
      if (threadIdx.y < 6) {
        rois_shared[threadIdx.y] = rois_offset[threadIdx.y];
      }

      int roi_pool_idx = n * pooled_height * pooled_width + ph * pooled_width + pw;
      int roi_pool_idx_shared = threadIdx.y;
      if (roi_pool_idx_shared < 8) {
        roi_pool_pts_shared[roi_pool_idx_shared] = roi_pool_pts[roi_pool_idx_shared * roi_pool_pt_num + roi_pool_idx];
      }
      __syncthreads();

      // compute line params
#if 0
      T line_params[4];
      for (int i = 0; i < 2; ++i)
      {
        line_params[i * 2] = roi_pool_pts_shared[((i + 1) * 2) % 8] - roi_pool_pts_shared[i * 2];
        line_params[i * 2 + 1] = roi_pool_pts_shared[((i + 1) * 2) % 8 + 1] - roi_pool_pts_shared[i * 2 + 1];
      }
#endif
      // if (roi_pool_idx_shared < 4) {
      //   line_params[roi_pool_idx_shared] = roi_pool_pts_shared[((roi_pool_idx_shared / 2) + 1) * 2 % 8 + roi_pool_idx_shared % 2] - roi_pool_pts_shared[roi_pool_idx_shared];
      // }
      if (roi_pool_idx_shared < 2) {
        line_params[roi_pool_idx_shared * 2] = roi_pool_pts_shared[((roi_pool_idx_shared + 1) * 2) % 8] - roi_pool_pts_shared[roi_pool_idx_shared * 2];
        line_params[roi_pool_idx_shared * 2 + 1] = roi_pool_pts_shared[((roi_pool_idx_shared + 1) * 2) % 8 + 1] - roi_pool_pts_shared[roi_pool_idx_shared * 2 + 1];
      }
      __syncthreads();

      int roi_batch_id = rois_shared[0];

      // Force malformed ROIs to be 1x1
      T roi_width = max(rois_shared[3] * spatial_scale, (T)1.);
      T roi_height = max(rois_shared[4] * spatial_scale, (T)1.);
      // We use roi_bin_grid to sample the grid and mimic integral
      int roi_bin_grid_h = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_height / pooled_height); // e.g., = 2
      int roi_bin_grid_w = (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

      const T mw = 1.0 / roi_bin_grid_w;
      const T mh = 1.0 / roi_bin_grid_h;

      // We do average (integral) pooling inside a bin
      const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

      T output_val = 0.;
      for (int iy = 0; iy < roi_bin_grid_h; iy ++) // e.g., iy = 0, 1
      {
        for (int ix = 0; ix < roi_bin_grid_w; ix ++)
        {
          const T x = roi_pool_pts_shared[0] + static_cast<T>(iy + 0.5) * line_params[0] * mh + static_cast<T>(ix + 0.5) * line_params[2] * mw;
          const T y = roi_pool_pts_shared[1] + static_cast<T>(iy + 0.5) * line_params[1] * mh + static_cast<T>(ix + 0.5) * line_params[3] * mw;

          T val = bilinear_interpolate_coalesced(bottom_data_coalesced, roi_batch_id, c, batch_size, channels, height, width, y, x);
          output_val += val;
        }
      }
      output_val /= count;

      int top_data_idx = (n * channels + c) * pooled_width * pooled_height + ph * pooled_width + pw;
      top_data[top_data_idx] = output_val;
    }
  }
}

// binlinear interpolation version of RROI align
void bp_rroi_align(
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

  unique_ptr_device<float> bottom_data_coalesced_d(nullptr);
  auto bottom_data_size = batch_size * channels * height * width;
  CUDA_CHECK(cudaMalloc((void **) &bottom_data_coalesced_d, bottom_data_size * sizeof(float)));

  {
    int block_x = TILE_DIM;
    int block_y = TILE_DIM;
    int grid_x = static_cast<int>(std::ceil(height * width * 1.0 / block_x));
    int grid_y = static_cast<int>(std::ceil(batch_size * channels * 1.0 / block_y));
    dim3 block(block_x, block_y);
    dim3 grid(grid_x, grid_y);
    compute_bottom_data_coalesced<float><<<grid, block, 0, stream>>>(
        bottom_data_coalesced_d.get(),
        bottom_data_d,
        batch_size,
        channels,
        height,
        width
        );
  }

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
  }

  CUDA_CHECK(cudaDeviceSynchronize());

  {
    // cudaDeviceProp deviceProperties;
    // int gpu_id = 0;
    // CUDA_CHECK(cudaGetDeviceProperties(&deviceProperties, gpu_id));

    int max_thread_num = 256;
    // int thread_num_x = std::min(max_thread_num / 8, pooled_width);
    // int thread_num_y = std::min(max_thread_num / thread_num_x, channels);
    int thread_num_y = std::min(channels, max_thread_num);
    // int thread_num_x = max_thread_num / thread_num_y;
    int thread_num_x = 1;
    // int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), deviceProperties.maxGridSize[0]);
    int block_num_x = std::min(static_cast<int>(std::ceil(pooled_width * pooled_height * num_rois * 1.0 / thread_num_x)), 65535);
    int block_num_y = static_cast<int>(std::ceil(channels * 1.0 / thread_num_y));
    dim3 block(thread_num_x, thread_num_y);
    dim3 grid(block_num_x, block_num_y);
    int sampling_ratio = 0; // default
    bp_rroi_align_kernel<float><<<grid, block, 0, stream>>>(
        top_data_d,
        bottom_data_coalesced_d.get(),
        roi_pool_pts_d.get(),
        rois_d,
        spatial_scale,
        sampling_ratio,
        num_rois,
        batch_size,
        channels,
        height,
        width,
        pooled_height,
        pooled_width);
    CUDA_CHECK(cudaDeviceSynchronize());
  }
}
