#include <cmath>
#include <algorithm>

#include <cuda_runtime.h>

#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>

#include <THC/THC.h>
#include <THC/THCAtomics.cuh>
#include <THC/THCDeviceUtils.cuh>

#include "rotate_rect_ops.h"


#define CUDA_CHECK(call) { \
  cudaError_t err; \
  if ((err = (call)) != cudaSuccess) { \
    fprintf(stderr, "Got error %s at %s:%d\n", cudaGetErrorString(err), \
        __FILE__, __LINE__); \
    exit(1); \
  } \
}

template <typename T>
__device__ inline T deg2rad(const T& deg)
{
    return deg / 180.0 * 3.1415926535;
}

template <typename T>
__global__ void compute_transform(
    T* __restrict__ matrix,
    const T* __restrict__ rois,
    const int num_rois,
    const float spatial_scale,
    const int pooled_height, const int pooled_width)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois; i += blockDim.x * gridDim.x) {
    int step = i * 6;
    T cx = rois[step + 1] * spatial_scale;
    T cy = rois[step + 2] * spatial_scale;
    // Force malformed ROIs to be 1x1
    T w = std::max(rois[step + 3] * spatial_scale, T(1));
    T h = std::max(rois[step + 4] * spatial_scale, T(1));
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
    int& leftMost, int& topMost,
    int& rightMost, int& bottomMost,
    const T* pts,
    const int pt_num,
    const int idx,
    const int width, const int height)
{
  leftMost =   int(std::max(std::min(std::min(pts[0*pt_num+idx], pts[2*pt_num+idx]), std::min(pts[4*pt_num+idx], pts[6*pt_num+idx])), 0.0));
  topMost =    int(std::max(std::min(std::min(pts[1*pt_num+idx], pts[3*pt_num+idx]), std::min(pts[5*pt_num+idx], pts[7*pt_num+idx])), 0.0));
  rightMost =  int(std::min(std::max(std::max(pts[0*pt_num+idx], pts[2*pt_num+idx]), std::max(pts[4*pt_num+idx], pts[6*pt_num+idx])) + 1, width - 1.0));
  bottomMost = int(std::min(std::max(std::max(pts[1*pt_num+idx], pts[3*pt_num+idx]), std::max(pts[5*pt_num+idx], pts[7*pt_num+idx])) + 1, height - 1.0));
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
    T* top_data,
    const T* bottom_data,
    const T* roi_pool_pts,
    const int num_rois,
    const int channels, const int height, const int width,
    const int pooled_height, const int pooled_width)
{
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_rois * channels * pooled_height * pooled_width; i += blockDim.x * gridDim.x) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = i % pooled_width;
    int ph = (i / pooled_width) % pooled_height;
    int c = (i / pooled_width / pooled_height) % channels;
    int n = i / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;  // batch_ind, xc, yc, w, h, angle
    int roi_batch_ind = offset_bottom_rois[0];

    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_bounding_box(leftMost, topMost, rightMost, bottomMost, roi_pool_pts, num_rois*pooled_height*pooled_width, n, width, height);
    T rbbox_area = get_rotated_bounding_box_area(spatial_scale, offset_bottom_data[4], offset_bottom_data[3], pooled_height, pooled_width);

    T output_val = 0.0;
    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        T pixel_rect_vertices[8] = {ww+0.0f,hh+0.0f,ww+1.0f,hh+0.0f,ww+1.0f,hh+1.0f,ww+0.0f,hh+1.0f};

        T inter_area = computeRectInterArea(pixel_rect_vertices, roi_pool_pts);
        T px_weight = inter_area / rbbox_area;
        output_val += px_weight * offset_bottom_data[hh * width + ww];
      }
    }
    top_data[i] = output_val;
  }
}

at::Tensor RROIAlign_forward_cuda(const at::Tensor& input,
                                const at::Tensor& rois,
                                const float spatial_scale,
                                const int pooled_height,
                                const int pooled_width)
{
  AT_ASSERTM(input.type().is_cuda(), "input must be a CUDA tensor");
  AT_ASSERTM(rois.type().is_cuda(), "rois must be a CUDA tensor");

  auto num_rois = rois.size(0);
  auto channels = input.size(1);
  auto height = input.size(2);
  auto width = input.size(3);

  auto output = at::empty({num_rois, channels, pooled_height, pooled_width}, input.options());
  auto output_size = num_rois * pooled_height * pooled_width * channels;
//  auto argmax = at::zeros({num_rois, channels, pooled_height, pooled_width}, input.options().dtype(at::kInt));

  cudaStream_t stream = at::cuda::getCurrentCUDAStream();

  scalar_t* transfrom_matrix_d;
  CUDA_CHECK(cudaMalloc(static_cast<void **>(&transfrom_matrix_d), 6*num_rois*sizeof(scalar_t)));
  {
    int grid = std::min(num_rois, 1024);
    int block = static_cast<int>(std::ceil(num_rois * 1.0 / grid));
    compute_transform<scalar_t><<<grid, block, 0, stream>>>(transfrom_matrix_d,
        rois.contiguous().data<scalar_t>(),
        num_rois,
        spatial_scale,
        pooled_height,
        pooled_width);
  }

  scalar_t* roi_pool_pts_d;
  CUDA_CHECK(cudaMalloc(static_cast<void **>(&roi_pool_pts_d), 6*output_size*sizeof(scalar_t)));
  {
    int roi_pool_pt_num = num_rois * pooled_height * pooled_width;
    int grid = std::min(roi_pool_pt_num, 1024);
    int block = static_cast<int>(std::ceil(roi_pool_pt_num * 1.0 / grid));
    compute_roi_pool_pts_coalesced<scalar_t><<<grid, block, 0, stream>>>(roi_pool_pts_d,
        transfrom_matrix_d,
        num_rois,
        pooled_height,
        pooled_width);
  }

  {
    int grid = std::min(output_size, 1024);
    int block = static_cast<int>(std::ceil(output_size * 1.0 / grid));
    compute_weight<scalar_t><<<grid, block, 0, stream>>>(
         output.data<scalar_t>(),
         input.contiguous().data<scalar_t>(),
         roi_pool_pts_d,
         num_rois,
         channels,
         height,
         width,
         pooled_height,
         pooled_width);
  }

  CUDA_CHECK(cudaFree(roi_pool_pts_d));
  CUDA_CHECK(cudaFree(transfrom_matrix_d));

  THCudaCheck(cudaGetLastError());

  return output;
}
