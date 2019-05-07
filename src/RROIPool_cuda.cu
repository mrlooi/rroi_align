#include <cstdio>
#include <algorithm>
#include <cuda.h>

#include "RROIPool_cuda.h"
#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


template <typename T>
__device__ inline void get_rotated_rect_bounding_box(const T* pts, int& leftMost, int& topMost,
  int& rightMost, int& bottomMost, const int width, const int height)
{
//  const T* P = pts;
//  leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
//  rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),width-1.0));
//  topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
//  bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),height-1.0));

  leftMost = int(max(min(min(pts[0], pts[2]), min(pts[4], pts[6])), 0.0));
  topMost = int(max(min(min(pts[1], pts[3]), min(pts[5], pts[7])), 0.0));
  rightMost = int(min(max(max(pts[0], pts[2]), max(pts[4], pts[6])) + 1, width - 1.0));
  bottomMost = int(min(max(max(pts[1], pts[3]), max(pts[5], pts[7])) + 1, height - 1.0));
}


template <typename T>
__global__ void RRoIPoolFForward(const int nthreads, const T* bottom_data,
    const float spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const T* bottom_rois, T* top_data//, int* argmax_data
  ) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const T* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    T P[8];
    compute_roi_pool_pts(offset_bottom_rois, P, spatial_scale, pooled_height, pooled_width, ph, pw);

    int leftMost, topMost, rightMost, bottomMost;
    get_rotated_rect_bounding_box(P, leftMost, topMost, rightMost, bottomMost, width, height);

    T maxval = 0;
    int maxidx = -1;
    const T* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    T AB[2];
    AB[0] = P[0] - P[2];
    AB[1] = P[1] - P[3];  
    T ABAB = AB[0]*AB[0] +AB[1]*AB[1];
    T AC[2];
    AC[0] = P[4] - P[2];
    AC[1] = P[5] - P[3];
    T ACAC = AC[0]*AC[0] + AC[1]*AC[1];

    for (int hh = topMost; hh < bottomMost+1; ++hh) {
      for (int ww = leftMost; ww < rightMost+1; ++ww) {
        T AP[2];
        AP[0] = ww - P[2];
        AP[1] = hh - P[3];
        T ABAP = AB[0]*AP[0] + AB[1]*AP[1];
        T ACAP = AC[0]*AP[0] + AC[1]*AP[1];
        if ( ABAP >= 1e-3 && (ABAB - ABAP) > -1e-3 && ACAP >= 1e-3 && (ACAC - ACAP) > -1e-3 )
        {
          int bottom_index = hh * width + ww;
          if (offset_bottom_data[bottom_index] > maxval) 
          {
            maxval = offset_bottom_data[bottom_index];
            maxidx = bottom_index;
          }
        }
      }
    }
    top_data[index] = maxval;
    // argmax_data[index] = maxidx;
  }
}


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
    cudaStream_t stream) 
{
  auto top_data_size = num_rois * channels * pooled_height * pooled_width;
  dim3 grid(std::min(static_cast<long>(std::ceil(top_data_size * 1.0 / 512L)), 4096L));
  dim3 block(512);

  RRoIPoolFForward<float><<<grid, block, 0, stream>>>(
       // output_size,
       // input.contiguous().data<float>(),
       // spatial_scale,
       // channels,
       // height,
       // width,
       // pooled_height,
       // pooled_width,
       // rois.contiguous().data<float>(),
       // output.data<float>(),
       // argmax.data<int>()
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
