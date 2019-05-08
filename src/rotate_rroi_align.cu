#include <cfloat>
#include <cmath>
#include <algorithm>
//#include <cstdio>
#include <cuda.h>

#include "rroi.h"
#include "rotate_rect_ops.h"

// TODO make it in a common file
#define CUDA_1D_KERNEL_LOOP(i, n)                            \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
       i += blockDim.x * gridDim.x)


//Definite equinox
template <typename Dtype>
__device__ inline float DexX(const Dtype* bottom_rois, int i_int, int j_int, const int pooled_height_int, const int pooled_width_int) {

  Dtype i = float(i_int);
  Dtype j = float(j_int);
  Dtype pooled_width = float(pooled_width_int);
  Dtype pooled_height = float(pooled_height_int);

  return (pooled_height - i) / pooled_height * (
	(pooled_width - j) / pooled_width * bottom_rois[1-1] + j / pooled_width * bottom_rois[3-1]) + i / pooled_height * (
	(pooled_width - j) / pooled_width * bottom_rois[7-1] + j / pooled_width * bottom_rois[5-1]);
}

template <typename Dtype>
__device__ inline float DexY(const Dtype* bottom_rois, int i_int, int j_int, const int pooled_height_int, const int pooled_width_int) {

  Dtype i = float(i_int);
  Dtype j = float(j_int);
  Dtype pooled_width = float(pooled_width_int);
  Dtype pooled_height = float(pooled_height_int);

  return (pooled_width - j) / pooled_width * (
	(pooled_height - i) / pooled_height * bottom_rois[2-1] + i / pooled_height * bottom_rois[8-1]) + j / pooled_width * (
	(pooled_height - i) / pooled_height * bottom_rois[4-1] + i / pooled_height * bottom_rois[6-1]);
}

template <typename Dtype>
__device__ inline Dtype cross_mul(Dtype *pt1,Dtype * pt2,Dtype *pt3){
  return pt2[0]*pt3[1]+pt3[0]*pt1[1]+pt1[0]*pt2[1]-pt2[0]*pt1[1]-pt3[0]*pt2[1]-pt1[0]*pt3[1];
}

template <typename Dtype>
__device__ inline bool inpoly(Dtype pt_x, Dtype pt_y, Dtype * pts) {
  bool flag = true;
  int cur_sign;
  Dtype pt[2];
  pt[0] = pt_x;
  pt[1] = pt_y;
  int sign;
  for(int i = 0 ;i<4;i++){
     Dtype val = cross_mul(pts+i*2,pts+((i+1)%4*2),pt);
     if(val<0.0f){
        cur_sign = -1;
     }else if(val>0.0f){
        cur_sign = 1;
     }else{
        cur_sign =0;
     }
     if(cur_sign !=0){
        if(flag){
            flag = false;
            sign = cur_sign;
        }else{
            if(sign!=cur_sign) return false;
        }
     }
  }
  return true;
}



template <typename Dtype>
__global__ void RotateROIAlignForward(const int nthreads, const Dtype* bottom_data,
    const Dtype spatial_scale, const int channels, const int height,
    const int width, const int pooled_height, const int pooled_width,
    const Dtype* bottom_rois, Dtype* top_data) {

    // The real spatial_scale should be depended on the true scale

    Dtype spatial_scale_h = spatial_scale;
    Dtype spatial_scale_w = spatial_scale;

    //Dtype spatial_scale = (spatial_scale_w + spatial_scale_h) / 2.0;
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // (n, c, ph, pw) is an element in the pooled output
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int c = (index / pooled_width / pooled_height) % channels;
    int n = index / pooled_width / pooled_height / channels;

    const Dtype* offset_bottom_rois = bottom_rois + n * 6;
    int roi_batch_ind = offset_bottom_rois[0];

    Dtype roi_pts[8]; 
    convert_region_to_pts(offset_bottom_rois + 1, roi_pts);

    // order of lt, rt, rb, lb
    Dtype P[8];
    P[0] = DexX(roi_pts, ph, pw, pooled_height, pooled_width) * spatial_scale_w;
    P[1] = DexY(roi_pts, ph, pw, pooled_height, pooled_width) * spatial_scale_h;
    P[2] = DexX(roi_pts, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_w;
    P[3] = DexY(roi_pts, ph, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[4] = DexX(roi_pts, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_w;
    P[5] = DexY(roi_pts, ph + 1, pw + 1, pooled_height, pooled_width) * spatial_scale_h;
    P[6] = DexX(roi_pts, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_w;
    P[7] = DexY(roi_pts, ph + 1, pw, pooled_height, pooled_width) * spatial_scale_h;

    //int leftMost = int(max(round(min(min(P[0],P[2]),min(P[4],P[6]))),0.0));
    //int rightMost= int(min(round(max(max(P[0],P[2]),max(P[4],P[6]))),imageWidth-1.0));
    //int topMost= int(max(round(min(min(P[1],P[3]),min(P[5],P[7]))),0.0));
    //int bottomMost= int(min(round(max(max(P[1],P[3]),max(P[5],P[7]))),imageHeight-1.0));

    // Exact position on feature map in type float
    Dtype leftMost = fmax(fmin(fmin(P[0],P[2]),fmin(P[4],P[6])),0.0);
    Dtype rightMost = fmin(fmax(fmax(P[0],P[2]),fmax(P[4],P[6])),width-1.0);
    Dtype topMost = fmax(fmin(fmin(P[1],P[3]),fmin(P[5],P[7])),0.0);
    Dtype bottomMost = fmin(fmax(fmax(P[1],P[3]),fmax(P[5],P[7])),height-1.0);

    float maxval = 0.0;

    float max_con_x = -1.0;
    float max_con_y = -1.0;

    const Dtype* offset_bottom_data = bottom_data + (roi_batch_ind * channels + c) * height * width;

    //Dtype AB[2];
    //AB[0] = P[2] - P[0];
    //AB[1] = P[3] - P[1];	
    //Dtype ABAB = AB[0]*AB[0] + AB[1]*AB[1];
    //Dtype AC[2];
    //AC[0] = P[4] - P[0];
    //AC[1] = P[5] - P[1];
    //Dtype ACAC = AC[0]*AC[0] + AC[1]*AC[1];

    Dtype h = topMost;
    
    while (h < bottomMost+1) {
      Dtype w = leftMost;
      while (w < rightMost+1) {
           
           if(inpoly(w, h, P)){
               //Performing blinear interpolation
               int bin_xs = int(floor(w));
               int bin_ys = int(floor(h));

               float rx = w - floor(w);
               float ry = h - floor(w);

               float wlt = (1.0 - rx) * (1.0 - ry);
               float wrt = rx * (1.0 - ry);
               float wrb = rx * ry;
               float wlb = (1.0 - rx) * ry;

               float inter_val = 0.0;

               int min_x = min(max(bin_xs, 0), width - 1);
               int min_y = min(max(bin_ys, 0), height - 1);
               int max_x = max(min(bin_xs + 1, width - 1), 0);
               int max_y = max(min(bin_ys + 1, height - 1), 0);

               int lt = min_y * width + min_x;
               int rt = min_y * width + max_x;
               int rb = max_y * width + max_x;
               int lb = max_y * width + min_x;               

               inter_val += offset_bottom_data[lt] * wlt;
               inter_val += offset_bottom_data[rt] * wrt;
               inter_val += offset_bottom_data[rb] * wrb;
               inter_val += offset_bottom_data[lb] * wlb;
               
               //inter_val = bottom_data[bin_ys * width + bin_xs];

               if (inter_val > maxval) {
                   maxval = inter_val;
                   
                   max_con_x = w;
                   max_con_y = h;
               }
          }
       
          w = w + 1.0;
      }
      h = h + 1.0;
    }
     
 

    top_data[index] = maxval;
    // con_idx_x[index] = max_con_x;
    // con_idx_y[index] = max_con_y;
  }
}


void rotate_rroi_align(
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
  RotateROIAlignForward<float><<<grid, block, 0, stream>>>(
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
