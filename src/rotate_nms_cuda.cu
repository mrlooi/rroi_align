#include "rotate_nms.h"

#include <vector>
#include <iostream>
#include <stdio.h>

#include <cmath>

#include "cuda_utils.h"
#include "rotate_rect_ops.h"

int const threadsPerBlock = sizeof(unsigned long long) * 8;

template <typename T>
__device__ inline float devRotateIoU(T const * const region1, T const * const region2) {

  return computeRectIoU(region1, region2);
}

__global__ void rotate_nms_kernel(const int n_boxes, const float nms_overlap_thresh,
                           const float *dev_boxes, unsigned long long *dev_mask) {
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  // if (row_start > col_start) return;

  const int row_size =
        min(n_boxes - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size =
        min(n_boxes - col_start * threadsPerBlock, threadsPerBlock);

  // cache all column data in this block
  __shared__ float block_boxes[threadsPerBlock * 5];
  __shared__ float cur_boxes[threadsPerBlock * 5];

  const int block_box_idx = threadsPerBlock * col_start + threadIdx.x;  // current column
  if (threadIdx.x < col_size) {
    block_boxes[threadIdx.x * 5 + 0] = dev_boxes[block_box_idx * 5 + 0];
    block_boxes[threadIdx.x * 5 + 1] = dev_boxes[block_box_idx * 5 + 1];
    block_boxes[threadIdx.x * 5 + 2] = dev_boxes[block_box_idx * 5 + 2];
    block_boxes[threadIdx.x * 5 + 3] = dev_boxes[block_box_idx * 5 + 3];
    block_boxes[threadIdx.x * 5 + 4] = dev_boxes[block_box_idx * 5 + 4];
  }

  const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;  // current row
  if (threadIdx.x < row_size) {
    cur_boxes[threadIdx.x * 5 + 0] = dev_boxes[cur_box_idx * 5 + 0];
    cur_boxes[threadIdx.x * 5 + 1] = dev_boxes[cur_box_idx * 5 + 1];
    cur_boxes[threadIdx.x * 5 + 2] = dev_boxes[cur_box_idx * 5 + 2];
    cur_boxes[threadIdx.x * 5 + 3] = dev_boxes[cur_box_idx * 5 + 3];
    cur_boxes[threadIdx.x * 5 + 4] = dev_boxes[cur_box_idx * 5 + 4];
  }

  __syncthreads();

  // iterate across each row in this block
  if (threadIdx.x < row_size) {
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start) {
      start = threadIdx.x + 1;  // if they are the same, skip to next (column)
    }

    // for this row, calculate all ious with each column
    for (i = start; i < col_size; i++) {
      float iou = devRotateIoU(cur_boxes + threadIdx.x * 5, block_boxes + i * 5);
      // printf("iou: %.3f\n", iou);
      if (iou > nms_overlap_thresh) {
        t |= 1ULL << i;  // basically storing all overlaps across the columns, hashed into one single ULL index
      }
    }

    const int col_blocks = DIVUP(n_boxes, threadsPerBlock);
    dev_mask[cur_box_idx * col_blocks + col_start] = t;
  }
}

void _rotate_nms_launcher(int64_t* keep_out, int* num_out, const float* boxes, int boxes_num,
          float nms_overlap_thresh, cudaStream_t stream)
{
  /**
  Inputs:
  boxes: N,5  (xc,yc,w,h,angle)  ASSUMES already sorted
  boxes_num: N
  nms_overlap_thresh: 0-1 e.g. 0.7

  Outputs:
  keep_out: N  (i.e. stores indices of valid boxes)
  num_out: total count of valid indices

  */

  const int col_blocks = DIVUP(boxes_num, threadsPerBlock);

  unsigned long long* mask_dev = NULL;
  // Get the IoUs between each element in the array (N**2 operation)
  // then store all the overlap results in the N*col_blocks array (mask_dev).
  // col_blocks represents the total number of column blocks (blockDim.x) made for the kernel computation
  // Each column block will store a hash of the iou overlaps between each column and row in the block. The hash is a ULL of bit overlaps between one row and all columns in the block
  // then copy the results to host code
  // Each result row is a col_block array, which contains all the iou overlap bool (as a hash) per column block.
  // Loop through the col_block array to aggregate all iou overlap results for that row
  CUDA_CHECK(cudaMalloc(&mask_dev,
                        boxes_num * col_blocks * sizeof(unsigned long long)));

  dim3 blocks(col_blocks, col_blocks);
  dim3 threads(threadsPerBlock);

  rotate_nms_kernel<<<blocks, threads, 0, stream>>>(boxes_num,
                                  nms_overlap_thresh,
                                  boxes,
                                  mask_dev);
  cudaThreadSynchronize();

  std::vector<unsigned long long> mask_host(boxes_num * col_blocks);
  CUDA_CHECK(cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * boxes_num * col_blocks,
                        cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  int num_to_keep = 0;
  for (int i = 0; i < boxes_num; i++) {
    int nblock = i / threadsPerBlock;  // get column block
    int inblock = i % threadsPerBlock;

    if (!(remv[nblock] & (1ULL << inblock))) {  // if not zero i.e. no overlap
      keep_out[num_to_keep++] = i;
      unsigned long long *p = &mask_host[0] + i * col_blocks;

      // Loop through the col_block array to aggregate all iou overlap results for that box
      for (int j = nblock; j < col_blocks; j++) {
        remv[j] |= p[j];
      }
    }
  }
  *num_out = num_to_keep;

  CUDA_CHECK(cudaFree(mask_dev));
}

int rotated_nms_golden(
    const float* rois_d,
    int64_t* out_keep,
    const int boxes_num, 
    const float nms_threshold, 
    const int max_output,
    cudaStream_t stream
    )
{
  if (boxes_num == 0)
    return 0; 


  std::vector<int64_t> keep(boxes_num, 0);

  int num_to_keep = 0;
  _rotate_nms_launcher(&keep[0], &num_to_keep, rois_d, boxes_num,
          nms_threshold, stream);

  if (max_output >= 0)
    num_to_keep = std::min(num_to_keep, max_output);

  CUDA_CHECK(cudaMalloc((void **) &out_keep, num_to_keep * sizeof(int64_t)));
  CUDA_CHECK(cudaMemcpy(out_keep,
                        &keep[0],
                        num_to_keep * sizeof(int64_t),
                        cudaMemcpyHostToDevice));

  return num_to_keep;
}
