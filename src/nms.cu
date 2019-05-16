#include <cuda.h>
#include <vector>
#include <iostream>

#include "rroi_helper.h"
#include "cuda_utils.h"
#include "rotate_rect_ops.h"

const int BLOCK_DIM = sizeof(unsigned long long) * 8;

#if 0
template <typename T>
__device__ float get_rotated_iou(
    const T* rectangle_pts1,
    const T* rectangle_pts2,
    const T area1,
    const T area2
    )
{
  T intersection_pts[16];
  int num_of_intersections = inter_pts(rectangle_pts1, rectangle_pts2, intersection_pts);
  // reorder_pts(intersection_pts, num_of_intersections);
  // float area_inter = area(intersection_pts, num_of_intersections);
  // return area_inter / (area1 + area2 - area_inter + 1e-8);
  return 0;
}
#endif

template <typename T>
__device__ T triangle_area(
    T a_x, T a_y,
    T b_x, T b_y,
    T c_x, T c_y)
{
  return ((a_x - c_x) * (b_y - c_y) - (a_y - c_y) * (b_x - c_x)) / 2.0;
}

template <typename T>
__device__ bool line_intersection(
    const T line1_start_x, const T line1_start_y,
    const T line1_end_x, const T line1_end_y,
    const T line2_start_x, const T line2_start_y,
    const T line2_end_x, const T line2_end_y,
    T* intersection_pts)
{
  T area_abc, area_abd, area_cda, area_cdb;

  area_abc = triangle_area(line1_start_x, line1_start_y, line1_end_x, line1_end_y, line2_start_x, line2_start_y);
  area_abd = triangle_area(line1_start_x, line1_start_y, line1_end_x, line1_end_y, line2_end_x, line2_end_y);
  if (area_abc * area_abd >= 0) {
    return false;
  }

  area_cda = triangle_area(line2_start_x, line2_start_y, line2_end_x, line2_end_y, line1_start_x, line1_start_y);
  area_cdb = area_cda + area_abc - area_abd;
  if (area_cda * area_cdb >= 0) {
    return false;
  }

  T t = area_cda / (area_abd - area_abc);
  T dx = t * (line1_end_x - line1_start_x);
  T dy = t * (line1_end_y - line1_start_y);
  intersection_pts[0] = line1_start_x + dx;
  intersection_pts[1] = line1_start_y + dy;

  return true;
}

__device__ bool in_rect_float(const float pt_x, const float pt_y, const float* pts)
{
  float ab[2];
  float ad[2];
  float ap[2];

  float abab;
  float abap;
  float adad;
  float adap;

  ab[0] = pts[2] - pts[0];
  ab[1] = pts[3] - pts[1];

  ad[0] = pts[6] - pts[0];
  ad[1] = pts[7] - pts[1];

  ap[0] = pt_x - pts[0];
  ap[1] = pt_y - pts[1];

  abab = ab[0] * ab[0] + ab[1] * ab[1];
  abap = ab[0] * ap[0] + ab[1] * ap[1];
  adad = ad[0] * ad[0] + ad[1] * ad[1];
  adap = ad[0] * ap[0] + ad[1] * ap[1];

  return abab >= abap && abap >= 0 && adad >= adap && adap >= 0;
}

__device__ int inter_pts_float(const float* pts1, const float* pts2, float* int_pts)
{
  int num_of_inter = 0;

  for (int i = 0;i < 4;i++) {
    if (in_rect_float(pts1[2 * i], pts1[2 * i + 1], pts2)) {
      int_pts[num_of_inter * 2] = pts1[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts1[2 * i + 1];
      num_of_inter++;
    }
    if (in_rect_float(pts2[2 * i], pts2[2 * i + 1], pts1)) {
      int_pts[num_of_inter * 2] = pts2[2 * i];
      int_pts[num_of_inter * 2 + 1] = pts2[2 * i + 1];
      num_of_inter++;
    }
  }

  float temp_pts[2];
  for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 4; j++) {
      bool has_pts = line_intersection(
          pts1[2 * i], pts1[2 * i + 1],
          pts1[2 * ((i + 1) % 4)], pts1[2 * ((i + 1) % 4) + 1],
          pts2[2 * j], pts2[2 * j + 1],
          pts2[2 * ((j + 1) % 4)], pts2[2 * ((j + 1) % 4) + 1],
          temp_pts);
      if (has_pts) {
        int_pts[num_of_inter * 2] = temp_pts[0];
        int_pts[num_of_inter * 2 + 1] = temp_pts[1];
        num_of_inter++;
      }
    }
  }

  return num_of_inter;
}

__device__ float get_rotated_iou(
    const float* rectangle_pts1,
    const float* rectangle_pts2,
    const float area1,
    const float area2
    )
{
  float intersection_pts[16];
  int num_of_intersections;
  // num_of_intersections = inter_pts(rectangle_pts1, rectangle_pts2, intersection_pts);
  num_of_intersections = inter_pts_float(rectangle_pts1, rectangle_pts2, intersection_pts);

  reorder_pts(intersection_pts, num_of_intersections);
  float area_inter = area(intersection_pts, num_of_intersections);
  return area_inter / (area1 + area2 - area_inter + 1e-8);
}

template <typename T>
__device__ T get_rbox_area(const T* boxes)
{
  return boxes[2] * boxes[3];
}

__global__ void rotated_nms_kernel(
    unsigned long long* __restrict__ mask,
    const int box_num,
    const float nms_overlap_threshold,
    const float* __restrict__ boxes
    )
{
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size = min(box_num - row_start * BLOCK_DIM, BLOCK_DIM);
  const int col_size = min(box_num - col_start * BLOCK_DIM, BLOCK_DIM);

  // cache all column data in this block
  // __shared__ float column_boxes[BLOCK_DIM * 5];
  // __shared__ float row_boxes[BLOCK_DIM * 5];

  __shared__ float column_pts[BLOCK_DIM * 8];
  __shared__ float row_pts[BLOCK_DIM * 8];

  __shared__ float column_box_areas[BLOCK_DIM];
  __shared__ float row_box_areas[BLOCK_DIM];

  const int column_box_idx = BLOCK_DIM * col_start + threadIdx.x;
  if (threadIdx.x < col_size) {
    // column_boxes[threadIdx.x * 5 + 0] = boxes[column_box_idx * 5 + 0];
    // column_boxes[threadIdx.x * 5 + 1] = boxes[column_box_idx * 5 + 1];
    // column_boxes[threadIdx.x * 5 + 2] = boxes[column_box_idx * 5 + 2];
    // column_boxes[threadIdx.x * 5 + 3] = boxes[column_box_idx * 5 + 3];
    // column_boxes[threadIdx.x * 5 + 4] = boxes[column_box_idx * 5 + 4];
    column_box_areas[threadIdx.x] = get_rbox_area(boxes + column_box_idx * 5);
    // convert_region_to_pts(column_boxes + threadIdx.x * 5, column_pts + threadIdx.x * 8);
    convert_region_to_pts(boxes + column_box_idx * 5, column_pts + threadIdx.x * 8);
  }

  const int row_box_idx = BLOCK_DIM * row_start + threadIdx.x;
  if (threadIdx.x < row_size) {
    // row_boxes[threadIdx.x * 5 + 0] = boxes[row_box_idx * 5 + 0];
    // row_boxes[threadIdx.x * 5 + 1] = boxes[row_box_idx * 5 + 1];
    // row_boxes[threadIdx.x * 5 + 2] = boxes[row_box_idx * 5 + 2];
    // row_boxes[threadIdx.x * 5 + 3] = boxes[row_box_idx * 5 + 3];
    // row_boxes[threadIdx.x * 5 + 4] = boxes[row_box_idx * 5 + 4];
    row_box_areas[threadIdx.x] = get_rbox_area(boxes + row_box_idx * 5);
    // convert_region_to_pts(row_boxes + threadIdx.x * 5, row_pts + threadIdx.x * 8);
    convert_region_to_pts(boxes + row_box_idx * 5, row_pts + threadIdx.x * 8);
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
      // float iou = computeRectIoU(row_boxes + threadIdx.x * 5, column_boxes + i * 5);
      float iou = get_rotated_iou(row_pts + threadIdx.x * 8, column_pts + i * 8, row_box_areas[threadIdx.x], column_box_areas[i]);

      // printf("iou: %.3f\n", iou);
      if (iou > nms_overlap_threshold) {
        t |= 1ULL << i;  // basically storing all overlaps across the columns, hashed into one single ULL index
      }
    }

    int col_blocks = DIVUP(box_num, BLOCK_DIM);
    mask[row_box_idx * col_blocks + col_start] = t;
  }
}

int rotated_nms(
    const float* rois_d,
    int64_t* keep_d,
    const int box_num,
    const int max_output,
    const float nms_overlap_threshold,
    cudaStream_t stream = 0
    )
{
  if (box_num == 0) {
    return 0;
  }

  // const int block_num = static_cast<int>(std::ceil(box_num * 1.0 / BLOCK_DIM));
  const int block_num = DIVUP(box_num, BLOCK_DIM);

  unique_ptr_host<unsigned long long> mask_h(nullptr);
  unique_ptr_device<unsigned long long> mask_d(nullptr);
  // long long mask_size = box_num * block_num;
  auto mask_size = box_num * block_num;
  CUDA_CHECK(cudaMallocHost((void **) &mask_h, mask_size * sizeof(unsigned long long)));
  CUDA_CHECK(cudaMalloc((void **) &mask_d, mask_size * sizeof(unsigned long long)));

  dim3 blocks(block_num, block_num);
  dim3 threads(BLOCK_DIM);
  rotated_nms_kernel<<<blocks, threads, 0, stream>>>(
      mask_d.get(),
      box_num,
      nms_overlap_threshold,
      rois_d
      );
  CUDA_CHECK(cudaDeviceSynchronize());

  CUDA_CHECK(cudaMemcpy(mask_h.get(), mask_d.get(), mask_size * sizeof(unsigned long long), cudaMemcpyDeviceToHost));

  std::vector<unsigned long long> remv(block_num);
  memset(&remv[0], 0, sizeof(unsigned long long) * block_num);

  unique_ptr_host<int64_t> keep_h(nullptr);
  CUDA_CHECK(cudaMallocHost((void **) &keep_h, box_num * sizeof(int64_t)));

  int num_to_keep = 0;
  for (int i = 0; i < box_num; i++) {
    int nblock = i / BLOCK_DIM;
    int inblock = i % BLOCK_DIM;

    if (!(remv[nblock] & (1ULL << inblock))) {  // if not zero i.e. no overlap
      keep_h[num_to_keep++] = i;
      unsigned long long *p = mask_h.get() + i * BLOCK_DIM;

      // Loop through the col_block array to aggregate all iou overlap results for that box
      for (int j = nblock; j < BLOCK_DIM; j++) {
        remv[j] |= p[j];
      }
    }
  }

  if (max_output >= 0) {
    num_to_keep = std::min(num_to_keep, max_output);
  }

  std::cout << "num_to_keep: " << num_to_keep << std::endl;

  // CUDA_CHECK(cudaMalloc((void **) &keep_d, num_to_keep * sizeof(int64_t)));
  // CUDA_CHECK(cudaMemcpy(keep_d, keep_h.get(), num_to_keep * sizeof(int64_t), cudaMemcpyHostToDevice));

  CUDA_CHECK(cudaMemcpy(keep_d, keep_h.get(), box_num * sizeof(int64_t), cudaMemcpyHostToDevice));

  return num_to_keep;
}
