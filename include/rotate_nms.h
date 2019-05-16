#pragma once

#include <cuda_runtime.h>

int rotated_nms_golden(
    const float* rois_d,
    int64_t* out_keep,
    const int boxes_num,
    const float nms_threshold,
    const int max_output,
    cudaStream_t stream = 0
    );

int rotated_nms(
    const float* rois_d,
    int64_t* keep_d,
    const int box_num,
    const int max_output,
    const float nms_overlap_threshold,
    cudaStream_t stream = 0
    );
