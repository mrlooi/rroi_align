#pragma once

#include <cuda_runtime.h>

int rotate_nms_cuda(
    const float* rois_d,
    int64_t* out_keep,
    const int boxes_num, 
    const float nms_threshold, 
    const int max_output,
    bool train = false,
    cudaStream_t stream = 0
    );
