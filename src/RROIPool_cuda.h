#pragma once

#include <cuda_runtime.h>

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
    cudaStream_t stream = 0
    );

