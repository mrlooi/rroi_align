#ifndef RROIALIGN_CUDA_H
#define RROIALIGN_CUDA_H

#include <cuda_runtime.h>

void RROIAlign_forward_golden(
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

#endif /* RROIALIGN_CUDA_H */
