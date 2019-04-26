#ifndef RROIALIGN_CUDA_H
#define RROIALIGN_CUDA_H

void RROIAlign_forward_cuda(
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
    float* top_data_d
    );

#endif /* RROIALIGN_CUDA_H */
