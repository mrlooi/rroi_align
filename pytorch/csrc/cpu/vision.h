#pragma once
#include <torch/extension.h>

#include "nms_method_enum.h"

at::Tensor nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

at::Tensor soft_nms_cpu(at::Tensor& dets,
                   at::Tensor& scores,
                   const float nms_thresh=0.3,
                   const float sigma=0.5,
                   const float score_thresh=0.001,
                   const int method=NMS_METHOD::GAUSSIAN
                   );

at::Tensor rotate_nms_cpu(const at::Tensor& dets,
                   const at::Tensor& scores,
                   const float threshold);

at::Tensor rotate_soft_nms_cpu(at::Tensor& dets,
                   at::Tensor& scores,
                   const float nms_thresh=0.3,
                   const float sigma=0.5,
                   const float score_thresh=0.001,
                   const int method=NMS_METHOD::GAUSSIAN
                   );

at::Tensor rotate_mask_iou_cpu(
    const at::Tensor& gt_masks, const at::Tensor& proposals
);