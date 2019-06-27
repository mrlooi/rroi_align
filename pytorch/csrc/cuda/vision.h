#pragma once
#include <torch/extension.h>

#include "nms_method_enum.h"

at::Tensor nms_cuda(const at::Tensor& dets,
                   float threshold);

at::Tensor soft_nms_cuda(const at::Tensor& dets,
                   at::Tensor& scores,
                   const float nms_thresh=0.3,
                   const float sigma=0.5,
                   const float score_thresh=0.001,
                   const int method=NMS_METHOD::GAUSSIAN
                   );