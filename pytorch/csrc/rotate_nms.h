#pragma once

#include "cpu/vision.h"

at::Tensor rotate_nms(const at::Tensor& dets,
               const at::Tensor& scores,
               const float threshold) {

  if (dets.type().is_cuda()) {
    AT_ERROR("Not compiled with GPU support");
  }

  at::Tensor result = rotate_nms_cpu(dets, scores, threshold);
  return result;
}

at::Tensor rotate_soft_nms(at::Tensor& dets,
                at::Tensor& scores,
                const float nms_thresh=0.3,
                const float sigma=0.5,
                const float score_thresh=0.001,
                const int method=NMS_METHOD::GAUSSIAN) 
{

  if (dets.type().is_cuda()) {
    AT_ERROR("Not compiled with GPU support");
  }

  at::Tensor result = rotate_soft_nms_cpu(dets, scores, nms_thresh, sigma, score_thresh, method);
  return result;
}
