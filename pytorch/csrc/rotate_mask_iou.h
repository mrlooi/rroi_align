#pragma once

#include "cpu/vision.h"


// Interface for Python
at::Tensor rotate_mask_iou(
    const at::Tensor& gt_masks, const at::Tensor& proposals
)
{
  if (gt_masks.type().is_cuda())
  {
#ifdef WITH_CUDA
#else
    AT_ERROR("Not compiled with GPU support");
#endif
  }

  return rotate_mask_iou_cpu(gt_masks, proposals);
}