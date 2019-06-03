#pragma once

#include <vector>
#include <torch/extension.h>

#include "rotate_mask_iou_impl.h"


// Interface for Python
at::Tensor rotate_mask_iou(
    const at::Tensor& gt_masks, const at::Tensor& proposals
)
{
  if (gt_masks.type().is_cuda())
  {
    AT_ERROR("Not compiled with GPU support");
  }

  return rotate_mask_iou_cpu(gt_masks, proposals);
}