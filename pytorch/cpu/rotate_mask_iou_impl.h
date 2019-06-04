#pragma once
#include <torch/extension.h>

at::Tensor rotate_mask_iou_cpu(
    const at::Tensor& gt_masks, const at::Tensor& proposals
);