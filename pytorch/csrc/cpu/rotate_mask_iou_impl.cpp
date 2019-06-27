// #include <opencv2/opencv.hpp>

#include <omp.h>

#include "cpu/vision.h"


template <typename scalar_t>
void rotate_mask_iou_cpu_kernel(const at::Tensor& gt_masks,
                          const at::Tensor& proposals,
                          at::Tensor& iou_matrix)
{
  AT_ASSERTM(!gt_masks.type().is_cuda(), "r_boxes1 must be a CPU tensor");
  AT_ASSERTM(!proposals.type().is_cuda(), "r_boxes2 must be a CPU tensor");

  if (gt_masks.numel() == 0 || proposals.numel() == 0)
    return;

  int N = gt_masks.size(0);
  int N2 = proposals.size(0);

  AT_ASSERTM(N == N2, "gt_masks and proposals size must match");

  for (int64_t i = 0; i < N; i++)
  {
  	
  }
}

at::Tensor rotate_mask_iou_cpu(
    const at::Tensor& gt_masks, const at::Tensor& proposals
)
{
  int N = gt_masks.size(0);
  int N2 = proposals.size(0);

  AT_ASSERTM(N == N2, "gt_masks and proposals size must match");

  at::Tensor mask_ious = at::zeros({N}, proposals.options());

  if (N == 0)
    return mask_ious;
  
  AT_DISPATCH_FLOATING_TYPES(gt_masks.type(), "rotate_mask_iou", [&] {
    rotate_mask_iou_cpu_kernel<scalar_t>(gt_masks, proposals, mask_ious);
  });

  return mask_ious;
}
