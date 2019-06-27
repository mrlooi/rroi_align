#include "rotate_mask_iou.h"
#include "nms.h"
#include "rotate_nms.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_mask_iou", &rotate_mask_iou, "rotate_mask_iou");
  m.def("nms", &nms, "nms");
  m.def("soft_nms", &soft_nms, "soft_nms");
  m.def("rotate_nms", &rotate_nms, "rotate_nms");
  m.def("rotate_soft_nms", &rotate_soft_nms, "rotate_soft_nms");
}