#include "rotate_mask_iou.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {

  m.def("rotate_mask_iou", &rotate_mask_iou, "rotate_mask_iou");

}