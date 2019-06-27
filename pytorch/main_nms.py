import torch
import _C

import numpy as np
# import pycocotools.mask as mask_util
# import cv2

def soft_nms(boxes, scores, nms_thresh=0.3, sigma=0.5, thresh=0.001, method=1):
    # method: 1) linear, 2) gaussian, else) original NMS
    scores2 = scores.clone()
    keep = _C.soft_nms(boxes, scores2, nms_thresh, sigma, thresh, method)
    scores[:] = scores2[:]
    return keep

if __name__ == '__main__':
    boxes = np.array([[0, 0, 100, 100], [200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

    nms_thresh = 0.3

    t_boxes = torch.from_numpy(boxes)
    t_boxscores = torch.from_numpy(boxscores)
    t_dets = torch.cat((t_boxes, t_boxscores.unsqueeze(1)), 1)

    keep = _C.nms(t_dets[:, :-1], t_dets[:, -1], nms_thresh)
    print(keep)
    print(t_dets[keep])

    t_dets_cu = t_dets.to("cuda")
    keep = _C.nms(t_dets_cu[:, :-1], t_dets_cu[:, -1], nms_thresh)
    keep = keep.cpu()
    print(keep)
    print(t_dets[keep])

    t_dets2 = t_dets.clone()
    soft_keep = soft_nms(t_dets2[:, :-1], t_dets2[:, -1], nms_thresh=nms_thresh, method=1)

    print(soft_keep)
    with np.printoptions(precision=3):
        print(t_dets2[soft_keep].numpy())