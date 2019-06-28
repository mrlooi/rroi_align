import torch
import _C

import numpy as np
# import pycocotools.mask as mask_util
# import cv2

def rotate_soft_nms(boxes, scores, nms_thresh=0.3, sigma=0.5, thresh=0.001, method=1):
    # method: 1) linear, 2) gaussian, else) original NMS
    boxes2 = boxes.clone()
    scores2 = scores.clone()
    keep = _C.rotate_soft_nms(boxes2, scores2, nms_thresh, sigma, thresh, method)
    scores[:] = scores2[:]
    boxes[:] = boxes2[:]
    return keep

if __name__ == '__main__':
    np.set_printoptions(suppress=True, precision=4)

    # boxes = np.array([
    #     [0, 0, 100, 100, 0], 
    #     [300, 300, 200, 200, 0], 
    #     [320, 320, 200, 200, 0], 
    #     [300, 340, 200, 200, 0], 
    #     [340, 300, 200, 200, 0], 
    #     [1, 1, 2, 2, 0]
    # ], dtype=np.float32)
    # boxscores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)
    
    boxes = np.array([
        [50, 50, 100, 100, 0], 
        [80, 80, 100, 100, 0], 
        [100, 100, 100, 100, 0], 
        [105, 105, 90, 90, 0], 
    ], dtype=np.float32)
    boxscores = np.array([0.95, 0.9, 0.85, 0.9], dtype=np.float32)

    nms_thresh = 0.3
    METHOD = 1

    t_boxes = torch.from_numpy(boxes)
    t_boxscores = torch.from_numpy(boxscores)
    t_dets = torch.cat((t_boxes, t_boxscores.unsqueeze(1)), 1)

    keep = _C.rotate_nms(t_dets[:, :-1], t_dets[:, -1], nms_thresh)
    print(keep)
    print(t_dets[keep])

    t_dets2 = t_dets.clone()
    soft_keep = rotate_soft_nms(t_dets2[:, :-1], t_dets2[:, -1], nms_thresh=nms_thresh, method=METHOD)

    print(soft_keep)
    with np.printoptions(precision=3):
        print(t_dets2[soft_keep].numpy())