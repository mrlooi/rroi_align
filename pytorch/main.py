import torch
# import _C

import numpy as np
# import pycocotools.mask as mask_util
import cv2

from rotate_ops import paste_rotated_roi_in_image, draw_anchors

def compute_rotated_proposal_gt_iou(gt_mask, proposal):
    img_h, img_w = gt_mask.shape[:2]

    xc, yc, w, h, angle = proposal
    h, w = np.round([h, w]).astype(np.int32)

    if h <= 0 or w <= 0:
        return 0.0
    img_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    proposal_mask = np.ones((h, w), dtype=np.uint8)
    proposal_mask = paste_rotated_roi_in_image(img_mask, proposal_mask, proposal)
    proposal_mask = gt_mask * proposal_mask

    full_area = np.sum(gt_mask == 1)
    box_area = np.sum(proposal_mask == 1)
    mask_iou = float(box_area) / full_area
    # rle_for_fullarea = mask_util.encode(np.asfortranarray(gt_mask))
    # rle_for_box_area = mask_util.encode(np.asfortranarray(proposal_mask))
    # full_area = mask_util.area(rle_for_fullarea).sum().astype(float)
    # box_area = mask_util.area(rle_for_box_area).sum().astype(float)
    # mask_iou = box_area / full_area

    return mask_iou


if __name__ == '__main__':
    import time

    ITERS = 1000

    img_h = 800
    img_w = 800

    RED = (0,0,255)

    proposal = np.array([img_h//2, img_w//2, img_h//3, img_w//3, 30])

    gt_mask = np.zeros((img_h, img_w), dtype=np.uint8)
    gt_mask[img_h//4:img_h//4*3,img_w//4:img_w//4*3] = 1.0

    t_gtm = torch.Tensor(ITERS, img_h, img_w)
    t_pp = torch.Tensor(ITERS, 5)
    t_gtm[:] = torch.from_numpy(gt_mask)
    t_pp[:] = torch.from_numpy(proposal)

    # t = time.time()
    # tx = _C.rotate_mask_iou(t_gtm, t_pp)
    # print("%.3f s"%(time.time() - t))
    # print(tx[0])

    t = time.time()
    for i in range(ITERS):
        mask_iou = compute_rotated_proposal_gt_iou(gt_mask, proposal)
    print("%.3f s"%(time.time() - t))
    print(mask_iou)

    # gt_mask_color = gt_mask * 255
    # gt_mask_color = cv2.cvtColor(gt_mask_color, cv2.COLOR_GRAY2BGR)
    # gt_mask_color = draw_anchors(gt_mask_color, [proposal], [RED])
    # cv2.imshow("gt_mask", gt_mask_color)
    # cv2.waitKey(0)