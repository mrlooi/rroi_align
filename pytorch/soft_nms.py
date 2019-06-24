import numpy as np

def cpu_soft_nms(boxes, sigma=0.5, Nt=0.3, threshold=0.001, method=0):
    """
    https://github.com/bharatsingh430/soft-nms/blob/master/lib/nms/cpu_nms.pyx
    """
    N = boxes.shape[0]

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection 
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt: 
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt: 
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]
            
                    # if box score falls below threshold, discard the box by swapping with last box
                    # update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    keep = [i for i in range(N)]
    return keep

def py_cpu_softnms(dets, Nt=0.3, sigma=0.5, thresh=0.001, method=2):
    """
    https://github.com/DocF/Soft-NMS/blob/master/soft_nms.py

    py_cpu_softnms
    :param dets:   boexs 坐标矩阵 format [y1, x1, y2, x2]
    :param sc:     每个 boxes 对应的分数
    :param Nt:     iou 交叠门限
    :param sigma:  使用 gaussian 函数的方差
    :param thresh: 最后的分数门限
    :param method: 使用的方法
    :return:       留下的 boxes 的 index
    """

    # indexes concatenate boxes with the last column
    N = dets.shape[0]
    # indexes = np.array([np.arange(N)])
    # dets = np.concatenate((dets, indexes.T), axis=1)

    for i in range(N):
        # intermediate parameters for later parameters exchange
        tBD = dets[i, :].copy()
        pos = i + 1

        # # SORT
        # scores = dets[:, 4]#.copy()
        # if i != N-1:
        #     maxscore = np.max(scores[pos:], axis=0)
        #     maxpos = np.argmax(scores[pos:], axis=0)
        # else:
        #     maxscore = scores[-1]
        #     maxpos = 0
        # if scores[i] < maxscore:
        #     dets[i, :] = dets[maxpos + i + 1, :]
        #     dets[maxpos + i + 1, :] = tBD

        # IoU calculate
        xx1 = np.maximum(dets[i, 0], dets[pos:, 0])
        yy1 = np.maximum(dets[i, 1], dets[pos:, 1])
        xx2 = np.minimum(dets[i, 2], dets[pos:, 2])
        yy2 = np.minimum(dets[i, 3], dets[pos:, 3])

        y1 = dets[i, 1]
        x1 = dets[i, 0]
        y2 = dets[i, 3]
        x2 = dets[i, 2]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        areas = (dets[pos:, 2] - dets[pos:, 0] + 1) * (dets[pos:, 3] - dets[pos:, 1] + 1) 

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (area + areas - inter)

        # Three methods: 1.linear 2.gaussian 3.original NMS
        if method == 1:  # linear
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = weight[ovr > Nt] - ovr[ovr > Nt]
        elif method == 2:  # gaussian
            weight = np.exp(-(ovr * ovr) / sigma)
        else:  # original NMS
            weight = np.ones(ovr.shape)
            weight[ovr > Nt] = 0

        dets[:, 4][pos:] = weight * dets[:, 4][pos:]

    # select the boxes and keep the corresponding indexes
    inds = np.arange(N)[dets[:, 4] > thresh]
    keep = inds.astype(int)

    return keep


if __name__ == '__main__':
    import cv2
    np.set_printoptions(suppress=True)

    # boxes and scores
    boxes = np.array([[0, 0, 100, 100], [200, 200, 400, 400], [220, 220, 420, 420], [200, 240, 400, 440], [240, 200, 440, 400], [1, 1, 2, 2]], dtype=np.float32)
    boxscores = np.array([0.95, 0.9, 0.8, 0.7, 0.6, 0.5], dtype=np.float32)

    nms_thresh = 0.5
    METHOD = 2

    dets = np.hstack((boxes, boxscores[:,np.newaxis]))
    keep = cpu_soft_nms(dets, Nt=nms_thresh, method=METHOD)

    # boxes2 = boxes.copy()
    # boxscores2 = boxscores.copy()
    dets2 = np.hstack((boxes, boxscores[:,np.newaxis]))
    keep2 = py_cpu_softnms(dets2, Nt=nms_thresh, method=METHOD)

    print(dets[keep])
    print(dets2[keep2])

