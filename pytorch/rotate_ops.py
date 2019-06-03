import numpy as np
import cv2

def convert_rect_to_pts(anchor):
    x_c, y_c, w, h, theta = anchor
    rect = ((x_c, y_c), (w, h), theta)
    rect = cv2.boxPoints(rect)
    # rect = np.int0(np.round(rect))
    return rect

def get_bounding_box(pts):
    """
    pts: (N, 2) array
    """
    bbox = np.zeros(4, dtype=pts.dtype)
    bbox[:2] = np.min(pts, axis=0)
    bbox[2:] = np.max(pts, axis=0)
    return bbox

def get_random_color():
    return (np.random.randint(255), np.random.randint(255), np.random.randint(255))

def draw_anchors(img, anchors, color_list=[], fill=False, line_sz=2):
    """
    img: (H,W,3) np.uint8 array
    anchors: (N,5) np.float32 array, where each row is [xc,yc,w,h,angle]
    """
    if isinstance(color_list, tuple):
        color_list = [color_list]

    img_copy = img.copy()
    Nc = len(color_list)
    N = len(anchors)
    if Nc == 0:
        color_list = [get_random_color() for a in anchors]
    elif Nc != N:
        color_list = [color_list[n % Nc] for n in range(N)]

    for ix,anchor in enumerate(anchors):
        color = color_list[ix]
        rect = anchor
        if len(anchor) != 8:
            rect = convert_rect_to_pts(anchor)
        rect = np.round(rect).astype(np.int32)
        if fill:
            cv2.fillConvexPoly(img_copy, rect, color)
        else:
            cv2.drawContours(img_copy, [rect], 0, color, line_sz)
    return img_copy

def get_rotated_roi_pixel_mapping(roi):
    assert len(roi) == 5  # xc yc w h angle

    xc, yc, w, h, angle = roi

    center = (xc, yc)
    theta = np.deg2rad(angle)

    # paste mask onto image via rotated rect mapping
    v_x = (np.cos(theta), np.sin(theta))
    v_y = (-np.sin(theta), np.cos(theta))
    s_x = center[0] - v_x[0] * ((w - 1) / 2) - v_y[0] * ((h - 1) / 2)
    s_y = center[1] - v_x[1] * ((w - 1) / 2) - v_y[1] * ((h - 1) / 2)

    M = np.array([[v_x[0], v_y[0], s_x],
                  [v_x[1], v_y[1], s_y]])

    return M

def paste_rotated_roi_in_image(image, roi_image, roi):
    assert len(roi) == 5  # xc yc w h angle

    w = roi[2]
    h = roi[3]

    w = int(np.round(w))
    h = int(np.round(h))
    rh, rw = roi_image.shape[:2]
    if rw != w or rh != h:
        roi_image = cv2.resize(roi_image, (w, h))

    # generate the mapping of points from roi_image to an image
    M = get_rotated_roi_pixel_mapping(roi)

    x_grid, y_grid = np.meshgrid(np.arange(w), np.arange(h))
    x_grid = x_grid.reshape(-1)
    y_grid = y_grid.reshape(-1)
    map_pts_x = x_grid * M[0, 0] + y_grid * M[0, 1] + M[0, 2]
    map_pts_y = x_grid * M[1, 0] + y_grid * M[1, 1] + M[1, 2]
    map_pts_x = np.round(map_pts_x).astype(np.int32)
    map_pts_y = np.round(map_pts_y).astype(np.int32)

    # stick onto image
    im_h, im_w = image.shape[:2]

    valid_x = np.logical_and(map_pts_x >= 0, map_pts_x < im_w)
    valid_y = np.logical_and(map_pts_y >= 0, map_pts_y < im_h)
    valid = np.logical_and(valid_x, valid_y)
    image[map_pts_y[valid], map_pts_x[valid]] = roi_image[y_grid[valid], x_grid[valid]]

    # close holes that arise due to rounding from the pixel mapping phase
    kernel = np.ones((5, 5), np.uint8)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    return image