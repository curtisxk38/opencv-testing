import numpy as np
import cv2

def order_points(pts):
    """Takes a numpy array (4, 2) of coordinates of a rectangle
        and orders it in [top left, top right, bottom right, bottom left]
        """
    # to return
    rect = np.zeros((4, 2), dtype="float32")

    # For each point in pts, find x+y
    pts_sum = pts.sum(axis=1)
    rect[0] = pts[np.argmin(pts_sum)] # the top left will have the lowest sum
    rect[2] = pts[np.argmax(pts_sum)] # the bottom right will have the highest sum

    # For each point in pts, find y-x
    pts_diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(pts_diff)] # the top right will have smallest difference
    rect[3] = pts[np.argmax(pts_diff)] # the bottom left will have the largest difference

    return rect

def four_point_transform(image, pts):
    """Given an image and a numpy array of points,
        transform the persepctive so that the rectangle described by pts is
        viewed head on"""
    rect = order_points(pts)
    tl, tr, br, bl = rect

    # compute the width of the new image, which will be the
    # maximum distance between the bottom right and bottom left
    # x coords or the top right and top left x coords
    width_A = np.linalg.norm(br - bl)
    width_B = np.linalg.norm(tr - tl)
    max_width = int(max(width_A, width_B))
    
    # likewise for the y, the height of the new image is the maximum between the two height measurements
    height_A = np.linalg.norm(tl - bl)
    height_B = np.linalg.norm(tr - br)
    max_height = int(max(height_A, height_B))

    # now, with the new dimensions construct destination points
    dest = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    # compute perspective transform matrix
    matrix = cv2.getPerspectiveTransform(rect, dest)
    # apply transform
    warped = cv2.warpPerspective(image, matrix, (max_width, max_height))
    return warped