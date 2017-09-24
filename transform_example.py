from transform import four_point_transform
import numpy as np
import cv2
import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", help="path to the image file")
ap.add_argument("-c", "--coords", help="comm separated list of source points, example: [[5,4],[1,2],[3,8],[3,2]")
args = vars(ap.parse_args())

image = cv2.imread(args["image"])
pts = np.array(json.loads(args["coords"]), dtype="float32")

warped = four_point_transform(image, pts)

cv2.imshow("Original", image)
cv2.imshow("Warped", warped)
cv2.waitKey(0)