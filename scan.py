import numpy as np
import cv2
import argparse
import imutils

from transform import four_point_transform

class Scan:
    def __init__(self, image):
        self.image = image
        self.orig = image.copy()

    def edge_detection(self):
        # Resize to 500, to speed up edge detection
        self.ratio = self.image.shape[0] / 500.0 # keep track of ratio of old height to new height
        self.image = imutils.resize(self.image, height=500)

        # Convert the image to grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # blur it
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        # find edges in image
        self.edged = cv2.Canny(gray, 75, 200)

    def find_contour(self):
        # simple assumption: largest contour with exactly four points
        # is our piece of paper to scan
        _, contours, _ = cv2.findContours(self.edged, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        # sort contours by calling contourArea() on them
        # then keep only the largest 5 (performance reasons)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]

        screen_contour = None
        cont = []
        for c in contours:
            # approximate the contour
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, .02 * peri, True)
            # if our approximated contour has 4 points,
            #  assume that we have found the it!
            cont.append(approx)
            if len(approx) == 4:
                screen_contour = approx
                break # don't need to check the rest of the contours

        if screen_contour is None:
            raise ValueError("Can't detect a contour with 4 points")

        self.screen_contour = screen_contour

    def transform_perspective(self):
        # transform image perspective by the screen contour coordinates
        warped = four_point_transform(self.orig, self.screen_contour.reshape(4, 2) * self.ratio)

        # convert to grayscale
        warped = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # threshold the image
        # cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C)
        warped = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 251, 11)

        self.final = warped

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="path to the image to be scanned")
    args = vars(ap.parse_args())
    image = cv2.imread(args["image"])
    scan = Scan(image)

    print("STEP 1: Edge Detection")
    scan.edge_detection()
    cv2.imshow("Image", scan.image)
    cv2.imshow("Edged", scan.edged)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("STEP 2: Find contours of paper")
    scan.find_contour()
    cv2.drawContours(scan.image, [scan.screen_contour], -1, (0, 255, 0), 2)
    cv2.imshow("Outline", scan.image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    print("STEP 3: Apply perspective transform")
    scan.transform_perspective()
    cv2.imshow("Original", imutils.resize(scan.orig, height=650))
    cv2.imshow("Scanned", imutils.resize(scan.final, height=650))
    cv2.waitKey(0)

if __name__ == "__main__":
    main()
