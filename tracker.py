import argparse
import cv2
import numpy

def select_roi(event, x, y, flags, param):
    # param refers to the Tracker object
    if param.input_mode and event == cv2.EVENT_LBUTTONDOWN and len(param.roi_points) < 4:
        param.roi_points.append((x, y))
        cv2.circle(param.frame, (x, y), 4, (0, 255, 0), 2)
        cv2.imshow("frame", param.frame)

class Tracker:
    def __init__(self, camera):
        self.camera = camera
        self.frame = None # current frame to process
        self.roi_points = [] # list of points corresponding to region of interest
        self.input_mode = False # whether we are currently selecting the object to track
        # termination criteria for camshift
        #  maximum of ten iterations or movement by at least one pixel
        #  along bounding box of ROI
        self.termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)
        # setup mouse callback
        cv2.namedWindow("frame")
        cv2.setMouseCallback("frame", select_roi, param=self)

    def run(self):
        
        roi_box = None
        roi_hist = None
        while True:
            grabbed, self.frame = self.camera.read()

            if not grabbed:
                break

            if roi_box is not None:
                self.process_roi(roi_box, roi_hist)

            cv2.imshow("frame", self.frame)
            key = cv2.waitKey(1) & 0xFF

            # if i key is pressed go into ROI selection mode
            if key == ord("i") and len(self.roi_points) < 4:
                self.input_mode = True
                original = self.frame.copy()

                # wait for 4 points to be selected
                while len(self.roi_points) < 4:
                    cv2.imshow("frame", self.frame)
                    cv2.waitKey(0)

                roi_box, roi_hist = self.process_input(original)

            elif key == ord("q"):
                break

    def process_input(self, original):
        # determine top-left and bottom-right points
        self.roi_points = numpy.array(self.roi_points)
        points_sum = self.roi_points.sum(axis=1)
        tl = self.roi_points[numpy.argmin(points_sum)]
        br = self.roi_points[numpy.argmax(points_sum)]

        # grab ROI for boundinb box and covert to HSV
        roi = original[tl[1]:br[1], tl[0]:br[0]]
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # compute histogram and store the bounding box
        roi_hist = cv2.calcHist([roi], [0], None, [16], [0, 180])
        roi_hist = cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
        roi_box = (tl[0], tl[1], br[0], br[1])

        return roi_box, roi_hist


    def process_roi(self, roi_box, roi_hist):
        # convert current frame to HSV color space and perform mean shift
        hsv = cv2.cvtColor(self.frame, cv2.COLOR_BGR2HSV)
        # only use the hue component for histogram back projection
        back_proj = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

        # apply cam shift to the back projection, convert
        #  the points to a bounding box and draw them
        # returns: 1st: estimated position, size, orientation of object
        #   2nd: newly estimated position of ROI
        r, roi_box = cv2.CamShift(back_proj, roi_box, self.termination)
        points = numpy.int0(cv2.boxPoints(r))
        cv2.polylines(self.frame, [points], True, (0, 255, 0), 2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", required=False, help="path to video file")
    args = ap.parse_args()
    if args.video is not None:
        camera = cv2.VideoCapture(args.video)
    else:
        camera = cv2.VideoCapture(0)

    t = Tracker(camera)
    t.run()
    camera.release()
    cv2.destroyAllWindows()