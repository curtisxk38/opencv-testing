import argparse
import cv2
import time
import imutils

class Motion2:
    def __init__(self, camera, min_area, show):
        self.camera = camera
        self.min_area = min_area
        self.average = None
        self.show = show

    def run(self):
        while True:
            grabbed, frame = self.camera.read()

            if not grabbed:
                break

            frame = self.process_frame(frame)
            if self.average is None:
                self.average = frame.copy().astype("float")
            else:
                cv2.accumulateWeighted(frame, self.average, .05)
                self.process_contours(frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                    break

    def process_frame(self, frame):
        # resize for faster processing
        frame = imutils.resize(frame, width=500)
        # convert to grayscale
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # blur
        frame = cv2.GaussianBlur(frame, (21, 21), 0)
        return frame

    def process_contours(self, frame):
        # compute absolute difference
        frame_delta = cv2.absdiff(frame, cv2.convertScaleAbs(self.average))
        # then threshold
        _, thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)

        thresh = cv2.dilate(thresh, None, iterations=15)
        thresh = cv2.erode(thresh, None, iterations=10)

        _, contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        large_contours = []

        for con in contours:
            if cv2.contourArea(con) > self.min_area:
                x, y, w, h = cv2.boundingRect(con)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                large_contours.append(con)

        cv2.drawContours(frame, large_contours, -1, (0, 255, 255), 2)

        if self.show:
            cv2.imshow("Feed", frame)
            cv2.imshow("thresh", thresh)
            cv2.imshow("delta", frame_delta)



if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-a", "--min-area", type=int,
                     default=800, help="minimum area size")
    ap.add_argument("-s", "--show", action="store_true",
                     help="Whether to show the processed images")
    args = ap.parse_args()

    camera = cv2.VideoCapture(0)

    Motion2(camera, args.min_area, args.show).run()

    camera.release()
    cv2.destroyAllWindows()