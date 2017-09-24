
import argparse
import cv2
import time
import imutils

class MotionDetector:
    def __init__(self, camera, min_area):
        self.camera = camera
        self.avg_bg = None
        self.min_area = min_area

    def run(self):
        while True:
            try:
                self.main()
            except KeyBoardInterruption:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    def main(self):
        grabbed, frame = self.camera.read()
        text = "Unoccupied"

        if not grabbed:
            break #reached end of video

        # resize, grayscale, and blur
        frame = imutils.resize(frame, width=500)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if self.avg_bg is None:
            self.avg_bg = gray.copy().astype('float')

        else:
            cv2.accumulateWeighted(gray, self.avg_bg, .5)
            self.detect(gray)

    def detect(self, frame):
        # compute absolute difference
        frame_delta = cv2.absdiff(frame, cv2.convertScaleAbs(self.avg_bg))
        # then threshold
        _, thresh = cv2.threshold(frame_delta, 15, 255, cv2.THRESH_BINARY)

        # dilate to fill in holes then find contours
        thresh = cv2.dilate(thresh, None, iterations=2)
        _, cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for c in cnts:
            if cv2.contourArea(c) > self.min_area:
                x, y, w, h = cv2.boundingRect(c)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Feed", frame)
        cv2.imshow("thresh", thresh)
        cv2.imshow("d", frame_delta)




if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-v", "--video", help="path to vid file")
    ap.add_argument("-a", "--min-area", type=int,
                     default=500, help="minimum area size")
    args = vars(ap.parse_args())

    if args.get("video") is None:
        camera =cv2.VideoCapture(0)
        time.sleep(.25)

    else:
        camera = cv2.VideoCapture(args["video"])

    MotionDetector(camera, args["min_area"]).run()
    camera.release()
    cv2.destroyAllWindows()
