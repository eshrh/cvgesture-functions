import numpy as np
import cv2
import imutils
from sklearn.metrics import pairwise


class cvContour:
    # define a class to make contours sortable.
    def __init__(self, contour):
        self.contour = contour

    def __lt__(self, ocontour):
        # might be other way around?
        return cv2.contourArea(self.contour) < cv2.contourArea(ocontour.contour)


class NumberRecognition:
    def __init__(self, thresh=15):
        self.top, self.right, self.bottom, self.left = 10, 350, 225, 590
        # region of interest(roi)

        self.accumulate = 0.5
        self.thresh = thresh
        self.vid = cv2.VideoCapture(0)
        self.bg = None
        self.cnt = None

        # use the first 50 frames to get the running average of the bg.
        self.initweights(50)

        totalcnt = 0
        cur = 0

        while True:
            contour = self.getContour()
            if contour:
                thresholded, segmented = contour

                # use the average finger count of the past 5 frames.
                totalcnt += self.count(thresholded, segmented)
                if cur >= 5:
                    self.cnt = round(totalcnt / cur)
                    cur = 0
                    totalcnt = 0
            cur += 1
            self.draw()

    def getContour(self):
        # basically just a thin wrapper func.
        maxcontour = self.contourDetect(self.bwroi())
        if maxcontour is not -1:
            return maxcontour

    def bwroi(self):
        # crop out the roi from the full frame, blur it to make contouring easier.
        frame = self.getframe()
        cropped = frame[self.top : self.bottom, self.right : self.left]
        return cv2.GaussianBlur(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    def getframe(self):
        # general processing
        _, frame = self.vid.read()
        if frame is not None:
            resized = imutils.resize(frame, width=700)
            frame = cv2.flip(imutils.resize(frame, width=700), 1)
            self.clone = frame.copy()
        return frame

    def initweights(self, nframes):
        # wrapper for accbg
        for i in range(nframes):
            self.accbg(self.bwroi())

    def accbg(self, frame):
        # set the running average of the background
        if self.bg is None:
            self.bg = frame.copy().astype("float")
            return 0

        cv2.accumulateWeighted(frame, self.bg, self.accumulate)

    def contourDetect(self, frame):
        # compute thresholded and contours
        diff = cv2.absdiff(self.bg.astype("uint8"), frame)
        thresholded = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)[1]
        contours, hierarchy = cv2.findContours(
            thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        # lots of overhead...
        if contours == []:
            return -1
        # pick the biggest contour
        return (thresholded, max([cvContour(i) for i in contours]).contour)

    def draw(self):
        cv2.rectangle(
            self.clone, (self.left, self.top), (self.right, self.bottom), (0, 255, 0), 2
        )
        if self.cnt is not None:
            cv2.putText(
                self.clone,
                str(self.cnt),
                (70, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        cv2.imshow("opencv", self.clone)
        cv2.waitKey(1)

    def count(self, thresholded, segmented):
        # the algorithm
        # compute complex hull
        chull = cv2.convexHull(segmented)

        # find extrema of the hull
        extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
        extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
        extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

        # palm center
        cX = int((extreme_left[0] + extreme_right[0]) / 2)
        cY = int((extreme_top[1] + extreme_bottom[1]) / 2)

        distance = pairwise.euclidean_distances(
            [(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
        )[0]
        maximum_distance = distance[distance.argmax()]

        # estimate the palm as a circular region
        radius = int(0.8 * maximum_distance)
        circumference = 2 * np.pi * radius
        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (cX, cY), radius, 255, 1)

        # bitwise and the circle and the full image to get the fingers.
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
        cnts, _ = cv2.findContours(
            circular_roi.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        count = 0
        for c in cnts:
            (x, y, w, h) = cv2.boundingRect(c)

            # it is a finger only if it is outside the palm and not below the palm.
            if ((cY + (cY * 0.25)) > (y + h)) and ((circumference * 0.25) > c.shape[0]):
                count += 1

        return count

    def kill(self):
        self.vid.release()
        cv2.destroyAllWindows()


numrec = NumberRecognition()
