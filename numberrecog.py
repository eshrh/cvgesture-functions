import numpy as np
import cv2
import imutils
from sklearn.metrics import pairwise
import yaml
import keyboard
import mouse


class cvContour:
    # define a class to make contours sortable.
    def __init__(self, contour):
        self.contour = contour

    def __lt__(self, ocontour) -> bool:
        # might be other way around?
        return cv2.contourArea(self.contour) < cv2.contourArea(ocontour.contour)


def imageGrid(images, rows=2, columns=3, cell_width=320, cell_height=240):
    # Convert all gray images to BGR
    images = [
        (cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image)
        for image in images
    ]
    # Scale all images
    images = [cv2.resize(image, (cell_width, cell_height)) for image in images]
    # Write numbers onto images
    for image_i in range(len(images)):
        cv2.putText(
            images[image_i],
            f"{image_i}",
            (10, 50),
            cv2.FONT_HERSHEY_PLAIN,
            4,
            (255, 255, 255),
            2,
        )
    # Create empty images as needed
    for _ in range((rows * columns) - len(images)):
        images.append(np.zeros_like(images[0]))
    # Generate image rows
    img_rows = [
        cv2.hconcat([images[(columns * row) + column] for column in range(columns)])
        for row in range(rows)
    ]
    # Concatinate image rows and return
    return cv2.vconcat(img_rows)


class NumberRecognition:
    def __init__(self, thresh=15, camera=0):
        self.settings = self.parseSettings()

        # region of interest(roi)
        self.top, self.right, self.bottom, self.left = 10, 350, 225, 590

        self.accumulate = 0.5

        self.thresh = thresh
        self.vid = cv2.VideoCapture(camera)
        self.bg = None
        self.cnt = None

        self.thresholded = None
        self.hand_cnt = None
        self.chull = None

        # use the first 50 frames to get the running average of the bg.
        self.initWeights(50)

    # Gesture Recognition
    def bwroi(self) -> np.ndarray:
        """Crop out the roi (region of interest) from the full frame, blur it to make contouring easier."""
        frame = self.getFrame()
        cropped = frame[self.top : self.bottom, self.right : self.left]
        return cv2.GaussianBlur(cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY), (7, 7), 0)

    def contourDetect(self, frame: np.ndarray) -> tuple:
        """Compute thresholded and contours. Returns -1 if no contours found."""
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

    def countFingers(self, thresholded: np.ndarray, segmented: np.ndarray) -> int:
        """Counts the fingers held up on a hand."""
        # the algorithm
        # http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.454.3689&rep=rep1&type=pdf
        # compute complex hull
        chull = cv2.convexHull(segmented)
        self.chull = chull

        # find extrema of the hull
        extreme_top = tuple(chull[chull[:, :, 1].argmin()][0])
        extreme_bottom = tuple(chull[chull[:, :, 1].argmax()][0])
        extreme_left = tuple(chull[chull[:, :, 0].argmin()][0])
        extreme_right = tuple(chull[chull[:, :, 0].argmax()][0])

        # palm center
        cX = int((extreme_left[0] + extreme_right[0]) / 2)
        cY = int((extreme_top[1] + extreme_bottom[1]) / 2)
        self.cX,self.cY = cX,cY

        # Find max distance between the extrema
        distance = pairwise.euclidean_distances(
            [(cX, cY)], Y=[extreme_left, extreme_right, extreme_top, extreme_bottom]
        )[0]
        maximum_distance = distance[distance.argmax()]

        # estimate the palm as a circular region
        radius = int(0.8 * maximum_distance)

        circumference = 2 * np.pi * radius

        circular_roi = np.zeros(thresholded.shape[:2], dtype="uint8")
        cv2.circle(circular_roi, (cX, cY), radius, 255, 1)
        self.palm_circle = circular_roi

        # bitwise and the circle and the full image to get the fingers.
        circular_roi = cv2.bitwise_and(thresholded, thresholded, mask=circular_roi)
        self.severed_fingers = circular_roi

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

    def getContour(self) -> tuple:
        """Thin wrapper function for contourDetect. Returns the thresholded image and the largest contour if a contour is found, otherwise no return."""
        thresh_and_max_contour = self.contourDetect(self.bwroi())
        if thresh_and_max_contour != -1:
            return thresh_and_max_contour

    def getFrame(self) -> np.ndarray:
        """Reads the current frame, processes it's size, flips it, and updates the display and debug images."""
        _, frame = self.vid.read()
        if frame is not None:
            resized = imutils.resize(frame, width=700)
            frame = cv2.flip(imutils.resize(frame, width=700), 1)
            self.display_img = frame.copy()
            self.debug_img = frame.copy()
        return frame

    def initWeights(self, nframes: int) -> None:
        """Accumulates the first nframes to effectively remove the background."""
        for _ in range(nframes):
            frame = self.bwroi()
            # set the running average of the background
            if self.bg is None:
                self.bg = frame.copy().astype("float")
                continue
            cv2.accumulateWeighted(frame, self.bg, self.accumulate)

    # Command Execution
    def runCommand(self, command: str) -> None:
        """Run the given input simulation command."""
        if command == "lmb":
            mouse.click(button="left")
            print("leftclicked")
        elif command == "rmb":
            mouse.click(button="right")
        else:
            keyboard.press_and_release(command)
        return

    # User Display
    def drawDebug(self, additional_imgs: list = None) -> None:
        """Draws the debug image previously initialized in getFrame. Saved as debug_img."""
        # self.debug_img = cv2.cvtColor(self.debug_img, cv2.COLOR_BGR2GRAY)
        cv2.rectangle(
            self.debug_img,
            (self.left, self.top),
            (self.right, self.bottom),
            (0, 255, 0),
            2,
        )
        if self.cnt is not None:
            cv2.putText(
                self.debug_img,
                str(self.cnt),
                (70, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

        if self.chull is not None:
            # chull = np.array([[[cnt[0][0] + self.right, cnt[0][1]]] for cnt in self.chull])
            chull = self.chull
            thresholded = cv2.cvtColor(self.thresholded, cv2.COLOR_GRAY2BGR)
            #for contour_i in range(len(chull)):
                #cv2.drawContours(thresholded, chull, contour_i, (255,0,0), 8)
            cv2.drawContours(thresholded, chull, -1, (255, 0, 0), 8)
            # # Move hand contour right by self.right
            # hand_cnt = np.array([[[cnt[0][0] + self.right, cnt[0][1]]] for cnt in self.hand_cnt])
            # # Draw hand contour
            # cv2.drawContours(self.debug_img, hand_cnt, -1, (255,255,255), 2)

            # Draw palm circle
            #cv2.drawContours(thresholded, self.hand_cnt, -1, (0, 0, 255), 2)
            #
            # Add thresholded image to debug image grid.
            severed_fingers = cv2.cvtColor(self.severed_fingers, cv2.COLOR_GRAY2BGR)
            cv2.circle(severed_fingers, (self.cX,self.cY), 1, (0,255,255), 2)

            # print(np.where((severed_fingers==[255, 255, 255])))
            severed_fingers[
                np.where((severed_fingers == [255, 255, 255]).all(axis=2))
            ] = [0, 0, 255]

            self.debug_img = imageGrid(
                [severed_fingers, thresholded], rows=1, columns=2
            )

    def drawDisplay(self) -> None:
        """Draws the display image previously initialized in getFrame. Saved as display_img."""
        cv2.rectangle(
            self.display_img,
            (self.left, self.top),
            (self.right, self.bottom),
            (0, 255, 0),
            2,
        )
        if self.cnt is not None:
            cv2.putText(
                self.display_img,
                str(self.cnt),
                (70, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2,
            )

    def show(self, debug: bool = True) -> None:
        """Launches and updates the video window."""
        self.drawDisplay()
        if debug == True:
            self.drawDebug()
            width = self.display_img.shape[1]
            height = self.display_img.shape[0]
            cv2.imshow(
                "OpenGesture - Debug",
                imageGrid(
                    [self.display_img, self.debug_img],
                    rows=1,
                    columns=2,
                    cell_width=width,
                    cell_height=height,
                ),
            )
        else:
            cv2.imshow("OpenGesture", self.display_img)
        if cv2.waitKey(1) == ord("q"):
            self.kill()

    # User Settings
    def parseSettings(self) -> dict:
        """Returns the settings.yaml file as a dictionary."""
        settings = None
        with open("settings.yaml", "r") as f:
            settings = yaml.safe_load(f)
        return settings

    # NumberRecognition API
    def kill(self) -> None:
        """Stop video stream and close windows"""
        self.vid.release()
        cv2.destroyAllWindows()

    def run(self) -> None:
        """Main loop of NumberRecognition."""
        totalcnt = 0
        cur = 0
        prevcnt = -1
        while True:
            # Updates frame clone and retrieves contoured image.
            contour = self.getContour()
            if contour:
                self.thresholded, self.hand_cnt = contour
                # use the average finger count of the past 5 frames.
                totalcnt += self.countFingers(
                    self.thresholded.copy(), self.hand_cnt.copy()
                )
                if cur >= 5:
                    self.cnt = round(totalcnt / cur)
                    cur = 0
                    totalcnt = 0
                if self.cnt != prevcnt:
                    if self.cnt != 0 and self.cnt in self.settings:
                        self.runCommand(self.settings[self.cnt])
                    prevcnt = self.cnt
            else:
                self.cnt = 0
                cur = 0
                totalcnt = 0
            cur += 1
            self.show()


numrec = NumberRecognition(thresh=45, camera=0)
numrec.run()
