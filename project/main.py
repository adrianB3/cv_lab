import cv2
import os
import sys

VIDEO_PATH = os.path.abspath("D:/recs/crackedLensIon/My Recording_4.mp4")


class System:
    def __init__(self):
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            raise FileNotFoundError("Video file not found.")
        self.algo = Algo()

    def process(self):
        while self.capture.isOpened():
            ret, frame = self.capture.read()

            if ret:
                self.algo.process(frame)
                cv2.imshow("System", frame)
                if cv2.waitKey(25) & 0xFF == ord('q'):
                    break
            else:
                break
        self.capture.release()
        cv2.destroyAllWindows()


class Algo:
    def __init__(self):
        pass

    def process(self, frame):
        pass


if __name__ == "__main__":
    sys = System()
    sys.process()
