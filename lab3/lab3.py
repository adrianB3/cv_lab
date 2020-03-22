import cv2
import numpy as np

FOREGROUND_VIDEO_PATH = "drone.mp4"
BACKGROUND_VIDEO_PATH = "earth_view.mp4"
PROCESSED_VIDEO_PATH = "output1.mp4"
PROCESSED_VIDEO_PATH_2 = "output2.mp4"


class GreenScreen:
    def __init__(self):
        self.cap1 = cv2.VideoCapture(FOREGROUND_VIDEO_PATH)
        self.cap2 = cv2.VideoCapture(BACKGROUND_VIDEO_PATH)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.frame_count = int(self.cap1.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap1.get(cv2.CAP_PROP_FPS)
        # will use foreground video as size ref
        self.out = cv2.VideoWriter(PROCESSED_VIDEO_PATH, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.out2 = cv2.VideoWriter(PROCESSED_VIDEO_PATH_2, fourcc, self.fps, (self.frame_width, self.frame_height))
        self.frame = np.zeros((self.frame_width, self.frame_height))
        self.frame2 = np.zeros((self.frame_width, self.frame_height))
        self.lower_green = (36, 50, 50)
        self.upper_green = (86, 255, 255)

    def process(self):
        while True:
            ret1, foreground_frame = self.cap1.read()
            ret2, background_frame = self.cap2.read()
            if not ret1 or not ret2:
                break
            background_frame = cv2.resize(background_frame, (self.frame_width, self.frame_height))
            self.frame = self.replace_background(foreground_frame, background_frame)

            back_frame_th = self.threshold(background_frame)
            for_frame_blur = self.add_blur(foreground_frame)

            self.frame2 = self.replace_background(for_frame_blur, back_frame_th)

            cv2.imshow("Video", self.frame)
            cv2.imshow("Video2", self.frame2)
            self.out.write(self.frame)
            self.out2.write(self.frame2)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        self.cap1.release()
        self.cap2.release()
        self.out.release()
        self.out2.release()
        cv2.destroyAllWindows()

    def replace_background(self, foreground, background):
        hsv = cv2.cvtColor(foreground, cv2.COLOR_BGR2HSV)
        green_mask = cv2.inRange(hsv, self.lower_green, self.upper_green)
        foreground_cpy = foreground.copy()
        background_cpy = background.copy()
        foreground_cpy[green_mask > 0] = [0, 0, 0]
        if len(background_cpy.shape) == 2:
            background_cpy = cv2.cvtColor(background_cpy, cv2.COLOR_GRAY2BGR)
        background_cpy[green_mask == 0] = [0, 0, 0]
        foreground_cpy += background_cpy
        return foreground_cpy

    def add_blur(self, frame):
        kernel = np.ones((15, 15), np.float32) / 225
        blurred = cv2.filter2D(frame, -1, kernel)
        return blurred

    def threshold(self, frame):
        grayscaled = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        th = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)
        return th


if __name__ == "__main__":
    grs = GreenScreen()
    grs.process()
