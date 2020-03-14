import cv2
import numpy as np
from queue import Queue

VIDEO_PATH = "input.mp4"


def affine_linear_transform(x, range_a: tuple, range_b: tuple):
    y = (((x - range_a[0]) * (range_b[1] - range_b[0])) / (range_a[1] - range_a[0])) + range_b[0]
    return y


class Animation:
    def __init__(self):
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.current_frame_nb = 0
        self.prev_frame_delay = 2

    def process(self):
        frame_buff = Queue(maxsize=self.prev_frame_delay)
        prev_frame = None
        delay_counter = 0
        isFirstSeq = True
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            self.current_frame_nb += 1

            self.add_pulsating_circle(frame, debug_vis=True)
            self.add_translating_line(frame, debug_vis=True)
            # self.add_text_seq(frame, "OpenCV")
            self.add_frame_counter(frame)
            frame_buff.put(frame.copy()) # TODO
            if delay_counter == self.prev_frame_delay:
                # prev_frame = frame_buff.pop()
                # delay_counter = 0
                isFirstSeq = False
            else:
                delay_counter += 1
            if not isFirstSeq:
                self.add_prev_frame_over(frame=frame, prev_frame=frame_buff.get(), roi_dim=(250, 250))
            cv2.imshow("Animation", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            cv2.waitKey(1000)

        self.cap.release()
        cv2.destroyAllWindows()

    def add_pulsating_circle(self, frame, debug_vis=False):
        radius = affine_linear_transform(self.current_frame_nb, (0, self.frame_count), (5, 50))
        center = (int(self.frame_width / 2), int(self.frame_height / 2))
        cv2.circle(img=frame,
                   center=center,
                   radius=int(radius),
                   color=(0, 255, 0),
                   thickness=1)
        if debug_vis:
            cv2.putText(img=frame,
                        text="Circle radius: {0:.2f}".format(radius),
                        org=center,
                        color=(0, 255, 0),
                        thickness=1,
                        fontFace=cv2.QT_FONT_NORMAL, fontScale=0.4,
                        lineType=cv2.LINE_AA)

    def add_translating_line(self, frame, debug_vis=False):
        line_dim = 100
        pt_y = affine_linear_transform(self.current_frame_nb, (0, self.frame_count), (0, self.frame_height))
        cv2.line(img=frame,
                 pt1=(int(self.frame_width / 2) - line_dim, int(pt_y)),
                 pt2=(int(self.frame_width / 2) + line_dim, int(pt_y)),
                 color=(255, 0, 0),
                 thickness=1)
        if debug_vis:
            cv2.putText(img=frame,
                        text="y = {0:.2f}".format(pt_y),
                        org=(int(self.frame_width / 2) + line_dim + 20, int(pt_y)),
                        color=(0, 255, 0),
                        thickness=1,
                        fontFace=cv2.QT_FONT_NORMAL, fontScale=0.4,
                        lineType=cv2.LINE_AA)

    def add_text_seq(self, frame, text):
        # TODO
        if self.current_frame_nb % self.fps == 0:
            for ch in text:
                cv2.putText(img=frame,
                            text=ch,
                            org=(100, 100),
                            color=(255, 0, 0),
                            thickness=2,
                            fontFace=cv2.QT_FONT_NORMAL, fontScale=1,
                            lineType=cv2.LINE_AA)

    def add_prev_frame_over(self, frame, prev_frame, roi_dim: tuple = (100, 100)):
        prev = prev_frame.copy()
        if prev is not None:
            prev = cv2.resize(src=prev_frame, dsize=roi_dim)
            # cv2.imshow("prev", prev)
            frame[self.frame_height - roi_dim[0]: self.frame_height, self.frame_width - roi_dim[1]: self.frame_width, :] = prev

    def add_frame_counter(self, frame):
        cv2.putText(img=frame,
                    text=str(self.current_frame_nb),
                    org=(10, 20),
                    color=(0, 255, 0),
                    thickness=1,
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=0.5,
                    lineType=cv2.LINE_AA)


if __name__ == "__main__":
    anim = Animation()
    anim.process()
