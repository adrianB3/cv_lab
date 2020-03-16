import cv2
import os
import sys
import numpy as np
from queue import Queue

VIDEO_PATH = os.path.abspath("E:\\recs\\blockage_part.mp4")
isIonsRecs = False
patch_size = 1
ncc_mask_acc_no = 3


class System:
    def __init__(self):
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            raise FileNotFoundError("Video file not found.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.algo = Algo(640, 320)
        self.frameCountGlobal = 0
        self.prev_frame_delay = 30
        self.isPlaying = True

    def process(self):
        frame_buff = Queue(maxsize=self.prev_frame_delay + 1)
        delay_counter = 0
        is_first_seq = True

        while self.capture.isOpened():
            # if cv2.waitKey(1) & 0xFF == ord('c'):  # Continue
            #     self.isPlaying = True
            if self.isPlaying:
                ret, current_frame = self.capture.read()
                if ret:
                    self.frameCountGlobal += 1
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    if isIonsRecs:
                        current_frame = current_frame[0:current_frame.shape[0], 0:current_frame.shape[1] - 192]
                    current_frame = cv2.resize(current_frame, (640, 320))
                    frame_buff.put(current_frame.copy())
                    self.add_frame_nb(current_frame)
                    if delay_counter == self.prev_frame_delay:
                        is_first_seq = False
                    else:
                        delay_counter += 1

                    if not is_first_seq:
                        self.algo.accumulate_ncc_masks(prev_frame=frame_buff.get(), frame=current_frame)
                        cv2.imshow("Processed", self.algo.accumulated_mask)
                    cv2.imshow("Current", current_frame)
                    # if cv2.waitKey(45) & 0xFF == ord('p'):  # Pause
                    #     self.isPlaying = False

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break
                else:
                    break
        cv2.waitKey()
        self.capture.release()
        cv2.destroyAllWindows()

    def add_frame_nb(self, frame):
        cv2.putText(img=frame,
                    text=str(self.frameCountGlobal),
                    org=(10, 20),
                    color=(255, 255, 255),
                    thickness=1,
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=0.5,
                    lineType=cv2.LINE_AA)


class Algo:
    def __init__(self, width, height):
        self.patchSize = patch_size
        self.accumulated_mask = np.zeros((height, width))

    def corr_coef(self, patch1, patch2, frame):
        product = (patch1 - patch1.mean()) * (patch2 - patch2.mean())
        stds = patch1.std() * patch2.std()
        sumss = np.sum(product/stds)
        if stds == 0:
            return 0
        else:
            coeff = sumss / (frame.shape[0] * frame.shape[1])
            return coeff

    def get_ncc_mask(self, prev_frame, frame):
        ncc_mask = np.zeros_like(frame)
        for i in range(self.patchSize, frame.shape[0] - (self.patchSize + 1)):
            for j in range(self.patchSize, frame.shape[1] - (self.patchSize + 1)):
                ncc_mask[i, j] = self.corr_coef(prev_frame[
                                                i - self.patchSize: i + self.patchSize + 1,
                                                j - self.patchSize: j + self.patchSize + 1],
                                                frame[
                                                i - self.patchSize: i + self.patchSize + 1,
                                                j - self.patchSize: j + self.patchSize + 1], frame)
        return ncc_mask

    def accumulate_ncc_masks(self, prev_frame, frame):
        current_mask = self.get_ncc_mask(prev_frame, frame)
        self.accumulated_mask += current_mask


if __name__ == "__main__":
    sys = System()
    sys.process()
