import cv2
import os
import sys
import numpy as np
from queue import Queue

VIDEO_PATH = os.path.abspath("D:/recs/adis/vid2.mp4")
FRAMES_BUFFER_LIMIT = 30
isIonsRecs = False
patch_size = 2
ncc_mask_acc_no = 3


class System:
    def __init__(self):
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            raise FileNotFoundError("Video file not found.")
        self.algo = Algo()
        self.frameCountGlobal = 0
        self.prevFrame = None
        self.isPlaying = True

    def process(self):
        while self.capture.isOpened():
            # if cv2.waitKey(1) & 0xFF == ord('c'):  # Continue
            #     self.isPlaying = True
            if self.isPlaying:
                ret, current_frame = self.capture.read()

                if ret:

                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    if isIonsRecs:
                        current_frame = current_frame[0:current_frame.shape[0], 0:current_frame.shape[1] - 192]
                    current_frame = cv2.resize(current_frame, (640, 320))
                    self.frameCountGlobal += 1
                    if self.frameCountGlobal == 1:
                        self.prevFrame = current_frame
                    if self.frameCountGlobal == FRAMES_BUFFER_LIMIT:
                        proc_frame = self.algo.process(prev_frame=self.prevFrame, frame=current_frame)
                        self.frameCountGlobal = 0
                        self.prevFrame = None
                        cv2.imshow("Processed", proc_frame)
                        cv2.imshow("Accumulated", self.algo.get_acc(current_frame))
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


class Algo:
    def __init__(self):
        self.patchSize = patch_size
        self.acc_count = 0
        self.ncc_masks_acc = []

    def corr_coef(self, patch1, patch2, frame):
        product = (patch1 - patch1.mean()) * (patch2 - patch2.mean())
        stds = patch1.std() * patch2.std()
        sumss = np.sum(product/stds)
        if stds == 0:
            return 0
        else:
            coeff = sumss / (frame.shape[0] * frame.shape[1])
            return coeff

    def process(self, prev_frame, frame):
        ncc_mask = np.zeros_like(frame)
        for i in range(self.patchSize, frame.shape[0] - (self.patchSize + 1)):
            for j in range(self.patchSize, frame.shape[1] - (self.patchSize + 1)):
                ncc_mask[i, j] = self.corr_coef(prev_frame[
                                                i - self.patchSize: i + self.patchSize + 1,
                                                j - self.patchSize: j + self.patchSize + 1],
                                                frame[
                                                i - self.patchSize: i + self.patchSize + 1,
                                                j - self.patchSize: j + self.patchSize + 1], frame)
        self.ncc_masks_acc.append(ncc_mask)
        return ncc_mask

    def get_acc(self, frame):
        ncc_mask_acc = np.zeros_like(frame)
        for mask in self.ncc_masks_acc:
            ncc_mask_acc += mask
        return ncc_mask_acc


if __name__ == "__main__":
    sys = System()
    sys.process()
