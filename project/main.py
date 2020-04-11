import cv2
import os
import sys
import numpy as np
from queue import Queue
import enum


class Mode(enum.Enum):
    compute_every_frame = 1
    compute_first_last_frame = 2
    compute_with_fixed_frame = 3


VIDEO_PATH = os.path.abspath("E:\\recs\\clear.mp4")
isIonsRecs = False
g_patch_size = 1
ncc_mask_acc_no = 10
current_mode = Mode.compute_every_frame
thresh_clear = 55000


class System:
    def __init__(self):
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            raise FileNotFoundError("Video file not found.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        self.algo = Algo(90, 64, g_patch_size)
        self.frameCountGlobal = 0
        self.frame_delay = 10
        self.isPlaying = True

    def process(self):
        frame_buff = Queue(maxsize=self.frame_delay + 1)
        delay_counter = 0
        is_first_seq = True
        is_seq_complete = False
        fixed = None
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
                    current_frame = cv2.resize(current_frame, (90, 64))
                    frame_buff.put(current_frame.copy())
                    to_show = current_frame.copy()
                    self.add_frame_nb(to_show)
                    if self.frameCountGlobal == self.frame_delay:
                        is_first_seq = False

                    if delay_counter == self.frame_delay:
                        delay_counter = 0
                        is_seq_complete = True
                    else:
                        delay_counter += 1
                        is_seq_complete = False

                    if current_mode == Mode.compute_every_frame:
                        if not is_first_seq:
                            prev = frame_buff.get()
                            cv2.imshow("F1", cv2.resize(to_show, (640, 320)))
                            cv2.imshow("F2", cv2.resize(prev, (640, 320)))
                            res = self.algo.process(prev, current_frame)
                            cv2.imshow("Processed", cv2.resize(res, (640, 320)))

                    if current_mode == Mode.compute_first_last_frame:
                        if is_seq_complete:
                            prev = frame_buff.get()
                            cv2.imshow("F1", cv2.resize(to_show, (640, 320)))
                            cv2.imshow("F2", cv2.resize(prev, (640, 320)))
                            res = self.algo.process(prev, current_frame)
                            cv2.imshow("Processed", cv2.resize(res, (640, 320)))
                            frame_buff.queue.clear()

                    if current_mode == Mode.compute_with_fixed_frame:
                        if not is_seq_complete:
                            cv2.imshow("F1", cv2.resize(to_show, (640, 320)))
                            if delay_counter == 1:
                                fixed = current_frame.copy()
                            else:
                                cv2.imshow("F2", cv2.resize(fixed, (640, 320)))
                                res = self.algo.process(fixed, current_frame)
                                cv2.imshow("Processed", cv2.resize(res, (640, 320)))
                        else:
                            frame_buff.queue.clear()

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
                    org=(5, 5),
                    color=(255, 255, 255),
                    thickness=1,
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=0.1,
                    lineType=cv2.LINE_AA)


class Algo:
    def __init__(self, width, height, patch_size):
        self.patchSize = patch_size
        self.accumulated_mask = np.zeros((height, width))
        self.acc_count = 0

    def corr_coef(self, patch1, patch2, frame):
        product = (patch1 - patch1.mean()) * (patch2 - patch2.mean())
        stds = patch1.std() * patch2.std()
        sumss = np.sum(product / stds)
        if stds == 0:
            return 0
        else:
            coeff = sumss / (frame.shape[0] * frame.shape[1])
            return coeff

    def intensity_dif(self, pix1, pix2):
        return np.abs(pix1 - pix2)

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

    def get_ncc_mask_cv2(self, prev_frame, frame):
        ncc_mask = np.zeros_like(frame)
        for i in range(self.patchSize, frame.shape[0] - (self.patchSize + 1)):
            for j in range(self.patchSize, frame.shape[1] - (self.patchSize + 1)):
                ncc_mask[i, j] = cv2.matchTemplate(prev_frame[
                                                   i - self.patchSize: i + self.patchSize + 1,
                                                   j - self.patchSize: j + self.patchSize + 1],
                                                   frame[
                                                   i - self.patchSize: i + self.patchSize + 1,
                                                   j - self.patchSize: j + self.patchSize + 1],
                                                   cv2.TM_CCOEFF_NORMED)
        return ncc_mask

    def process(self, prev_frame, frame):
        self.acc_count += 1
        if self.acc_count == ncc_mask_acc_no:
            self.accumulated_mask = np.zeros((frame.shape[0], frame.shape[1]))
            self.acc_count = 0
        mask = self.get_ncc_mask_cv2(prev_frame, frame)
        cv2.accumulateWeighted(mask, self.accumulated_mask, 0.5)
        cv2.convertScaleAbs(self.accumulated_mask, self.accumulated_mask)
        ret, thresh = cv2.threshold(self.accumulated_mask, 0, 255, cv2.THRESH_BINARY)
        sum = np.sum(thresh)
        status = ""
        if sum < thresh_clear:
            status = "clear"
        if sum > thresh_clear:
            status = "artifacts"
        print(str(sum) + ": " + status)
        return thresh


if __name__ == "__main__":
    sys = System()
    sys.process()
