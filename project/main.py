import cv2
import os
import sys
import numpy as np
from queue import Queue
import enum
import time
import matplotlib.pyplot as plt

class Mode(enum.Enum):
    compute_every_frame = 1
    compute_first_last_frame = 2
    compute_with_fixed_frame = 3


class AlgoType(enum.Enum):
    ncc_algo = 1
    b_dist_algo = 2


VIDEO_PATH = os.path.abspath("D:\\recs\\test\\cc0ffeed-291914b0.mov")
isIonsRecs = False
g_patch_size = 1
ncc_mask_acc_no = 10
frame_delay = 50
current_mode = Mode.compute_every_frame
current_algo = AlgoType.ncc_algo
thresh_clear = 55000
width = 240
height = 160

topLeft = (20, 20)
bottomRight = (100, 250)
x, y = topLeft[0], topLeft[1]
w, h = bottomRight[0] - topLeft[0], bottomRight[1] - topLeft[1]

ROW_DIVISOR = 20
COL_DIVISOR = 40

def add_blurness(frame):
    ROI = frame[y:y+h, x:x+w]
    blur = cv2.GaussianBlur(ROI, (51, 51), 0)
    frame[y:y+h, x:x+w] = blur
    return frame


class System:
    def __init__(self):
        self.capture = cv2.VideoCapture(VIDEO_PATH)
        if not self.capture.isOpened():
            raise FileNotFoundError("Video file not found.")
        self.frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.capture.get(cv2.CAP_PROP_FPS)
        if current_algo == AlgoType.b_dist_algo:
            self.algo = BDistAlgo(COL_DIVISOR, ROW_DIVISOR)
        if current_algo == AlgoType.ncc_algo:
            self.algo = Algo(width, height, g_patch_size)
        self.frameCountGlobal = 0
        self.frame_delay = frame_delay
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
                    if isIonsRecs:
                        current_frame = current_frame[0:current_frame.shape[0], 0:current_frame.shape[1] - 192]
                    current_frame = cv2.resize(current_frame, (width, height))
                    color = current_frame.copy()
                    current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    # current_frame = add_blurness(current_frame)
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
                            color[res > 0] = [0, 0, 255]
                            cv2.imshow("Processed", cv2.resize(color, (640, 320)))

                    if current_mode == Mode.compute_first_last_frame:
                        if is_seq_complete:
                            prev = frame_buff.get()
                            cv2.imshow("F1", cv2.resize(to_show, (640, 320)))
                            cv2.imshow("F2", cv2.resize(prev, (640, 320)))
                            res = self.algo.process(prev, current_frame)
                            color[res > 0] = [0, 0, 255]
                            cv2.imshow("Processed", cv2.resize(color, (640, 320)))
                            frame_buff.queue.clear()

                    if current_mode == Mode.compute_with_fixed_frame:
                        if not is_seq_complete:
                            cv2.imshow("F1", cv2.resize(to_show, (640, 320)))
                            if delay_counter == 1:
                                fixed = current_frame.copy()
                            else:
                                cv2.imshow("F2", cv2.resize(fixed, (640, 320)))
                                res = self.algo.process(fixed, current_frame)
                                color[res > 0] = [0, 0, 255]
                                cv2.imshow("Processed", cv2.resize(color, (640, 320)))
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
                    org=(10, 10),
                    color=(255, 255, 255),
                    thickness=1,
                    fontFace=cv2.QT_FONT_NORMAL, fontScale=0.3,
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
        # prev_frame_edgy = cv2.Canny(prev_frame, 100, 200)
        # frame_edgy = cv2.Canny(frame, 100, 200)
        mask = self.get_ncc_mask_cv2(prev_frame, frame)
        cv2.accumulateWeighted(mask, self.accumulated_mask, 1)
        cv2.convertScaleAbs(self.accumulated_mask, self.accumulated_mask)
        # ret, thresh = cv2.threshold(self.accumulated_mask, 0, 255, cv2.THRESH_BINARY)
        sum = np.sum(self.accumulated_mask)
        status = ""
        if sum < thresh_clear:
            status = "clear"
        if sum > thresh_clear:
            status = "artifacts"
        # result = frame.copy()
        # result[self.accumulated_mask > 0] = [255]
        return self.accumulated_mask


class BDistAlgo:
    def __init__(self, col_divisor, row_divisor):
        self.row_divisor = row_divisor
        self.col_divisor = col_divisor

    def segment_image(self, frame):
        cell_list = []
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
        if frame_width % self.col_divisor == 0 and frame_height % self.row_divisor == 0:
            idx = 0
            for Y in range(0, frame_width, int(frame_width / self.col_divisor)):
                for X in range(0, frame_height, int(frame_height / self.row_divisor)):
                    idx += 1
                    patch = (Y, X, int(width / self.col_divisor), int(height / self.row_divisor))
                    img = frame[X:X + int(frame_width / self.col_divisor), Y:Y + int(frame_height / self.row_divisor)]
                    # cv2.putText(img, str(idx), (5, 5), fontFace=cv2.QT_FONT_NORMAL, fontScale=0.2, color=(0, 0, 0))
                    cell_list.append(img)

        elif frame_width % self.col_divisor != 0:
            print("Use another col divisor")
        elif frame_height % self.row_divisor != 0:
            print("Use another row divisor")

        return cell_list

    def process(self, prev_frame, frame):
        # segment the images in patches of patch size
        prev_frame_grid = self.segment_image(prev_frame)
        frame_grid = self.segment_image(frame)
        cmp_res = []
        c = 0
        for c in range(0, (self.row_divisor * self.col_divisor)):
            # calculate histogram of coresponding patches
            histPrev = cv2.calcHist([prev_frame_grid[c]], [0], None, [256], [0, 256], accumulate=False)
            hist = cv2.calcHist([frame_grid[c]], [0], None, [256], [0, 256], accumulate=False)
            cv2.normalize(histPrev, histPrev, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
            # calculate the Bhattacharyya distance for every pair of patches
            comp_res = cv2.compareHist(histPrev, hist, method=cv2.HISTCMP_BHATTACHARYYA)
            cmp_res.append(comp_res)

        result = np.zeros((self.row_divisor, self.col_divisor))
        arr = np.asarray(cmp_res)
        result = np.reshape(arr, (self.row_divisor, self.col_divisor))

        cv2.normalize(result, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
        # if dist small => similar else different
        # img_arr = np.split(np.array(frame_grid), 10)
        # igm = cv2.vconcat([cv2.hconcat(ims) for ims in img_arr])

        return result


if __name__ == "__main__":
    sys = System()
    sys.process()
