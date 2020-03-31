import cv2
import numpy as np

if __name__ == "__main__":
    img1 = cv2.imread("opencv-feature-matching-template.jpg", 0)
    img2 = cv2.imread("opencv-feature-matching-image.jpg", 0)
    sift = cv2.xfeatures2d.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    #FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches_flann = flann.knnMatch(des1, des2, k=2)
    matchesMask = [[0, 0] for i in range(len(matches_flann))]

    for i, (m, n) in enumerate(matches_flann):
        if m.distance < 0.7*n.distance:
            matchesMask[i] = [1, 0]

    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv2.DrawMatchesFlags_DEFAULT)

    img3_flann = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches_flann, None, **draw_params)
    cv2.imshow("Matches FLANN", img3_flann)

    bf = cv2.BFMatcher()

    matches = bf.knnMatch(des1, des2, k=2)

    good = []
    for m, n in matches:
        if m.distance < 0.75*n.distance:
            good.append([m])

    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow("Matches bf", img3)

    cv2.waitKey()
