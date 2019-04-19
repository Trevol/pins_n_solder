import numpy as np
import cv2


def cvHsvToBgr(hsv):
    img = np.reshape(np.uint8(hsv), (1, 1, -1))
    return cv2.cvtColor(img, cv2.COLOR_HSV2BGR, dst=img)[0, 0]


def makeBgrColors(n, s=100, v=100):
    return [cvHsvToBgr([(i * 180) // (n - 1), s, v]) for i in range(n)]
