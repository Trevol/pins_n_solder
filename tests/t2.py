import numpy as np
import cv2
import colorsys
from utils.color_generator import makeBgrColors, cvHsvToBgr
import time


def main():
    img = np.ones([4, 4, 3], dtype=np.uint8)
    ch0 = img[..., 0]
    ch1 = np.left_shift(img[..., 1], 8, dtype=np.uint16)
    ch2 = np.left_shift(img[..., 2], 16, dtype=np.uint32)
    s1 = ch0 + ch1 + ch2


if __name__ == '__main__':
    main()
