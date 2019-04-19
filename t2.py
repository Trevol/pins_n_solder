import numpy as np
import cv2
import colorsys
from utils.color_generator import makeBgrColors, cvHsvToBgr


def main():
    dst = np.full((5, 5), 2, np.float32)
    src = np.ones((5, 5), np.uint8)
    mask = np.zeros((5, 5), np.uint8)
    mask[1:4, 1:4] = 12
    cv2.accumulateWeighted(src, dst, 0.5, mask)
    print(dst)


if __name__ == '__main__':
    main()
