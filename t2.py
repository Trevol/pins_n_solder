import numpy as np
import cv2
import colorsys
from utils.color_generator import makeBgrColors, cvHsvToBgr
import time

def main():
    cnt = 10
    dtype = np.uint32
    rr = 1020//2
    cc = 1980//2
    # 1020*1980
    # np.uint32 - 3716 942 1732
    # np.int16 - 1867
    # np.uint8 - 942
    # np.float32 - 3716


    dst = np.full((cnt, rr, cc, 3), 2, dtype)
    t0 = time.time()
    for img in dst:
        for r in img:
            for c in r:
                for color in c:
                    color
    print(time.time()-t0)
    t0 = time.time()
    for img in range(cnt):
        for r in range(rr):
            for c in range(cc):
                for color in range(3):
                    dst[img, r, c, color]

    print(time.time() - t0)


if __name__ == '__main__':
    main()
