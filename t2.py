import numpy as np
import cv2
import colorsys
from utils.color_generator import makeBgrColors, cvHsvToBgr


def colorPatch(h, w, color):
    patch = np.empty((h, w, 3), np.uint8)
    patch[..., :] = color
    return patch


def main():
    img = np.vstack([
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 256, 256)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 255, 255)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 200, 200)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 180, 180)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 140, 140)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 100, 100)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 90, 90)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 80, 80)]),
        np.hstack([colorPatch(50, 50, color) for color in makeBgrColors(15, 60, 255)]),
    ])
    cv2.imshow('uint', img)
    cv2.waitKey()

def main():
    hsvColors = [
        [i, 200, 200] for i in range(179, 360)
    ]
    for hsv in hsvColors:
        print(hsv, cvHsvToBgr(hsv), np.uint8(hsv))

    img = np.hstack([
        colorPatch(50, 50, cvHsvToBgr([179, 200, 200])),
        colorPatch(50, 50, cvHsvToBgr([178, 200, 200])),
        colorPatch(50, 50, cvHsvToBgr([182, 200, 200]))
    ])

    cv2.imshow('uint', img)
    cv2.waitKey()


if __name__ == '__main__':
    main()
