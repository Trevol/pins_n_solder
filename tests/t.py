import numpy as np
import cv2

import utils.keyboard_keys as keyboard_keys


def putPatch(img, patch, xy):
    xy = np.asarray(xy)
    imgH, imgW = img.shape[:2]
    patchH, patchW = patch.shape[:2]

    patchPt1 = np.clip(-xy, [0, 0], [patchW, patchH])
    patchPt2 = np.clip(-xy + [imgW, imgH], [0, 0], [patchW, patchH])
    (patchX1, patchY1), (patchX2, patchY2) = patchPt1, patchPt2
    resultPatch = patch[patchY1:patchY2, patchX1:patchX2]
    resultPatchH, resultPatchW = resultPatch.shape[:2]

    imgPt1 = np.clip(xy, [0, 0], [imgW, imgH])
    imgPt2 = np.clip(imgPt1 + [resultPatchW, resultPatchH], [0, 0], [imgW, imgH])

    # --------------------------------------
    (imgX1, imgY1), (imgX2, imgY2) = imgPt1, imgPt2

    # print((imgY1, imgY2), (imgX1, imgX2), ' << ', (patchY1, patchY2), (patchX1, patchX2))

    img[imgY1:imgY2, imgX1:imgX2] = resultPatch
    return img


def main():
    controls = dict(zip(
        [ord('w'), ord('s'), ord('a'), ord('d'),
         keyboard_keys.UpArrow, keyboard_keys.DownArrow, keyboard_keys.LeftArrow, keyboard_keys.RightArrow],
        np.int8([[0, -1], [0, 1], [-1, 0], [1, 0]] * 2) * 5  # unitVector * magnitude
    ))

    img = np.zeros((200, 400, 3), np.uint8)
    patch = np.zeros((50, 50, 3), np.uint8) + (0, 150, 0)  # broadcast color to black image

    patchPos = np.array([11, 5])
    while True:
        patchedImg = putPatch(img.copy(), patch, patchPos)
        cv2.imshow('uint', patchedImg)
        key = keyboard_keys.wait(None, 27, *controls)
        if key == 27:
            break
        stepVector = controls[key]
        patchPos = patchPos + stepVector


if __name__ == '__main__':
    main()
