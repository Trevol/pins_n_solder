import numpy as np
import cv2
from scene import noisyBackground, controlledFrames
from utils.cv_named_window import imshow, CvNamedWindow as Wnd


def notnone(iterable):
    return (item for item in iterable if item is not None)


def process(prevFrame, currentFrame):
    colorDiff = cv2.absdiff(prevFrame, currentFrame)
    grayDiff = cv2.cvtColor(colorDiff, cv2.COLOR_BGR2GRAY)
    binaryDiff = cv2.threshold(grayDiff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return colorDiff, grayDiff, binaryDiff


def calcObjectMask(prevObjectMask, prevFrame, currentFrame):
    # return
    # objectMask, (colorDiff, grayDiff, binaryDiff, commonPart, newDiff)
    colorDiff = cv2.absdiff(prevFrame, currentFrame)
    grayDiff = cv2.cvtColor(colorDiff, cv2.COLOR_BGR2GRAY)
    if not np.any(grayDiff):  # use prevObjectMask as currentObjectMask if no scene changes detected
        return prevObjectMask, (colorDiff, grayDiff, grayDiff, prevObjectMask, grayDiff)
    binaryDiff = cv2.threshold(grayDiff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    commonPart = np.where(np.subtract(prevObjectMask, binaryDiff, dtype=np.int16) == 255, np.uint8(255), np.uint8(0))
    newDiff = np.where(np.subtract(binaryDiff, prevObjectMask, dtype=np.int16) == 255, np.uint8(255), np.uint8(0))
    objectMask = np.bitwise_or(commonPart, newDiff)

    return objectMask, (colorDiff, grayDiff, binaryDiff, commonPart, newDiff)


def main():
    frameShape = [200, 400, 3]
    frames = controlledFrames(noisyBackground(frameShape), 20, [-1111, 0], 3)

    frameWnd = Wnd('frame')
    frameWnd.ensureWndCreated()

    prevObjectMask = np.zeros(frameShape[:2], np.uint8)
    frame = prevFrame = next(frames)
    frameWnd.imshow(frame)

    for frame in notnone(frames):
        frameWnd.imshow(frame)
        _r = calcObjectMask(prevObjectMask, prevFrame, frame)
        objectMask, (colorDiff, grayDiff, binaryDiff, commonPart, newDiff) = _r
        objectOnFrame = np.bitwise_and(frame, objectMask.reshape(*objectMask.shape[:2], 1))

        imshow(colorDiff=colorDiff, grayDiff=grayDiff, binaryDiff=binaryDiff, commonPart=commonPart, newDiff=newDiff,
               objectMask=objectMask, objectOnFrame=objectOnFrame)
        prevFrame = frame
        prevObjectMask = objectMask


if __name__ == '__main__':
    main()
