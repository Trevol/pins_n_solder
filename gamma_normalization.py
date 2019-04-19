import cv2
import numpy as np

from utils.buffered_gamma_normalizer import BufferedGammaNormalizer
from utils.buffered_resizer import BufferedResizer


def getSize(video: cv2.VideoCapture):
    h = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    w = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    return np.int16([w, h])


def setFramePos(video: cv2.VideoCapture, pos):
    video.set(cv2.CAP_PROP_POS_FRAMES, pos)


def main():
    video = cv2.VideoCapture('D:/DiskE/Computer_Vision_Task/Video_6.mp4')
    ret, prevFrame = video.read()
    if not ret:
        return

    gamma = BufferedGammaNormalizer(prevFrame.shape)
    resizer = BufferedResizer(prevFrame.shape, .5)

    prevNormalizedFrame = gamma.normalize(prevFrame)
    frame = normalizedDiff = diff = None
    while True:
        ret, frame = video.read(image=frame)
        if not ret:
            break

        patch = frame[140:852, 350:1501]
        print(np.round(np.mean(patch, axis=(0, 1)), 3))
        print(np.round(np.std(patch, axis=(0, 1)), 3))

        normalizedFrame = gamma.normalize(frame)

        normalizedDiff = cv2.absdiff(prevNormalizedFrame, normalizedFrame, dst=normalizedDiff)
        diff = cv2.absdiff(prevFrame, frame, dst=diff)

        frame__ = frame.copy()
        cv2.rectangle(frame__, (350, 140), (1500, 850), (0, 255, 0))
        cv2.imshow('frame', resizer.resize(frame__))
        cv2.imshow('normalized frame', resizer.resize(normalizedFrame))

        cv2.imshow('normalizedDiff', resizer.resize(normalizedDiff))
        cv2.imshow('diff', resizer.resize(diff))

        if cv2.waitKey() == 27:
            break
        prevNormalizedFrame = normalizedFrame.copy()
        np.copyto(prevFrame, frame, casting='no')

    video.release()


def notes():
    # как отличить тени/отблески от объекта?
    #
    #
    #
    pass


if __name__ == '__main__':
    main()
