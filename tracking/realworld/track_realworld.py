import numpy as np
import cv2
from utils.keyboard_keys import wait
from utils.video_capture import VideoCapture, putFramePos
from utils.video_controller import VideoController
from utils.cv_named_window import CvNamedWindow as Wnd, imshow
from tracking.realworld.video_sources import video_6, video_6_work_area_markers
from utils.image_resizer import ImageResizer
from utils.workarea_view import WorkAreaView

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
    video = VideoCapture(video_6)
    vc = VideoController(1, state='pause')
    workarea = WorkAreaView(video_6_work_area_markers)
    wnd = Wnd('frame')
    resizer = ImageResizer(video.resolution(), .5)

    frames = workarea.skip_non_area(video.frames(), denoise=False)
    prevObjectMask = resizer.resize(np.zeros(video.size(), np.uint8))

    frame = prevFrame = resizer.resize(next(frames)[0])
    wnd.imshow(putFramePos(frame.copy(), video.frame_pos() - 1))
    if vc.wait_key() in (27, -1):
        return

    for frame, _ in frames:
        frame = resizer.resize(frame)

        _r = calcObjectMask(prevObjectMask, prevFrame, frame)
        objectMask, (colorDiff, grayDiff, binaryDiff, commonPart, newDiff) = _r
        objectOnFrame = np.bitwise_and(frame, objectMask.reshape(*objectMask.shape[:2], 1))

        imshow(colorDiff=colorDiff, grayDiff=grayDiff, binaryDiff=binaryDiff, commonPart=commonPart, newDiff=newDiff,
               objectMask=objectMask, objectOnFrame=objectOnFrame)
        prevFrame = frame
        prevObjectMask = objectMask


        wnd.imshow(putFramePos(frame.copy(), video.frame_pos() - 1))
        if vc.wait_key() in (27, -1):
            break


if __name__ == '__main__':
    main()
