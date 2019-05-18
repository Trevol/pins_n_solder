import numpy as np
import cv2
from utils.video_capture import VideoCapture, putFramePos
from utils.video_controller import VideoController
from utils.cv_named_window import CvNamedWindow as Wnd, imshow
from tracking.realworld.video_sources import video_6, video_6_work_area_markers
from utils.workarea_view import WorkAreaView, NonSlicingWorkAreaView
from utils.image_resizer import resize


def roiForVideo6(frame):
    x1, y1 = (360, 109)
    x2, y2 = (1530, 833)
    return frame[y1:y2 + 1, x1:x2 + 1]


def debugDiff(frame0, frame1):
    colorDiff = cv2.absdiff(frame0, frame1)
    grayDiff = cv2.cvtColor(colorDiff, cv2.COLOR_BGR2GRAY)
    val, binaryDiff = cv2.threshold(grayDiff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU) # 11
    print('grayDiff', np.min(grayDiff), np.max(grayDiff), np.mean(grayDiff), np.std(grayDiff), np.var(grayDiff))
    print('OTSU', val)
    print('-----------------')
    return grayDiff, binaryDiff


def main():
    video = VideoCapture(video_6)
    vc = VideoController(1, state='pause')
    # workarea = NonSlicingWorkAreaView(video_6_work_area_markers, video.size())
    workarea = WorkAreaView(video_6_work_area_markers)
    wnd = Wnd('frame')

    frames = video.frames()
    # frames = ((f, f) for f in frames)  # like skip_non_area result
    frames = workarea.skip_non_area(frames, denoise=True)

    prevFrame, prevOriginalFrame = next(frames)
    for frame, originalFrame in frames:
        pos = video.frame_pos() - 1
        wnd.imshow(putFramePos(frame.copy(), pos))
        grayDiff, binaryDiff = debugDiff(roiForVideo6(prevFrame), roiForVideo6(frame))
        imshow(grayDiff=resize(grayDiff, .5), binaryDiff=resize(binaryDiff, .5))
        # debugDiff(roiForVideo6(prevOriginalFrame), roiForVideo6(originalFrame))
        print('################################# ', pos)
        if vc.wait_key() in (27, -1):
            break
        prevFrame, prevOriginalFrame = frame, originalFrame


if __name__ == '__main__':
    main()
