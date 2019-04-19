import cv2

UpArrow = 2490368
DownArrow = 2621440
LeftArrow = 2424832
RightArrow = 2555904

def wait(delay=None, *keys):
    waitAnyKey = not any(keys)
    delay = delay or 0
    while True:
        key = cv2.waitKeyEx(delay)
        if waitAnyKey:
            return key
        if key == -1 or key in keys:
            return key