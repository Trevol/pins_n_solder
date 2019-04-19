import numpy as np
import cv2
import utils.keyboard_keys as kbd


def noisyImage(shape, colorMean, colorStd):
    normal = np.random.normal(colorMean, colorStd, shape)
    return np.round(normal, 0, out=normal).astype(np.uint8)


def noisyBackground(shape):
    mean = [132.967, 137.186, 140.115]
    std = [4.609, 4.963, 4.466]
    return noisyImage(shape, mean, std)


def drawNoisyCircle(img, center, r, colorMean, colorStd):
    src = np.zeros_like(img)
    cv2.circle(src, center, r, colorMean, -1)
    _, mask = cv2.threshold(src[..., 0], 1, 255, cv2.THRESH_BINARY)
    dst = img.astype(np.float32)
    dst = cv2.accumulateWeighted(src, dst, 1, mask=mask)
    return dst.astype(np.uint8)
    # cv2.circle(img, center, r, 0, -1)
    # return img | mask



def syntheticClip(framesCount=None):
    r = 25
    objColor = (0, 200, 0)
    bg = noisyBackground((200, 400, 3))

    center0 = np.int16([-r - 2, 100])
    step = np.int16([5, 0])

    yield 0, bg.copy()
    for i in range(framesCount or 1000):
        center = tuple(center0 + step * i)
        yield i + 1, cv2.circle(bg.copy(), center, r, objColor, -1)


def main():
    cv2.namedWindow('clip')
    framesIter = syntheticClip()
    i, prevFrame = next(framesIter)
    for i, frame in framesIter:
        diff = cv2.absdiff(prevFrame, frame)

        cv2.imshow('diff', diff)
        cv2.imshow('clip', frame)
        cv2.setWindowTitle('clip', f'clip {i}')

        if cv2.waitKey() == 27: break
        prevFrame = frame


def main():
    img = noisyBackground([200, 400, 3])
    img = drawNoisyCircle(img, (100, 100), 50, (0, 180, 0), [2, 2, 2])
    cv2.imshow('', img)
    kbd.wait()


if __name__ == '__main__':
    main()
