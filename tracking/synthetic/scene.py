import cv2
import numpy as np

from utils import keyboard_keys


def noisyImage(shape, colorMean, colorStd):
    normal = np.random.normal(colorMean, colorStd, shape)
    normal = np.round(normal, 0, out=normal)
    # normal = np.abs(normal, out=normal).astype(np.uint8)
    normal = np.clip(normal, 0, 255, out=normal)
    return normal.astype(np.uint8)


def noisyBackground(shape):
    mean = [132.967, 137.186, 140.115]
    std = [4.609, 4.963, 4.466]
    return noisyImage(shape, mean, std)


def drawNoisyCircle(img, center, r, colorMean, colorStd):
    mask = cv2.circle(np.zeros_like(img), center, r, (255, 255, 255), -1)
    inverseMask = cv2.bitwise_not(mask)
    img = cv2.bitwise_and(img, inverseMask, dst=img)  # img with black hole
    colorObj = noisyImage(mask.shape, colorMean, colorStd)
    colorObj = np.bitwise_and(mask, colorObj, out=colorObj, dtype=np.uint8, casting='unsafe')
    img = cv2.bitwise_or(img, colorObj, dst=img)
    return img


def throttleKeyEvents(delay=1, iterations=1):
    for _ in range(iterations):
        cv2.waitKey(delay)


class UnitVector:
    (Up, Down, Left, Right) = np.int8([[0, -1], [0, 1], [-1, 0], [1, 0]])
    (UpLeft, UpRight, DownLeft, DownRight) = (Up + Left, Up + Right, Down + Left, Down + Right)


def controlledFrames(bg, r, center, step):
    controls = dict([
        (ord('w'), UnitVector.Up * step),
        (ord('s'), UnitVector.Down * step),
        (ord('a'), UnitVector.Left * step),
        (ord('d'), UnitVector.Right * step),
        (ord('8'), UnitVector.Up * step),
        (ord('2'), UnitVector.Down * step),
        (ord('4'), UnitVector.Left * step),
        (ord('6'), UnitVector.Right * step),
        (keyboard_keys.UpArrow, UnitVector.Up * step),
        (keyboard_keys.DownArrow, UnitVector.Down * step),
        (keyboard_keys.LeftArrow, UnitVector.Left * step),
        (keyboard_keys.RightArrow, UnitVector.Right * step),

        (ord('7'), UnitVector.UpLeft * step),
        (ord('9'), UnitVector.UpRight * step),
        (ord('1'), UnitVector.DownLeft * step),
        (ord('3'), UnitVector.DownRight * step),
    ])

    center = np.asarray(center)
    aMin, aMax = [-r - 1, -r - 1], [bg.shape[1] + r + 1, bg.shape[0] + r + 1]
    np.clip(center, aMin, aMax, center)  # clip boundaries

    # yield bg.copy()
    while True:
        frame = drawNoisyCircle(bg.copy(), tuple(center), r, [0, 150, 0], [3, 4, 1])
        yield frame
        throttleKeyEvents()  # small delay for window refresh
        key = keyboard_keys.wait(None, 27, *controls)
        if key in (27, -1):
            yield None
            break
        center = np.add(center, controls[key], out=center)  # center = center + stepVector
        np.clip(center, aMin, aMax, center)  # clip boundaries

# def controlledFrames__(bg, r, center, step):
#     controls = dict(zip(
#         [ord('w'), ord('s'), ord('a'), ord('d'),
#          keyboard_keys.UpArrow, keyboard_keys.DownArrow, keyboard_keys.LeftArrow, keyboard_keys.RightArrow],
#         np.int8([[0, -1], [0, 1], [-1, 0], [1, 0]] * 2) * step  # unitVector * step(magnitude of vector)
#     ))
#     center = np.asarray(center)
#     aMin, aMax = [-r - 1, -r - 1], [bg.shape[1] + r + 1, bg.shape[0] + r + 1]
#     np.clip(center, aMin, aMax, center)  # clip boundaries
#     index = 0
#     # yield bg.copy()
#     while True:
#         frame = drawNoisyCircle(bg.copy(), tuple(center), r, [0, 150, 0], [3, 4, 1])
#         cv2.imshow('frame', frame)
#         throttleKeyEvents()  # small delay for window refresh
#         index += 1
#         cv2.setWindowTitle('frame', f'frame {index}')
#         key = keyboard_keys.wait(None, 27, *controls)
#         if key in (27, -1):
#             yield None
#             break
#         center = np.add(center, controls[key], out=center)  # center = center + stepVector
#         np.clip(center, aMin, aMax, center)  # clip boundaries
#         yield frame
