import cv2
import numpy as np


class BufferedResizer:
    def __init__(self, originalShape, factor, buffer=None):
        shape, self.__size = self.__dimensions(originalShape, factor)
        self.__buffer = buffer or np.empty(shape, np.uint8)

    @staticmethod
    def __dimensions(originalShape, resizeFactor):
        shape = np.uint16(originalShape)
        shape[:2] = shape[:2] * resizeFactor
        size = shape[1::-1]
        return shape, tuple(size)

    def resize(self, image):
        return cv2.resize(image, self.__size, dst=self.__buffer)