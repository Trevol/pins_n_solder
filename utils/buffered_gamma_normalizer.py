import numpy as np


class BufferedGammaNormalizer:
    __coef = 255 / np.sqrt(255)

    def __init__(self, imageShape, floatBuffer=None, frameBuffer=None):
        self.__floatBuffer = floatBuffer or np.empty(imageShape, np.float32)
        self.__frameBuffer = frameBuffer or np.empty(imageShape, np.uint8)

    def normalize(self, src):
        floatBuffer = self.__floatBuffer
        np.copyto(floatBuffer, src, casting='unsafe')
        np.multiply(np.sqrt(floatBuffer, out=floatBuffer), self.__coef, out=floatBuffer)
        np.copyto(self.__frameBuffer, floatBuffer, 'unsafe')
        return self.__frameBuffer