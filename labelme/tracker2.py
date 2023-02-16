import cv2
import numpy as np
import copy

class Tracker():

    def __init__(self, *args, **kwargs):
        self.shape = None

    @property
    def isRunning(self):
        return (self.shape is not None)


    def initTracker(self, qimg,shape):
        status = False
        if qimg.isNull() or not shape:
            return status
        else:
            self.shape = shape
            status = True

        return status

    def updateTracker(self, qimg, shape):

        assert (shape and shape.label == self.shape.label),"Inalid tracker state!"

        status = False

        result = shape

        if not self.isRunning or qimg.isNull():
            return result, status

        result = copy.deepcopy(shape)
        status = True

        return result , status

    def __reset__(self):
        self.shape = None

    def stopTracker(self):
        self.__reset__()
