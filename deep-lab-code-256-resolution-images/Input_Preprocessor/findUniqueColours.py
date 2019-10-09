import cv2 as cv
import numpy


if __name__ == "__main__":
    img = cv.imread("/home/vignesh/PycharmProjects/SEGNET/LabelsRaw/I1_C1.png")
    print (numpy.unique(img.reshape(-1, img.shape[2]), axis=0))
