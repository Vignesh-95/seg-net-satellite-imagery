import cv2 as cv
import numpy


if __name__ == "__main__":
    img = cv.imread("/home/vignesh/Documents/COS700/Code/06SeptemberBackup/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png")
    print (numpy.unique(img.reshape(-1, img.shape[2]), axis=0))
    img2 = cv.imread("/home/vignesh/Documents/COS700/Code/06SeptemberBackup/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClass/2007_000032.png", cv.IMREAD_GRAYSCALE)
    img3 = cv.imread("/home/vignesh/Documents/COS700/Code/06SeptemberBackup/deeplab/datasets/pascal_voc_seg/VOCdevkit/VOC2012/SegmentationClassRaw/2007_000032.png")
    print (numpy.unique(img3.reshape(-1, img3.shape[2]), axis=0))
    cv.imshow("img", img)
    cv.imshow("img2", img2)
    cv.imshow("img3", img3)
    cv.waitKey(0)  # Waits for the next key to be pressed
    cv.destroyAllWindows()

