import numpy as np
import cv2 as cv
import glob


if __name__ == "__main__":

    image_file_names = glob.glob("/home/vignesh/Documents/COS700/Datasets/ISPRS/Potsdam/5_Labels_all/*.tif")
    image_count = 0
    image_file_names.sort()
    for img_name in image_file_names:
        img = cv.imread(img_name)
        image_count = image_count + 1
        y = 0
        x = 0
        iterator_end = 6000 - 512
        cut_count = 0
        while y + 512 < iterator_end:
            while x + 512 < iterator_end:
                new_img = img[y:y+512, x:x+512]
                cut_count = cut_count + 1
                name = "/home/vignesh/Documents/COS700/Datasets/ISPRS/Sliced_Potsdam_RGB/Labels/I" + str(image_count)\
                       + "_C" + str(cut_count) + ".png"
                cv.imwrite(name, new_img)
                x = x + 512
            x = 0
            y = y + 512
        y = 0
    # cv.imshow('image', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
