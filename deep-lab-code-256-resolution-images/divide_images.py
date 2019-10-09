import numpy as np
import cv2 as cv
import glob


if __name__ == "__main__":

    image_file_names = glob.glob("./SATELLITE_PNG_FOLDER/*.png")
    image_count = 0
    image_file_names.sort()
    # TO BE CHANGED
    resolution = 256
    for img_name in image_file_names:
        img = cv.imread(img_name)
        image_count = image_count + 1
        y = 0
        x = 0
        iterator_end_y = img.shape[0] - resolution
        iterator_end_x = img.shape[1] - resolution
        cut_count = 0
        while y + resolution <= iterator_end_y:
            while x + resolution <= iterator_end_x:
                new_img = img[y:y+resolution, x:x+resolution]
                cut_count = cut_count + 1
                name = "./CUT_PNG_FOLDER/" + str(cut_count)\
                       + ".png"
                cv.imwrite(name, new_img)
                x = x + resolution
            x = 0
            y = y + resolution
