import numpy as np
import cv2 as cv
import glob


if __name__ == "__main__":

    image_file_names = glob.glob("./CUT_PNG_FOLDER/*.png")
    image_count = 0
    image_file_names.sort()
    resolution = 256
    calculation = 60000 - (int(60000/resolution) * resolution)
    new_img = np.zeros(calculation, calculation, 3))
    y = 0
    x = 0
    num_images = len(image_file_names)
    while image_count < num_images:
        file_name = image_file_names[image_count]
        img = cv.imread(file_name)
        new_img[y: y + 256, x: x + 256] = img
        if x == 59648:
            x = 0
            y = y + 256
        else:
            x = x + 256
        image_count = image_count + 1
    cv.imwrite("segmented_image.png", new_img)
