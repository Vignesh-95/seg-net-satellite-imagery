import numpy as np
import cv2 as cv
import glob


if __name__ == "__main__":

    image_file_names = glob.glob("/home/vignesh/Documents/COS700/Code/models/research/deeplab/datasets/potsdam/exp/train_on_trainval_set_mobilenetv2/vis/segmentation_results/*prediction.png")
    image_count = 0
    image_file_names.sort()
    new_img = np.zeros((5888, 5888, 3))
    y = 0
    x = 0
    while image_count < 484:
        file_name = image_file_names[image_count]
        img = cv.imread(file_name)
        new_img[y: y + 256, x: x + 256] = img
        if x == 5632:
            x = 0
            y = y + 256
        else:
            x = x + 256
        image_count = image_count + 1
    cv.imwrite("image.png", new_img)
