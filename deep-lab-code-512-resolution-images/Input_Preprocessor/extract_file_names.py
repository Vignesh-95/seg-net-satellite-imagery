import glob
import os
import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    image_file_names = glob.glob("/home/cs/students/u15031625/Code/models/research/deeplab/datasets/potsdam/Images/*.png")

    for count in range(len(image_file_names)):
         image_file_names[count] = os.path.basename(image_file_names[count])
    sorted_images = sorted_nicely(image_file_names)
    count = 1
    file = open("/home/cs/students/u15031625/Code/models/research/deeplab/datasets/potsdam/Index/val.txt", "w")
    for x in sorted_images:
        if count > 15488:
            file.write(x.split(".")[0] + "\n")
        count = count + 1
