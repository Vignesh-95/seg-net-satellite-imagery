import glob
import os
import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


if __name__ == "__main__":
    image_file_names = glob.glob("/home/cs/students/u15031625/256_SegNet/Images/*.png")
    label_file_names = glob.glob("/home/cs/students/u15031625/256_SegNet/LabelsRaw/*.png")

    # for count in range(len(image_file_names)):
    #     image_file_names[count] = os.path.basename(image_file_names[count])
    sorted_images = sorted_nicely(image_file_names)
    sorted_labels = sorted_nicely(label_file_names)
    sorted_both = zip(sorted_images, sorted_labels)
    count = 1
    f = open("val.txt", "a") 
    for x in sorted_both:
        if count > 15488:
            f.write(x[0] + " " + x[1] + "\n")
        count = count + 1
    f.close()

