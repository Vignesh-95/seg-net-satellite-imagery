from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import cv2 as cv
import glob
import os

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 256
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, model_name):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()
    with tf.gfile.GFile(model_name, "rb") as f:
      graph_def = tf.GraphDef.FromString(f.read())
    if graph_def is None:
      raise RuntimeError('Cannot find inference graph.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map[0]
    return resized_image, seg_map


def create_potsdam_label_colormap():
  """Creates a label colormap used in potsdam segmentation benchmark.

  Returns:
    A colormap for visualizing segmentation results.
  """
  return np.asarray([[0, 0, 255], [0, 255, 0], [255, 255, 0], [255, 0, 0], [0, 255, 255], [255, 255, 255]])


def label_to_color_image(label):
  """Adds color defined by the dataset colormap to the label.

  Args:
    label: A 2D array with integer type, storing the segmentation label.

  Returns:
    result: A 2D array with floating type. The element of the array
      is the color indexed by the corresponding element in the input label
      to the PASCAL color map.

  Raises:
    ValueError: If label is not of rank 2 or its value is larger than color
      map maximum entry.
  """
  if label.ndim != 2:
    raise ValueError('Expect 2-D input label')

  colormap = create_potsdam_label_colormap()

  if np.max(label) >= len(colormap):
    raise ValueError('label value too large.')

  return colormap[label]


LABEL_NAMES = np.asarray([
    'Clutter', 'Trees', 'Low-vegetation', 'Building', 'Car', 'Impervious'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL = DeepLabModel("./datasets/potsdam/exp/train_on_trainval_set_mobilenetv2/export/frozen_inference_graph.pb")
print('model loaded successfully!')


# Run on sample images
def run_visualization(image_path, c_count):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(image_path)
  except IOError:
    print('Cannot retrieve image:' + image_path)
    return

  #print('running deeplab on image %s...' % image_path)
  resized_im, seg_map = MODEL.run(original_im)
  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  # resized_im.save("inference_image_satellite.png")
  # cv.imwrite("./Segmented_Images/"  + str(c_count) + ".png", seg_image)
  return seg_image

if __name__ == "__main__":
  image_file_names = glob.glob("./SATELLITE_PNG_FOLDER/*.png")
  image_count = 0
  image_file_names.sort()
  # TO BE CHANGED
  resolution = 256
  for img_name in image_file_names:
    img = cv.imread(img_name)
    # TO BE CHANGED
    calculation = int(6000/resolution) * resolution
    new_img = np.zeros((calculation, calculation, 3), dtype=np.uint8)
    image_count = image_count + 1
    y = 0
    x = 0
    iterator_end_y = img.shape[0] - resolution
    iterator_end_x = img.shape[1] - resolution
    cut_count = 0
    while y + resolution <= iterator_end_y:
      while x + resolution <= iterator_end_x:
        new = img[y:y+resolution, x:x+resolution]
        cut_count = cut_count + 1
        name = "./CUT_PNG_FOLDER/Image.png"
        cv.imwrite(name, new)
        seg = run_visualization("./CUT_PNG_FOLDER/Image.png", cut_count)
        new_img[y:y+resolution, x:x+resolution] = seg
        x = x + resolution
      x = 0
      y = y + resolution
    cv.imwrite("Segmented_" + os.path.basename(img_name), new_img)
