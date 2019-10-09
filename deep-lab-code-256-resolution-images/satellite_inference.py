from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import tensorflow as tf
from io import BytesIO
import cv2 as cv

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 6000
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
  return np.asarray([[0, 255, 0], [0, 255, 255], [255, 255, 255], [255, 255, 0], [0, 0, 255], [255, 0, 0]])


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


def vis_segmentation(image, seg_map):
  """Visualizes input image, segmentation map and overlay view."""
  plt.figure(figsize=(15, 5))
  grid_spec = gridspec.GridSpec(1, 4, width_ratios=[6, 6, 6, 1])

  plt.subplot(grid_spec[0])
  plt.imshow(image)
  plt.axis('off')
  plt.title('input image')

  plt.subplot(grid_spec[1])
  seg_image = label_to_color_image(seg_map).astype(np.uint8)
  plt.imshow(seg_image)
  plt.axis('off')
  plt.title('segmentation map')

  plt.subplot(grid_spec[2])
  plt.imshow(image)
  plt.imshow(seg_image, alpha=0.7)
  plt.axis('off')
  plt.title('segmentation overlay')

  unique_labels = np.unique(seg_map)
  ax = plt.subplot(grid_spec[3])
  plt.imshow(
      FULL_COLOR_MAP[unique_labels].astype(np.uint8), interpolation='nearest')
  ax.yaxis.tick_right()
  plt.yticks(range(len(unique_labels)), LABEL_NAMES[unique_labels])
  plt.xticks([], [])
  ax.tick_params(width=0.0)
  plt.grid('off')
  plt.show()


LABEL_NAMES = np.asarray([
    'Clutter', 'Trees', 'Low-vegetation', 'Building', 'Car', 'Impervious'
])

FULL_LABEL_MAP = np.arange(len(LABEL_NAMES)).reshape(len(LABEL_NAMES), 1)
FULL_COLOR_MAP = label_to_color_image(FULL_LABEL_MAP)

MODEL = DeepLabModel("./datasets/potsdam/exp/train_on_trainval_set_mobilenetv2/export/frozen_inference_graph.pb")
print('model loaded successfully!')


# Run on sample images
def run_visualization(image_path):
  """Inferences DeepLab model and visualizes result."""
  try:
    original_im = Image.open(image_path)
  except IOError:
    print('Cannot retrieve image:' + image_path)
    return

  print('running deeplab on image %s...' % image_path)
  resized_im, seg_map = MODEL.run(original_im)
  seg_image = label_to_color_image(seg_map).astype(np.uint8)

  #cv.imwrite("inference_image.png", resized_im)
  
  #resized_im.save("inference_image.png")
  #cv.imwrite("inference_seg_map.png", seg_image)
  
  resized_im.save("inference_image_satellite.png")
  cv.imwrite("inference_seg_map_satellite.png", seg_image)

  # vis_segmentation(resized_im, seg_map)

#run_visualization("./datasets/potsdam/Images/I1_C1.png")
run_visualization("./top_potsdam_2_10_RGB.png")
