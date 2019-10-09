# Exit immediately if a command exits with a non-zero status.
set -e

python divide_images.py
python deeplab_demo.py
