"""This tempory script is to develop GUI.

- You should mannually set screen size

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import cv2
from spikefuel import gui

# Global Parameters

data_path = os.environ["SPIKEFUEL_DATA"]
stats_path = os.path.join(data_path, "sf_data")

screen_width = 720
screen_height = 540
# base_dir = "/Users/dgyHome/Downloads/256_ObjectCategories"
base_dir = os.path.join(data_path, "256_ObjectCategories")
fps = 30
bg_color = [127, 127, 127]

# Load VOT dataset stats

f = file(os.path.join(stats_path, "caltech256_stats.pkl"), mode="r")
caltech256_stats = pickle.load(f)
f.close()

caltech256_list = caltech256_stats["caltech256_list"]
class_name = caltech256_list[2]
image_name = caltech256_stats[class_name][29]

# load image
img = cv2.imread(os.path.join(base_dir, class_name, image_name))
img_new = gui.rescale_image(img, screen_height, screen_width, color=bg_color)
cv2.imwrite(os.path.join(data_path, "rescaled_img.png"), img_new)
