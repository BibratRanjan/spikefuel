"""Recollect damaged recordings from DVS Caltech256 Experiment.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import numpy as np
import cv2

from spikefuel import tools, gui, helpers

# caltech256_dir = "/Users/dgyHome/Downloads/256_ObjectCategories"
data_path = os.environ["SPIKEFUEL_DATA"]
caltech256_dir = os.path.join(data_path, "256_ObjectCategories")
recording_save_path = os.path.join(data_path, "caltech256_recordings_10fps")
error_path = "/home/inilab/data/mis_caltech_recordings.pkl"
viewer_id = 2
screen_height = 540
screen_width = 720
saccade_size = 3
work_win_scale = 0.9
bg_color = [127, 127, 127]
fps = 10

# create data folder if not existed
if not os.path.exists(recording_save_path):
    os.mkdir(recording_save_path)
# Load UCF-50 stats
# f = file(error_path, mode="r")
# error_list = pickle.load(f)
# f.close()

error_list = ["063.electric-guitar-101/063_0063"]

# Create full background
background = (np.ones((screen_height,
                       screen_width, 3)) * bg_color).astype(np.uint8)

# Setup OpenCV display window
window_title = "DVS-CALTECH-256-EXP-EXTRA"
cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)

# Experiment setup calibration
# Not without tuning images
swin_h, swin_w = helpers.calibration(win_h=screen_height,
                                     win_w=screen_width,
                                     scale=work_win_scale,
                                     window_title=window_title,
                                     bg_color=bg_color)

# Main routine
s = tools.init_dvs()
tools.reset_dvs_time(s)
for img_name in error_list:
    img_path = os.path.join(caltech256_dir, img_name+".jpg")

    frames, num_frames = gui.gen_image_frames(img_path, fps, 1)
    frames = gui.create_saccade_sequence(frames, saccade_size, bg_color)

    new_frames = gui.rescale_image_sequence(frames, swin_h, swin_w, bg_color)
    new_frames = gui.create_border_sequence(new_frames, screen_height,
                                            screen_width, bg_color)

    cv2.imshow(window_title, new_frames[0])
    print "[MESSAGE] Adapting image %s" % str(img_name)
    cv2.waitKey(delay=1000)
    tools.start_log_dvs(s, recording_save_path, img_name,
                        viewer_id)
    for frame in new_frames:
        cv2.imshow(window_title, frame)
        key = cv2.waitKey(delay=int(1000 / fps)) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            quit()

    cv2.imshow(window_title, new_frames[-1])
    tools.stop_log_dvs(s, viewer_id)
    print "[MESSAGE] Releasing image %s" % str(img_name)
    cv2.waitKey(delay=1000)
    cv2.imshow(window_title, background)
    cv2.waitKey(delay=1000)
    tools.reset_dvs_time(s)
    print "[MESSAGE] Image %s is logged." % str(img_name)

# Destory both scoket and opencv window
tools.destroy_dvs(s)
cv2.destroyAllWindows()
