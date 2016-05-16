"""Testing of LCD monitor.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import numpy as np
import cv2
from spikefuel import helpers, gui

screen_height = 540
screen_width = 720
work_win_scale = 0.9
window_title = "LED TEST"
bg_color = [127, 127, 127]
fps = 15

cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)

background = np.ones((screen_height, screen_width, 3), dtype=np.uint8)*bg_color
background = np.array(background, dtype=np.uint8)
swin_h, swin_w = helpers.calibration(win_h=screen_height,
                                     win_w=screen_width,
                                     scale=work_win_scale,
                                     window_title=window_title,
                                     bg_color=bg_color)

# First stage test
# Flashing boxes

# from gray to white
print "[MESSAGE] Adapting"
cv2.imshow(window_title, background)
cv2.waitKey(delay=3000)

# s = tools.init_dvs()
# tools.reset_dvs_time(s)
# tools.start_log_dvs(s, "/home/inilab/data/",
#                     "led_test",
#                     1)

print "[MESSAGE] Displaying from gray to white"
frames = []
box_height = 100
box_width = 100
for color in xrange(127, 256, 8):
    frame = np.ones((box_height, box_width, 3), dtype=np.uint8)*color
    frames.append(frame)
frames = gui.create_border_sequence(frames, screen_height, screen_width,
                                    bg_color)
for frame in frames:
    cv2.imshow(window_title, background)
    cv2.waitKey(delay=int(1000/fps)*3)
    cv2.imshow(window_title, frame)
    key = cv2.waitKey(delay=int(1000/fps))
    if key == 27:
        break

print "[MESSAGE] Releasing"
cv2.imshow(window_title, background)
cv2.waitKey(delay=3000)

# from gray to black

print "[MESSAGE] Displaying from gray to black"
frames = []
for color in reversed(xrange(0, 128, 8)):
    frame = np.ones((box_height, box_width, 3), dtype=np.uint8)*color
    frames.append(frame)
frames = gui.create_border_sequence(frames, screen_height, screen_width,
                                    bg_color)
for frame in frames:
    cv2.imshow(window_title, background)
    cv2.waitKey(delay=int(1000/fps)*3)
    cv2.imshow(window_title, frame)
    key = cv2.waitKey(delay=int(1000/fps))
    if key == 27:
        break

# Second stage test
# boxes moving around
print "[MESSAGE] Adapting"
cv2.imshow(window_title, background)
cv2.waitKey(delay=3000)

# white box moving around
print "[MESSAGE] Displaying from white box moving around"
frames = []
num_frames = 1000
fps *= 4
ran_win_height = screen_height*0.6
ran_win_width = screen_width*0.6

idx_x = screen_height/2
idx_y = screen_width/2

box = np.ones((box_height, box_width, 3), dtype=np.uint8)*255

frames = []
for i in xrange(num_frames):
    frame = background.copy()
    rand_x = int(idx_x + (ran_win_width/2-box_width)*np.cos(i/5.0))
    rand_y = int(idx_y/2 + (ran_win_height/2-box_height)*np.sin(i/5.0))

    frame[rand_y:rand_y+box_height, rand_x:rand_x+box_width] = box
    frame = np.array(frame, dtype=np.uint8)
    frames.append(frame)

for frame in frames:
    cv2.imshow(window_title, frame)
    key = cv2.waitKey(delay=int(1000/fps))
    if key == 27:
        break

# black box moving around
print "[MESSAGE] Displaying from black box moving around"
box = np.zeros((box_height, box_width, 3), dtype=np.uint8)
frames = []
for i in xrange(num_frames):
    frame = background.copy()
    rand_x = int(idx_x + (ran_win_width/2-box_width)*np.cos(i/5.0))
    rand_y = int(idx_y/2 + (ran_win_height/2-box_height)*np.sin(i/5.0))

    frame[rand_y:rand_y+box_height, rand_x:rand_x+box_width] = box
    frame = np.array(frame, dtype=np.uint8)
    frames.append(frame)

for frame in frames:
    cv2.imshow(window_title, frame)
    key = cv2.waitKey(delay=int(1000/fps))
    if key == 27:
        break

print "[MESSAGE] Releasing"
cv2.imshow(window_title, background)
cv2.waitKey(delay=3000)

# tools.stop_log_dvs(s, 1)
# tools.destroy_dvs(s)
cv2.destroyAllWindows()
