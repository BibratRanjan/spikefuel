"""This script carries out analyis that exposes hidden property of recordings.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import cPickle as pickle

import numpy as np
from spikefuel import dvsproc, helpers, gui
import cv2

tracking_dir = "/Users/dgyHome/Downloads/TrackingDataset/"
tracking_stats_path = "./data/tracking_stats.pkl"
save_path = "./data/tracking_recordings_30fps/"
image_save_path = "./data/tracking_dvs_frames/"

# Load VOT stats
f = file(tracking_stats_path, mode="r")
tracking_stats = pickle.load(f)
f.close()

primary_list = tracking_stats["primary_list"]
pcg = primary_list[0]
scg = tracking_stats["secondary_list"][pcg][0]

# Load groundtruth and image lists
print "[MESSAGE] Loading ground truth"
groundtruth = np.loadtxt(tracking_dir+pcg+"/"+scg+"/groundtruth.txt",
                         dtype=np.float, delimiter=",")
gt = helpers.trans_groundtruth(groundtruth)
gt = np.reshape(gt, (gt.shape[0], 4, 2))
print "[MESSAGE] Ground truths and image lists are loaded."

base_path = tracking_dir+pcg+"/"+scg+"/"
origin_frame = cv2.imread(base_path+tracking_stats[scg][0])
num_frames = len(tracking_stats[scg])

(timestamps, xaddr, yaddr, pol) = dvsproc.loadaerdat(
                                            save_path+scg+".aedat")

(timestamps, xaddr, yaddr, pol) = dvsproc.clean_up_events(timestamps, xaddr,
                                                          yaddr, pol,
                                                          window=1000)

frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, xaddr, yaddr,
                                       pol, num_frames, fs=3)

print len(frames)

shift = helpers.cal_img_shift(origin_frame.shape, frames[0].shape)
ratio = helpers.cal_bound_box_ratio(gt, origin_frame.shape[0],
                                    origin_frame.shape[1])
new_pts = helpers.cal_bound_box_position(ratio,
                                         frames[0].shape[0]-shift[1],
                                         frames[0].shape[1]-shift[0])

new_pts[:, :, 0] += shift[0]/2.
new_pts[:, :, 1] += shift[1]/2.

new_frames = gui.draw_poly_box_sequence(frames, new_pts, color=[0, 255, 0])

# for i in xrange(len(new_frames)):
#     new_frames[i] = np.array(new_frames[i]/float(fs)*255., dtype=np.uint8)
#
# tools.create_image_sequence(new_frames, image_save_path, scg, form=".png")

for frame in new_frames:
    cv2.imshow("frame", (frame)/float(fs))
    key = cv2.waitKey(delay=0)

cv2.destroyAllWindows()
