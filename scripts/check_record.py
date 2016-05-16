"""Check if damaged recordings are stored properly.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import h5py
import cPickle as pickle
import cv2

from spikefuel import dvsproc, helpers

# For UCF-50 TESTING

error_path = "/home/inilab/data/mis_ucf50_recordings.pkl"
ucf50_data_path = "/home/inilab/data/UCF50"
db_path = "/home/inilab/data/ARCHIVE/UCF-50-ARCHIVE/UCF50_30fps_20160409.hdf5"
# error_path = "./data/mis_ucf50_recordings.pkl"

# load error list
f = file(error_path, mode="r")
error_list = pickle.load(f)
f.close()

db = h5py.File(db_path, mode="r")

for rec in error_list:
    timestamps = db[rec]["timestamps"][()]
    x_pos = db[rec]["x_pos"][()]
    y_pos = db[rec]["y_pos"][()]
    pol = db[rec]["pol"][()]
    print "[MESSAGE] Checking " + rec

    vid_path = os.path.join(ucf50_data_path, rec+".avi")
    num_frames = helpers.count_video_frames(vid_path)

    (timestamps, x_pos,
     y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                           y_pos, pol, window=1000)
    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                           pol, num_frames, fs=5)
    for frame in frames:
        cv2.imshow("test", (frame+fs)/float(2*fs))
        cv2.waitKey(delay=10)
