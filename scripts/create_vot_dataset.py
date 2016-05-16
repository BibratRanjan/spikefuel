"""Create HDF5 dataset for VOT Challenge Dataset.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import h5py
import numpy as np
import cv2
from spikefuel import dataset, dvsproc, helpers


def printname(name):
    """print name."""
    print name

# paths
db_name = "VOT_30fps_20160409"
save_path = "./data"
vot_data_path = "/Users/dgyHome/Downloads/vot2015/"
vot_path = "./data/vot_recordings_30fps"
vot_stats_path = "./data/vot_stats.pkl"

# reading dataset statistaic
f = file(vot_stats_path, mode="r")
vot_stats = pickle.load(f)
f.close()

vot_list = vot_stats["vot_list"]
num_frames = vot_stats['num_frames']
num_seq = len(vot_list)

bounding_boxes = {}

for i in xrange(num_seq):
    # load groundtruth
    gt_path = os.path.join(vot_data_path, vot_list[i]+"/groundtruth.txt")
    gt = np.loadtxt(gt_path, dtype=float, delimiter=",")
    gt = np.reshape(gt, (gt.shape[0], 4, 2))

    # load a frame as reference
    frame_path = os.path.join(vot_data_path, vot_list[i]+"/00000001.jpg")
    origin_frame = cv2.imread(frame_path)
    print "[MESSAGE] Loading sequence %s" % (vot_list[i])
    (timestamps, xaddr,
     yaddr, pol) = dvsproc.loadaerdat(os.path.join(vot_path,
                                                   vot_list[i]+".aedat"))
    (timestamps, xaddr,
     yaddr, pol) = dvsproc.clean_up_events(timestamps, xaddr, yaddr,
                                           pol, window=1000)
    frames, _, ts = dvsproc.gen_dvs_frames(timestamps, xaddr, yaddr,
                                           pol, num_frames[i], fs=3)
    ts = np.array(ts)

    print "[MESSAGE] Number of frames (original): %i" % (num_frames[i])
    print "[MESSAGE] Number of frames: %i" % (len(frames))
    print "[MESSAGE] Size of TS: %i" % (ts.shape[0])

    shift = helpers.cal_img_shift(origin_frame.shape, frames[0].shape)
    ratio = helpers.cal_bound_box_ratio(gt, origin_frame.shape[0],
                                        origin_frame.shape[1])
    gt = helpers.cal_bound_box_position(ratio,
                                        frames[0].shape[0]-shift[1],
                                        frames[0].shape[1]-shift[0])
    gt[:, :, 0] += shift[0]/2.
    gt[:, :, 1] += shift[1]/2.

    gt = np.reshape(gt, (gt.shape[0], 8))
    print "[MESSAGE] Size of groundtruth: "+str(gt.shape)

    gt = np.vstack((ts, gt.T)).T

    bounding_boxes[vot_list[i]] = gt


# inite dataset
dataset.create_vot_db(db_name, save_path, vot_path, vot_stats,
                      vot_data_path, bounding_boxes)

db = h5py.File(os.path.join(save_path, db_name+".hdf5"), mode="r")
db.visit(printname)
