"""Create HDF5 dataset for TrackingDataset.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import cPickle as pickle
import h5py
import numpy as np
import cv2
from spikefuel import dataset, helpers, dvsproc


def printname(name):
    """print name."""
    print name

# paths
db_name = "TrackingDataset"
save_path = "./data"
td_path = "/Users/dgyHome/Downloads/TrackingDataset"
tracking_path = "./data/tracking_recordings_30fps/"
tracking_stats_path = "./data/tracking_stats.pkl"

# reading dataset statistaic
f = file(tracking_stats_path, mode="r")
tracking_stats = pickle.load(f)
f.close()

# Produce bounding boxes

primary_list = tracking_stats["primary_list"]
secondary_list = tracking_stats["secondary_list"]

bounding_boxes = {}

for pc in primary_list:
    if pc != "Kalal":
        for sc in secondary_list[pc]:
            # load groundtruth
            gt_path = os.path.join(td_path, pc, sc, "groundtruth.txt")
            gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=",")
            gt = helpers.trans_groundtruth(gt, method="size")
            gt = np.reshape(gt, (gt.shape[0], 4, 2))

            # load one original frame
            frame_path = os.path.join(td_path, pc, sc, tracking_stats[sc][0])
            origin_frame = cv2.imread(frame_path)

            # loading and process DVS recordings
            print "[MESSAGE] Loading sequence %s" % (sc)
            num_frames = len(tracking_stats[sc])
            (timestamps, xaddr,
             yaddr, pol) = dvsproc.loadaerdat(os.path.join(tracking_path,
                                                           sc+".aedat"))
            (timestamps, xaddr,
             yaddr, pol) = dvsproc.clean_up_events(timestamps, xaddr, yaddr,
                                                   pol, window=1000)
            frames, _, ts = dvsproc.gen_dvs_frames(timestamps, xaddr, yaddr,
                                                   pol, num_frames, fs=3)
            ts = np.array(ts)

            print "[MESSAGE] Number of frames (original): %i" % (num_frames)
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

            bounding_boxes[sc] = gt

# inite dataset
dataset.create_tracking_db(db_name, save_path,
                           tracking_path, tracking_stats,
                           bounding_boxes)

db = h5py.File("./data/TrackingDataset.hdf5", mode="r")
db.visit(printname)
