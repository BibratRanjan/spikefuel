"""Create DVS figures for examples.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
from os.path import join
import h5py
import cPickle as pickle
import numpy as np
import cv2
from spikefuel import dvsproc, gui, tools, helpers


def check_folder_path(path):
    """Check folder path, if not existed, then create it."""
    if not os.path.isdir(path):
        os.mkdir(path)

# options:
# VOT DVS Dataset figures         : "vot-dvs-figure"
# VOT Datset figures              : "vot-figure"
# TrackingDataset DVS figures     : "tracking-dvs-figure"
# TrackingDataset figures         : "tracking-figure"
# UCF-50 DVS Dataset figures      : "ucf50-dvs-figure"
# UCF-50 Dataset figures          : "ucf50-figure"
# Caltech-256 DVS Dataset figures : "caltech256-dvs-figure"

option = "vot-dvs-figure"
data_path = os.environ["SPIKEFUEL_DATA"]
stats_path = os.path.join(data_path, "sf_data")

if option == "vot-dvs-figure":
    vot_fn = "INI_VOT_30fps_20160424.hdf5"
    vot_path = os.path.join(data_path, vot_fn)
    vot_db = h5py.File(vot_path, mode="r")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']
    vidseq = vot_list[1]

    seq_save_path = join(data_path, "vot_dvs_figs")
    check_folder_path(seq_save_path)
    num_frames = int(vot_db[vidseq].attrs["num_frames"])

    timestamps = vot_db[vidseq]["timestamps"][()]
    x_pos = vot_db[vidseq]["x_pos"][()]
    y_pos = vot_db[vidseq]["y_pos"][()]
    pol = vot_db[vidseq]["pol"][()]
    bounding_box = vot_db[vidseq]["bounding_box"][()]
    gt = bounding_box[:, 1:]
    gt = np.reshape(gt, (gt.shape[0], 4, 2))
    (timestamps, x_pos,
     y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                           y_pos, pol, window=1000)

    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                           pol, num_frames, fs=3)

    new_frames = []
    for frame in frames:
        tmp_frame = (((frame+fs)/float(2*fs))*255).astype(np.uint8)
        new_frames.append(tmp_frame)

    rgb_frames = []
    height = new_frames[0].shape[0]
    width = new_frames[0].shape[1]
    for frame in new_frames:
        temp_frame = np.zeros((height, width, 3))
        temp_frame[:, :, 0] = frame
        temp_frame[:, :, 1] = frame
        temp_frame[:, :, 2] = frame
        rgb_frames.append(temp_frame)
    new_frames = gui.draw_poly_box_sequence(rgb_frames, gt,
                                            color=[0, 0, 255])

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print("Sequence %s is saved at %s" % (vidseq, seq_save_path))
elif option == "vot-figure":
    vot_path = os.path.join(data_path, "vot2015")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']
    no_seq = 0
    vidseq = vot_list[no_seq]
    seq_save_path = join(data_path, "vot_figs")
    check_folder_path(seq_save_path)

    list_path = join(vot_path, vidseq)
    img_list = tools.create_vot_image_list(list_path, num_frames[no_seq])
    gts = np.loadtxt(join(list_path, "groundtruth.txt"),
                     dtype=np.float32, delimiter=",")
    gts = np.reshape(gts, (gts.shape[0], 4, 2))

    print("[MESSAGE] Ground truths and image lists are loaded.")

    frames = []
    for img_name in img_list:
        frames.append(cv2.imread(img_name))

    print("[MESSAGE] Images are loaded")
    new_frames = gui.draw_poly_box_sequence(frames, gts)

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print("Sequence %s is saved at %s" % (vidseq, seq_save_path))
elif option == "tracking-dvs-figure":
    tracking_fn = "INI_TrackingDataset_30fps_20160424.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]
    pc = pl[1]
    sc = sl[pc][7]

    seq_save_path = os.path.join(data_path, "tracking_dvs_figs")
    check_folder_path(seq_save_path)
    num_frames = int(tracking_db[pc][sc].attrs["num_frames"])

    timestamps = tracking_db[pc][sc]["timestamps"][()]
    x_pos = tracking_db[pc][sc]["x_pos"][()]
    y_pos = tracking_db[pc][sc]["y_pos"][()]
    pol = tracking_db[pc][sc]["pol"][()]
    bounding_box = tracking_db[pc][sc]["bounding_box"][()]
    gt = bounding_box[:, 1:]
    gt = np.reshape(gt, (gt.shape[0], 4, 2))
    (timestamps, x_pos,
     y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                           y_pos, pol, window=1000)
    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos, pol,
                                           num_frames, fs=3)
    new_frames = []
    for frame in frames:
        tmp_frame = (((frame+fs) / float(2*fs))*255).astype(np.uint8)
        new_frames.append(tmp_frame)

    rgb_frames = []
    height = new_frames[0].shape[0]
    width = new_frames[0].shape[1]
    for frame in new_frames:
        temp_frame = np.zeros((height, width, 3))
        temp_frame[:, :, 0] = frame
        temp_frame[:, :, 1] = frame
        temp_frame[:, :, 2] = frame
        rgb_frames.append(temp_frame)
    new_frames = gui.draw_poly_box_sequence(rgb_frames, gt, color=[0, 0, 255])

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print("Sequence %s is saved at %s" % (sc, seq_save_path))
elif option == "tracking-figure":
    tracking_path = os.path.join(data_path, "TrackingDataset")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]
    pc = pl[6]
    sc = sl[pc][3]

    seq_save_path = os.path.join(data_path, "tracking_figs")
    check_folder_path(seq_save_path)
    frames = []
    for img_name in tracking_stats[sc]:
        img_path = join(tracking_path, pc, sc, img_name)
        frames.append(cv2.imread(img_path))

    gt_path = os.path.join(tracking_path, pc, sc, "groundtruth.txt")
    gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=",")
    gt = helpers.trans_groundtruth(gt, method="size")
    gt = np.reshape(gt, (gt.shape[0], 4, 2))

    print("[MESSAGE] Images are loaded")
    new_frames = gui.draw_poly_box_sequence(frames, gt)
    new_frames = gui.rescale_image_sequence(new_frames, 270, 360, [0, 0, 0])

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print("Sequence %s is saved at %s" % (sc, seq_save_path))
elif option == "ucf50-dvs-figure":
    ucf50_fn = "INI_UCF50_30fps_20160424.hdf5"
    ucf50_path = join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 10

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = "RopeClimbing"

    vid_name = ucf50_stats[cn][vid_num-1]
    vid_n, vid_ex = os.path.splitext(vid_name)
    seq_save_path = os.path.join(data_path, "ucf50_dvs_figs")
    check_folder_path(seq_save_path)
    num_frames = int(ucf50_db[cn][vid_n].attrs["num_frames"])

    timestamps = ucf50_db[cn][vid_n]["timestamps"][()]
    x_pos = ucf50_db[cn][vid_n]["x_pos"][()]
    y_pos = ucf50_db[cn][vid_n]["y_pos"][()]
    pol = ucf50_db[cn][vid_n]["pol"][()]

    (timestamps, x_pos,
     y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                           y_pos, pol, window=1000)

    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                           pol, num_frames, fs=3)

    new_frames = []
    for frame in frames:
        tmp_frame = (((frame+fs)/float(2*fs))*255).astype(np.uint8)
        new_frames.append(tmp_frame)

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print("Sequence %s is saved at %s" % (vid_name, seq_save_path))
elif option == "ucf50-figure":
    ucf50_path = join(data_path, "UCF50", "UCF50")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 10

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = "RopeClimbing"

    seq_save_path = os.path.join(data_path, "ucf50_figs")
    check_folder_path(seq_save_path)
    vid_name = ucf50_stats[cn][vid_num-1]
    frames, num_frames = helpers.read_video(join(ucf50_path, cn, vid_name))

    for i in xrange(num_frames):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, frames[i])

    print("Sequence %s is saved at %s" % (vid_name, seq_save_path))
elif option == "caltech256-dvs-figure":
    caltech_fn = "INI_Caltech256_10fps_20160424.hdf5"
    caltech_path = join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")
    img_num = 30

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()
    caltech_list = caltech_stats["caltech256_list"]
    cn = caltech_list[1]
    img_name = caltech_stats[cn][img_num-1]
    img_n, img_ex = os.path.splitext(img_name)
    num_frames = int(caltech_db[cn][img_n].attrs["num_frames"])
    timestamps = caltech_db[cn][img_n]["timestamps"][()]
    x_pos = caltech_db[cn][img_n]["x_pos"][()]
    y_pos = caltech_db[cn][img_n]["y_pos"][()]
    pol = caltech_db[cn][img_n]["pol"][()]

    (timestamps, x_pos,
     y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                           y_pos, pol, window=1000)
    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                           pol, num_frames, fs=3)
    new_frames = []
    for frame in frames:
        tmp_frame = (((frame+fs)/float(2*fs))*255).astype(np.uint8)
        new_frames.append(tmp_frame)

    seq_save_path = os.path.join(data_path, "caltech256_dvs_figs")
    check_folder_path(seq_save_path)
    for i in xrange(num_frames):
        img_name_temp = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name_temp, new_frames[i])

    print("Sequence %s is saved at %s" % (img_name, seq_save_path))
