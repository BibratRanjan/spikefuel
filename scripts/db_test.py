"""Testing dataset stats generation.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
import numpy as np
import cPickle as pickle
import h5py
from spikefuel import dvsproc
from time import gmtime, strftime

option = "export-td-bounding-boxes"
data_path = os.environ["SPIKEFUEL_DATA"]
stats_path = os.path.join(data_path, "sf_data")

if option == "vot":
    # Load VOT Challenge Dataset
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

    avg_num_frames = np.average(np.asarray(num_frames))
    print "Average Number of Frames: %.2f" % (avg_num_frames)

    tot_t = 0.
    tot_freq = 0.
    avg_freq = 0.
    for vidseq in vot_list:
        timestamps = vot_db[vidseq]["timestamps"][()]
        tot_t += (timestamps[-1]-timestamps[0])/1e6
        event_arr = dvsproc.cal_event_count(timestamps)
        event_freq = dvsproc.cal_event_freq(event_arr, window=1000000)
        tot_freq += np.max(event_freq[:, 1])
        t = float(timestamps[-1]-timestamps[0])/1e6
        avg_freq += float(timestamps.shape[0])/float(t)
        print "Video sequence %s is processed" % (vidseq)

    print "Average Recording Length: %.2f s" % (tot_t/len(vot_list))
    print "Average Maximum Firing Rate: %.2f K" % (tot_freq/len(vot_list)/1e3)
    print "Average Firing Rate: %.2f K" % (avg_freq/len(vot_list)/1e3)

if option == "tracking":
    tracking_fn = "INI_TrackingDataset_30fps_20160424.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]

    num_videos = 0.
    tot_frames = 0.
    tot_t = 0.
    tot_freq = 0.
    avg_freq = 0.
    for pc in pl:
        # remove sequence Kalal until I got more memory
        if pc != "Kalal":
            for sc in sl[pc]:
                num_videos += 1
                tot_frames += int(tracking_db[pc][sc].attrs["num_frames"])
                timestamps = tracking_db[pc][sc]["timestamps"][()]
                tot_t += (timestamps[-1]-timestamps[0])/1e6
                event_arr = dvsproc.cal_event_count(timestamps)
                event_freq = dvsproc.cal_event_freq(event_arr, window=1000000)
                tot_freq += np.max(event_freq[:, 1])
                t = float(timestamps[-1]-timestamps[0])/1e6
                avg_freq += float(timestamps.shape[0])/float(t)
                print "Video sequence %s is processed" % (sc)

    print "Total Number of Videos: %.2f" % (num_videos)
    print "Average Number of Frames: %.2f" % (tot_frames/num_videos)
    print "Average Recording Length: %.2f s" % (tot_t/num_videos)
    print "Average Maximum Firing Rate: %.2f K" % (tot_freq/num_videos/1e3)
    print "Average Firing Rate: %.2f K" % (avg_freq/num_videos/1e3)

if option == "ucf50":
    ucf50_fn = "INI_UCF50_30fps_20160424.hdf5"
    ucf50_path = os.path.join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]

    num_videos = 0.
    tot_frames = 0.
    tot_t = 0.
    tot_freq = []
    avg_freq = []
    for cn in ucf50_list:
        for vid_name in ucf50_stats[cn]:
            vid_n, vid_ex = os.path.splitext(vid_name)
            num_videos += 1
            tot_frames += int(ucf50_db[cn][vid_n].attrs["num_frames"])
            timestamps = ucf50_db[cn][vid_n]["timestamps"][()]

            tot_t += (timestamps[-1]-timestamps[0])/1e6
            event_arr = dvsproc.cal_event_count(timestamps)
            event_freq = dvsproc.cal_event_freq(event_arr, window=1000000)
            t = float(timestamps[-1]-timestamps[0])/1e6
            tot_freq.append(np.max(event_freq[:, 1]))
            avg_freq.append(float(timestamps.shape[0])/float(t))
            print "Video sequence %s is processed" % (vid_n)

    average_freq = np.average(np.asarray(tot_freq))
    mean_freq = np.average(np.asarray(avg_freq))
    print "Total Number of Videos: %.2f" % (num_videos)
    print "Average Number of Frames: %.2f" % (tot_frames/num_videos)
    print "Average Recording Length: %.2f s" % (tot_t/num_videos)
    print "Average Maximum Firing Rate: %.2f K" % (average_freq/1e3)
    print "Average Firing Rate: %.2f K" % (mean_freq/1e3)

if option == "caltech256":
    caltech_fn = "INI_Caltech256_10fps_20160424.hdf5"
    caltech_path = os.path.join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()

    caltech_list = caltech_stats["caltech256_list"]

    num_videos = 0.
    tot_t = 0.
    tot_freq = []
    avg_freq = []
    wrong_recordings = []
    for cn in caltech_list:
        for img_name in caltech_stats[cn]:
            img_n, img_ex = os.path.splitext(img_name)
            num_videos += 1
            timestamps = caltech_db[cn][img_n]["timestamps"][()]
            if timestamps.size != 0:
                tot_t += (timestamps[-1]-timestamps[0])/1e6
                event_arr = dvsproc.cal_event_count(timestamps)
                event_freq = dvsproc.cal_event_freq(event_arr, window=1000000)
                t = float(timestamps[-1]-timestamps[0])/1e6
                tot_freq.append(np.max(event_freq[:, 1]))
                avg_freq.append(float(timestamps.shape[0])/float(t))
            else:
                wrong_recordings.append(img_n)
            print "Video sequence %s is processed" % (img_n)

    average_freq = np.average(np.asarray(tot_freq))
    mean_freq = np.average(np.asarray(avg_freq))
    print "Total Number of Videos: %.2f" % (num_videos)
    print "Average Recording Length: %.2f s" % (tot_t/num_videos)
    print "Average Maximum Firing Rate: %.2f K" % (average_freq/1e3)
    print "Average Firing Rate: %.2f K" % (mean_freq/1e3)
    print wrong_recordings

if option == "export-vot-bounding-boxes":
    vot_fn = "INI_VOT_30fps_20160424.hdf5"
    vot_path = os.path.join(data_path, vot_fn)
    vot_db = h5py.File(vot_path, mode="r")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")
    vot_gt_path = os.path.join(data_path, "vot-gt")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']
    for vidseq in vot_list:
        gt_filename = vidseq+"-groundtruth.txt"
        gt_savepath = os.path.join(vot_gt_path, gt_filename)
        header = "File is created at: "
        sys_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
        header += sys_time+"\n"
        header += "Each line is a bounding box and has 9 values.\n"
        header += "The structure is as follows:\n"
        header += "[Timestamps] [X1, Y1] [X2, Y2] [X3, Y3] [X4, Y4]"
        gt = vot_db[vidseq]["bounding_box"][()]

        np.savetxt(gt_savepath, gt, fmt='%.2f', delimiter=',', header=header)
        print "Ground Truth for %s is saved at %s" % (vidseq, gt_savepath)

if option == "export-td-bounding-boxes":
    tracking_fn = "INI_TrackingDataset_30fps_20160424.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")
    tracking_gt_path = os.path.join(data_path, "tracking-gt")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]

    for pc in pl:
        # remove sequence Kalal until I got more memory
        if pc != "Kalal":
            for sc in sl[pc]:
                pc_path = os.path.join(tracking_gt_path, pc)
                if not os.path.isdir(pc_path):
                    os.mkdir(pc_path)
                sc_path = os.path.join(pc_path, sc+"-groundtruth.txt")
                header = "File is created at: "
                sys_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
                header += sys_time+"\n"
                header += "Each line is a bounding box and has 9 values. \n"
                header += "The structure is as follows:\n"
                header += "[Timestamps] [X1, Y1] [X2, Y2] [X3, Y3] [X4, Y4]"

                gt = tracking_db[pc][sc]["bounding_box"][()]

                np.savetxt(sc_path, gt, fmt='%.2f', delimiter=',',
                           header=header)

                print "Ground Truth for %s is saved at %s" % (sc, sc_path)
