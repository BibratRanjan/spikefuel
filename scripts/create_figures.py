"""Create figures for visualization purposes.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

import os
from os.path import join
import h5py
import cPickle as pickle
import numpy as np
from moviepy.editor import ImageSequenceClip
import cv2
import matplotlib
import matplotlib.pylab as plt

from spikefuel import dvsproc, gui, tools, helpers

# matplotlib.rcParams.update({'font.size': 100})

# options:
# "vot", "tracking", "ucf50", "caltech256"
# "caltech256-identity-wrong-files"
# "vot-ps", "tracking-ps", "ucf50-ps", "caltech256-ps"
# "event-frequency"
# "mnist-dvs", "mnist-dvs-ps", "nmnist", "ncaltech101", "ncaltech101-ps"
# "white-test"
# "vot-dvs-figure" "vot-figure" "tracking-dvs-figure" "tracking-figure"
# "ucf50-figure", "ucf50-dvs-figure"
# "gui-show"
option = "y-time-figure"
data_path = os.environ["SPIKEFUEL_DATA"]
stats_path = os.path.join(data_path, "sf_data")

if option == "ucf50-dvs-figure":
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
    seq_save_path = os.path.join(data_path, "all_imgs", "ucf50_dvs_figs")
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

    print "Sequence %s is saved at %s" % (vid_name, seq_save_path)

if option == "ucf50-figure":
    ucf50_path = join(data_path, "UCF50", "UCF50")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 10

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = "RopeClimbing"

    seq_save_path = os.path.join(data_path, "all_imgs", "ucf50_figs")
    vid_name = ucf50_stats[cn][vid_num-1]
    frames, num_frames = helpers.read_video(join(ucf50_path, cn, vid_name))

    for i in xrange(num_frames):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, frames[i])

    print "Sequence %s is saved at %s" % (vid_name, seq_save_path)

if option == "tracking-figure":
    tracking_path = os.path.join(data_path, "TrackingDataset")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]
    pc = pl[6]
    sc = sl[pc][3]
    print sc

    seq_save_path = os.path.join(data_path, "all_imgs", "tracking_figs")
    frames = []
    for img_name in tracking_stats[sc]:
        img_path = join(tracking_path, pc, sc, img_name)
        frames.append(cv2.imread(img_path))

    gt_path = os.path.join(tracking_path, pc, sc, "groundtruth.txt")
    gt = np.loadtxt(gt_path, dtype=np.float32, delimiter=",")
    gt = helpers.trans_groundtruth(gt, method="size")
    gt = np.reshape(gt, (gt.shape[0], 4, 2))

    print "[MESSAGE] Images are loaded"
    new_frames = gui.draw_poly_box_sequence(frames, gt)
    new_frames = gui.rescale_image_sequence(new_frames, 270, 360, [0, 0, 0])

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print "Sequence %s is saved at %s" % (sc, seq_save_path)

if option == "tracking-dvs-figure":
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
    print sc

    seq_save_path = os.path.join(data_path, "all_imgs", "tracking_dvs_figs")
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

    print "Sequence %s is saved at %s" % (sc, seq_save_path)

if option == "vot-figure":
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
    seq_save_path = join(data_path, "all_imgs", "vot_figs")

    list_path = join(vot_path, vidseq)
    img_list = tools.create_vot_image_list(list_path, num_frames[no_seq])
    gts = np.loadtxt(join(list_path, "groundtruth.txt"),
                     dtype=np.float32, delimiter=",")
    gts = np.reshape(gts, (gts.shape[0], 4, 2))

    print "[MESSAGE] Ground truths and image lists are loaded."

    frames = []
    for img_name in img_list:
        frames.append(cv2.imread(img_name))

    print "[MESSAGE] Images are loaded"
    new_frames = gui.draw_poly_box_sequence(frames, gts)

    for i in xrange(len(new_frames)):
        img_name = join(seq_save_path, "%08d" % (i+1,)+".png")
        cv2.imwrite(img_name, new_frames[i])

    print "Sequence %s is saved at %s" % (vidseq, seq_save_path)


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

    seq_save_path = join(data_path, "all_imgs", "vot_dvs_figs")
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

    print "Sequence %s is saved at %s" % (vidseq, seq_save_path)

if option == "white-test":
    test_path = os.path.join(data_path, "test.aedat")

    (timestamps, xaddr, yaddr, pol) = dvsproc.loadaerdat(test_path)

    event_arr = dvsproc.cal_event_count(timestamps)
    event_freq = dvsproc.cal_event_freq(event_arr, window=1000)

    plt.figure(figsize=(18, 8))
    plt.plot(event_freq[:, 0]/1e3, event_freq[:, 1], linewidth=2)
    plt.xlabel("Time (s)")
    plt.ylabel("Event Frequency")
    plt.savefig(os.path.join(data_path, "event_freq.pdf"))

    timestamps = timestamps-timestamps[0]
    timestamps = timestamps[:10000]
    tend = timestamps[-1]
    vv = np.zeros((tend+1,))
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(24, 10))
    # plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 10))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Events")
    plt.savefig(os.path.join(data_path, "white_test_ps.pdf"))

if option == "caltech256-identity-wrong-files":
    caltech_fn = "Caltech256_10fps_20160411.hdf5"
    caltech_path = os.path.join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")
    img_num = 30

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()
    caltech_list = caltech_stats["caltech256_list"]

    cn = caltech_list[62]
    img_name = caltech_stats[cn][63 - 1]
    print img_name

    img_n, img_ex = os.path.splitext(img_name)
    seq_save_path = os.path.join(data_path, "caltech256_figs_exp",
                                            img_n + ".gif")
    if not os.path.isfile(seq_save_path):
        num_frames = int(caltech_db[cn][img_n].attrs["num_frames"])
        print "Number of frames: ", num_frames
        timestamps = caltech_db[cn][img_n]["timestamps"][()]
        x_pos = caltech_db[cn][img_n]["x_pos"][()]
        y_pos = caltech_db[cn][img_n]["y_pos"][()]
        pol = caltech_db[cn][img_n]["pol"][()]

        print timestamps
        print x_pos
        print y_pos.shape
        print pol.shape

        (timestamps, x_pos,
         y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                               y_pos, pol, window=1000)

        frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                               pol, num_frames, fs=3)
        print "Length of produced frames: ", len(frames)
        new_frames = []
        for frame in frames:
            tmp_frame = (((frame+fs)/float(2*fs))*255).astype(np.uint8)
            new_frames.append(tmp_frame)

        clip = ImageSequenceClip(new_frames, fps=20)
        clip.write_gif(seq_save_path, fps=30)

        print "Sequence %s is saved at %s" % (img_name, seq_save_path)
elif option == "caltech256-ps":
    caltech_fn = "INI_Caltech256_10fps_20160424.hdf5"
    caltech_path = os.path.join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")
    caltech_save_path = os.path.join(data_path, "caltech256_ps.eps")
    img_num = 60

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()

    caltech_list = caltech_stats["caltech256_list"]
    cn = caltech_list[0]
    img_name = caltech_stats[cn][img_num-1]
    img_n, img_ex = os.path.splitext(img_name)

    timestamps = caltech_db[cn][img_n]["timestamps"][()]

    print "[MESSAGE] DATA IS LOADED."

    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(50, 45))
    # plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    # plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 2.0e-1, 0.3e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    # plt.ylabel("Events", fontsize=100)
    plt.savefig(caltech_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    print "[MESSAGE] Power Spectrum is saved at %s" % (caltech_save_path)
elif option == "ucf50-ps":
    ucf50_fn = "INI_UCF50_30fps_20160424.hdf5"
    ucf50_path = os.path.join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 11
    ucf50_save_path = os.path.join(data_path, "ucf50_ps.eps")

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = ucf50_list[0]
    vid_name = ucf50_stats[cn][vid_num-1]
    vid_n, vid_ex = os.path.splitext(vid_name)

    timestamps = ucf50_db[cn][vid_n]["timestamps"][()]

    print "[MESSAGE] DATA IS LOADED."

    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(50, 45))
    # plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    # plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 2.5e-1, 0.4e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    # plt.ylabel("Events", fontsize=100)
    plt.savefig(ucf50_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    print "[MESSAGE] Power Spectrum is saved at %s" % (ucf50_save_path)
elif option == "tracking-ps":
    tracking_fn = "INI_TrackingDataset_30fps_20160424.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")
    tracking_save_path = os.path.join(data_path, "tracking_ps.eps")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]

    pc = pl[0]
    sc = sl[pc][1]

    timestamps = tracking_db[pc][sc]["timestamps"][()]

    print "[MESSAGE] DATA IS LOADED."

    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(50, 45))
    # plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    # plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 7e-2+2e-2, 1.5e-2))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    # plt.ylabel("Events", fontsize=100)
    plt.savefig(tracking_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    print "[MESSAGE] Power Spectrum is saved at %s" % (tracking_save_path)
elif option == "vot-ps":
    vot_fn = "INI_VOT_30fps_20160424.hdf5"
    vot_path = os.path.join(data_path, vot_fn)
    vot_db = h5py.File(vot_path, mode="r")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")
    vot_save_path = os.path.join(data_path, "vot_ps.eps")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']
    vidseq = vot_list[9]
    timestamps = vot_db[vidseq]["timestamps"][()]

    print "[MESSAGE] DATA IS LOADED."

    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(50, 45))
    # plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    # plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 1.8e-1+0.3e-1, 0.3e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    # plt.ylabel("Events", fontsize=100)
    plt.savefig(vot_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    # plt.show()
    print "[MESSAGE] Power Spectrum is saved at %s" % (vot_save_path)
elif option == "event-frequency":
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
    vidseq = vot_list[2]

    timestamps = vot_db[vidseq]["timestamps"][()]

    event_arr = dvsproc.cal_event_count(timestamps)
    event_freq = dvsproc.cal_event_freq(event_arr, window=1000)

    plt.figure(figsize=(54, 24))
    plt.plot(event_freq[:, 0]/1e6, event_freq[:, 1], linewidth=10)
    plt.xlabel("Time (s)", fontsize=100)
    plt.ylabel("Event Frequency", fontsize=100)
    plt.savefig(os.path.join(data_path, "event_freq.eps"),
                format="eps", dpi=1200, bbox_inches='tight',
                pad_inches=0.5)
elif option == "mnist-dvs":
    mnist_path = os.path.join(data_path, "MNIST_DVS")

    for i in xrange(10):
        base_path = os.path.join(mnist_path, str(i))
        s4_path = os.path.join(base_path, "mnist_"+str(i)+"_scale04.aedat")
        s8_path = os.path.join(base_path, "mnist_"+str(i)+"_scale08.aedat")
        s16_path = os.path.join(base_path, "mnist_"+str(i)+"_scale16.aedat")

        for p in [s4_path, s8_path, s16_path]:
            p_n, p_ex = os.path.splitext(p)
            (timestamps, xaddr,
             yaddr, pol) = dvsproc.loadaerdat(p, camera='DVS128')
            frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, xaddr, yaddr,
                                                   pol, num_frames=10, fs=5,
                                                   platform="linux2",
                                                   device="DVS128")
            frame = ((frames[1]+fs)/float(2*fs)*256).astype(np.uint8)
            cv2.imwrite(p_n+".png", frame)
            print "[MESSAGE] Image for recording %s is generated" % p
elif option == "mnist-dvs-ps":
    mnist_path = os.path.join(data_path, "MNIST_DVS")
    mnist_save_path = os.path.join(mnist_path, "ps_mnist_dvs.pdf")
    i = 4
    base_path = os.path.join(mnist_path, str(i))
    s4_path = os.path.join(base_path, "mnist_"+str(i)+"_scale04.aedat")
    s8_path = os.path.join(base_path, "mnist_"+str(i)+"_scale08.aedat")
    s16_path = os.path.join(base_path, "mnist_"+str(i)+"_scale16.aedat")
    (timestamps, xaddr,
     yaddr, pol) = dvsproc.loadaerdat(s4_path, camera='DVS128')

    print "[MESSAGE] DATA IS LOADED."

    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(timestamps.shape[0]):
        vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f[:450]
    ff_draw = 2*np.abs(ff[:450])

    plt.figure(figsize=(18, 8))
    plt.ylim([0, 3e-3])
    plt.xlim([0, 100])
    plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(np.min(f_draw), np.max(f_draw)+1, 10))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Events")
    plt.savefig(mnist_save_path)
    # plt.show()
    print "[MESSAGE] Power Spectrum is saved at %s" % (mnist_save_path)
elif option == "nmnist":
    nmnist_path = os.path.join(data_path, "N_MNIST")

    for i in xrange(10):
        file_path = os.path.join(nmnist_path, str(i)+".bin")
        f_n, f_ex = os.path.splitext(file_path)
        print "[MESSAGE] Loading %s" % (file_path)

        file_handle = open(file_path, 'rb')
        raw_data = np.fromfile(file_handle, dtype=np.uint8)
        file_handle.close()
        raw_data = np.uint16(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | \
                 (raw_data[3::5] << 8) | (raw_data[4::5])

        frames, fs, _ = dvsproc.gen_dvs_frames(all_ts, all_x, all_y, all_p, 3,
                                               fs=3, platform="linux2",
                                               device="ATIS")
        frame = frames[1]
        frame = ((frame[:28, :28]+fs)/float(2*fs)*256).astype(np.uint8)
        cv2.imwrite(f_n+".png", frame)
        print "[MESSAGE] Image for recording %s is generated" % (file_path)
elif option == "ncaltech101":
    n_caltech_path = os.path.join(data_path, "N_Caltech101")
    for i in xrange(16):
        file_path = os.path.join(n_caltech_path,
                                 "image_"+"%04d" % (i+1,)+".bin")
        f_n, f_ex = os.path.splitext(file_path)
        print "[MESSAGE] Loading %s" % (file_path)

        file_handle = open(file_path, 'rb')
        raw_data = np.fromfile(file_handle, dtype=np.uint8)
        file_handle.close()
        raw_data = np.uint16(raw_data)
        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7
        all_ts = ((raw_data[2::5] & 127) << 16) | \
                 (raw_data[3::5] << 8) | (raw_data[4::5])

        max_y = np.max(all_y)
        max_x = np.max(all_x)

        frames, fs, _ = dvsproc.gen_dvs_frames(all_ts, all_x, all_y, all_p, 3,
                                               fs=3, platform="linux2",
                                               device="ATIS")
        frame = frames[2][:max_y, :max_x]
        frame = ((frame+fs)/float(2*fs)*256).astype(np.uint8)
        cv2.imwrite(f_n+".png", frame)
        print "[MESSAGE] Image for recording %s is generated" % (file_path)
elif option == "ncaltech101-ps":
    n_caltech_path = os.path.join(data_path, "N_Caltech101")
    n_caltech_save_path = os.path.join(n_caltech_path, "ps_ncaltech101.pdf")

    timestamps = np.array([])
    for i in xrange(100):
        file_path = os.path.join(n_caltech_path,
                                 "image_" + "%04d" % (i + 1,) + ".bin")

        print "[MESSAGE] Loading %s" % (file_path)

        file_handle = open(file_path, 'rb')
        raw_data = np.fromfile(file_handle, dtype=np.uint8)
        file_handle.close()
        raw_data = np.uint16(raw_data)
        all_ts = ((raw_data[2::5] & 127) << 16) | \
                 (raw_data[3::5] << 8) | (raw_data[4::5])

        all_ts = all_ts.astype(np.float64)
        if not timestamps.size:
            timestamps = all_ts
        else:
            # all_ts -= all_ts[0]
            all_ts += timestamps[-1]
            timestamps = np.hstack((timestamps, all_ts))

    num_data = timestamps.shape[0]
    tend = timestamps[-1]
    vv = np.zeros(tend+1)
    for i in xrange(num_data):
        if timestamps[i] < tend:
            vv[timestamps[i]] += 1

    fs = 1e6
    L = vv.shape[0]
    vv = vv - np.sum(vv)/L
    NFFT = int(2**np.ceil(np.log2(L)))
    ff = np.fft.fft(vv, NFFT)/L
    f = fs/2*(np.arange(NFFT/2)/float(NFFT/2))

    f_draw = f
    ff_draw = 2*np.abs(ff[:NFFT/2])

    plt.figure(figsize=(18, 8))
    # plt.ylim([0, 2e-5])
    plt.xlim([0, 100])
    plt.grid(True)
    plt.plot(f_draw, ff_draw, 'b', linewidth=2)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 10))
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Events")
    plt.savefig(n_caltech_save_path)
    # plt.show()
    print "[MESSAGE] Power Spectrum is saved at %s" % (n_caltech_save_path)
elif option == "vot":
    # Load VOT Challenge Dataset
    vot_fn = "VOT_30fps_20160409.hdf5"
    vot_path = os.path.join(data_path, vot_fn)
    vot_db = h5py.File(vot_path, mode="r")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']

    for vidseq in vot_list:
        seq_save_path = os.path.join(data_path, "vot_gifs", vidseq+".gif")
        if not os.path.isfile(seq_save_path):
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

            new_frames = gui.draw_poly_box_sequence(new_frames, gt,
                                                    color=[0, 255, 0])

            clip = ImageSequenceClip(new_frames, fps=20)
            clip.write_gif(seq_save_path, fps=30)

            print "Sequence %s is saved at %s" % (vidseq, seq_save_path)
elif option == "tracking":
    tracking_fn = "TrackingDataset_30fps_20160401.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]

    for pc in pl:
        # remove sequence Kalal until I got more memory
        if pc != "Kalal":
            for sc in sl[pc]:
                seq_save_path = os.path.join(data_path, "tracking_gifs",
                                             sc+".gif")

                if not os.path.isfile(seq_save_path):
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
                                                           y_pos, pol,
                                                           window=1000)

                    frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos,
                                                           y_pos, pol,
                                                           num_frames, fs=3)
                    new_frames = []
                    for frame in frames:
                        tmp_frame = (((frame+fs) /
                                     float(2*fs))*255).astype(np.uint8)
                        new_frames.append(tmp_frame)

                    new_frames = gui.draw_poly_box_sequence(new_frames, gt,
                                                            color=[0, 255, 0])

                    clip = ImageSequenceClip(new_frames, fps=20)
                    clip.write_gif(seq_save_path, fps=30)

                    print "Sequence %s is saved at %s" % (sc, seq_save_path)
elif option == "ucf50":
    ucf50_fn = "UCF50_30fps_20160409.hdf5"
    ucf50_path = os.path.join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 10

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]

    for cn in ucf50_list:
        vid_name = ucf50_stats[cn][vid_num-1]
        vid_n, vid_ex = os.path.splitext(vid_name)
        seq_save_path = os.path.join(data_path, "ucf50_gifs",
                                     vid_n+".gif")
        if not os.path.isfile(seq_save_path):
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

            clip = ImageSequenceClip(new_frames, fps=20)
            clip.write_gif(seq_save_path, fps=30)

            print "Sequence %s is saved at %s" % (vid_name, seq_save_path)
elif option == "caltech256":
    caltech_fn = "Caltech256_10fps_20160411.hdf5"
    caltech_path = os.path.join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")
    img_num = 30

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()
    caltech_list = caltech_stats["caltech256_list"]

    for cn in caltech_list:
        img_name = caltech_stats[cn][img_num-1]
        img_n, img_ex = os.path.splitext(img_name)
        seq_save_path = os.path.join(data_path, "caltech256_figs_exp",
                                     img_n+".gif")
        if not os.path.isfile(seq_save_path):
            num_frames = int(caltech_db[cn][img_n].attrs["num_frames"])
            print "Number of frames: ", num_frames
            timestamps = caltech_db[cn][img_n]["timestamps"][()]
            x_pos = caltech_db[cn][img_n]["x_pos"][()]
            y_pos = caltech_db[cn][img_n]["y_pos"][()]
            pol = caltech_db[cn][img_n]["pol"][()]

            (timestamps, x_pos,
             y_pos, pol) = dvsproc.clean_up_events(timestamps, x_pos,
                                                   y_pos, pol, window=1000)

            frames, fs, _ = dvsproc.gen_dvs_frames(timestamps, x_pos, y_pos,
                                                   pol, num_frames, fs=3)
            print "Length of produced frames: ", len(frames)
            new_frames = []
            for frame in frames:
                tmp_frame = (((frame+fs)/float(2*fs))*255).astype(np.uint8)
                new_frames.append(tmp_frame)

            clip = ImageSequenceClip(new_frames, fps=20)
            clip.write_gif(seq_save_path, fps=30)

            print "Sequence %s is saved at %s" % (img_name, seq_save_path)


if option == "gui-show":
    # Put text on screen
    image_path = os.path.join(data_path, "vot2015", "motocross1",
                              "00000106.jpg")
    frame = cv2.imread(image_path)
    win_w = 720
    win_h = 540
    scale = 0.9
    window_title = "DVS-VOT-EXP"
    bg_color = [127, 127, 127]

    message = "Experiment Setup Calibration"
    # Check if input window is 4:3
    if float(win_h)/float(win_w) != 0.75:
        raise ValueError("the input window is not in ratio 4:3")

    # get stats of smaller window
    swin_h = int(scale*win_h)
    swin_w = int(scale*win_w)
    frame = gui.rescale_image(frame, swin_h, swin_w, color=bg_color)

    window = np.ones((win_h, win_w, 3))*bg_color
    diff_y = (win_h-swin_h)/2
    diff_x = (win_w-swin_w)/2
    window[diff_y:swin_h+diff_y, diff_x:swin_w+diff_x, :] = frame
    window = np.array(window, dtype=np.uint8)

    flag = True
    while (1):
        # draw such window
        if flag is True:
            temp_win = window.copy()
            cv2.rectangle(temp_win, (diff_x, diff_y),
                          (diff_x+swin_w, diff_y+swin_h), color=[0, 255, 0],
                          thickness=2)
            flag = True
        elif flag is False:
            temp_win = window.copy()
            flag = True

        cv2.imshow(window_title, temp_win)

        k = cv2.waitKey(delay=10) & 0xFF
        if k == 27:
            break

    print "[MESSAGE] Experiment setup calibration is finished."

if option == "y-time-figure":
    ucf50_fn = "INI_UCF50_30fps_20160424.hdf5"
    ucf50_path = join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 50

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = "Drumming"

    vid_name = ucf50_stats[cn][vid_num-1]
    vid_n, vid_ex = os.path.splitext(vid_name)
    seq_save_path = os.path.join(data_path, "all_imgs", "ucf50_dvs_figs")
    num_frames = int(ucf50_db[cn][vid_n].attrs["num_frames"])

    timestamps = ucf50_db[cn][vid_n]["timestamps"][()]
    x_pos = ucf50_db[cn][vid_n]["x_pos"][()]
    y_pos = ucf50_db[cn][vid_n]["y_pos"][()]
    pol = ucf50_db[cn][vid_n]["pol"][()]

    time = timestamps[3000:4000]
    x_idx = x_pos[3000:4000]
    y_idx = y_pos[3000:4000]

    plt.figure(figsize=(30, 6))
    plt.plot(time/1e3, y_idx, ".", linewidth=2)
    plt.ylim([0, 180])
    plt.xlabel("Time (ms)")
    plt.ylabel("y")
    plt.savefig(os.path.join(data_path, "y-time-figure.png"))

    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(30, 15))
    ax = fig.gca(projection='3d')
    ax.plot(time/1e3, x_idx, y_idx, ".", linewidth=2)
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('X')
    ax.set_zlabel('Y')

    fig.savefig(os.path.join(data_path, "x-y-time-figure.png"))
