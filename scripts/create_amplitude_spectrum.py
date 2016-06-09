"""Amplitude Spetrum produced in the paper.

DVS Benchmark Datasets for Object Tracking, Action Recognition and Object
Recognition

Please set your environment variable SPIKEFUEL_DATA:
export SPIKEFUEL_DATA=/path/to/data

and then place all HDF5 format data in this folder, then
create a folder `sf_data` in `/path/to/data`, and place stats file in there.
stats file can be found at:
https://github.com/duguyue100/spikefuel/tree/master/data

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from __future__ import print_function
import os
import h5py
import cPickle as pickle
import numpy as np
import matplotlib
import matplotlib.pylab as plt

matplotlib.rcParams.update({'font.size': 100})

# options
# VOT dataset amplitude spectrum    : vot-as
# TrackingDataset amplitude spectrum: traking-as
# UCF-50 amplitude spectrum         : ucf50-as
# Caltech-256 amplitude spectrum    : caltech256-as

option = "caltech256-as"
data_path = os.environ["SPIKEFUEL_DATA"]
stats_path = os.path.join(data_path, "sf_data")

if option == "vot-as":
    vot_fn = "INI_VOT_30fps_20160424.hdf5"
    vot_path = os.path.join(data_path, vot_fn)
    vot_db = h5py.File(vot_path, mode="r")
    vot_stats_path = os.path.join(stats_path, "vot_stats.pkl")
    vot_save_path = os.path.join(data_path, "vot_as.eps")

    # load vot stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()
    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']
    vidseq = vot_list[9]
    timestamps = vot_db[vidseq]["timestamps"][()]

    print("[MESSAGE] DATA IS LOADED.")

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
    plt.xlim([0, 100])
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 1.8e-1+0.3e-1, 0.3e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    plt.savefig(vot_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    print("[MESSAGE] Amplitude Spectrum is saved at %s" % (vot_save_path))
elif option == "tracking-as":
    tracking_fn = "INI_TrackingDataset_30fps_20160424.hdf5"
    tracking_path = os.path.join(data_path, tracking_fn)
    tracking_db = h5py.File(tracking_path, mode="r")
    tracking_stats_path = os.path.join(stats_path, "tracking_stats.pkl")
    tracking_save_path = os.path.join(data_path, "tracking_as.eps")

    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    pl = tracking_stats["primary_list"]
    sl = tracking_stats["secondary_list"]

    pc = pl[0]
    sc = sl[pc][1]

    timestamps = tracking_db[pc][sc]["timestamps"][()]

    print("[MESSAGE] DATA IS LOADED.")

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
    plt.xlim([0, 100])
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 7e-2+2e-2, 1.5e-2))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    plt.savefig(tracking_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    print("[MESSAGE] Amplitude Spectrum is saved at %s" % (tracking_save_path))
elif option == "ucf50-as":
    ucf50_fn = "INI_UCF50_30fps_20160424.hdf5"
    ucf50_path = os.path.join(data_path, ucf50_fn)
    ucf50_db = h5py.File(ucf50_path, mode="r")
    ucf50_stats_path = os.path.join(stats_path, "ucf50_stats.pkl")
    vid_num = 11
    ucf50_save_path = os.path.join(data_path, "ucf50_as.eps")

    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]
    cn = ucf50_list[0]
    vid_name = ucf50_stats[cn][vid_num-1]
    vid_n, vid_ex = os.path.splitext(vid_name)

    timestamps = ucf50_db[cn][vid_n]["timestamps"][()]

    print("[MESSAGE] DATA IS LOADED.")

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
    plt.xlim([0, 100])
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 2.5e-1, 0.4e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    plt.savefig(ucf50_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    print("[MESSAGE] Amplitude Spectrum is saved at %s" % (ucf50_save_path))
elif option == "caltech256-as":
    caltech_fn = "INI_Caltech256_10fps_20160424.hdf5"
    caltech_path = os.path.join(data_path, caltech_fn)
    caltech_db = h5py.File(caltech_path, mode="r")
    caltech_stats_path = os.path.join(stats_path, "caltech256_stats.pkl")
    caltech_save_path = os.path.join(data_path, "caltech256_as.eps")
    img_num = 60

    f = file(caltech_stats_path, mode="r")
    caltech_stats = pickle.load(f)
    f.close()

    caltech_list = caltech_stats["caltech256_list"]
    cn = caltech_list[0]
    img_name = caltech_stats[cn][img_num-1]
    img_n, img_ex = os.path.splitext(img_name)

    timestamps = caltech_db[cn][img_n]["timestamps"][()]

    print("[MESSAGE] DATA IS LOADED.")

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
    plt.xlim([0, 100])
    plt.plot(f_draw, ff_draw, 'b', linewidth=10)
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))
    plt.xticks(np.arange(0, 100+1, 20))
    plt.yticks(np.arange(0, 2.0e-1, 0.3e-1))
    plt.xlabel("Frequency [Hz]", fontsize=150)
    plt.savefig(caltech_save_path, format="eps", dpi=1200,
                bbox_inches='tight', pad_inches=0.5)
    print("[MESSAGE] Amplitude Spectrum is saved at %s" % (caltech_save_path))
