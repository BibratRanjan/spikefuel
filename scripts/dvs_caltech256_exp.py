"""Experiment script for setting up DVS recording with Caltech-256 dataset.

This script uses Sacred from IDSIA lab to setup the experiment.
This allows me to configure experiment by JSON file.

Author: Yuhuang Hu
Email : duguyue100@gmail.com
"""

from sacred import Experiment

import os
import cPickle as pickle
import numpy as np
import cv2

from spikefuel import tools, gui, helpers

exp = Experiment("DVS Recording - Caltech-256")

exp.add_config({
    "caltech256_dir": "",
    "caltech256_stats_path": "",
    "recording_save_path": "",
    "viewer_id": 1,
    "screen_height": 0,
    "screen_width": 0,
    "saccade_size": 0,
    "work_win_scale": 0.9,
    "bg_color": [255, 0, 0],
    "fps": 0,
    "start_class": 0
})


@exp.automain
def dvs_ucf50_exp(caltech256_dir,
                  caltech256_stats_path,
                  recording_save_path,
                  viewer_id,
                  screen_height,
                  screen_width,
                  saccade_size,
                  work_win_scale,
                  bg_color,
                  fps,
                  start_class):
    """Setup an experiment for Caltech-256 dataset.

    Parameters
    ----------
    caltech256_dir : string
        absolute path of Caltech-256 dataset
        e.g. /home/user/Caltech-256
    caltech256_stats_path : string
        path to Caltech-256 dataset stats
    recording_save_path : string
        path to logged recording data
    viewer_id : int
        the ID of jAER viewer, for Linux is 1, Mac OS X is 2
    screen_height : int
        height of the screen in pixel
    screen_width : int
        width of the screen in pixel
    saccade_size : int
        the step length of each saccade
    work_win_scale : float
        the scaling factor that calculates working window size
    bg_color : list
        background color definition
    fps : int
        frame per second while displaying the video,
        will round to closest number
    start_class : int
        select which class to start
    """
    # create data folder if not existed
    if not os.path.exists(recording_save_path):
        os.mkdir(recording_save_path)
    # Load UCF-50 stats
    f = file(caltech256_stats_path, mode="r")
    caltech256_stats = pickle.load(f)
    f.close()

    caltech256_list = caltech256_stats["caltech256_list"]
    caltech256_list = caltech256_list[(start_class-1):]

    # Create full background

    background = (np.ones((screen_height,
                          screen_width, 3))*bg_color).astype(np.uint8)

    # Setup OpenCV display window
    window_title = "DVS-CALTECH-256-EXP"
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
    for class_name in caltech256_list:
        class_path = os.path.join(recording_save_path, class_name)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        for img_name in caltech256_stats[class_name]:
            img_path = os.path.join(caltech256_dir, class_name, img_name)
            img_n, img_ex = os.path.splitext(img_name)

            frames, num_frames = gui.gen_image_frames(img_path, fps, 1)
            frames = gui.create_saccade_sequence(frames, saccade_size,
                                                 bg_color)

            new_frames = gui.rescale_image_sequence(frames, swin_h, swin_w,
                                                    bg_color)
            new_frames = gui.create_border_sequence(new_frames, screen_height,
                                                    screen_width, bg_color)

            cv2.imshow(window_title, new_frames[0])
            print "[MESSAGE] Adapting image %s" % str(img_n)
            cv2.waitKey(delay=1000)
            tools.start_log_dvs(s, recording_save_path,
                                str(class_name+"/"+img_n),
                                viewer_id)
            for frame in new_frames:
                cv2.imshow(window_title, frame)
                key = cv2.waitKey(delay=int(1000/fps)) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    quit()

            cv2.imshow(window_title, new_frames[-1])
            tools.stop_log_dvs(s, viewer_id)
            print "[MESSAGE] Releasing image %s" % str(img_n)
            cv2.waitKey(delay=1000)
            cv2.imshow(window_title, background)
            cv2.waitKey(delay=1000)
            tools.reset_dvs_time(s)
            print "[MESSAGE] Image %s is logged." % str(img_n)

    # Destory both scoket and opencv window
    tools.destroy_dvs(s)
    cv2.destroyAllWindows()
