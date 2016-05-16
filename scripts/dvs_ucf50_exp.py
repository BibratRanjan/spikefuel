"""Experiment script for setting up DVS recording with UCF-50 dataset.

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

exp = Experiment("DVS Recording - UCF-50")

exp.add_config({
    "ucf50_dir": "",
    "ucf50_stats_path": "",
    "recording_save_path": "",
    "viewer_id": 1,
    "screen_height": 0,
    "screen_width": 0,
    "work_win_scale": 0.9,
    "bg_color": [255, 0, 0],
    "fps": 0
})


@exp.automain
def dvs_ucf50_exp(ucf50_dir,
                  ucf50_stats_path,
                  recording_save_path,
                  viewer_id,
                  screen_height,
                  screen_width,
                  work_win_scale,
                  bg_color,
                  fps):
    """Setup an experiment for UCF-50 dataset.

    Parameters
    ----------
    ucf50_dir : string
        absolute path of UCF-50 dataset
        e.g. /home/user/UCF50
    ucf50_stats_path : string
        path to vot dataset stats
    recording_save_path : string
        path to logged recording data
    viewer_id : int
        the ID of jAER viewer, for Linux is 1, Mac OS X is 2
    screen_height : int
        height of the screen in pixel
    screen_width : int
        width of the screen in pixel
    work_win_scale : float
        the scaling factor that calculates working window size
    bg_color : list
        background color definition
    fps : int
        frame per second while displaying the video,
        will round to closest number
    """
    # create data folder if not existed
    if not os.path.exists(recording_save_path):
        os.mkdir(recording_save_path)
    # Load UCF-50 stats
    f = file(ucf50_stats_path, mode="r")
    ucf50_stats = pickle.load(f)
    f.close()

    ucf50_list = ucf50_stats["ucf50_list"]

    # Set read video function based on platform
    read_video = helpers.read_video

    # Create full background
    background = (np.ones((screen_height,
                          screen_width, 3))*bg_color).astype(np.uint8)

    # Setup OpenCV display window
    window_title = "DVS-UCF50-EXP"
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
    for class_name in ucf50_list:
        class_path = os.path.join(recording_save_path, class_name)
        if not os.path.exists(class_path):
            os.mkdir(class_path)
        for video_name in ucf50_stats[class_name]:
            video_path = str(os.path.join(ucf50_dir, class_name, video_name))

            frames, num_frames = read_video(video_path)
            new_frames = gui.rescale_image_sequence(frames, swin_h, swin_w,
                                                    bg_color)
            new_frames = gui.create_border_sequence(new_frames, screen_height,
                                                    screen_width, bg_color)

            cv2.imshow(window_title, new_frames[0])
            print "[MESSAGE] Adapting video sequence %s" % str(video_name)
            cv2.waitKey(delay=2000)
            tools.start_log_dvs(s, recording_save_path,
                                str(class_name+"/"+video_name[:-4]),
                                viewer_id)
            for i in xrange(num_frames):
                cv2.imshow(window_title, new_frames[i])
                key = cv2.waitKey(delay=int(1000/fps)) & 0xFF
                if key == 27:
                    cv2.destroyAllWindows()
                    quit()

            cv2.imshow(window_title, new_frames[-1])
            tools.stop_log_dvs(s, viewer_id)
            print "[MESSAGE] Releasing video sequence %s" % str(video_name)
            cv2.waitKey(delay=2000)
            cv2.imshow(window_title, background)
            cv2.waitKey(delay=1000)
            tools.reset_dvs_time(s)
            print "[MESSAGE] Video sequence %s is logged." % str(video_name)

    # Destory both scoket and opencv window
    tools.destroy_dvs(s)
    cv2.destroyAllWindows()
