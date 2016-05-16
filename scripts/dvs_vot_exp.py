"""A experimental script for setting up DVS recording with VOT dataset.

This script uses Sacred from IDSIA lab to setup the experiment.
This allows me to configure experiment by JSON file.

Author: Yuhuang Hu
Email : duugyue100@gmail.com
"""

from sacred import Experiment

import os
import cPickle as pickle
import numpy as np
import cv2
from spikefuel import tools, gui, helpers

exp = Experiment("DVS Recording - VOT")

exp.add_config({
    "vot_dir": "",
    "vot_stats_path": "",
    "recording_save_path": "",
    "viewer_id": 1,
    "screen_height": 0,
    "screen_width": 0,
    "work_win_scale": 0.9,
    "bg_color": [255, 0, 0],
    "fps": 0
})


@exp.automain
def dvs_vot_exp(vot_dir,
                vot_stats_path,
                recording_save_path,
                viewer_id,
                screen_height,
                screen_width,
                work_win_scale,
                bg_color,
                fps):
    """Setup an experiment for VOT dataset.

    Parameters
    ----------
    vot_dir : string
        absolute path of VOT dataset
        e.g. /home/user/vot2015
    vot_stats_path : string
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
    if not os.path.exists(str(recording_save_path)):
        os.mkdir(str(recording_save_path))
    # Load VOT stats
    f = file(vot_stats_path, mode="r")
    vot_stats = pickle.load(f)
    f.close()

    vot_list = vot_stats['vot_list']
    num_frames = vot_stats['num_frames']

    # Load groundtruth and image lists
    print "[MESSAGE] Loading image lists."
    lists = []
    for i in xrange(len(num_frames)):
        list_path = os.path.join(vot_dir, vot_list[i])
        temp_list = tools.create_vot_image_list(list_path, num_frames[i])
        lists.append(temp_list)
    print "[MESSAGE] Ground truths and image lists are loaded."

    # Create full background
    background = (np.ones((screen_height,
                          screen_width, 3))*bg_color).astype(np.uint8)

    # Setup OpenCV display window
    window_title = "DVS-VOT-EXP"
    cv2.namedWindow(window_title, cv2.WND_PROP_FULLSCREEN)

    # Experiment setup calibration
    # Not without tuning images
    swin_h, swin_w = helpers.calibration(win_h=screen_height,
                                         win_w=screen_width,
                                         scale=work_win_scale,
                                         window_title=window_title,
                                         bg_color=bg_color)

    # Init a general UDP socket
    s = tools.init_dvs()
    tools.reset_dvs_time(s)
    for k in xrange(len(num_frames)):
        print "[MESSAGE] Display video sequence %i" % (k+1)
        frames = []
        for i in xrange(num_frames[k]):
            frames.append(cv2.imread(lists[k][i]))

        new_frames = gui.rescale_image_sequence(frames, swin_h, swin_w,
                                                bg_color)
        new_frames = gui.create_border_sequence(new_frames,
                                                screen_height, screen_width,
                                                bg_color)
        cv2.imshow(window_title, new_frames[0])
        print "[MESSAGE] Adapting video sequence %i" % (k+1)
        cv2.waitKey(delay=2000)
        tools.start_log_dvs(s, recording_save_path, vot_list[k], viewer_id)
        for i in xrange(num_frames[k]):
            cv2.imshow(window_title, new_frames[i])
            key = cv2.waitKey(delay=int(1000/fps)) & 0xFF
            if key == 27:
                cv2.destroyAllWindows()
                quit()

        cv2.imshow(window_title, new_frames[-1])
        tools.stop_log_dvs(s, viewer_id)
        print "[MESSAGE] Releasing video sequence %i" % (k+1)
        cv2.waitKey(delay=2000)
        cv2.imshow(window_title, background)
        cv2.waitKey(delay=1000)
        tools.reset_dvs_time(s)
        print "[MESSAGE] Video sequence %i is logged." % (k+1)
    # Destory both scoket and opencv window
    tools.destroy_dvs(s)
    cv2.destroyAllWindows()
