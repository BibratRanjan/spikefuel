"""A experimental script for setting up DVS recording with Tracking dataset.

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

exp = Experiment("DVS Recording - TrackingDataset")

exp.add_config({
    "tracking_dir": "",
    "tracking_stats_path": "",
    "recording_save_path": "",
    "viewer_id": 1,
    "screen_height": 0,
    "screen_width": 0,
    "work_win_scale": 0.9,
    "bg_color": [255, 0, 0],
    "fps": 0
})


@exp.automain
def dvs_vot_exp(tracking_dir,
                tracking_stats_path,
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
    tracking_dir : string
        absolute path of Tracking dataset
        e.g. /home/user/vot2015
    tracking_stats_path : string
        path to tracking dataset stats
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
    f = file(tracking_stats_path, mode="r")
    tracking_stats = pickle.load(f)
    f.close()

    # primary list
    pl = tracking_stats["primary_list"]
    # secondary list
    sl = tracking_stats["secondary_list"]

    # Create full background
    background = (np.ones((screen_height,
                          screen_width, 3))*bg_color).astype(np.uint8)

    # Setup OpenCV display window
    window_title = "DVS-TRACKING-EXP"
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
    for pcg in pl:
        # remove sequence Kalal until I got more memory
        if pcg != "Kalal":
            for scg in sl[pcg]:
                print "[MESSAGE] Display video sequence "+scg
                seq_base_path = os.path.join(tracking_dir, pcg, scg)
                frames = []
                for fn in tracking_stats[scg]:
                    frames.append(cv2.imread(os.path.join(seq_base_path, fn)))

                frames = gui.rescale_image_sequence(frames, swin_h, swin_w,
                                                    bg_color)
                frames = gui.create_border_sequence(frames, screen_height,
                                                    screen_width, bg_color)
                cv2.imshow(window_title, frames[0])
                print "[MESSAGE] Adapting video sequence "+scg
                cv2.waitKey(delay=2000)
                tools.start_log_dvs(s, recording_save_path, scg, viewer_id)
                for frame in frames:
                    cv2.imshow(window_title, frame)
                    key = cv2.waitKey(delay=int(1000/fps)) & 0xFF
                    if key == 27:
                        cv2.destroyAllWindows()
                        quit()

                cv2.imshow(window_title, frames[-1])
                tools.stop_log_dvs(s, viewer_id)
                print "[MESSAGE] Releasing video sequence "+scg
                cv2.waitKey(delay=2000)
                cv2.imshow(window_title, background)
                cv2.waitKey(delay=1000)
                tools.reset_dvs_time(s)
                print "[MESSAGE] Video sequence "+scg+" is logged."
    # Destory both scoket and opencv window
    tools.destroy_dvs(s)
    cv2.destroyAllWindows()
